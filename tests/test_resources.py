"""Resource governor tests.

The load-bearing assertion is that reasoning and execution pools scale differently.
Collapsing them into one number either starves cheap remote concurrency or thrashes
the machine, and both look like "the harness is slow".
"""

from __future__ import annotations

import pytest

from hive.core.resources import (
    CHECKPOINT_BELOW_GIB,
    EXECUTION_WORKER_RAM_GIB,
    STOP_NEW_WORKERS_BELOW_GIB,
    HardwareProfile,
    Pressure,
    PressureAction,
    ResourceClass,
    ResourceGovernor,
    WorkerKind,
    detect_hardware,
    sample_pressure,
)


def _profile(cores=6, ram=16.0, vram=0.0, gpu="") -> HardwareProfile:
    p = HardwareProfile(
        os_name="test", arch="x86_64", physical_cores=cores, logical_cores=cores,
        ram_total_gib=ram, gpu_name=gpu, gpu_vram_gib=vram,
    )
    p.gpu_usable_for_inference = vram >= 8.0
    return p


# -------------------------------------------------------------- classification


def test_resource_class_by_shape():
    assert _profile(2, 8.0).resource_class is ResourceClass.TINY
    assert _profile(6, 16.0).resource_class is ResourceClass.STANDARD
    assert _profile(16, 64.0).resource_class is ResourceClass.WORKSTATION
    assert _profile(32, 128.0, vram=48.0, gpu="A6000").resource_class is ResourceClass.GPU_SERVER


def test_weak_gpu_does_not_count_as_inference_capable():
    """An old low-VRAM card advertises CUDA while being useless for modern inference.
    Treating it as capable routes work somewhere it will crawl."""
    weak = _profile(6, 16.0, vram=2.0, gpu="GT 710")
    assert weak.gpu_usable_for_inference is False
    assert weak.prefer_remote_inference is True
    assert weak.resource_class is ResourceClass.STANDARD


def test_strong_gpu_enables_local_inference():
    strong = _profile(16, 64.0, vram=24.0, gpu="RTX 4090")
    assert strong.gpu_usable_for_inference is True
    assert strong.prefer_remote_inference is False


# -------------------------------------------------------------------- limits


def test_reasoning_pool_is_larger_than_execution_pool():
    """The central structural claim: remote thinking is cheap locally, builds are not."""
    limits = ResourceGovernor(_profile(6, 16.0)).static_limits()
    assert limits.reasoning > limits.execution


def test_execution_limit_respects_ram_budget():
    """4 GiB reserved from 16 leaves 12; at 2.5 GiB per execution worker that is 4,
    but 6 cores at 2 cores each caps it at 3."""
    limits = ResourceGovernor(_profile(6, 16.0), os_reserve_gib=4.0).static_limits()
    assert limits.execution == 3


def test_tiny_machine_still_gets_at_least_one_of_each():
    """A cramped machine must remain usable rather than deadlock at zero workers."""
    limits = ResourceGovernor(_profile(2, 4.0)).static_limits()
    assert limits.reasoning >= 1
    assert limits.execution >= 1


def test_workstation_scales_execution_up_to_the_hard_cap():
    limits = ResourceGovernor(_profile(32, 128.0), hard_max_execution=4).static_limits()
    assert limits.execution == 4


def test_hard_caps_are_honoured():
    limits = ResourceGovernor(
        _profile(64, 256.0), hard_max_reasoning=3, hard_max_execution=2
    ).static_limits()
    assert limits.reasoning == 3
    assert limits.execution == 2


# ------------------------------------------------------------------ pressure


def test_healthy_pressure_proceeds():
    p = Pressure(ram_available_gib=8.0, cpu_percent=30.0, swap_percent=0.0, sampled=True)
    assert p.action is PressureAction.PROCEED
    assert p.healthy


def test_low_ram_stops_new_workers_but_does_not_kill_anything():
    """Stop issuing first. Killing a working agent discards tokens already paid for."""
    p = Pressure(ram_available_gib=STOP_NEW_WORKERS_BELOW_GIB - 0.1, sampled=True)
    assert p.action is PressureAction.NO_NEW_WORKERS


def test_very_low_ram_asks_for_a_checkpoint():
    p = Pressure(ram_available_gib=CHECKPOINT_BELOW_GIB - 0.1, sampled=True)
    assert p.action is PressureAction.CHECKPOINT_NOW


def test_swapping_suspends_because_everything_is_now_slower():
    """Once swapping starts, N workers are slower than one would have been."""
    p = Pressure(ram_available_gib=6.0, swap_percent=40.0, sampled=True)
    assert p.action is PressureAction.SUSPEND_LOWEST_VALUE


def test_pinned_cpu_stops_new_workers():
    p = Pressure(ram_available_gib=8.0, cpu_percent=95.0, sampled=True)
    assert p.action is PressureAction.NO_NEW_WORKERS


def test_unsampled_pressure_is_not_treated_as_a_crisis():
    """Failure to measure must not look like an emergency, or a machine without
    psutil would refuse to do any work at all."""
    assert Pressure().action is PressureAction.PROCEED


# --------------------------------------------------------------- live limits


def test_live_limits_narrow_execution_under_memory_pressure():
    gov = ResourceGovernor(_profile(6, 16.0))
    healthy = gov.live_limits(Pressure(ram_available_gib=10.0, sampled=True))
    squeezed = gov.live_limits(Pressure(ram_available_gib=2.0, sampled=True))
    assert squeezed.execution < healthy.execution
    assert squeezed.execution == 1
    assert squeezed.index == 0


def test_live_limits_keep_reasoning_available_under_pressure():
    """Local pressure is not a reason to stop remote thinking."""
    gov = ResourceGovernor(_profile(6, 16.0))
    squeezed = gov.live_limits(Pressure(ram_available_gib=1.0, sampled=True))
    assert squeezed.reasoning >= 1


def test_execution_slots_never_exceed_what_free_ram_can_hold():
    gov = ResourceGovernor(_profile(32, 128.0))
    limits = gov.live_limits(Pressure(ram_available_gib=EXECUTION_WORKER_RAM_GIB, sampled=True))
    assert limits.execution == 1


def test_may_start_blocks_execution_under_pressure_only():
    gov = ResourceGovernor(_profile(6, 16.0))
    squeezed = Pressure(ram_available_gib=1.0, sampled=True)
    assert gov.may_start(WorkerKind.EXECUTION, running=0, pressure=squeezed) is False
    assert gov.may_start(WorkerKind.REASONING, running=0, pressure=squeezed) is True


def test_may_start_respects_the_cap_when_healthy():
    gov = ResourceGovernor(_profile(6, 16.0))
    ok = Pressure(ram_available_gib=12.0, cpu_percent=10.0, sampled=True)
    limits = gov.live_limits(ok)
    assert gov.may_start(WorkerKind.EXECUTION, running=limits.execution - 1, pressure=ok) is True
    assert gov.may_start(WorkerKind.EXECUTION, running=limits.execution, pressure=ok) is False


# ------------------------------------------------------------------- policy


def test_policy_document_is_complete():
    pol = ResourceGovernor(_profile(6, 16.0, vram=2.0, gpu="GT 710")).policy()
    assert pol["machine"]["resource_class"] == "standard"
    assert pol["inference"]["prefer_remote"] is True
    assert "difficult_coding" in pol["inference"]["local_discouraged_for"]


def test_local_inference_discouraged_without_a_capable_gpu():
    assert ResourceGovernor(_profile(6, 16.0)).local_inference_advised() is False
    assert ResourceGovernor(_profile(16, 64.0, vram=24.0)).local_inference_advised() is True


# -------------------------------------------------------- real machine probe


def test_detection_works_on_this_machine():
    prof = detect_hardware()
    assert prof.physical_cores >= 1
    assert prof.os_name
    if prof.detected:
        assert prof.ram_total_gib > 0


def test_sampling_works_on_this_machine():
    p = sample_pressure()
    if p.sampled:
        assert p.ram_available_gib >= 0
        assert 0 <= p.ram_percent <= 100


def test_governor_produces_usable_limits_on_this_machine():
    """Whatever this machine is, the governor must yield at least one of each."""
    limits = ResourceGovernor().live_limits()
    assert limits.reasoning >= 1
    assert limits.execution >= 1
