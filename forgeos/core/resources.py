"""Hardware-aware concurrency. Detect the machine, then size the pools to it.

The structural point, and the thing a single `max_parallel` gets wrong:

    **Reasoning workers and execution workers are not the same resource.**

An API model thinking consumes almost no local CPU or RAM — the work happens on
someone else's hardware. A build, a test run, or a language server consumes a great
deal. Capping both with one number means either starving the cheap concurrency or
thrashing the machine. So there are two pools with independent limits.

The pressure ladder never kills a working agent outright. It stops *issuing* new
work first, then asks for a checkpoint, and only suspends as a last resort — killing
an agent mid-task discards tokens already paid for, which is the opposite of the goal.
"""

from __future__ import annotations

import os
import platform
import shutil
from enum import Enum

from pydantic import BaseModel

from ..diagnostics import record_degradation

# Leave this much for the OS and the operator's own editor/browser.
DEFAULT_OS_RESERVE_GIB = 4.0
# Below this, starting another local worker risks swapping — which is slower than
# running the work sequentially would have been.
STOP_NEW_WORKERS_BELOW_GIB = 3.0
# Below this, ask running workers to checkpoint so nothing is lost if we must stop.
CHECKPOINT_BELOW_GIB = 1.5

# Rough per-worker local footprints. Execution workers are the expensive kind.
EXECUTION_WORKER_RAM_GIB = 2.5
REASONING_WORKER_RAM_GIB = 0.15  # a subprocess holding a socket, essentially


class WorkerKind(str, Enum):
    REASONING = "reasoning"  # API/CLI model calls — remote compute
    EXECUTION = "execution"  # builds, tests, language servers — local compute


class ResourceClass(str, Enum):
    TINY = "tiny"
    STANDARD = "standard"
    WORKSTATION = "workstation"
    GPU_SERVER = "gpu_server"


class PressureAction(str, Enum):
    """Ordered least to most disruptive. Never skip straight to suspending."""

    PROCEED = "proceed"
    NO_NEW_WORKERS = "no_new_workers"
    CHECKPOINT_NOW = "checkpoint_now"
    SUSPEND_LOWEST_VALUE = "suspend_lowest_value"


class HardwareProfile(BaseModel):
    os_name: str = ""
    arch: str = ""
    physical_cores: int = 1
    logical_cores: int = 1
    ram_total_gib: float = 0.0
    disk_free_gib: float = 0.0
    gpu_name: str = ""
    gpu_vram_gib: float = 0.0
    gpu_usable_for_inference: bool = False
    detected: bool = False

    @property
    def resource_class(self) -> ResourceClass:
        if self.gpu_usable_for_inference and self.gpu_vram_gib >= 24 and self.ram_total_gib >= 64:
            return ResourceClass.GPU_SERVER
        if self.ram_total_gib >= 64 and self.physical_cores >= 16:
            return ResourceClass.WORKSTATION
        if self.ram_total_gib >= 15 and self.physical_cores >= 6:
            return ResourceClass.STANDARD
        return ResourceClass.TINY

    @property
    def prefer_remote_inference(self) -> bool:
        """A weak or absent GPU means local LLM inference is false economy.

        Free tokens that take ten times as long, and steal the CPU a build needs,
        cost more than the API call they replaced.
        """
        return not self.gpu_usable_for_inference


def detect_hardware() -> HardwareProfile:
    """Best-effort detection. Never raises — an undetected machine still has to run."""
    prof = HardwareProfile(
        os_name=f"{platform.system()} {platform.release()}".strip(),
        arch=platform.machine(),
        logical_cores=os.cpu_count() or 1,
        physical_cores=os.cpu_count() or 1,
    )
    try:
        import psutil

        prof.physical_cores = psutil.cpu_count(logical=False) or prof.logical_cores
        prof.logical_cores = psutil.cpu_count() or prof.logical_cores
        prof.ram_total_gib = round(psutil.virtual_memory().total / 2**30, 2)
        prof.detected = True
    except Exception as exc:
        record_degradation(
            "resources", "psutil hardware detection failed", exc,
            consequence="hardware profile uses defaults -- resource_class and pool sizing may be wrong for this machine",
        )

    try:
        prof.disk_free_gib = round(shutil.disk_usage(os.getcwd()).free / 2**30, 2)
    except Exception as exc:
        record_degradation(
            "resources", "disk usage probe failed", exc,
            consequence="disk_free_gib defaults to 0 -- disk-based checks see no free space",
        )

    prof.gpu_name, prof.gpu_vram_gib = _detect_gpu()
    # A GPU only counts if it can actually hold a useful model. Old low-VRAM cards
    # advertise CUDA support while being useless for modern inference, and treating
    # them as capable is how a scheduler routes work somewhere it will crawl.
    prof.gpu_usable_for_inference = prof.gpu_vram_gib >= 8.0
    return prof


def _detect_gpu() -> tuple[str, float]:
    """Query nvidia-smi if present. Absence of a GPU is normal, not an error."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "", 0.0
    try:
        import subprocess

        out = subprocess.run(  # noqa: S603 - fixed binary, no shell
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        first = (out.stdout or "").strip().splitlines()
        if not first:
            return "", 0.0
        name, mib = (p.strip() for p in first[0].split(",")[:2])
        return name, round(float(mib) / 1024, 2)
    except Exception:
        return "", 0.0


class Pressure(BaseModel):
    """A live sample. Cheap enough to take every few seconds."""

    ram_available_gib: float = 0.0
    ram_percent: float = 0.0
    cpu_percent: float = 0.0
    swap_percent: float = 0.0
    sampled: bool = False

    @property
    def action(self) -> PressureAction:
        if self.ram_available_gib and self.ram_available_gib < CHECKPOINT_BELOW_GIB:
            return PressureAction.CHECKPOINT_NOW
        if self.swap_percent > 25.0:
            # Swapping means every worker is now slower than one worker would be.
            return PressureAction.SUSPEND_LOWEST_VALUE
        if self.ram_available_gib and self.ram_available_gib < STOP_NEW_WORKERS_BELOW_GIB:
            return PressureAction.NO_NEW_WORKERS
        if self.cpu_percent >= 90.0:
            return PressureAction.NO_NEW_WORKERS
        return PressureAction.PROCEED

    @property
    def healthy(self) -> bool:
        return self.action is PressureAction.PROCEED


def sample_pressure() -> Pressure:
    try:
        import psutil

        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        return Pressure(
            ram_available_gib=round(vm.available / 2**30, 2),
            ram_percent=vm.percent,
            # interval=None returns since-last-call, which is non-blocking.
            cpu_percent=psutil.cpu_percent(interval=None),
            swap_percent=sw.percent,
            sampled=True,
        )
    except Exception:
        return Pressure()


class Limits(BaseModel):
    reasoning: int = 1
    execution: int = 1
    index: int = 1
    reason: str = ""


class ResourceGovernor:
    def __init__(
        self,
        profile: HardwareProfile | None = None,
        *,
        os_reserve_gib: float = DEFAULT_OS_RESERVE_GIB,
        hard_max_reasoning: int = 8,
        hard_max_execution: int = 4,
    ):
        self.profile = profile or detect_hardware()
        self.os_reserve_gib = os_reserve_gib
        self.hard_max_reasoning = hard_max_reasoning
        self.hard_max_execution = hard_max_execution

    # --------------------------------------------------------------- limits

    def static_limits(self) -> Limits:
        """Limits from the machine's shape alone, before live pressure."""
        p = self.profile
        budget = max(0.0, p.ram_total_gib - self.os_reserve_gib)

        by_ram_exec = int(budget // EXECUTION_WORKER_RAM_GIB)
        # Builds and test runs are CPU-hungry; give each roughly two physical cores.
        by_cpu_exec = max(1, p.physical_cores // 2)
        execution = max(1, min(by_ram_exec, by_cpu_exec, self.hard_max_execution))

        # Reasoning workers are limited by sockets and provider quota, not by this
        # machine, so they scale far higher than execution workers.
        reasoning = max(1, min(self.hard_max_reasoning, max(2, p.physical_cores)))

        index = 1 if p.resource_class in (ResourceClass.TINY, ResourceClass.STANDARD) else 2

        return Limits(
            reasoning=reasoning,
            execution=execution,
            index=index,
            reason=(
                f"{p.resource_class.value}: {p.physical_cores} cores, "
                f"{p.ram_total_gib:.0f} GiB total, {budget:.0f} GiB agent budget"
            ),
        )

    def live_limits(self, pressure: Pressure | None = None) -> Limits:
        """Static limits narrowed by what the machine is doing right now."""
        pressure = pressure or sample_pressure()
        limits = self.static_limits()

        if not pressure.sampled:
            limits.reason += " (no live sample; static limits)"
            return limits

        action = pressure.action
        if action is PressureAction.PROCEED:
            # Never promise more execution slots than free RAM can actually hold.
            affordable = int(max(0.0, pressure.ram_available_gib) // EXECUTION_WORKER_RAM_GIB)
            limits.execution = max(1, min(limits.execution, max(1, affordable)))
            limits.reason += f" | {pressure.ram_available_gib:.1f} GiB free"
            return limits

        # Under pressure, execution concurrency collapses first — it is the pool
        # actually consuming the scarce resource. Reasoning work can continue.
        limits.execution = 1
        limits.index = 0
        limits.reason = f"{action.value}: {pressure.ram_available_gib:.1f} GiB free, swap {pressure.swap_percent:.0f}%"
        return limits

    def may_start(self, kind: WorkerKind, running: int, pressure: Pressure | None = None) -> bool:
        pressure = pressure or sample_pressure()
        limits = self.live_limits(pressure)
        cap = limits.reasoning if kind is WorkerKind.REASONING else limits.execution
        if kind is WorkerKind.EXECUTION and pressure.action is not PressureAction.PROCEED:
            return False
        return running < cap

    # ------------------------------------------------------------- policy

    def local_inference_advised(self) -> bool:
        return not self.profile.prefer_remote_inference

    def policy(self) -> dict:
        """The whole profile as a document, for the dashboard and the ledger."""
        limits = self.static_limits()
        return {
            "machine": {
                "os": self.profile.os_name,
                "arch": self.profile.arch,
                "physical_cores": self.profile.physical_cores,
                "ram_total_gib": self.profile.ram_total_gib,
                "disk_free_gib": self.profile.disk_free_gib,
                "gpu": self.profile.gpu_name or None,
                "gpu_vram_gib": self.profile.gpu_vram_gib,
                "resource_class": self.profile.resource_class.value,
            },
            "limits": {
                "reasoning_workers": limits.reasoning,
                "execution_workers": limits.execution,
                "index_workers": limits.index,
            },
            "inference": {
                "prefer_remote": self.profile.prefer_remote_inference,
                "local_advised_for": (
                    ["classification", "dedup", "log_reduction", "embeddings"]
                    if self.profile.gpu_usable_for_inference
                    else ["classification", "log_reduction"]
                ),
                "local_discouraged_for": ["difficult_coding", "architecture", "large_context"],
            },
        }


__all__ = [
    "CHECKPOINT_BELOW_GIB",
    "EXECUTION_WORKER_RAM_GIB",
    "HardwareProfile",
    "Limits",
    "Pressure",
    "PressureAction",
    "ResourceClass",
    "ResourceGovernor",
    "STOP_NEW_WORKERS_BELOW_GIB",
    "WorkerKind",
    "detect_hardware",
    "sample_pressure",
]


if __name__ == "__main__":  # pragma: no cover
    import json

    g = ResourceGovernor()
    print(json.dumps(g.policy(), indent=2))
    p = sample_pressure()
    print(f"\nlive: {p.ram_available_gib:.1f} GiB free, cpu {p.cpu_percent:.0f}%, "
          f"swap {p.swap_percent:.0f}% -> {p.action.value}")
    print("live limits:", g.live_limits(p).model_dump())
