from forgeos.bench import Layer, bench


def test_full_benchmark_uses_prompt_cache_contract(monkeypatch):
    calls = []

    class FakeCache:
        def lookup(self, provider, model, prompt):
            calls.append(("lookup", provider, model, prompt))
            return None

        def store(self, provider, model, prompt, response, **kwargs):
            calls.append(("store", provider, model, prompt, response, kwargs))

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr("forgeos.prompt_cache.PromptCache", FakeCache)
    results = bench("measure a local retry helper", iterations=1)

    assert [result.layer for result in results] == list(Layer)
    assert calls[0][:3] == ("lookup", "openrouter", "sonnet")
    assert calls[1][0] == "store"
    assert calls[1][-1] == {"tokens_out": 100}
    assert calls[-1] == ("close",)
