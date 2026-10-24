"""Candidate comparisons restore all scenario inputs before each invocation."""

import torch
from types import SimpleNamespace


def _record(tokens=4):
    return dict(
        x=torch.arange(tokens * 16, dtype=torch.bfloat16).reshape(tokens, 16),
        ids=torch.tensor([[1, 4]] * tokens, dtype=torch.int32),
        weights=torch.tensor([[0.25, 0.75]] * tokens, dtype=torch.float32),
    )


def test_candidate_producers_restore_complete_inputs_for_large_batches(monkeypatch):
    from benchmarks.experimental.b12x import benchmark_moe as benchmark

    records = [_record(32), _record(32)]
    records[1]["x"].add_(1)
    records[1]["ids"].add_(1)
    records[1]["weights"].mul_(0.5)
    scenarios = tuple(
        tuple(record[key] for key in ("x", "ids", "weights")) for record in records
    )
    output = torch.empty_like(scenarios[0][0])
    state = SimpleNamespace(scratch=SimpleNamespace(scratch_specs=lambda: ()))
    observed = []

    def bind(**kwargs):
        def run():
            observed.append(
                tuple(kwargs[key].clone() for key in ("a", "topk_ids", "topk_weights"))
            )
            return output

        return SimpleNamespace(run=run)

    state.bind = bind
    declaration = SimpleNamespace(
        token_counts=(32,), selection=SimpleNamespace(source="test", config={})
    )
    monkeypatch.setattr(
        benchmark.fused_moe, "plan_execution", lambda **kwargs: declaration
    )
    monkeypatch.setattr(
        benchmark, "request_for_capacity", lambda declaration, **kwargs: kwargs
    )

    def prepare(requests):
        call = requests[0]["benchmark_calls"][32](state)
        assert len(call.benchmark_producers) == 2
        for produce in call.benchmark_producers:
            produce()
            call.invoke()

    benchmark.prepare_moe_execution(
        session=SimpleNamespace(prepare=prepare),
        name="synthetic-inputs",
        experts=SimpleNamespace(has_csf_blocks=False),
        top_k=2,
        inputs={32: scenarios[0]},
        outputs={32: output},
        benchmark_inputs={32: scenarios},
    )
    for actual, expected in zip(observed, scenarios, strict=True):
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e)
