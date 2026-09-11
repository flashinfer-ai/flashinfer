# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from flashinfer.deepseek_v41 import (
    deepseek_v41_decode,
    deepseek_v41_decode_bf16x3,
    deepseek_v41_decode_fp32,
    deepseek_v41_quantize_cache,
    deepseek_v41_window_decode,
)
from .test_deepseek_v41 import gate


@pytest.mark.parametrize("arithmetic", ["bf16x3", "tf32x3"])
@pytest.mark.parametrize("mixed", [True, False])
def test_frost_plan_replay_route_and_recipe(monkeypatch, arithmetic, mixed):
    gate()
    from flashinfer.experimental.deepseek_v41 import decode as implementation

    def forbidden(*args, **kwargs):
        raise AssertionError("Frost must not execute the FlashMLA provider")

    monkeypatch.setattr(implementation, "decode", forbidden)
    torch.manual_seed(415)
    q = torch.randn(2, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    values = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    swa = deepseek_v41_quantize_cache(values, format="swa_mxfp8")
    main = deepseek_v41_quantize_cache(values, format="main_kv_fp4") if mixed else None
    swa_ids = (
        torch.arange(128, device="cuda", dtype=torch.int32)
        .expand(2, 1, 128)
        .contiguous()
    )
    main_ids = swa_ids.clone() if mixed else None
    sink = torch.randn(64, device="cuda")
    args = (q, swa, main, swa_ids, main_ids, sink)
    out, lse, plan = deepseek_v41_decode(*args, backend="frost", arithmetic=arithmetic)
    assert plan.backend == "frost" and plan.arithmetic == arithmetic
    pointers = (
        out.data_ptr(),
        lse.data_ptr(),
        *(value.data_ptr() for value in plan.workspace.values()),
    )

    def run():
        if mixed:
            return deepseek_v41_decode(
                *args, backend="frost", arithmetic=arithmetic, plan=plan
            )
        return deepseek_v41_window_decode(
            q, swa, swa_ids, sink, backend="frost", arithmetic=arithmetic, plan=plan
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    reference = (
        deepseek_v41_decode_bf16x3
        if arithmetic == "bf16x3"
        else deepseek_v41_decode_fp32
    )
    for step in range(3):
        q.normal_()
        swa_ids[..., (96 - step * 32 if step < 2 else 0) :] = -1
        if main_ids is not None:
            main_ids[..., (96 - step * 32 if step < 2 else 0) :] = -1
        graph.replay()
        expected, expected_lse, _ = reference(*args)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)
        result = run()
        assert result[0] is out and result[1] is lse and result[2] is plan
        assert pointers == (
            out.data_ptr(),
            lse.data_ptr(),
            *(value.data_ptr() for value in plan.workspace.values()),
        )
    other = "tf32x3" if arithmetic == "bf16x3" else "bf16x3"
    with pytest.raises(ValueError, match="arithmetic mismatch"):
        deepseek_v41_decode(*args, backend="frost", arithmetic=other, plan=plan)
    with pytest.raises(ValueError, match="cannot be used"):
        deepseek_v41_decode(*args, backend="flashmla", plan=plan)
    for name in ("cutedsl", "triton", "auto", "unknown"):
        with pytest.raises(ValueError, match="backend"):
            deepseek_v41_decode(*args, backend=name)
    with pytest.raises(ValueError, match="declaration/arithmetic mismatch"):
        deepseek_v41_decode(
            q[:1],
            swa,
            main,
            swa_ids[:1],
            None if main_ids is None else main_ids[:1],
            sink,
            backend="frost",
            arithmetic=arithmetic,
            plan=plan,
        )
