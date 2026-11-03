"""Persisted cuDNN winners should not rebuild the heuristic search portfolio."""

import pytest
import torch

import flashinfer
from flashinfer.autotuner import AutoTuner
from flashinfer.gemm import gemm_base


@pytest.mark.parametrize("family", ["bf16", "fp8", "bf16_fp4"])
def test_cudnn_saved_winner_builds_one_plan_and_captures(tmp_path, monkeypatch, family):
    if (
        not torch.cuda.is_available()
        or not gemm_base._is_cudnn_override_shape_available()
    ):
        pytest.skip("requires CUDA and cuDNN shape override support")
    major, minor = torch.cuda.get_device_capability()
    operation = {
        "bf16": flashinfer.bmm_bf16,
        "fp8": flashinfer.bmm_fp8,
        "bf16_fp4": flashinfer.mm_bf16_fp4,
    }[family]
    if not operation.is_backend_supported("cudnn", major * 10 + minor):
        pytest.skip(f"cuDNN {family} is unavailable on SM{major}{minor}")
    import cudnn

    torch.manual_seed(7)
    a = torch.randn(1, 16, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, 256, 256, device="cuda", dtype=torch.bfloat16).transpose(-1, -2)
    out = torch.empty(1, 16, 256, device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda")
    if family == "fp8":
        a, b = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
    elif family == "bf16_fp4":
        # Integers in [-2, 2] are exactly representable with this FP4 scaling.
        b = torch.randint(-2, 3, (1, 256, 256), device="cuda").bfloat16()
        global_scale = torch.full((1,), 1344.0, device="cuda")
        packed, sf = flashinfer.nvfp4_quantize(b[0].T.contiguous(), global_scale)
        packed, sf, alpha = flashinfer.prepare_bf16_fp4_weights(
            packed, sf, 1 / global_scale, backend="cudnn"
        )

    def run():
        if family == "fp8":
            return flashinfer.bmm_fp8(
                a, b, scale, scale, out.dtype, out=out, backend="cudnn"
            )
        if family == "bf16_fp4":
            return flashinfer.mm_bf16_fp4(
                a[0], packed, sf, alpha, out=out[0], backend="cudnn"
            )
        return flashinfer.bmm_bf16(a, b, out=out, backend="cudnn")

    reference = a.float() @ b.float()
    tuner = AutoTuner.get()
    tuner.clear_cache()
    gemm_base.clear_cudnn_graph_cache()
    try:
        with flashinfer.autotune(True, tuning_buckets=(16, 32)):
            run()
        # Capture immediately: no extra eager call may be needed to materialize
        # a second graph after profiling built the candidate portfolio.
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            run()
        out.fill_(float("nan"))
        capture.replay()
        torch.testing.assert_close(out.float(), reference, atol=0.125, rtol=0.02)

        filename = str(tmp_path / "winners.json")
        tuner.save_configs(filename)
        gemm_base.clear_cudnn_graph_cache()
        tuner.clear_cache()
        assert tuner.load_configs(filename)

        def unexpected(*args, **kwargs):
            pytest.fail(
                "loading a saved winner rebuilt or profiled the search portfolio"
            )

        monkeypatch.setattr(cudnn.pygraph, "create_execution_plans", unexpected)
        monkeypatch.setattr(tuner, "_profile_single_kernel", unexpected)
        run()
        torch.testing.assert_close(out.float(), reference, atol=0.125, rtol=0.02)
        replay = torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay):
            run()
        out.fill_(float("nan"))
        replay.replay()
        torch.testing.assert_close(out.float(), reference, atol=0.125, rtol=0.02)
    finally:
        gemm_base.clear_cudnn_graph_cache()
        tuner.clear_cache()
