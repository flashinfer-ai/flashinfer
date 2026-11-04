"""Public-dispatch tests for the CuTe C++ FP8 prefill specializations."""

import pytest
import torch

from tests.attention.dspark_mla_utils import dspark_fixture, numerical_metrics


@pytest.mark.parametrize("heads", [8, 16, 32, 64])
@pytest.mark.parametrize("rank", [0, 3])
def test_public_prefill_specializations(heads, rank):
    f = dspark_fixture(
        2,
        "prefill",
        "v41_fp8",
        heads=heads,
        prefill_tokens=65,
        prefill_cp_size=4,
        prefill_cp_rank=rank,
        history_tokens=128,
        with_sink=heads != 8,
    )

    def mega():
        return f["wrapper"].run_mega(
            f["q"],
            f["caches"][0],
            f["indices"][0],
            f["out"],
            512**-0.5,
            positions=f["positions"],
            cos_sin=f["cs"],
            output_scales=f["scales"],
            topk_length=f["lengths"][0],
            attn_sink=f["sink"],
            extra_kv_cache=f["caches"][1],
            extra_indices=f["indices"][1],
            extra_topk_length=f["lengths"][1],
            return_lse=True,
            lse_scale=0.6931471805599453,
        )

    mega()
    prepared = next(iter(f["wrapper"]._prepared_calls.values()))
    assert prepared.plan.inspect()["variant"] != 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mega()
    rows = torch.tensor([0, 1, 2, 32, 64, 65, 66, 97, 129], device="cuda")
    for length in (0, 1, 31, 32, 33, 63, 65):
        for lengths, ids in zip(f["lengths"], f["indices"], strict=True):
            lengths[0] = 0
            lengths[1] = length
            ids[2, ::7] = -1
            ids[66, :] = -1
        f["positions"].add_(1)
        graph.replay()
        metrics = numerical_metrics(f, prepared.lse, rows)
        assert metrics["finite_output"]
        assert metrics["relative_l2"] < 0.07, metrics
        assert metrics["lse_max_abs"] < 0.06, metrics
        fused = f["out"].float().clone(), f["scales"].clone(), prepared.lse.clone()
        # This calls Mega=false, with separate RoPE and output quantization.
        _, _, lse = f["baseline"]()
        metrics = numerical_metrics(f, lse, rows)
        assert metrics["relative_l2"] < 0.07, metrics
        assert metrics["lse_max_abs"] < 0.06, metrics
        torch.testing.assert_close(lse, fused[2], atol=2e-6, rtol=1e-6)
        # Compiler contraction may change values at a quantization boundary.
        disagreement = (f["out"].float() != fused[0]).float().mean().item()
        assert disagreement < 0.001, disagreement
        torch.testing.assert_close(f["scales"], fused[1], atol=0, rtol=0)


def test_pitched_inline_rows_and_lse_stride():
    f = dspark_fixture(
        1,
        "prefill",
        "v41_fp8",
        heads=16,
        prefill_tokens=65,
        prefill_cp_size=4,
        prefill_cp_rank=3,
        history_tokens=128,
    )

    def mega(out_lse=None):
        return f["wrapper"].run_mega(
            f["q"],
            f["caches"][0],
            f["indices"][0],
            f["out"],
            512**-0.5,
            positions=f["positions"],
            cos_sin=f["cs"],
            output_scales=f["scales"],
            topk_length=f["lengths"][0],
            attn_sink=f["sink"],
            extra_kv_cache=f["caches"][1],
            extra_indices=f["indices"][1],
            extra_topk_length=f["lengths"][1],
            return_lse=True,
            lse_scale=0.6931471805599453,
            out_lse=out_lse,
        )

    lse = mega()
    assert (
        next(iter(f["wrapper"]._prepared_calls.values())).plan.inspect()["variant"] != 0
    )
    expected = f["out"].float().clone(), f["scales"].clone(), lse.clone()
    backing = []
    for i, cache in enumerate(f["caches"]):
        padded = torch.full(
            (*cache.shape[:-1], 544), 0xA5, device=cache.device, dtype=torch.uint8
        )
        view = padded[..., :528]
        view.copy_(cache)
        backing.append(padded)
        f["caches"][i] = view
    storage = torch.full((65, 24), 1234.0, device="cuda")
    lse = storage[:, :16]
    mega(lse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mega(lse)
    graph.replay()
    for actual, wanted in zip(
        (f["out"].float(), f["scales"], lse), expected, strict=True
    ):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert (storage[:, 16:] == 1234.0).all()
    for padded in backing:
        assert (padded[..., 528:] == 0xA5).all()
