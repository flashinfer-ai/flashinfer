"""Public-dispatch tests for the CuTe C++ FP8 prefill specializations."""

import pytest
import torch

from tests.attention.sparse_mla_prefill_utils import (
    prefill_fixture,
    numerical_metrics,
    poison_mega_destinations,
    assert_mega_outputs,
)


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("heads", [8, 16, 32, 64])
@pytest.mark.parametrize("rank", [0, 3])
def test_public_prefill_specializations(heads, rank, inline):
    f = prefill_fixture(
        2,
        "v41_fp8",
        heads=heads,
        prefill_tokens=65,
        prefill_cp_size=4,
        prefill_cp_rank=rank,
        history_tokens=128,
        with_sink=heads != 8,
        inline=inline,
    )

    actual_lse = torch.empty(f["q"].shape[:2], device="cuda")

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
            out_lse=actual_lse,
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
        # Keep the reference before poisoning every captured destination.
        expected = tuple(x.clone() for x in f["baseline"]())
        metrics = numerical_metrics(f, expected[2], rows)
        assert metrics["finite_output"], metrics
        assert metrics["relative_l2"] < 0.07, metrics
        assert metrics["lse_max_abs"] < 0.06, metrics
        poison_mega_destinations(f["out"], f["scales"], actual_lse, expected[1])
        graph.replay()
        metrics = numerical_metrics(f, actual_lse, rows)
        assert metrics["finite_output"]
        assert metrics["relative_l2"] < 0.07, metrics
        assert metrics["lse_max_abs"] < 0.06, metrics
        assert_mega_outputs((f["out"], f["scales"], actual_lse), expected)


@pytest.mark.parametrize("impl", ["auto", "decode-splitkv", "decode"])
def test_pitched_inline_rows_and_lse_stride(impl):
    """Cache/LSE strides and graph replay must work for each scheduling choice."""
    f = prefill_fixture(
        1,
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
            impl=impl,
        )

    lse = mega()
    # Compare layouts within the same implementation; this is independent of
    # which implementation the automatic scheduling policy selects.
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
    poison_mega_destinations(f["out"], f["scales"], lse, expected[1])
    graph.replay()
    for actual, wanted in zip(
        (f["out"].float(), f["scales"], lse), expected, strict=True
    ):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert (storage[:, 16:] == 1234.0).all()
    for padded in backing:
        assert (padded[..., 528:] == 0xA5).all()


@pytest.mark.parametrize("tokens", [1, 4, 16, 32, 64])
@pytest.mark.parametrize("inline", [False, True])
def test_explicit_small_query_prefill(tokens, inline):
    """Small Q is legal for prefill even when auto dispatch prefers decode."""
    from tests.attention.sparse_mla_prefill_utils import fixture, rope, epilogue
    from flashinfer.mla._sparse_mla_sm120._execution import (
        get_sparse_mla_sm120_module,
        resolve_attention,
        resolve_model_type,
    )

    f = fixture(tokens, 32, "v41_fp8", inline=inline, extra_page_size=2)
    q, caches, ids = f["q"], f["caches"], f["indices"]
    if not inline:
        # Preserve footer packing while leaving sentinel bytes between pages.
        for i, cache in enumerate(caches):
            flat = cache.reshape(cache.shape[0], -1)
            storage = torch.full(
                (flat.shape[0], flat.shape[1] + 16),
                0xA5,
                dtype=torch.uint8,
                device="cuda",
            )
            view = storage[:, : flat.shape[1]].view_as(cache)
            view.copy_(cache)
            caches[i] = view
    props = torch.cuda.get_device_properties(q.device)
    lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    plan = resolve_attention(
        model=resolve_model_type(512, "ue8m0_g32_inline" if inline else "ue8m0_g32"),
        tokens=tokens,
        heads=32,
        topk=128,
        extra_topk=512,
        page_size=64,
        extra_page_size=2,
        page_stride_bytes=caches[0].stride(0),
        extra_page_stride_bytes=caches[1].stride(0),
        row_stride_bytes=528,
        extra_row_stride_bytes=528,
        indices_stride=128,
        extra_indices_stride=512,
        lse_stride=32,
        has_lengths=True,
        has_extra_lengths=True,
        has_sink=True,
        extra_fp4=False,
        variant=1,
        precision="fp8",
        cpb=0,
        sm_count=props.multi_processor_count,
        max_shared_bytes=props.shared_memory_per_block_optin,
    )
    module = get_sparse_mla_sm120_module()

    def args(query):
        return (
            plan,
            query,
            caches[0],
            ids[0],
            f["context"],
            lse,
            f["context"],
            lse,
            512**-0.5,
            f["lengths"][0],
            f["sink"],
            caches[1],
            ids[1],
            f["lengths"][1],
            0.6931471805599453,
        )

    def mega():
        module.execute_attention_rope_fp8(
            *args(q), f["positions"], f["cs"], f["out"].view(torch.uint8), f["scales"]
        )

    mega()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mega()
    poison_mega_destinations(f["out"], f["scales"], lse, f["scales"].clone())
    graph.replay()
    metrics = numerical_metrics(f, lse, torch.arange(tokens, device="cuda"))
    assert metrics["relative_l2"] < 0.07, metrics
    assert metrics["lse_max_abs"] < 0.06, metrics
    expected = f["out"].float().clone(), f["scales"].clone(), lse.clone()
    rope(q, f["positions"], f["cs"], f["rotated"])
    module.execute_attention(*args(f["rotated"]))
    epilogue(f["context"], f["positions"], f["cs"], f["out"], f["scales"])
    torch.testing.assert_close(lse, expected[2], rtol=1e-6, atol=2e-6)
    assert (f["out"].float() != expected[0]).float().mean().item() < 0.001
    torch.testing.assert_close(f["scales"], expected[1], rtol=0, atol=0)
