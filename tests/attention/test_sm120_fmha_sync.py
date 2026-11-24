"""Synchronization of the SM120 FP8 TMA FMHA prefill kernel
(flashinfer/cute_dsl/attention/fmha/sm120/fmha_prefill_fp8_tma.py), checked in
its generated ``sm_120a`` PTX.

Compute warps read TMA-populated K/V shared-memory stages with generic-proxy
``ldmatrix`` and release them via ``mbar_consumed`` so the load warp can refill
the same bytes with async-proxy ``cp.async.bulk.tensor``. Under the PTX memory
model, every compute lane must fence its shared-memory reads across the
generic/async proxy boundary and arrive on ``mbar_consumed`` itself (count 1
per lane, rather than one elected lane arriving with count 32).
"""

import re

import pytest

_PROXY_FENCE_RE = re.compile(
    r"fence\.proxy\.async(?:\.shared::cta|::generic\.release\.sync_restrict::shared::cluster::read\.cluster)\s*;"
)
_ARRIVE_RE = re.compile(r"^\s*mbarrier\.arrive\.shared(?:::cta)?\.b64\b(.*)$")


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((64, 64, 64, False), id="d64-two-buffer-ragged"),
        pytest.param((128, 128, 128, False), id="d128-direct-ring-ragged"),
        pytest.param((256, 128, 128, False), id="d256-distributed-ring-ragged"),
        pytest.param((128, 128, 128, True), id="d128-direct-ring-paged"),
    ],
)
def ptx(request, tmp_path_factory) -> str:
    from flashinfer.cute_dsl.availability import is_cute_dsl_experimental_available

    if not is_cute_dsl_experimental_available():
        pytest.skip("nvidia-cutlass-dsl with cutlass.experimental is not available.")
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

    from flashinfer.cute_dsl.attention.fmha.sm120.fmha_prefill_fp8_tma import (
        SM120FusedMultiHeadAttentionFP8ForwardTMA,
    )

    head_dim, kv_tile, q_tile, use_paged_kv = request.param
    fmha = SM120FusedMultiHeadAttentionFP8ForwardTMA(
        in_dtype=cutlass.Float8E4M3FN,
        out_dtype=cutlass.BFloat16,
        is_causal=False,
        head_tile=head_dim,
        kv_tile=kv_tile,
        q_tile=q_tile,
        use_paged_kv=use_paged_kv,
        num_tokens_per_page=16 if use_paged_kv else None,
    )

    sym_total_q = cute.sym_int()
    sym_cu = cute.sym_int()
    fake_q = make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (sym_total_q, 8, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_o = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_total_q, 8, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_cu_q = make_fake_compact_tensor(Int32, (sym_cu,), assumed_align=4)
    if use_paged_kv:
        sym_pages = cute.sym_int()
        sym_b = cute.sym_int()
        sym_max_pages = cute.sym_int()
        fake_k = make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (sym_pages, 2, 16, head_dim),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        fake_v = make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (sym_pages, 2, 16, head_dim),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        fake_seqlens_kv = make_fake_compact_tensor(Int32, (sym_b,), assumed_align=4)
        fake_block_tables = make_fake_compact_tensor(
            Int32, (sym_b, sym_max_pages), stride_order=(1, 0), assumed_align=4
        )
        fake_cu_k = None
    else:
        sym_total_k = cute.sym_int()
        fake_k = make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (sym_total_k, 2, head_dim),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
        fake_v = make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (sym_total_k, 2, head_dim),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
        fake_seqlens_kv = None
        fake_block_tables = None
        fake_cu_k = make_fake_compact_tensor(Int32, (sym_cu,), assumed_align=4)

    dump_dir = tmp_path_factory.mktemp(
        f"d{head_dim}_kv{kv_tile}_q{q_tile}_{'paged' if use_paged_kv else 'ragged'}"
    )
    compiled = cute.compile(
        fmha,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        None,
        cutlass.Float32(1.0),
        cutlass.Float32(1.0),
        None,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        fake_seqlens_kv,
        fake_cu_q,
        fake_block_tables,
        fake_cu_k,
        cutlass.Int32(1),
        True,
        options=f"--gpu-arch sm_120a --keep-ptx --dump-dir {dump_dir}",
    )
    assert compiled.__ptx__ is not None
    return compiled.__ptx__


def _consumer_arrives(ptx: str) -> list[tuple[int, str]]:
    return [
        (idx, m.group(1))
        for idx, line in enumerate(ptx.splitlines())
        if "expect_tx" not in line and (m := _ARRIVE_RE.match(line)) is not None
    ]


def test_every_compute_lane_releases_kv_stages(ptx):
    lines = ptx.splitlines()
    arrives = _consumer_arrives(ptx)
    assert len(arrives) >= 2, "Expected K and V consumer stage-release mbarrier.arrive"

    for idx, operands in arrives:
        # Per-lane arrive takes 2 operands (%state, [%mbar]) with implicit count 1,
        # not a 3-operand warp-elected arrive (%state, [%mbar], 32).
        operand_count = len([op for op in operands.split(",") if op.strip()])
        assert operand_count == 2, (
            f"Consumer mbarrier.arrive at PTX line {idx + 1} uses an explicit "
            f"multi-thread count ({operands.strip()}) instead of per-lane arrive"
        )
        window = "\n".join(lines[max(0, idx - 6) : idx])
        assert "elect.sync" not in window, (
            f"Consumer mbarrier.arrive at PTX line {idx + 1} is gated by elect.sync:\n"
            f"{window}\n{lines[idx]}"
        )


def test_kv_stage_releases_fence_async_proxy(ptx):
    lines = ptx.splitlines()
    arrives = _consumer_arrives(ptx)
    assert len(arrives) >= 2, "Expected K and V consumer stage-release mbarrier.arrive"

    for idx, _ in arrives:
        window = "\n".join(lines[max(0, idx - 6) : idx])
        assert _PROXY_FENCE_RE.search(window), (
            f"Missing async-proxy fence before consumer mbarrier.arrive at PTX line {idx + 1}:\n"
            f"{window}\n{lines[idx]}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
