# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from pathlib import Path

import pytest

_CTA_PROXY_FENCE_RE = re.compile(r"\bfence\.proxy\.async\.shared::cta\s*;")
_SYNC_RESTRICT_PROXY_FENCE_RE = re.compile(
    r"\bfence\.proxy\.async::generic\.release\.sync_restrict::shared::cluster::read\.cluster\s*;"
)
_PROXY_FENCE_RE = re.compile(
    r"\bfence\.proxy\.async(?:\.shared::cta|::generic\.release\.sync_restrict::shared::cluster::read\.cluster)\s*;"
)
_ARRIVE_RE = re.compile(
    r"^\s*(?:(@!?%\w+)\s+)?"
    r"mbarrier\.arrive(?:\.[a-z0-9_]+)*\.shared(?:::cta)?\.b64\b(.*)$"
)
_LABEL_RE = re.compile(r"^\s*(?:\$?L[\w$.]*|\w+):\s*$")


def _compile_sm120_fmha_ptx(
    head_dim: int,
    kv_tile: int,
    q_tile: int,
    use_paged_kv: bool,
    is_causal: bool,
    use_skip_softmax: bool,
    dump_dir: Path,
) -> str:
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

    fmha = SM120FusedMultiHeadAttentionFP8ForwardTMA(
        in_dtype=cutlass.Float8E4M3FN,
        out_dtype=cutlass.BFloat16,
        is_causal=is_causal,
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
    fake_skip = (
        make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        if use_skip_softmax
        else None
    )
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

    dump_dir.mkdir(parents=True, exist_ok=True)
    compiled = cute.compile(
        fmha,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        None,
        cutlass.Float32(1.0),
        cutlass.Float32(1.0),
        fake_skip,
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


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((64, 64, 64, False, False, False), id="d64-two-buffer-ragged"),
        pytest.param(
            (128, 128, 128, False, False, False), id="d128-direct-ring-ragged"
        ),
        pytest.param(
            (256, 128, 128, False, False, False), id="d256-distributed-ring-ragged"
        ),
        pytest.param((128, 128, 128, True, False, False), id="d128-direct-ring-paged"),
        pytest.param(
            (128, 128, 128, False, True, True), id="d128-direct-ring-causal-skip"
        ),
    ],
)
def ptx(request, tmp_path_factory) -> str:
    head_dim, kv_tile, q_tile, use_paged_kv, is_causal, use_skip_softmax = request.param
    dump_dir = tmp_path_factory.mktemp(
        f"d{head_dim}_kv{kv_tile}_q{q_tile}"
        f"_{'paged' if use_paged_kv else 'ragged'}"
        f"_{'causal' if is_causal else 'noncausal'}"
        f"_{'skip' if use_skip_softmax else 'noskip'}"
    )
    return _compile_sm120_fmha_ptx(
        head_dim,
        kv_tile,
        q_tile,
        use_paged_kv,
        is_causal,
        use_skip_softmax,
        dump_dir,
    )


def _consumer_arrives(ptx: str) -> list[tuple[int, str | None, str]]:
    return [
        (idx, m.group(1), m.group(2))
        for idx, line in enumerate(ptx.splitlines())
        if "expect_tx" not in line and (m := _ARRIVE_RE.match(line)) is not None
    ]


def _block_start(lines: list[str], idx: int) -> int:
    for pos in range(idx - 1, -1, -1):
        if _LABEL_RE.match(lines[pos]) or lines[pos].strip() == "{":
            return pos + 1
    return 0


def _assert_per_lane_consumer_arrives(ptx: str) -> None:
    lines = ptx.splitlines()
    arrives = _consumer_arrives(ptx)
    assert len(arrives) >= 2, "Expected K and V consumer stage-release mbarrier.arrive"

    for idx, predicate, operands in arrives:
        assert predicate is None, (
            f"Consumer mbarrier.arrive at PTX line {idx + 1} is predicated "
            f"({predicate}) instead of unconditional per-lane arrive:\n{lines[idx]}"
        )
        # Per-lane arrive takes 2 operands (%state, [%mbar]) with implicit count 1,
        # not a 3-operand warp-elected arrive (%state, [%mbar], 32).
        operand_count = len([op for op in operands.split(",") if op.strip()])
        assert operand_count == 2, (
            f"Consumer mbarrier.arrive at PTX line {idx + 1} uses an explicit "
            f"multi-thread count ({operands.strip()}) instead of per-lane arrive"
        )
        block = "\n".join(lines[_block_start(lines, idx) : idx])
        assert "elect.sync" not in block, (
            f"Consumer mbarrier.arrive at PTX line {idx + 1} is preceded by "
            f"elect.sync in the same basic block:\n{block}\n{lines[idx]}"
        )


def _assert_fenced_consumer_arrives(
    ptx: str, fence_re: re.Pattern[str] = _PROXY_FENCE_RE
) -> None:
    lines = ptx.splitlines()
    arrives = _consumer_arrives(ptx)
    assert len(arrives) >= 2, "Expected K and V consumer stage-release mbarrier.arrive"

    for idx, _, _ in arrives:
        start = _block_start(lines, idx)
        fence_idx = None
        for pos in range(idx - 1, start - 1, -1):
            if fence_re.search(lines[pos]):
                fence_idx = pos
                break
        context = "\n".join(lines[max(start, idx - 8) : idx + 1])
        assert fence_idx is not None, (
            f"Missing async-proxy fence before consumer mbarrier.arrive at PTX line {idx + 1}:\n"
            f"{context}"
        )
        between = "\n".join(lines[fence_idx + 1 : idx])
        assert "ldmatrix." not in between, (
            f"Found ldmatrix between async-proxy fence (line {fence_idx + 1}) and "
            f"consumer mbarrier.arrive (line {idx + 1}):\n{between}"
        )


def test_every_compute_lane_releases_kv_stages(ptx):
    _assert_per_lane_consumer_arrives(ptx)


def test_kv_stage_releases_fence_async_proxy(ptx):
    _assert_fenced_consumer_arrives(ptx)


def test_cuda_lt_13_4_fallback_emits_cta_async_proxy_fence(monkeypatch, tmp_path):
    from flashinfer.cute_dsl.availability import is_cute_dsl_experimental_available

    if not is_cute_dsl_experimental_available():
        pytest.skip("nvidia-cutlass-dsl with cutlass.experimental is not available.")
    import cutlass

    monkeypatch.setattr(cutlass, "CUDA_VERSION", type(cutlass.CUDA_VERSION)("13.2"))
    fallback_ptx = _compile_sm120_fmha_ptx(
        128, 128, 128, False, False, False, tmp_path / "fallback_13_2"
    )
    _assert_per_lane_consumer_arrives(fallback_ptx)
    _assert_fenced_consumer_arrives(fallback_ptx, fence_re=_CTA_PROXY_FENCE_RE)
    assert _SYNC_RESTRICT_PROXY_FENCE_RE.search(fallback_ptx) is None


def test_ptx_sync_checker_rejects_broken_patterns():
    predicated = """
    $L__BB0_1:
        fence.proxy.async.shared::cta;
        mbarrier.arrive.shared.b64 %rd1, [%r1];
        fence.proxy.async.shared::cta;
        mbarrier.arrive.shared.b64 %rd2, [%r2];
        @%p1 mbarrier.arrive.shared.b64 %rd3, [%r3], 32;
    """
    with pytest.raises(AssertionError, match="predicated"):
        _assert_per_lane_consumer_arrives(predicated)

    distant_elect = (
        "$L__BB0_1:\n"
        "    elect.sync %r1|%p1, -1;\n"
        + "    add.s32 %r2, %r2, 1;\n"
        * 20
        + "    mbarrier.arrive.shared.b64 %rd1, [%r1];\n"
        "$L__BB0_2:\n"
        "    mbarrier.arrive.shared.b64 %rd2, [%r2];\n"
    )
    with pytest.raises(AssertionError, match="elect.sync"):
        _assert_per_lane_consumer_arrives(distant_elect)

    ldmatrix_after_fence = """
    $L__BB0_1:
        fence.proxy.async.shared::cta;
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%r5];
        mbarrier.arrive.shared.b64 %rd1, [%r6];
    $L__BB0_2:
        fence.proxy.async.shared::cta;
        mbarrier.arrive.shared.b64 %rd2, [%r7];
    """
    with pytest.raises(AssertionError, match="ldmatrix"):
        _assert_fenced_consumer_arrives(ldmatrix_after_fence)


if __name__ == "__main__":
    pytest.main([__file__])
