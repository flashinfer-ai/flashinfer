"""LSE base selection for the SM120/SM121 sparse MLA backend (issue #4485).

Companion to ``test_mla_lse_base.py``, which covers trtllm-gen and cute-dsl.
``return_lse_base`` is a *guarantee* about the units of the returned LSE,
not a transformation request:

    None   -> the backend's default base (base-2 for sparse, per the docstring
              of ``trtllm_batch_decode_with_kv_cache_mla``)
    "base2" -> base-2, whichever backend ran
    "basee" -> base-e, whichever backend ran

The SM120 sparse kernels are log2 throughout and store base-2, so ``None`` and
``"base2"`` must stay bit-identical to the pre-#4485 output; only ``"basee"`` scales.

There are four user-facing LSE stores reachable through
``backend="sparse"`` (a fifth, DOTS3_SWA prefill, is reachable only through
``SparseMLASm120Wrapper`` -- see the section at the end of this file), and each
needs the scale applied independently:

  * ``decode_dsv4_kernel.cuh`` -- ``sparse_mla_decode_dsv4_merge_kernel``,
    shared by both decode entries (``csrc/sparse_mla_sm120/decode_dsv3_2.cu``
    and ``..._dsv4.cu``). Reached when ``num_tokens <= 64``.
  * ``prefill_sg_kernel.cuh``  -- SG prefill, ``num_heads <= 16``.
  * ``prefill_mg_kernel.cuh`` -- MG prefill, ``num_heads in {32, 64, 128}``.
  * ``prefill_swapab_kernel.cuh`` -- swapAB prefill, ``num_heads in {64, 128}``.

``_CASES`` below covers decode, SG, and MG; a dedicated test verifies auto
and forced swapAB dispatch. Each prefill template has its own LSE epilogue.

Only the v32/DSv3.2 public path (``head_dim_qk == 576``, packed uint8 KV) can
return LSE -- ``_trtllm_batch_decode_sparse_mla_dsv4_sm120`` hardcodes
``return_lse=False``, so the DSv4 and dual-cache kernels have no user-facing
LSE surface to test.
"""

from __future__ import annotations

import math

import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm12x_supported
from tests.attention.sparse_mla_test_utils import (
    _ref_sparse_attn,
    dequantize_kv_dots3_swa,
    dequantize_kv_dsv3_2,
    quantize_kv_dots3_swa,
    quantize_kv_dsv3_2,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")),
    reason="Sparse-MLA SM120 requires SM12x; the LSE base is kernel-side and "
    "cannot be checked on other architectures.",
)

LOG2E = math.log2(math.e)  # 1.4426950408889634

D_QK = 576
D_V = 512
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
PAGE_BLOCK_SIZE = 64
SM_SCALE = D_QK**-0.5
WORKSPACE_BYTES = 64 << 20

# LSE magnitudes here are O(log2(topk)) ~ 7-11, so base-2 and base-e differ by
# ~44% -- far outside this tolerance. The negative assertions below rely on that.
_LSE_TOL = {"atol": 5e-2, "rtol": 5e-2}

# (num_tokens, num_heads, topk) -> one case per user-facing LSE store.
# Decode needs num_tokens <= 64 and (num_heads, topk) in _DECODE_DSV3_2_DISPATCH;
# v32 prefill only instantiates topk == 2048.
_CASES = [
    pytest.param(4, 8, 512, id="decode_nh8_topk512"),
    pytest.param(4, 32, 2048, id="decode_nh32_topk2048"),  # 32 split-K partitions
    pytest.param(128, 8, 2048, id="prefill_sg_nh8"),
    pytest.param(128, 32, 2048, id="prefill_mg_nh32"),
]

_DECODE_CASE = (4, 8, 512)
_PREFILL_MG_CASE = (128, 32, 2048)

_UNSET = object()  # distinguishes "use the case default" from an explicit None


def _is_prefill(num_tokens: int) -> bool:
    """Return whether the request is outside the decode-form token limit."""
    return num_tokens > 64


def _varying_lengths(num_tokens: int, topk: int, device: torch.device) -> torch.Tensor:
    """Per-token valid-candidate counts cycling topk/8, topk/4, topk/2, topk.

    Never zero -- the all-masked row writes a sentinel instead of an LSE and is
    covered separately by
    ``test_sparse_mla_lse_base_preserves_masked_row_sentinel``.
    """
    cycle = torch.tensor([topk // 8, topk // 4, topk // 2, topk], device=device)
    reps = (num_tokens + cycle.numel() - 1) // cycle.numel()
    return cycle.repeat(reps)[:num_tokens].to(torch.int32).contiguous()


class _Inputs:
    """Public-API kwargs for one sparse call plus its PyTorch reference."""

    def __init__(
        self,
        num_tokens: int,
        num_heads: int,
        topk: int,
        *,
        with_sink: bool = False,
        seed: int = 0,
        build_reference: bool = True,
        vary_lengths: bool = False,
    ) -> None:
        torch.manual_seed(seed)
        device = torch.device("cuda")
        # 2x slack over topk so the random indices span more than the top-k window,
        # matching the shapes proven in tests/attention/test_sparse_mla_sm120.py.
        num_blocks = 2 * topk // PAGE_BLOCK_SIZE
        s_kv = num_blocks * PAGE_BLOCK_SIZE

        kv_bf16 = (
            torch.randn(
                num_blocks,
                PAGE_BLOCK_SIZE,
                1,
                D_QK,
                device=device,
                dtype=torch.bfloat16,
            )
            / 10.0
        ).clamp(-1, 1)
        kv_packed = quantize_kv_dsv3_2(kv_bf16)

        self.q = (
            torch.randn(
                num_tokens, num_heads, D_QK, device=device, dtype=torch.bfloat16
            )
            / 10.0
        ).clamp(-1, 1)
        self.indices = torch.randint(
            0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
        )
        self.sink = (
            torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
            if with_sink
            else None
        )

        self.topk = topk
        self.device = device
        self.kv_hnd = kv_packed.transpose(1, 2)  # [num_pages, 1, page_size, 656]
        self.workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        # With all topk candidates valid and the small-magnitude inputs above,
        # every row's LSE collapses to ~log2(topk); vary_lengths spreads it over
        # ~log2(topk/8)..log2(topk) so a bug that returns a constant is visible.
        self.topk_lengths = (
            _varying_lengths(num_tokens, topk, device) if vary_lengths else None
        )
        if self.topk_lengths is not None:
            self.seq_lens = self.topk_lengths
        elif _is_prefill(num_tokens):
            # Prefill and decode differ only in how seq_lens is spelled; both
            # mean "all topk candidates are valid" here.
            self.seq_lens = torch.full(
                (num_tokens,), topk, dtype=torch.int32, device=device
            )
        else:
            self.seq_lens = None

        # The dense reference materializes num_tokens x topk x 576 floats
        # (~600 MB at topk=2048), so skip it for tests that only inspect LSE
        # sentinels.
        self.ref_out = None
        self.ref_lse_base2 = None
        if build_reference:
            self.ref_out, self.ref_lse_base2 = _ref_sparse_attn(
                self.q,
                dequantize_kv_dsv3_2(kv_packed),
                self.indices,
                SM_SCALE,
                D_V,
                attn_sink=self.sink,
                topk_length=self.topk_lengths,
            )

    def expected_lse(self, return_lse_base: str | None) -> torch.Tensor:
        """Reference LSE in the base the selector promises."""
        if return_lse_base == "basee":
            return self.ref_lse_base2 / LOG2E
        return self.ref_lse_base2

    def wrong_lse(self, return_lse_base: str | None) -> torch.Tensor:
        """The *other* base, to catch a scale that was never applied."""
        if return_lse_base == "basee":
            return self.ref_lse_base2
        return self.ref_lse_base2 / LOG2E


def _run(
    inputs: _Inputs,
    *,
    return_lse_base: str | None,
    return_lse: bool = True,
    lse: torch.Tensor | None = None,
    seq_lens=_UNSET,
    clone: bool = True,
):
    """Call the public sparse path.

    Results are cloned by default so a later call can't alias them -- the runner
    owns its internal LSE buffer. Pass ``clone=False`` to inspect the objects the
    API actually returned, which is what the caller-buffer identity check needs.
    """
    result = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        query=inputs.q.unsqueeze(1),
        kv_cache=inputs.kv_hnd,
        workspace_buffer=inputs.workspace,
        qk_nope_head_dim=D_V,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=inputs.indices.unsqueeze(1),
        seq_lens=inputs.seq_lens if seq_lens is _UNSET else seq_lens,
        max_seq_len=inputs.topk,
        sparse_mla_top_k=inputs.topk,
        bmm1_scale=SM_SCALE,
        bmm2_scale=1.0,
        sinks=None if inputs.sink is None else [inputs.sink],
        backend="sparse",
        lse=lse,
        return_lse=return_lse,
        return_lse_base=return_lse_base,
    )
    if not return_lse:
        out = result.squeeze(1)
        return out.clone() if clone else out
    out, out_lse = result
    out = out.squeeze(1)
    if clone:
        return out.clone(), out_lse.clone()
    return out, out_lse


# --------------------------------------------------------------------------------------
# Base selection, one case per LSE store
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("return_lse_base", [None, "base2", "basee"])
@pytest.mark.parametrize("num_tokens,num_heads,topk", _CASES)
def test_sparse_mla_lse_base(
    num_tokens: int, num_heads: int, topk: int, return_lse_base: str | None
) -> None:
    inputs = _Inputs(num_tokens, num_heads, topk)
    out, lse = _run(inputs, return_lse_base=return_lse_base)

    torch.testing.assert_close(out, inputs.ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse, inputs.expected_lse(return_lse_base), **_LSE_TOL)
    assert not torch.allclose(lse, inputs.wrong_lse(return_lse_base), **_LSE_TOL), (
        f"return_lse_base={return_lse_base} returned the other base"
    )


@pytest.mark.parametrize("num_tokens,num_heads,topk", _CASES)
def test_sparse_mla_lse_base_none_matches_base2(
    num_tokens: int, num_heads: int, topk: int
) -> None:
    """base-2 is the sparse default, so None and "base2" must not merely agree
    numerically -- they must be the same store with the same 1.0 multiplier."""
    inputs = _Inputs(num_tokens, num_heads, topk)
    _, lse_none = _run(inputs, return_lse_base=None)
    _, lse_base2 = _run(inputs, return_lse_base="base2")

    assert torch.equal(lse_none, lse_base2), (
        "return_lse_base=None and base2 must be bit-identical on the "
        "sparse backend (both are base-2)"
    )


@pytest.mark.parametrize("num_tokens,num_heads,topk", _CASES)
def test_sparse_mla_lse_base_leaves_output_unchanged(
    num_tokens: int, num_heads: int, topk: int
) -> None:
    """The scale must land on the LSE store only. The kernels reuse the running
    max/sum to normalize the output, so a scale applied too early corrupts it."""
    inputs = _Inputs(num_tokens, num_heads, topk)
    out_none, _ = _run(inputs, return_lse_base=None)
    out_base2, _ = _run(inputs, return_lse_base="base2")
    out_basee, _ = _run(inputs, return_lse_base="basee")

    assert torch.equal(out_none, out_base2)
    assert torch.equal(out_none, out_basee), (
        "return_lse_base must not change the attention output"
    )


@pytest.mark.parametrize("return_lse_base", ["base2", "basee"])
@pytest.mark.parametrize(
    "num_tokens,num_heads,topk",
    [
        pytest.param(*_DECODE_CASE, id="decode"),
        pytest.param(*_PREFILL_MG_CASE, id="prefill_mg"),
    ],
)
def test_sparse_mla_lse_base_with_sink(
    num_tokens: int, num_heads: int, topk: int, return_lse_base: str
) -> None:
    """attn_sink is merged into the LSE in log2 space immediately before the
    store (``lse += log2f(1 + exp2f(sink_log2 - lse))``), so the base scale has
    to be applied after the merge, not to the pre-sink value."""
    inputs = _Inputs(num_tokens, num_heads, topk, with_sink=True)
    out, lse = _run(inputs, return_lse_base=return_lse_base)

    torch.testing.assert_close(out, inputs.ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse, inputs.expected_lse(return_lse_base), **_LSE_TOL)
    assert not torch.allclose(lse, inputs.wrong_lse(return_lse_base), **_LSE_TOL), (
        f"return_lse_base={return_lse_base} returned the other base"
    )


@pytest.mark.parametrize("return_lse_base", [None, "base2", "basee"])
@pytest.mark.parametrize(
    "num_tokens,num_heads,topk",
    [
        pytest.param(*_DECODE_CASE, id="decode"),
        pytest.param(*_PREFILL_MG_CASE, id="prefill_mg"),
    ],
)
def test_sparse_mla_lse_base_varying_topk_length(
    num_tokens: int, num_heads: int, topk: int, return_lse_base: str | None
) -> None:
    """Same check as ``test_sparse_mla_lse_base``, but with per-token
    topk_length so the LSE actually varies across rows -- a scale applied to a
    stale or shared register would still match a constant reference."""
    inputs = _Inputs(num_tokens, num_heads, topk, vary_lengths=True)
    out, lse = _run(inputs, return_lse_base=return_lse_base)

    assert lse.std() > 0.1, "topk_length variation did not reach the LSE"
    torch.testing.assert_close(out, inputs.ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse, inputs.expected_lse(return_lse_base), **_LSE_TOL)
    assert not torch.allclose(lse, inputs.wrong_lse(return_lse_base), **_LSE_TOL), (
        f"return_lse_base={return_lse_base} returned the other base"
    )


# --------------------------------------------------------------------------------------
# Plumbing contracts
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,num_heads,topk",
    [
        pytest.param(*_DECODE_CASE, id="decode"),
        pytest.param(*_PREFILL_MG_CASE, id="prefill_mg"),
    ],
)
def test_sparse_mla_lse_base_ignored_without_return_lse(
    num_tokens: int, num_heads: int, topk: int
) -> None:
    """A valid return_lse_base is a no-op when no LSE is requested."""
    inputs = _Inputs(num_tokens, num_heads, topk)
    out = _run(inputs, return_lse_base="basee", return_lse=False)

    torch.testing.assert_close(out, inputs.ref_out, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("return_lse_base", [None, "base2", "basee"])
@pytest.mark.parametrize("nested", [False, True])
def test_sparse_mla_lse_base_writes_user_lse_buffer(
    nested: bool, return_lse_base: str | None
) -> None:
    """A caller-supplied lse buffer is returned verbatim by
    ``_trtllm_batch_decode_sparse_mla_sm120`` (the ``user_lse`` branch), so the
    scale has to reach the caller's memory, not a runner-owned copy."""
    num_tokens, num_heads, topk = _DECODE_CASE
    inputs = _Inputs(num_tokens, num_heads, topk)
    shape = (num_tokens, 1, num_heads) if nested else (num_tokens, num_heads)
    user_lse = torch.full(
        shape, float("nan"), dtype=torch.float32, device=inputs.device
    )

    _, returned = _run(
        inputs, return_lse_base=return_lse_base, lse=user_lse, clone=False
    )

    # The contract is the caller's buffer itself, not a copy: the sparse path
    # returns ``user_lse`` unchanged. Identity is what pins that down -- an
    # implementation that scaled a runner-owned copy would still match on values.
    assert returned is user_lse
    assert returned.shape == shape
    expected = inputs.expected_lse(return_lse_base).reshape(shape)
    torch.testing.assert_close(returned, expected, **_LSE_TOL)


@pytest.mark.parametrize("return_lse_base", [None, "base2", "basee"])
@pytest.mark.parametrize(
    "num_tokens,num_heads,topk",
    [
        pytest.param(*_DECODE_CASE, id="decode"),
        pytest.param(128, 8, 2048, id="prefill_sg"),
    ],
)
def test_sparse_mla_lse_base_preserves_masked_row_sentinel(
    num_tokens: int, num_heads: int, topk: int, return_lse_base: str | None
) -> None:
    """Empty KV without a sink has negative-infinite LSE in either base."""
    inputs = _Inputs(num_tokens, num_heads, topk, build_reference=False)
    zero_lens = torch.zeros(num_tokens, dtype=torch.int32, device=inputs.device)
    _, lse = _run(inputs, return_lse_base=return_lse_base, seq_lens=zero_lens)

    torch.testing.assert_close(lse, torch.full_like(lse, -float("inf")))


# --------------------------------------------------------------------------------------
# DOTS3_SWA prefill -- the fifth LSE store
# --------------------------------------------------------------------------------------
#
# ``sparse_mla_prefill_math_pc`` (prefill_mg_kernel.cuh) is a producer/consumer
# path that only instantiates when ``PrefillTileCfg<MT>::SPLIT_QK_XV`` holds,
# i.e. ``MATH_WARPS != QK_WARPS``. That is true for DOTS3_SWA alone (QK_WARPS=4
# vs MATH_WARPS=8); the DeepSeek family has 8/8 and runs the serial path. So it
# is a distinct LSE store from the four above and needs its own coverage.
#
# It is not reachable through ``trtllm_batch_decode_with_kv_cache_mla``:
# ``_trtllm_batch_decode_sparse_mla_v32_sm120`` requires ``query.size(-1) == 576``
# and DOTS3_SWA is 1088. The public surface that reaches it is
# ``flashinfer.mla.SparseMLASm120Wrapper.run``, which takes ``lse_scale``
# directly rather than ``return_lse_base``.

D_QK_DOTS3 = 1088
D_V_DOTS3 = 1024
SM_SCALE_DOTS3 = D_QK_DOTS3**-0.5
# BI=32 for DOTS3_SWA (see _make_decode_scratch), and the binding requires whole
# index tiles, so topk must be a multiple of 32. 576 is the tightest such value
# above the 513-token sliding window.
_DOTS3_TOPK = 576
# > _DECODE_MAX_TOKENS so the planner routes to prefill rather than decode.
_DOTS3_PREFILL_TOKENS = 128
# The head counts DISPATCH_DOTS3_SWA_SG instantiates.
_DOTS3_HEADS = [8, 16, 32, 64]


class _Dots3SwaInputs:
    """Inputs for one DOTS3_SWA prefill call plus its PyTorch reference."""

    def __init__(
        self,
        num_tokens: int = _DOTS3_PREFILL_TOKENS,
        num_heads: int = 8,
        topk: int = _DOTS3_TOPK,
        *,
        seed: int = 0,
        build_reference: bool = True,
    ) -> None:
        torch.manual_seed(seed)
        device = torch.device("cuda")
        num_blocks = 2 * topk // PAGE_BLOCK_SIZE
        s_kv = num_blocks * PAGE_BLOCK_SIZE

        kv_bf16 = (
            torch.randn(
                num_blocks,
                PAGE_BLOCK_SIZE,
                1,
                D_QK_DOTS3,
                device=device,
                dtype=torch.bfloat16,
            )
            / 10.0
        ).clamp(-1, 1)
        self.kv_packed = quantize_kv_dots3_swa(kv_bf16)

        self.q = (
            torch.randn(
                num_tokens, num_heads, D_QK_DOTS3, device=device, dtype=torch.bfloat16
            )
            / 10.0
        ).clamp(-1, 1)
        self.indices = torch.randint(
            0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
        )
        self.topk = topk
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.device = device

        self.ref_out = None
        self.ref_lse_base2 = None
        if build_reference:
            self.ref_out, self.ref_lse_base2 = _ref_sparse_attn(
                self.q,
                dequantize_kv_dots3_swa(self.kv_packed),
                self.indices,
                SM_SCALE_DOTS3,
                D_V_DOTS3,
            )


def _run_dots3_swa(
    inputs: _Dots3SwaInputs,
    *,
    lse_scale: float,
    topk_length: torch.Tensor | None = None,
):
    """Call the DOTS3_SWA prefill path through the public wrapper."""
    runner = flashinfer.mla.SparseMLASm120Wrapper(d_v=D_V_DOTS3, device=inputs.device)
    output = torch.empty(
        inputs.num_tokens,
        inputs.num_heads,
        D_V_DOTS3,
        device=inputs.device,
        dtype=torch.bfloat16,
    )
    out_lse = torch.empty(
        inputs.num_tokens,
        inputs.num_heads,
        device=inputs.device,
        dtype=torch.float32,
    )
    lse = runner.run(
        inputs.q,
        inputs.kv_packed,
        inputs.indices,
        output,
        SM_SCALE_DOTS3,
        out_lse=out_lse,
        topk_length=topk_length,
        return_lse=True,
        lse_scale=lse_scale,
    )
    return output.clone(), lse.clone()


@pytest.mark.parametrize("num_heads", _DOTS3_HEADS)
def test_dots3_swa_prefill_lse_scale_is_applied(num_heads: int) -> None:
    """lse_scale must reach the DOTS3_SWA producer/consumer store.

    The regression this pins: ``sparse_mla_prefill_math_pc`` stored ``lse``
    verbatim while every sibling store multiplied by ``cold.lse_scale``, so a
    caller asking for base-e silently received base-2.
    """
    inputs = _Dots3SwaInputs(num_heads=num_heads)

    _, lse_base2 = _run_dots3_swa(inputs, lse_scale=1.0)
    _, lse_base_e = _run_dots3_swa(inputs, lse_scale=1.0 / LOG2E)

    torch.testing.assert_close(lse_base2, inputs.ref_lse_base2, **_LSE_TOL)
    torch.testing.assert_close(lse_base_e, inputs.ref_lse_base2 / LOG2E, **_LSE_TOL)
    assert not torch.allclose(lse_base_e, lse_base2, **_LSE_TOL), (
        "lse_scale was not applied: base-e and base-2 came back identical"
    )


def test_dots3_swa_prefill_lse_scale_leaves_output_unchanged() -> None:
    """The scale lands on the LSE store only. The kernel reuses the running
    max/sum to normalize the output, so a scale applied too early corrupts it."""
    inputs = _Dots3SwaInputs(build_reference=False)

    out_unit, _ = _run_dots3_swa(inputs, lse_scale=1.0)
    out_scaled, _ = _run_dots3_swa(inputs, lse_scale=1.0 / LOG2E)

    assert torch.equal(out_unit, out_scaled), (
        "lse_scale must not change the attention output"
    )


def test_dots3_swa_prefill_preserves_masked_row_sentinel() -> None:
    """DOTS3_SWA preserves negative-infinite LSE for empty KV rows."""
    inputs = _Dots3SwaInputs(build_reference=False)
    zero_lens = torch.zeros(inputs.num_tokens, dtype=torch.int32, device=inputs.device)

    _, lse = _run_dots3_swa(inputs, lse_scale=1.0 / LOG2E, topk_length=zero_lens)

    torch.testing.assert_close(lse, torch.full_like(lse, -float("inf")))


@pytest.mark.parametrize("num_heads", [64, 128])
@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_mla_lse_base_swapab(monkeypatch, num_heads, with_sink):
    """Public auto and forced swapAB scale only final LSE, including empty rows."""
    from flashinfer.mla import _sparse_mla_sm120 as sm
    from flashinfer.mla._sparse_mla_sm120 import _prepared, KernelVariant

    inputs = _Inputs(
        65,
        num_heads,
        512,
        with_sink=with_sink,
        vary_lengths=True,
        build_reference=False,
    )
    inputs.topk_lengths[0] = 0
    kv_packed = inputs.kv_hnd.transpose(1, 2)
    ref_out, ref_lse = _ref_sparse_attn(
        inputs.q,
        dequantize_kv_dsv3_2(kv_packed),
        inputs.indices,
        SM_SCALE,
        D_V,
        attn_sink=inputs.sink,
        topk_length=inputs.topk_lengths,
    )
    selected = []
    real_execute = _prepared.PreparedCall.execute

    def spy_execute(prepared, *args, **kwargs):
        variant = prepared.plan.inspect()["variant"]
        assert variant == int(KernelVariant.PREFILL_SWAPAB)
        selected.append(variant)
        return real_execute(prepared, *args, **kwargs)

    monkeypatch.setattr(_prepared.PreparedCall, "execute", spy_execute)
    previous_out = previous_lse = None
    for lse_base in (None, "base2", "basee"):
        scale = 1 / LOG2E if lse_base == "basee" else 1.0
        out, lse = _run(inputs, return_lse_base=lse_base)
        forced_out = torch.empty_like(out)
        forced_lse = torch.empty_like(lse)
        sm._sparse_mla_sm120_paged_attention(
            inputs.q,
            kv_packed,
            inputs.indices,
            forced_out,
            forced_lse,
            SM_SCALE,
            topk_length=inputs.topk_lengths,
            attn_sink=inputs.sink,
            prefill_impl="swapab",
            lse_scale=scale,
        )
        expected_lse = ref_lse * scale
        torch.testing.assert_close(out, ref_out, **_LSE_TOL)
        torch.testing.assert_close(lse, expected_lse, **_LSE_TOL)
        assert torch.equal(forced_out, out)
        assert torch.equal(forced_lse, lse)
        if previous_out is not None:
            assert torch.equal(out, previous_out)
        if lse_base == "base2":
            assert torch.equal(lse, previous_lse)
        if not with_sink:
            assert torch.equal(lse[0], torch.full_like(lse[0], -float("inf")))
        previous_out, previous_lse = out, lse
    assert len(selected) == 6
