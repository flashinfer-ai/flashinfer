"""LSE base selection for the SM120/SM121 sparse MLA backend (issue #4485).

Companion to ``test_mla_lse_base.py``, which covers trtllm-gen and cute-dsl.
``return_lse_base_on_e`` is a *guarantee* about the units of the returned LSE,
not a transformation request:

    None   -> the backend's default base (base-2 for sparse, per the docstring
              of ``trtllm_batch_decode_with_kv_cache_mla``)
    False  -> base-2, whichever backend ran
    True   -> base-e, whichever backend ran

The SM120 sparse kernels are log2 throughout and store base-2, so ``None`` and
``False`` must stay bit-identical to the pre-#4485 output; only ``True`` scales.

There are exactly three user-facing LSE stores reachable through
``backend="sparse"``, and each needs the scale applied independently:

  * ``decode_dsv4_kernel.cuh:912`` -- ``sparse_mla_decode_dsv4_merge_kernel``,
    shared by both decode entries (``csrc/sparse_mla_sm120_decode_dsv3_2.cu``
    and ``..._dsv4.cu``). Reached when ``num_tokens <= 64``.
  * ``prefill_kernel.cuh:619``  -- SG prefill, ``num_heads <= 16``.
  * ``prefill_kernel.cuh:1634`` -- MG prefill, ``num_heads in {32, 64, 128}``.

``_CASES`` below pins one shape per store. Testing one prefill kernel does not
cover the other: SG and MG are separate templates with their own epilogues.

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
from tests.attention.test_sparse_mla_sm120 import (
    _ref_sparse_attn,
    dequantize_kv_dsv3_2,
    quantize_kv_dsv3_2,
)

pytestmark = pytest.mark.skipif(
    not is_sm12x_supported(torch.device("cuda")),
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
    """Mirror ``_DECODE_MAX_TOKENS`` in ``flashinfer/mla/_sparse_mla_sm120.py``."""
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

    def expected_lse(self, return_lse_base_on_e: bool | None) -> torch.Tensor:
        """Reference LSE in the base the flag promises."""
        if return_lse_base_on_e is True:
            return self.ref_lse_base2 / LOG2E
        return self.ref_lse_base2

    def wrong_lse(self, return_lse_base_on_e: bool | None) -> torch.Tensor:
        """The *other* base, to catch a scale that was never applied."""
        if return_lse_base_on_e is True:
            return self.ref_lse_base2
        return self.ref_lse_base2 / LOG2E


def _run(
    inputs: _Inputs,
    *,
    return_lse_base_on_e: bool | None,
    return_lse: bool = True,
    lse: torch.Tensor | None = None,
    seq_lens=_UNSET,
):
    """Call the public sparse path; clone results so later calls can't alias them."""
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
        return_lse_base_on_e=return_lse_base_on_e,
    )
    if not return_lse:
        return result.squeeze(1).clone()
    out, out_lse = result
    return out.squeeze(1).clone(), out_lse.clone()


# --------------------------------------------------------------------------------------
# Base selection, one case per LSE store
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("return_lse_base_on_e", [None, False, True])
@pytest.mark.parametrize("num_tokens,num_heads,topk", _CASES)
def test_sparse_mla_lse_base(
    num_tokens: int, num_heads: int, topk: int, return_lse_base_on_e: bool | None
) -> None:
    inputs = _Inputs(num_tokens, num_heads, topk)
    out, lse = _run(inputs, return_lse_base_on_e=return_lse_base_on_e)

    torch.testing.assert_close(out, inputs.ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        lse, inputs.expected_lse(return_lse_base_on_e), **_LSE_TOL
    )
    assert not torch.allclose(
        lse, inputs.wrong_lse(return_lse_base_on_e), **_LSE_TOL
    ), f"return_lse_base_on_e={return_lse_base_on_e} returned the other base"


@pytest.mark.parametrize("num_tokens,num_heads,topk", _CASES)
def test_sparse_mla_lse_base_none_matches_false(
    num_tokens: int, num_heads: int, topk: int
) -> None:
    """base-2 is the sparse default, so None and False must not merely agree
    numerically -- they must be the same store with the same 1.0 multiplier."""
    inputs = _Inputs(num_tokens, num_heads, topk)
    _, lse_none = _run(inputs, return_lse_base_on_e=None)
    _, lse_false = _run(inputs, return_lse_base_on_e=False)

    assert torch.equal(lse_none, lse_false), (
        "return_lse_base_on_e=None and False must be bit-identical on the "
        "sparse backend (both are base-2)"
    )


@pytest.mark.parametrize("num_tokens,num_heads,topk", _CASES)
def test_sparse_mla_lse_base_leaves_output_unchanged(
    num_tokens: int, num_heads: int, topk: int
) -> None:
    """The scale must land on the LSE store only. The kernels reuse the running
    max/sum to normalize the output, so a scale applied too early corrupts it."""
    inputs = _Inputs(num_tokens, num_heads, topk)
    out_none, _ = _run(inputs, return_lse_base_on_e=None)
    out_false, _ = _run(inputs, return_lse_base_on_e=False)
    out_true, _ = _run(inputs, return_lse_base_on_e=True)

    assert torch.equal(out_none, out_false)
    assert torch.equal(out_none, out_true), (
        "return_lse_base_on_e must not change the attention output"
    )


@pytest.mark.parametrize("return_lse_base_on_e", [False, True])
@pytest.mark.parametrize(
    "num_tokens,num_heads,topk",
    [
        pytest.param(*_DECODE_CASE, id="decode"),
        pytest.param(*_PREFILL_MG_CASE, id="prefill_mg"),
    ],
)
def test_sparse_mla_lse_base_with_sink(
    num_tokens: int, num_heads: int, topk: int, return_lse_base_on_e: bool
) -> None:
    """attn_sink is merged into the LSE in log2 space immediately before the
    store (``lse += log2f(1 + exp2f(sink_log2 - lse))``), so the base scale has
    to be applied after the merge, not to the pre-sink value."""
    inputs = _Inputs(num_tokens, num_heads, topk, with_sink=True)
    out, lse = _run(inputs, return_lse_base_on_e=return_lse_base_on_e)

    torch.testing.assert_close(out, inputs.ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        lse, inputs.expected_lse(return_lse_base_on_e), **_LSE_TOL
    )
    assert not torch.allclose(
        lse, inputs.wrong_lse(return_lse_base_on_e), **_LSE_TOL
    ), f"return_lse_base_on_e={return_lse_base_on_e} returned the other base"


@pytest.mark.parametrize("return_lse_base_on_e", [None, False, True])
@pytest.mark.parametrize(
    "num_tokens,num_heads,topk",
    [
        pytest.param(*_DECODE_CASE, id="decode"),
        pytest.param(*_PREFILL_MG_CASE, id="prefill_mg"),
    ],
)
def test_sparse_mla_lse_base_varying_topk_length(
    num_tokens: int, num_heads: int, topk: int, return_lse_base_on_e: bool | None
) -> None:
    """Same check as ``test_sparse_mla_lse_base``, but with per-token
    topk_length so the LSE actually varies across rows -- a scale applied to a
    stale or shared register would still match a constant reference."""
    inputs = _Inputs(num_tokens, num_heads, topk, vary_lengths=True)
    out, lse = _run(inputs, return_lse_base_on_e=return_lse_base_on_e)

    assert lse.std() > 0.1, "topk_length variation did not reach the LSE"
    torch.testing.assert_close(out, inputs.ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        lse, inputs.expected_lse(return_lse_base_on_e), **_LSE_TOL
    )
    assert not torch.allclose(
        lse, inputs.wrong_lse(return_lse_base_on_e), **_LSE_TOL
    ), f"return_lse_base_on_e={return_lse_base_on_e} returned the other base"


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
    """return_lse_base_on_e is meaningless without return_lse; it must be a
    silent no-op rather than a raise or a scale on a null LSE pointer."""
    inputs = _Inputs(num_tokens, num_heads, topk)
    out = _run(inputs, return_lse_base_on_e=True, return_lse=False)

    torch.testing.assert_close(out, inputs.ref_out, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("return_lse_base_on_e", [None, False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_sparse_mla_lse_base_writes_user_lse_buffer(
    nested: bool, return_lse_base_on_e: bool | None
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

    _, returned = _run(inputs, return_lse_base_on_e=return_lse_base_on_e, lse=user_lse)

    assert returned.shape == shape
    expected = inputs.expected_lse(return_lse_base_on_e).reshape(shape)
    torch.testing.assert_close(returned, expected, **_LSE_TOL)
    torch.testing.assert_close(user_lse, expected, **_LSE_TOL)


@pytest.mark.parametrize("return_lse_base_on_e", [None, False, True])
@pytest.mark.parametrize(
    "num_tokens,num_heads,topk",
    [
        pytest.param(*_DECODE_CASE, id="decode"),
        pytest.param(128, 8, 2048, id="prefill_sg"),
    ],
)
def test_sparse_mla_lse_base_preserves_masked_row_sentinel(
    num_tokens: int, num_heads: int, topk: int, return_lse_base_on_e: bool | None
) -> None:
    """A fully masked row stores the magic value -1e30, not a log-domain number
    (``sm_glse = (total_sum > 0.f) ? ... : -1e30f`` at decode_dsv4_kernel.cuh:865,
    and the -1e30 init at prefill).

    CONTRACT CHOICE: the sentinel is asserted to survive the base scale exactly.
    -1e30 is not an LSE in either base, and callers compare against the literal,
    so scaling it to -6.93e29 would silently break that comparison. If the
    implementation decides to scale it instead, this assertion is the one to
    change -- it is not a kernel bug either way.
    """
    inputs = _Inputs(num_tokens, num_heads, topk, build_reference=False)
    zero_lens = torch.zeros(num_tokens, dtype=torch.int32, device=inputs.device)
    _, lse = _run(inputs, return_lse_base_on_e=return_lse_base_on_e, seq_lens=zero_lens)

    torch.testing.assert_close(lse, torch.full_like(lse, -1e30))
