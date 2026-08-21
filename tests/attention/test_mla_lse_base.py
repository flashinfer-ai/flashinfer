"""LSE base selection for trtllm-gen MLA decode (issue #4485).

The returned log-sum-exp is base-2 by default on trtllm-gen. ``return_lse_base_on_e``
is a *guarantee* about the units of the returned tensor, not a transformation request:

    None   -> the backend's default base (base-2 for trtllm-gen)
    False  -> base-2, whichever backend ran
    True   -> base-e, whichever backend ran

The conversion is a float multiplier applied in ``ComputeLSEFromMDKernel``
(``include/flashinfer/trtllm/fmha/lse.cuh``), so the ``None``/``False`` paths multiply
by exactly ``1.0f`` and must stay bit-identical to the pre-#4485 output.

``test_trtllm_ragged_lse_is_base2`` guards a path that has *no* flag: the ragged
launcher shares ``TllmGenFmhaRunnerParams::lseScale`` with the paged one, and the
struct's constructor memsets itself, so a launcher that forgets to set the field
silently emits an all-zero LSE.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability

LOG2E = math.log2(math.e)  # 1.4426950408889634

WORKSPACE_BYTES = 128 * 1024 * 1024


def _require_trtllm_gen(device: torch.device) -> None:
    major, minor = get_compute_capability(device)
    # SM100 (B200) and SM103 (B300) only; an SM101/SM102 part would otherwise
    # fall through to an unsupported launch instead of skipping.
    if (major, minor) not in ((10, 0), (10, 3)):
        pytest.skip(
            "trtllm-gen requires SM100/SM103, got "
            f"sm{major}{minor}; the LSE base is kernel-side and cannot be checked here"
        )


def _workspace(device: torch.device) -> torch.Tensor:
    return torch.empty(WORKSPACE_BYTES, dtype=torch.int8, device=device)


# --------------------------------------------------------------------------------------
# MLA decode
# --------------------------------------------------------------------------------------

BATCH_SIZE = 4
NUM_HEADS = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576, post absorption
PAGE_SIZE = 64
SEQ_LEN = 256
# The kernel scales the QK product by bmm1_scale; use the pre-absorption head dim,
# matching tests/attention/test_trtllm_gen_mla.py.
BMM1_SCALE = 1.0 / ((QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** 0.5)

_FP8 = torch.float8_e4m3fn

# bf16 and fp8 select different monolithic kernels -- mla_decode_fp16.py and
# mla_decode_fp8.py are near-duplicate files, each carrying its own LSE stores, so
# neither is covered by testing the other.
_MLA_DTYPES = [
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(_FP8, id="fp8"),
]

# The monolithic kernel has two user-facing LSE stores on mutually exclusive paths,
# picked by split_kv. _get_split_kv_and_workspace_size normalizes the occupancy
# estimate to ceil_div(max_seq_len, 128) K tiles, so a max_seq_len that fits in one
# tile pins split_kv == 1 and the epilogue writes mLSE itself; anything longer hands
# the user-facing write to reduction_kernel. Both stores need the scale applied.
SEQ_LEN_SINGLE_TILE = 128


def _lse_tolerance(dtype: torch.dtype) -> dict:
    """Absolute LSE agreement with an fp32 reference, per input dtype."""
    return {"rtol": 0.1, "atol": 0.2} if dtype == _FP8 else {"rtol": 2e-2, "atol": 2e-2}


# The two bases differ by a factor of log2(e), which on these inputs is a gap of
# roughly 2.0 -- far outside the loosest tolerance above, so the negative checks stay
# discriminating even for fp8. Held at the loosest tolerance deliberately: a tighter
# one would make "not close" easier to satisfy and weaken the assertion.
_WRONG_BASE_TOL = {"rtol": 0.1, "atol": 0.2}


def _mla_decode_inputs(
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
    seq_len: int = SEQ_LEN,
):
    torch.manual_seed(seed)
    # torch.randn has no fp8 kernel, so every tensor is drawn in fp32 and cast.
    # e4m3 keeps 3 mantissa bits and saturates early, so fp8 inputs are damped
    # first -- same conditioning as tests/attention/test_cute_dsl_mla_decode.py.
    # The bf16 multiplier is exactly 1.0, leaving those draws bit-identical.
    damp = 0.1 if dtype == _FP8 else 1.0

    # One query token per request: no intra-request causal mask to mirror.
    query = (
        torch.randn(
            BATCH_SIZE, 1, NUM_HEADS, QK_HEAD_DIM, device=device, dtype=torch.float32
        )
        * damp
    ).to(dtype)

    blocks_per_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = blocks_per_seq * BATCH_SIZE
    kv_cache = (
        torch.randn(
            num_blocks, PAGE_SIZE, QK_HEAD_DIM, device=device, dtype=torch.float32
        )
        * damp
    ).to(dtype)
    # Shuffled page ids, so a reference that ignores block_tables cannot pass.
    block_tables = torch.randperm(num_blocks, device=device, dtype=torch.int32).reshape(
        BATCH_SIZE, blocks_per_seq
    )
    seq_lens = torch.full((BATCH_SIZE,), seq_len, dtype=torch.int32, device=device)
    return query, kv_cache, block_tables, seq_lens


def _mla_reference_lse_natural(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Natural-log LSE, [batch * q_len, num_heads], accumulated in fp32.

    MLA post-absorption is a single 576-wide dot product against the compressed KV
    row, so the reference needs no nope/rope split.
    """
    batch_size = query.shape[0]
    page_size = kv_cache.shape[1]
    q = query.float()
    # Upcast before gathering: advanced indexing on fp8 is patchy, and casting
    # before the gather is numerically identical to casting after it.
    kv_all = kv_cache.float()
    rows = []
    for b in range(batch_size):
        kv_len = int(seq_lens[b].item())
        num_pages = (kv_len + page_size - 1) // page_size
        pages = block_tables[b, :num_pages].long()
        kv = kv_all[pages].reshape(-1, kv_all.shape[-1])[:kv_len]
        scores = torch.einsum("qhd,ld->qhl", q[b], kv) * BMM1_SCALE
        rows.append(torch.logsumexp(scores, dim=-1))  # [q_len, num_heads]
    return torch.cat(rows, dim=0)


def _run_mla_decode(
    query,
    kv_cache,
    block_tables,
    seq_lens,
    *,
    return_lse_base_on_e,
    backend="trtllm-gen",
    cute_dsl_impl="auto",
    provide_lse=True,
    max_seq_len=SEQ_LEN,
):
    # cute-dsl writes LSE as [B, q_len, H]; trtllm-gen as [tokens, H]. With one
    # query token per request those are the same elements, so the caller-visible
    # tensor is reshaped to [tokens, H] before comparison. cute-dsl runs with
    # provide_lse=False so the test makes no claim about it accepting a 2D buffer.
    lse = (
        torch.full(
            (query.shape[0] * query.shape[1], NUM_HEADS),
            float("nan"),
            device=query.device,
            dtype=torch.float32,
        )
        if provide_lse
        else None
    )
    out, lse_out = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=_workspace(query.device),
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=BMM1_SCALE,
        bmm2_scale=1.0,
        backend=backend,
        cute_dsl_impl=cute_dsl_impl,
        lse=lse,
        return_lse=True,
        return_lse_base_on_e=return_lse_base_on_e,
    )
    if provide_lse:
        assert lse_out is lse
    lse_out = lse_out.reshape(-1, NUM_HEADS)
    assert torch.isfinite(lse_out).all(), "LSE contains NaN/Inf"
    return out, lse_out


def _run_cute_dsl_decode(
    query, kv_cache, block_tables, seq_lens, *, flag, max_seq_len=SEQ_LEN
):
    return _run_mla_decode(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        return_lse_base_on_e=flag,
        max_seq_len=max_seq_len,
        backend="cute-dsl",
        # Pin monolithic: the modular impl raises NotImplementedError on
        # return_lse, so "auto" would make the test depend on the dispatcher.
        cute_dsl_impl="monolithic",
        provide_lse=False,
    )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("return_lse_base_on_e", [None, False, True])
def test_trtllm_gen_mla_decode_lse_base(return_lse_base_on_e):
    """Each flag state lands on the base it promises, checked against fp32 softmax."""
    device = torch.device("cuda")
    _require_trtllm_gen(device)

    query, kv_cache, block_tables, seq_lens = _mla_decode_inputs(device, torch.bfloat16)
    _, lse = _run_mla_decode(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        return_lse_base_on_e=return_lse_base_on_e,
    )

    ref_natural = _mla_reference_lse_natural(query, kv_cache, block_tables, seq_lens)
    # None and False both mean base-2 on trtllm-gen; only True is base-e.
    expected = ref_natural if return_lse_base_on_e is True else ref_natural * LOG2E

    # bf16 inputs, fp32 accumulation: the two bases are 1.44x apart, so this
    # tolerance still rejects the wrong one by a wide margin.
    torch.testing.assert_close(lse, expected, rtol=2e-2, atol=2e-2)

    wrong_base = ref_natural * LOG2E if return_lse_base_on_e is True else ref_natural
    assert not torch.allclose(lse, wrong_base, rtol=0.1, atol=0.1), (
        f"return_lse_base_on_e={return_lse_base_on_e} returned the other base"
    )


@pytest.mark.arch_blackwell
def test_trtllm_gen_mla_decode_lse_base_relationship():
    """None == False bit-for-bit, and True is exactly the base-2 result over log2(e)."""
    device = torch.device("cuda")
    _require_trtllm_gen(device)

    query, kv_cache, block_tables, seq_lens = _mla_decode_inputs(device, torch.bfloat16)
    runs = {
        flag: _run_mla_decode(
            query, kv_cache, block_tables, seq_lens, return_lse_base_on_e=flag
        )[1]
        for flag in (None, False, True)
    }

    # The default path multiplies by literal 1.0f, so it must not merely be close
    # to the explicit base-2 path -- it must be the same bits.
    assert torch.equal(runs[None], runs[False]), (
        "return_lse_base_on_e=None and False must be bit-identical on trtllm-gen"
    )

    # True differs from False by one fp32 multiply by 1/log2(e).
    torch.testing.assert_close(runs[True] * LOG2E, runs[False], rtol=1e-6, atol=1e-6)


@pytest.mark.arch_blackwell
def test_trtllm_gen_mla_decode_lse_base_ignored_without_return_lse():
    """Passing the flag with return_lse=False is silently ignored, not an error."""
    device = torch.device("cuda")
    _require_trtllm_gen(device)

    query, kv_cache, block_tables, seq_lens = _mla_decode_inputs(device, torch.bfloat16)
    out = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=_workspace(device),
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=SEQ_LEN,
        bmm1_scale=BMM1_SCALE,
        bmm2_scale=1.0,
        backend="trtllm-gen",
        return_lse=False,
        return_lse_base_on_e=True,
    )
    assert isinstance(out, torch.Tensor)
    assert torch.isfinite(out.float()).all()


# --------------------------------------------------------------------------------------
# Ragged prefill: no flag, shares lseScale with the paged launcher
# --------------------------------------------------------------------------------------

RAGGED_BATCH = 2
RAGGED_Q_LEN = 8
RAGGED_KV_LEN = 128
RAGGED_NUM_HEADS = 128
RAGGED_HEAD_DIM_QK = 192
RAGGED_HEAD_DIM_VO = 128


@pytest.mark.arch_blackwell
def test_trtllm_ragged_lse_is_base2():
    """trtllm_ragged_attention_deepseek keeps base-2 LSE and never emits zeros.

    The ragged launcher has no lse_scale parameter; it pins
    ``runner_params.lseScale = 1.0f``. If that assignment is dropped, the
    memset in ``TllmGenFmhaRunnerParams``'s constructor leaves 0.0f and every
    LSE comes back as zero -- finite, correctly shaped, and wrong.
    """
    device = torch.device("cuda")
    _require_trtllm_gen(device)
    if not hasattr(flashinfer.prefill, "trtllm_ragged_attention_deepseek"):
        pytest.skip("trtllm_ragged_attention_deepseek is not available")

    torch.manual_seed(42)
    q_lens = torch.full((RAGGED_BATCH,), RAGGED_Q_LEN, dtype=torch.int32, device=device)
    kv_lens = torch.full(
        (RAGGED_BATCH,), RAGGED_KV_LEN, dtype=torch.int32, device=device
    )

    def _indptr(lens):
        return torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.int32),
                torch.cumsum(lens, dim=0, dtype=torch.int32),
            ]
        )

    q_indptr, kv_indptr = _indptr(q_lens), _indptr(kv_lens)
    total_q, total_kv = int(q_indptr[-1].item()), int(kv_indptr[-1].item())

    query = torch.randn(
        total_q,
        RAGGED_NUM_HEADS,
        RAGGED_HEAD_DIM_QK,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    key = torch.randn(
        total_kv,
        RAGGED_NUM_HEADS,
        RAGGED_HEAD_DIM_QK,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    value = torch.randn(
        total_kv,
        RAGGED_NUM_HEADS,
        RAGGED_HEAD_DIM_VO,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    bmm1_scale = 1.0 / (RAGGED_HEAD_DIM_QK**0.5)
    _, lse = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        query=query,
        key=key,
        value=value,
        workspace_buffer=_workspace(device),
        seq_lens=kv_lens,
        max_q_len=RAGGED_Q_LEN,
        max_kv_len=RAGGED_KV_LEN,
        bmm1_scale=bmm1_scale,
        bmm2_scale=1.0,
        o_sf_scale=1.0,
        batch_size=RAGGED_BATCH,
        window_left=-1,
        cum_seq_lens_q=q_indptr,
        cum_seq_lens_kv=kv_indptr,
        enable_pdl=False,
        is_causal=False,
        return_lse=True,
    )

    assert lse.shape == (total_q, RAGGED_NUM_HEADS)
    assert torch.isfinite(lse).all()
    assert (lse != 0).any(), "all-zero LSE: lseScale was left at its memset value"

    ref_rows = []
    for b in range(RAGGED_BATCH):
        q_slice = query[int(q_indptr[b]) : int(q_indptr[b + 1])].float()
        k_slice = key[int(kv_indptr[b]) : int(kv_indptr[b + 1])].float()
        scores = torch.einsum("qhd,lhd->qhl", q_slice, k_slice) * bmm1_scale
        ref_rows.append(torch.logsumexp(scores, dim=-1))
    ref_base2 = torch.cat(ref_rows, dim=0) * LOG2E

    torch.testing.assert_close(lse, ref_base2, rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------------------
# cute-dsl monolithic
# --------------------------------------------------------------------------------------
#
# These kernels compute LSE in base 2 internally and apply the caller's multiplier at
# each user-facing store, so the scale mapping is the inverse of trtllm-gen's: None and
# True are both 1.0 / log2e (today's behaviour, unchanged), and only False -- base 2 --
# is a new path.
#
# "monolithic" is two near-duplicate kernels, mla_decode_fp16.py and mla_decode_fp8.py,
# selected by input dtype, and each has two user-facing stores selected by split_kv.
# The tests below cross both axes; see _SPLIT_KV_LENS.


# Both dtypes and both split_kv regimes are covered, which is what reaches all four
# monolithic store sites: {mla_decode_fp16, mla_decode_fp8} x {reduction_kernel store,
# epilogue store}. A single-dtype, single-length test leaves three of the four dead.
_SPLIT_KV_LENS = [
    pytest.param(SEQ_LEN, id="split-kv"),
    pytest.param(SEQ_LEN_SINGLE_TILE, id="single-tile"),
]


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("dtype", _MLA_DTYPES)
@pytest.mark.parametrize("seq_len", _SPLIT_KV_LENS)
@pytest.mark.parametrize("return_lse_base_on_e", [None, False, True])
def test_cute_dsl_monolithic_lse_base(return_lse_base_on_e, seq_len, dtype):
    """None and True stay base-e on monolithic; False switches it to base-2."""
    device = torch.device("cuda")
    _require_trtllm_gen(device)

    query, kv_cache, block_tables, seq_lens = _mla_decode_inputs(
        device, dtype, seq_len=seq_len
    )
    _, lse = _run_cute_dsl_decode(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        flag=return_lse_base_on_e,
        max_seq_len=seq_len,
    )

    ref_natural = _mla_reference_lse_natural(query, kv_cache, block_tables, seq_lens)
    expected = ref_natural * LOG2E if return_lse_base_on_e is False else ref_natural
    torch.testing.assert_close(lse, expected, **_lse_tolerance(dtype))

    wrong_base = ref_natural if return_lse_base_on_e is False else ref_natural * LOG2E
    assert not torch.allclose(lse, wrong_base, **_WRONG_BASE_TOL), (
        f"return_lse_base_on_e={return_lse_base_on_e} returned the other base"
    )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("dtype", _MLA_DTYPES)
@pytest.mark.parametrize("seq_len", _SPLIT_KV_LENS)
def test_cute_dsl_monolithic_lse_base_relationship(seq_len, dtype):
    """None == True bit-for-bit (same scale), and False is that times log2(e).

    Reference-free, so it isolates the scalar plumbing from kernel numerics: all three
    runs are the same kernel on the same inputs with only the runtime multiplier
    differing, which makes the ratio exact even where fp8 quantization moves the
    absolute values well away from the fp32 reference.
    """
    device = torch.device("cuda")
    _require_trtllm_gen(device)

    query, kv_cache, block_tables, seq_lens = _mla_decode_inputs(
        device, dtype, seq_len=seq_len
    )
    runs = {
        flag: _run_cute_dsl_decode(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            flag=flag,
            max_seq_len=seq_len,
        )[1]
        for flag in (None, False, True)
    }

    # Both resolve to the same 1.0 / log2e multiplier, so this is exact -- it is the
    # check that the default path did not change when the scalar was threaded through.
    assert torch.equal(runs[None], runs[True]), (
        "None and True must be bit-identical on cute-dsl monolithic (same scale)"
    )
    torch.testing.assert_close(runs[None] * LOG2E, runs[False], rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------------------
# Cross-backend: the point of the parameter
# --------------------------------------------------------------------------------------


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("return_lse_base_on_e", [False, True])
def test_lse_base_agrees_across_backends(return_lse_base_on_e):
    """An explicit flag pins the units, so which backend ran stops mattering."""
    device = torch.device("cuda")
    _require_trtllm_gen(device)

    query, kv_cache, block_tables, seq_lens = _mla_decode_inputs(device, torch.bfloat16)
    _, trt = _run_mla_decode(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        return_lse_base_on_e=return_lse_base_on_e,
    )
    _, cute = _run_cute_dsl_decode(
        query, kv_cache, block_tables, seq_lens, flag=return_lse_base_on_e
    )
    # Two different kernels, so this is a numerical comparison, not a bitwise one.
    torch.testing.assert_close(trt, cute, rtol=2e-2, atol=2e-2)


@pytest.mark.arch_blackwell
def test_lse_base_default_still_differs_across_backends():
    """None preserves each backend's default, which are one log2(e) apart.

    Documents the status quo the parameter exists to work around: with no flag,
    trtllm-gen returns base-2 and cute-dsl monolithic returns base-e, so a caller
    on backend="auto" cannot know the units. Unifying the defaults is a follow-up.
    """
    device = torch.device("cuda")
    _require_trtllm_gen(device)

    query, kv_cache, block_tables, seq_lens = _mla_decode_inputs(device, torch.bfloat16)
    _, trt = _run_mla_decode(
        query, kv_cache, block_tables, seq_lens, return_lse_base_on_e=None
    )
    _, cute = _run_cute_dsl_decode(query, kv_cache, block_tables, seq_lens, flag=None)
    ratio = trt / cute
    torch.testing.assert_close(
        ratio, torch.full_like(ratio, LOG2E), rtol=2e-2, atol=2e-2
    )
