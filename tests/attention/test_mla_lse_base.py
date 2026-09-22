"""LSE base selection for trtllm-gen MLA decode (issue #4485).

The returned log-sum-exp is base-2 by default on trtllm-gen. ``return_lse_base``
is a *guarantee* about the units of the returned tensor, not a transformation request:

    None   -> the backend's default base (base-2 for trtllm-gen)
    "base2" -> base-2, whichever backend ran
    "basee" -> base-e, whichever backend ran

The conversion is a float multiplier applied in ``ComputeLSEFromMDKernel``
(``include/flashinfer/trtllm/fmha/lse.cuh``), so the ``None``/``"base2"`` paths multiply
by exactly ``1.0f`` and must stay bit-identical to the pre-#4485 output.

``test_trtllm_ragged_lse_is_base2`` guards a path that has *no* flag: the ragged
launcher shares ``TllmGenFmhaRunnerParams::lseScale`` with the paged one, and the
struct's constructor memsets itself, so a launcher that forgets to set the field
silently emits an all-zero LSE.
"""

import math
from types import SimpleNamespace

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability

LOG2E = math.log2(math.e)  # 1.4426950408889634

WORKSPACE_BYTES = 128 * 1024 * 1024


def _public_mla_entrypoint(entrypoint):
    if entrypoint == "decode":
        return flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla
    if entrypoint == "prefill":
        return flashinfer.mla.trtllm_prefill_with_kv_cache_mla
    return flashinfer.prefill.trtllm_prefill_with_kv_cache_mla


def _cpu_mla_arguments():
    """Small host tensors for validation/forwarding, never a kernel launch."""
    return dict(
        query=torch.empty(1, 1, 8, 576, dtype=torch.bfloat16),
        kv_cache=torch.empty(1, 1, 64, 576, dtype=torch.bfloat16),
        workspace_buffer=torch.empty(1, dtype=torch.uint8),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=torch.zeros(1, 1, dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int32),
        max_seq_len=1,
    )


@pytest.mark.parametrize("entrypoint", ["decode", "prefill", "prefill_alias"])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize(
    "selector", [True, False, 0, 1, -1, 2, "unknown", "none", "base_e", "base_2", ""]
)
def test_mla_lse_base_rejects_invalid_selector_before_device_probe(
    monkeypatch, entrypoint, return_lse, selector
):
    from flashinfer.mla import _core

    def unexpected_probe(*args, **kwargs):
        pytest.fail("invalid LSE selector reached device/backend probing")

    monkeypatch.setattr(_core, "get_compute_capability", unexpected_probe)
    monkeypatch.setattr(_core, "is_sm12x_supported", unexpected_probe)
    monkeypatch.setattr(torch.cuda, "current_stream", unexpected_probe)
    with pytest.raises(ValueError, match="return_lse_base"):
        _public_mla_entrypoint(entrypoint)(
            **_cpu_mla_arguments(),
            return_lse=return_lse,
            return_lse_base=selector,
        )


@pytest.mark.parametrize("entrypoint", ["decode", "prefill", "prefill_alias"])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("selector", ["omitted", None, "basee", "base2"])
def test_mla_lse_base_public_forwarding(monkeypatch, entrypoint, return_lse, selector):
    from flashinfer.mla import _core

    seen = []
    output = torch.empty(1, 1, 8, 512, dtype=torch.bfloat16)
    expected_result = (output, torch.empty(1, 8)) if return_lse else output

    def implementation(**kwargs):
        seen.append(kwargs)
        return expected_result

    monkeypatch.setattr(
        _core, "_trtllm_batch_decode_with_kv_cache_mla_impl", implementation
    )
    kwargs = _cpu_mla_arguments()
    if selector != "omitted":
        kwargs["return_lse_base"] = selector
    result = _public_mla_entrypoint(entrypoint)(**kwargs, return_lse=return_lse)
    assert result is expected_result
    assert len(seen) == 1
    assert seen[0]["return_lse_base"] == (None if selector == "omitted" else selector)
    assert seen[0]["return_lse"] is return_lse


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
    return_lse_base,
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
        return_lse_base=return_lse_base,
    )
    if provide_lse:
        assert lse_out is lse
    lse_out = lse_out.reshape(-1, NUM_HEADS)
    assert torch.isfinite(lse_out).all(), "LSE contains NaN/Inf"
    return out, lse_out


def _run_cute_dsl_decode(
    query, kv_cache, block_tables, seq_lens, *, lse_base, max_seq_len=SEQ_LEN
):
    return _run_mla_decode(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        return_lse_base=lse_base,
        max_seq_len=max_seq_len,
        backend="cute-dsl",
        # Pin monolithic: the modular impl raises NotImplementedError on
        # return_lse, so "auto" would make the test depend on the dispatcher.
        cute_dsl_impl="monolithic",
        provide_lse=False,
    )


@pytest.mark.parametrize("with_lse", [False, True])
def test_planned_trtllm_launch_lse_scale(with_lse):
    from flashinfer.mla._batch_mla._backends.trtllm_gen_backend import (
        _BatchMLAPagedAttentionTrtllmGenBackend,
    )

    calls = []
    backend = _BatchMLAPagedAttentionTrtllmGenBackend.__new__(
        _BatchMLAPagedAttentionTrtllmGenBackend
    )
    backend._module = SimpleNamespace(
        trtllm_paged_attention_decode=lambda *args: calls.append(args)
    )
    backend._float_workspace_buffer = torch.empty(16, dtype=torch.uint8)
    backend._multi_ctas_kv_counter_buffer = object()
    backend._block_tables = object()
    backend._seq_lens = object()
    backend._max_q_len = 1
    backend._max_seq_len = SEQ_LEN
    backend._batch_size = BATCH_SIZE
    backend._sm_count = 148
    backend._enable_pdl = False
    lse = object() if with_lse else None
    token_stride, head_stride = (NUM_HEADS, 1) if with_lse else (0, 0)
    backend._launch_native(
        out=object(),
        query=object(),
        kv_cache=object(),
        bmm1_scale=BMM1_SCALE,
        bmm2_scale=1.0,
        sinks=None,
        cum_seq_lens_q=None,
        skip_softmax_threshold_scale_factor=None,
        lse=lse,
        lse_stride_tokens=token_stride,
        lse_stride_heads=head_stride,
    )
    (args,) = calls
    assert len(args) == 36
    assert args[28] is lse
    assert args[29:] == (1.0, token_stride, head_stride, False, None, 0, None)


def test_planned_monolithic_launch_lse_scale():
    from flashinfer.mla._batch_mla._backends.cute_dsl_monolithic_backend import (
        _BatchMLAPagedAttentionCuteDslMonolithicBackend,
    )

    calls = []
    backend = _BatchMLAPagedAttentionCuteDslMonolithicBackend.__new__(
        _BatchMLAPagedAttentionCuteDslMonolithicBackend
    )
    backend._execution_state = SimpleNamespace(
        Int32=int, Float32=float, compiled_kernel=lambda *args: calls.append(args)
    )
    launch_args = tuple(object() for _ in range(13))
    backend._launch_compiled_kernel(launch_args, sinks=None)
    (args,) = calls
    assert len(args) == 17
    assert args[:10] == launch_args[:10]
    assert args[10:13] == (None, None, 0)
    assert args[13:16] == launch_args[10:]
    assert args[16] == math.log(2.0)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("backend", ["trtllm-gen", "cute-dsl-monolithic"])
@pytest.mark.parametrize("dtype", _MLA_DTYPES)
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("seq_len", [SEQ_LEN_SINGLE_TILE, SEQ_LEN])
def test_planned_mla_preserves_lse_base(backend, dtype, return_lse, seq_len):
    """Planned adapters retain their native LSE units after the ABI expansion."""
    device = torch.device("cuda")
    _require_trtllm_gen(device)
    query, kv_cache, block_tables, seq_lens = _mla_decode_inputs(
        device, dtype, seq_len=seq_len
    )
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        _workspace(device), backend=backend
    )
    natural = backend == "cute-dsl-monolithic"
    wrapper.plan(
        metadata=flashinfer.mla.MLAPlanMetadata.dense(
            cum_seq_lens_q=torch.arange(
                BATCH_SIZE + 1, dtype=torch.int32, device=device
            ),
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=1,
        ),
        num_heads=NUM_HEADS,
        head_dim_ckv=KV_LORA_RANK,
        head_dim_kpe=QK_ROPE_HEAD_DIM,
        page_size=PAGE_SIZE,
        causal=False,
        sm_scale=BMM1_SCALE,
        q_data_type=dtype,
        kv_data_type=dtype,
        output_dtype=torch.bfloat16,
        lse_mode=("basee" if natural else "base2") if return_lse else "none",
        scale_mode="bmm-scalar",
        enable_pdl=False,
    )
    out = torch.empty(
        (BATCH_SIZE, NUM_HEADS, KV_LORA_RANK), dtype=torch.bfloat16, device=device
    )
    lse = (
        torch.full(
            (BATCH_SIZE, NUM_HEADS), float("nan"), dtype=torch.float32, device=device
        )
        if return_lse
        else None
    )
    result = wrapper.run(
        query=query.reshape(BATCH_SIZE, NUM_HEADS, QK_HEAD_DIM),
        kv_cache=kv_cache,
        out=out,
        lse=lse,
        return_lse=return_lse,
        return_lse_base_on_e=natural and return_lse,
        bmm1_scale=BMM1_SCALE,
        bmm2_scale=1.0,
    )
    assert (result[0] if return_lse else result) is out
    expected_out = []
    for batch_idx in range(BATCH_SIZE):
        kv = kv_cache.float()[block_tables[batch_idx].long()].reshape(-1, QK_HEAD_DIM)
        scores = query[batch_idx, 0].float() @ kv.T * BMM1_SCALE
        expected_out.append(scores.softmax(dim=-1) @ kv[:, :KV_LORA_RANK])
    expected_out = torch.stack(expected_out)
    # FP8 values are damped by 0.1, giving outputs of order 0.1/sqrt(seq_len).
    # The LSE atol=0.2 would accept an all-zero attention output at these scales.
    output_tolerance = (
        {"rtol": 0.1, "atol": 1e-3} if dtype == _FP8 else _lse_tolerance(dtype)
    )
    assert not torch.allclose(
        torch.zeros_like(expected_out), expected_out, **output_tolerance
    )
    torch.testing.assert_close(out.float(), expected_out, **output_tolerance)
    if return_lse:
        assert result[1] is lse
        expected_lse = _mla_reference_lse_natural(
            query, kv_cache, block_tables, seq_lens
        )
        if not natural:
            expected_lse *= LOG2E
        torch.testing.assert_close(lse, expected_lse, **_lse_tolerance(dtype))


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("return_lse_base", [None, "base2", "basee"])
def test_trtllm_gen_mla_decode_lse_base(return_lse_base):
    """Each selector value lands on the base it promises, checked against fp32 softmax."""
    device = torch.device("cuda")
    _require_trtllm_gen(device)

    query, kv_cache, block_tables, seq_lens = _mla_decode_inputs(device, torch.bfloat16)
    _, lse = _run_mla_decode(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        return_lse_base=return_lse_base,
    )

    ref_natural = _mla_reference_lse_natural(query, kv_cache, block_tables, seq_lens)
    # None and "base2" both mean base-2 on trtllm-gen; only "basee" is base-e.
    expected = ref_natural if return_lse_base == "basee" else ref_natural * LOG2E

    # bf16 inputs, fp32 accumulation: the two bases are 1.44x apart, so this
    # tolerance still rejects the wrong one by a wide margin.
    torch.testing.assert_close(lse, expected, rtol=2e-2, atol=2e-2)

    wrong_base = ref_natural * LOG2E if return_lse_base == "basee" else ref_natural
    assert not torch.allclose(lse, wrong_base, rtol=0.1, atol=0.1), (
        f"return_lse_base={return_lse_base} returned the other base"
    )


@pytest.mark.arch_blackwell
def test_trtllm_gen_mla_decode_lse_base_relationship():
    """None matches base2 bit-for-bit, and basee is exactly the base-2 result over log2(e)."""
    device = torch.device("cuda")
    _require_trtllm_gen(device)

    query, kv_cache, block_tables, seq_lens = _mla_decode_inputs(device, torch.bfloat16)
    runs = {
        lse_base: _run_mla_decode(
            query, kv_cache, block_tables, seq_lens, return_lse_base=lse_base
        )[1]
        for lse_base in (None, "base2", "basee")
    }

    # The default path multiplies by literal 1.0f, so it must not merely be close
    # to the explicit base-2 path -- it must be the same bits.
    assert torch.equal(runs[None], runs["base2"]), (
        "return_lse_base=None and base2 must be bit-identical on trtllm-gen"
    )

    # basee differs from base2 by one fp32 multiply by 1/log2(e).
    torch.testing.assert_close(
        runs["basee"] * LOG2E, runs["base2"], rtol=1e-6, atol=1e-6
    )


@pytest.mark.arch_blackwell
def test_trtllm_gen_mla_decode_lse_base_ignored_without_return_lse():
    """Passing a valid selector with return_lse=False is silently ignored, not an error."""
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
        return_lse_base="basee",
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
# basee are both 1.0 / log2e (today's behaviour, unchanged), and only base2 -- base 2 --
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
@pytest.mark.parametrize("return_lse_base", [None, "base2", "basee"])
def test_cute_dsl_monolithic_lse_base(return_lse_base, seq_len, dtype):
    """None and basee stay base-e on monolithic; base2 selects base-2."""
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
        lse_base=return_lse_base,
        max_seq_len=seq_len,
    )

    ref_natural = _mla_reference_lse_natural(query, kv_cache, block_tables, seq_lens)
    expected = ref_natural * LOG2E if return_lse_base == "base2" else ref_natural
    torch.testing.assert_close(lse, expected, **_lse_tolerance(dtype))

    wrong_base = ref_natural if return_lse_base == "base2" else ref_natural * LOG2E
    assert not torch.allclose(lse, wrong_base, **_WRONG_BASE_TOL), (
        f"return_lse_base={return_lse_base} returned the other base"
    )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("dtype", _MLA_DTYPES)
@pytest.mark.parametrize("seq_len", _SPLIT_KV_LENS)
def test_cute_dsl_monolithic_lse_base_relationship(seq_len, dtype):
    """None matches basee bit-for-bit (same scale), and base2 is that times log2(e).

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
        lse_base: _run_cute_dsl_decode(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            lse_base=lse_base,
            max_seq_len=seq_len,
        )[1]
        for lse_base in (None, "base2", "basee")
    }

    # Both resolve to the same 1.0 / log2e multiplier, so this is exact -- it is the
    # check that the default path did not change when the scalar was threaded through.
    assert torch.equal(runs[None], runs["basee"]), (
        "None and basee must be bit-identical on cute-dsl monolithic (same scale)"
    )
    torch.testing.assert_close(runs[None] * LOG2E, runs["base2"], rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------------------
# Cross-backend: the point of the parameter
# --------------------------------------------------------------------------------------


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("return_lse_base", ["base2", "basee"])
def test_lse_base_agrees_across_backends(return_lse_base):
    """An explicit selector pins the units, so which backend ran stops mattering."""
    device = torch.device("cuda")
    _require_trtllm_gen(device)

    query, kv_cache, block_tables, seq_lens = _mla_decode_inputs(device, torch.bfloat16)
    _, trt = _run_mla_decode(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        return_lse_base=return_lse_base,
    )
    _, cute = _run_cute_dsl_decode(
        query, kv_cache, block_tables, seq_lens, lse_base=return_lse_base
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
        query, kv_cache, block_tables, seq_lens, return_lse_base=None
    )
    _, cute = _run_cute_dsl_decode(
        query, kv_cache, block_tables, seq_lens, lse_base=None
    )
    ratio = trt / cute
    torch.testing.assert_close(
        ratio, torch.full_like(ratio, LOG2E), rtol=2e-2, atol=2e-2
    )
