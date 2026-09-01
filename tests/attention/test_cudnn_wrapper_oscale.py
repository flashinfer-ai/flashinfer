"""o_scale plumbing tests for BatchPrefillWith{Ragged,Paged}KVCacheWrapper.run.

FlashInfer issue #4224 stage 2: both wrappers must forward ``o_scale`` to
``cudnn_batch_prefill_with_kv_cache`` so an fp8 output can be quantized
(without it, an fp8 ``o_data_type`` is emitted unit-scaled and saturates at
the e4m3 maximum of 448).  Both wrappers support the fp8 cudnn path via
``plan(..., q_data_type=fp8)`` (the paged path is exercised end-to-end by
``test_cudnn_prefill_fp8`` already), so both get dequantization coverage
here: the dequantized fp8 output must match a reference computed by the
identical fp8 graph with a bf16 output, which isolates output quantization
from input-quantization error.  The bf16 reference itself is cross-checked
against an fa2 bf16 wrapper.  ``o_scale`` is exercised both as a float
(converted internally to the ``(1, 1, 1, 1)`` fp32 GPU tensor cuDNN expects)
and as such a tensor passed through as-is.

The ragged fp8 runs pass ``out`` explicitly: the ragged wrapper's internal
allocation coerces fp8 inputs to a bf16 output, so an fp8 output buffer
(validated against the planned ``o_data_type``) must be supplied by the
caller.

Two rejection tests need no fp8 support: non-cudnn paged backends must raise
``NotImplementedError`` when ``o_scale`` is set, and a non-fp8 cudnn run must
surface ``cudnn_batch_prefill_with_kv_cache``'s ValueError (proof the
argument actually reaches the cudnn call).
"""

import pytest
import torch

import cudnn
import flashinfer
from flashinfer.utils import get_compute_capability


def _skip_unless_fp8_prefill_supported(device):
    if cudnn.backend_version() < 92600:
        pytest.skip("cuDNN FP8 prefill needs backend 9.26+, skipping test")
    major, _ = get_compute_capability(torch.device(device))
    if major != 10:
        pytest.skip(
            f"cuDNN FP8 prefill is not supported on compute capability {major}, "
            "skipping test"
        )


def _as_o_scale_arg(o_scale_value, o_scale_form, device):
    """o_scale in the two forms the wrappers accept: float or (1,1,1,1) tensor."""
    if o_scale_form == "float":
        return o_scale_value
    return torch.tensor(o_scale_value, device=device, dtype=torch.float32).reshape(
        1, 1, 1, 1
    )


def _quantize_e4m3(x):
    """Quantize to e4m3 with an amax/256 descale; returns (x_fp8, scale (1,1,1,1))."""
    scale = (x.float().abs().amax() / 256).reshape(1, 1, 1, 1)
    x_fp8 = (x.float() / scale.item()).to(torch.float8_e4m3fn)
    return x_fp8, scale


@pytest.mark.parametrize("o_scale_value", [4.0, 1.0 / 16.0])
@pytest.mark.parametrize("o_scale_form", ["float", "tensor"])
@pytest.mark.parametrize("causal", [False, True])
def test_ragged_wrapper_fp8_o_scale(o_scale_value, o_scale_form, causal):
    """Ragged wrapper end-to-end: fp8 q/k/v, fp8 output quantized by o_scale."""
    device = "cuda:0"
    _skip_unless_fp8_prefill_supported(device)

    torch.manual_seed(5)
    batch_size, s_qo, s_kv = 4, 32, 64
    num_qo_heads = num_kv_heads = 4
    head_dim = 128

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    total_q = int(actual_seq_lens_q.sum())
    total_kv = int(actual_seq_lens_kv.sum())

    q = torch.randn(
        total_q, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k = (
        torch.randn(
            total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * 0.05
    )
    v = (
        torch.randn(
            total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * 0.05
    )
    q_fp8, q_scale = _quantize_e4m3(q)
    k_fp8, k_scale = _quantize_e4m3(k)
    v_fp8, v_scale = _quantize_e4m3(v)

    zero = torch.zeros(1, dtype=torch.int64, device=device)
    qo_indptr_tok = torch.cat([zero, torch.cumsum(actual_seq_lens_q.view(-1), 0)]).int()
    kv_indptr_tok = torch.cat(
        [zero, torch.cumsum(actual_seq_lens_kv.view(-1), 0)]
    ).int()
    # The cudnn ragged path takes element-unit indptrs.
    q_indptr = qo_indptr_tok * num_qo_heads * head_dim
    kv_indptr = kv_indptr_tok * num_kv_heads * head_dim

    scale = float(1.0 / (head_dim**0.5))
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    def _plan_cudnn(o_data_type):
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, "NHD", backend="cudnn"
        )
        wrapper.plan(
            q_indptr,
            kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            causal=causal,
            sm_scale=scale,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=o_data_type,
            seq_lens=actual_seq_lens_kv,
            seq_lens_q=actual_seq_lens_q,
            max_token_per_sequence=s_qo,
            max_sequence_kv=s_kv,
        )
        return wrapper

    # Reference: identical fp8 compute, bf16 output (no output quantization).
    out_ref = _plan_cudnn(torch.bfloat16).run(
        q_fp8, k_fp8, v_fp8, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale
    )
    assert out_ref.dtype == torch.bfloat16
    assert out_ref.float().abs().amax() > 0

    # Cross-check the reference against a bf16 fa2 wrapper so a garbage fp8
    # graph cannot self-validate.
    wrapper_ref = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device), "NHD"
    )
    wrapper_ref.plan(
        qo_indptr_tok,
        kv_indptr_tok,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        sm_scale=scale,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    out_fa2 = wrapper_ref.run(q, k, v)
    torch.testing.assert_close(out_ref, out_fa2, atol=1e-2, rtol=1e-2)

    out_fp8_buf = torch.empty(
        total_q, num_qo_heads, head_dim, device=device, dtype=torch.float8_e4m3fn
    )
    out_fp8 = _plan_cudnn(torch.float8_e4m3fn).run(
        q_fp8,
        k_fp8,
        v_fp8,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        o_scale=_as_o_scale_arg(o_scale_value, o_scale_form, device),
        out=out_fp8_buf,
    )
    assert out_fp8.dtype == torch.float8_e4m3fn

    # Dequantized output must match the pre-quantization reference; the atol
    # floor covers e4m3's subnormal grid (2^-9) mapped back through o_scale.
    torch.testing.assert_close(
        out_fp8.float() / o_scale_value,
        out_ref.float(),
        atol=2**-9 / o_scale_value + 5e-3,
        rtol=1e-1,
    )


@pytest.mark.parametrize("o_scale_value", [4.0, 1.0 / 16.0])
@pytest.mark.parametrize("o_scale_form", ["float", "tensor"])
@pytest.mark.parametrize("causal", [False, True])
def test_paged_wrapper_fp8_o_scale(o_scale_value, o_scale_form, causal):
    """Paged wrapper end-to-end: fp8 q + fp8 paged KV, fp8 output via o_scale."""
    device = "cuda:0"
    _skip_unless_fp8_prefill_supported(device)

    torch.manual_seed(6)
    batch_size, s_qo, s_kv, page_size = 4, 32, 64, 16
    num_qo_heads = num_kv_heads = 4
    head_dim = 128

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    total_q = int(actual_seq_lens_q.sum())

    q = torch.randn(
        total_q, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    q_fp8, q_scale = _quantize_e4m3(q)

    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_cache = (
        torch.randn(
            total_num_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.05
    )
    k_cache = kv_cache[:, 0]
    v_cache = kv_cache[:, 1]
    k_cache_fp8, k_scale = _quantize_e4m3(k_cache)
    v_cache_fp8, v_scale = _quantize_e4m3(v_cache)

    zero = torch.zeros(1, dtype=torch.int64, device=device)
    qo_indptr_tok = torch.cat([zero, torch.cumsum(actual_seq_lens_q.view(-1), 0)]).int()
    # The cudnn paged path takes element-unit q offsets.
    q_indptr = qo_indptr_tok * num_qo_heads * head_dim
    kv_indptr = torch.cat(
        [
            zero,
            torch.cumsum((actual_seq_lens_kv.view(-1) + page_size - 1) // page_size, 0),
        ]
    ).int()
    kv_indices = torch.zeros(int(kv_indptr[-1]), device=device, dtype=torch.int32)
    for i in range(batch_size):
        start_idx, end_idx = int(kv_indptr[i]), int(kv_indptr[i + 1])
        kv_indices[start_idx:end_idx] = torch.arange(
            i * num_pages_per_seq,
            i * num_pages_per_seq + (end_idx - start_idx),
            device=device,
        )
    kv_last_page_len = torch.where(
        actual_seq_lens_kv.flatten() % page_size == 0,
        torch.full((batch_size,), page_size, device=device),
        actual_seq_lens_kv.flatten() % page_size,
    ).int()
    block_tables = torch.arange(
        total_num_pages, dtype=torch.int32, device=device
    ).reshape(batch_size, num_pages_per_seq)

    scale = float(1.0 / (head_dim**0.5))
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    def _plan_cudnn(o_data_type):
        # The cudnn wrapper only accepts the "NHD" layout string; the caches
        # are passed as an explicit (k, v) tuple whose (pages, heads,
        # page_size, dim) strides the cudnn graph reads directly, so the
        # layout string is inert here (mirrors test_cudnn_prefill_fp8).
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD", backend="cudnn"
        )
        wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            sm_scale=scale,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=o_data_type,
            seq_lens=actual_seq_lens_kv,
            seq_lens_q=actual_seq_lens_q,
            max_token_per_sequence=s_qo,
            max_sequence_kv=s_kv,
            block_tables=block_tables,
        )
        return wrapper

    # Reference: identical fp8 compute, bf16 output (no output quantization).
    out_ref = _plan_cudnn(torch.bfloat16).run(
        q_fp8,
        (k_cache_fp8, v_cache_fp8),
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    assert out_ref.dtype == torch.bfloat16
    assert out_ref.float().abs().amax() > 0

    # Cross-check the reference against a bf16 fa2 wrapper so a garbage fp8
    # graph cannot self-validate.
    wrapper_ref = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        "HND",
        backend="fa2",
    )
    wrapper_ref.plan(
        qo_indptr_tok,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        sm_scale=scale,
        q_data_type=torch.bfloat16,
    )
    out_fa2 = wrapper_ref.run(q, kv_cache)
    torch.testing.assert_close(out_ref, out_fa2, atol=1e-2, rtol=1e-2)

    # fp8 output allocated internally from the planned o_data_type.
    out_fp8 = _plan_cudnn(torch.float8_e4m3fn).run(
        q_fp8,
        (k_cache_fp8, v_cache_fp8),
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        o_scale=_as_o_scale_arg(o_scale_value, o_scale_form, device),
    )
    assert out_fp8.dtype == torch.float8_e4m3fn

    # Dequantized output must match the pre-quantization reference; the atol
    # floor covers e4m3's subnormal grid (2^-9) mapped back through o_scale.
    torch.testing.assert_close(
        out_fp8.float() / o_scale_value,
        out_ref.float(),
        atol=2**-9 / o_scale_value + 5e-3,
        rtol=1e-1,
    )


def test_paged_wrapper_o_scale_rejected_on_non_cudnn():
    """Non-cudnn paged backends must reject o_scale with NotImplementedError."""
    device = "cuda:0"
    s, page_size = 8, 8
    num_heads, head_dim = 4, 128

    q = torch.randn(s, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    kv_cache = torch.randn(
        1, 2, page_size, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    qo_indptr = torch.tensor([0, s], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indices = torch.tensor([0], dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([s], dtype=torch.int32, device=device)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        "NHD",
        backend="fa2",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_heads,
        num_heads,
        head_dim,
        page_size,
        causal=False,
        q_data_type=torch.bfloat16,
    )
    with pytest.raises(NotImplementedError, match="o_scale"):
        wrapper.run(q, kv_cache, o_scale=1.0)


def test_ragged_wrapper_o_scale_reaches_cudnn_validation():
    """o_scale with a non-fp8 q must surface the cudnn call's ValueError.

    cudnn_batch_prefill_with_kv_cache validates fp8-only arguments before any
    graph work, so this needs neither backend 9.26 nor SM100 -- and it proves
    the ragged wrapper actually forwards o_scale to the cudnn call.
    """
    device = "cuda:0"
    batch_size, s = 1, 8
    num_heads, head_dim = 4, 128

    q = torch.randn(s, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    seq_lens = torch.full((batch_size, 1, 1, 1), s, dtype=torch.int32, device=device)
    indptr = (
        torch.tensor([0, s], dtype=torch.int64, device=device) * num_heads * head_dim
    ).int()

    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device),
        "NHD",
        backend="cudnn",
    )
    wrapper.plan(
        indptr,
        indptr,
        num_qo_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim_qk=head_dim,
        causal=False,
        sm_scale=float(1.0 / (head_dim**0.5)),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        seq_lens=seq_lens,
        seq_lens_q=seq_lens,
        max_token_per_sequence=s,
        max_sequence_kv=s,
    )
    with pytest.raises(ValueError, match="fp8 query dtype"):
        wrapper.run(q, k, v, o_scale=2.0)
