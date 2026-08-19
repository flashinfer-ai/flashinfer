import warnings

import pytest
import torch

import flashinfer
import cudnn

from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 700])
@pytest.mark.parametrize("s_kv", [8, 32, 1066])
@pytest.mark.parametrize("page_size", [8, 16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True])
def test_cudnn_prefill(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    causal,
    return_lse,
    is_cuda_graph_compatible,
):
    head_dim = 128
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test")

    # test set up basics
    seed = 1
    torch.manual_seed(seed)
    device = "cuda:0"

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    q_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0) * head_dim * num_qo_heads,
        ]
    ).int()

    # Initialize KV Cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape, dtype=torch.bfloat16).to(device)
    kv_cache = kv_cache.as_strided(
        kv_cache.shape,
        (
            2 * page_size * num_kv_heads * head_dim,
            page_size * num_kv_heads * head_dim,
            head_dim,
            num_kv_heads * head_dim,
            1,
        ),
    )
    k_cache_view = kv_cache[:, 0, :, :, :]
    v_cache_view = kv_cache[:, 1, :, :, :]

    v_cache = v_cache_view.as_strided(
        v_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )
    k_cache = k_cache_view.as_strided(
        k_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )

    kv_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(
                (actual_seq_lens_kv.flatten() + page_size - 1) // page_size,
                dim=0,
            ),
        ]
    ).int()

    # kv_indices
    kv_indices = torch.zeros(kv_indptr[-1], device=device, dtype=torch.int32)
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = torch.arange(
            i * num_pages_per_seq,
            i * num_pages_per_seq + (end_idx - start_idx),
            device=device,
        )

    # kv_last_page_len
    kv_last_page_len = torch.where(
        actual_seq_lens_kv.flatten() % page_size == 0,
        torch.full((batch_size,), page_size, device=device),
        actual_seq_lens_kv.flatten() % page_size,
    ).int()

    # Now initialize the page tables
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )

    # Initialize scale
    scale = float(1.0 / (head_dim**0.5))

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_cudnn = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="cudnn"
    )
    wrapper_cudnn.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.bfloat16,
        seq_lens=actual_seq_lens_kv,
        seq_lens_q=actual_seq_lens_q,
        sm_scale=scale,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        block_tables=block_tables,
    )

    output = wrapper_cudnn.run(q, (k_cache, v_cache))

    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # Workspace buffer
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer_ref, "HND", backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.bfloat16,
    )

    output_ref = wrapper.run(q, kv_cache)
    torch.testing.assert_close(output, output_ref, atol=3e-3, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 700])
@pytest.mark.parametrize("s_kv", [8, 32, 1066])
@pytest.mark.parametrize("page_size", [8, 16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True])
def test_cudnn_prefill_fp8(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    causal,
    return_lse,
    is_cuda_graph_compatible,
):
    if cudnn.backend_version() < 91701:
        pytest.skip("cuDNN backend version is less than 9.17.1, skipping test")

    head_dim = 128
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test")

    # test set up basics
    seed = 1
    torch.manual_seed(seed)
    device = "cuda:0"

    major, _ = get_compute_capability(torch.device(device))

    if major != 10:
        pytest.skip(
            f"cuDNN FP8 prefill is not supported on compute capability {major}, skipping test"
        )

    if cudnn.backend_version() < 92600:
        pytest.xfail(
            "cuDNN FP8 prefill has known issues on Blackwell before cuDNN 9.26"
        )

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    q_scale = q.amax().item() / 256

    q_scale = torch.tensor(q_scale, device=device, dtype=torch.float32)
    q_fp8 = (q / q_scale).to(torch.float8_e4m3fn)

    q_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0) * head_dim * num_qo_heads,
        ]
    ).int()

    # Initialize KV Cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape, dtype=torch.bfloat16).to(device) * 0.05
    kv_cache = kv_cache.as_strided(
        kv_cache.shape,
        (
            2 * page_size * num_kv_heads * head_dim,
            page_size * num_kv_heads * head_dim,
            head_dim,
            num_kv_heads * head_dim,
            1,
        ),
    )
    k_cache_view = kv_cache[:, 0, :, :, :]
    v_cache_view = kv_cache[:, 1, :, :, :]

    v_cache = v_cache_view.as_strided(
        v_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )
    k_cache = k_cache_view.as_strided(
        k_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )

    k_scale = k_cache.amax().item() / 256
    v_scale = v_cache.amax().item() / 256
    k_cache_fp8 = (k_cache / k_scale).to(torch.float8_e4m3fn)
    v_cache_fp8 = (v_cache / v_scale).to(torch.float8_e4m3fn)

    k_scale_tensor = torch.tensor(k_scale, device=device, dtype=torch.float32)
    v_scale_tensor = torch.tensor(v_scale, device=device, dtype=torch.float32)

    kv_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(
                (actual_seq_lens_kv.flatten() + page_size - 1) // page_size,
                dim=0,
            ),
        ]
    ).int()

    # kv_indices
    kv_indices = torch.zeros(kv_indptr[-1], device=device, dtype=torch.int32)
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = torch.arange(
            i * num_pages_per_seq,
            i * num_pages_per_seq + (end_idx - start_idx),
            device=device,
        )

    # kv_last_page_len
    kv_last_page_len = torch.where(
        actual_seq_lens_kv.flatten() % page_size == 0,
        torch.full((batch_size,), page_size, device=device),
        actual_seq_lens_kv.flatten() % page_size,
    ).int()

    # Now initialize the page tables
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )

    # Initialize scale
    scale = float(1.0 / (head_dim**0.5))

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_cudnn = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="cudnn"
    )
    wrapper_cudnn.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
        seq_lens=actual_seq_lens_kv,
        seq_lens_q=actual_seq_lens_q,
        sm_scale=scale,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        block_tables=block_tables,
    )

    output = wrapper_cudnn.run(
        q_fp8,
        (k_cache_fp8, v_cache_fp8),
        q_scale=q_scale,
        k_scale=k_scale_tensor,
        v_scale=v_scale_tensor,
    )

    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # Workspace buffer
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer_ref, "HND", backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.bfloat16,
    )

    output_ref = wrapper.run(q, kv_cache)

    torch.testing.assert_close(output, output_ref, atol=1e-2, rtol=1e-2)


def _skip_unless_fp8_prefill_supported(device):
    if cudnn.backend_version() < 92600:
        pytest.skip("cuDNN FP8 prefill needs backend 9.26+, skipping test")
    major, _ = get_compute_capability(torch.device(device))
    if major != 10:
        pytest.skip(
            f"cuDNN FP8 prefill is not supported on compute capability {major}, skipping test"
        )


def _make_fp8_prefill_inputs(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    device,
    paged=True,
):
    """Random fp8 e4m3 prefill inputs for direct cudnn_batch_prefill_with_kv_cache calls.

    Returns a kwargs dict (fp8 q / fp8 kv caches, descales, seq lens,
    element-unit batch offsets, workspace) missing only causal / o_data_type /
    o_scale / amax_* arguments.  ``paged=True`` builds a 4-D paged KV cache
    with block tables (``page_size`` pages); ``paged=False`` builds packed 3-D
    KV caches addressed via element-unit batch_offsets_k/v.
    """
    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    q_scale = torch.tensor(
        q.float().abs().amax().item() / 256, device=device, dtype=torch.float32
    ).reshape(1, 1, 1, 1)
    q_fp8 = (q.float() / q_scale.item()).to(torch.float8_e4m3fn)

    if paged:
        num_pages_per_seq = (s_kv + page_size - 1) // page_size
        total_num_pages = num_pages_per_seq * batch_size
        k_cache = torch.randn(
            total_num_pages,
            num_kv_heads,
            page_size,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
    else:
        total_kv_tokens = int(torch.sum(actual_seq_lens_kv))
        k_cache = torch.randn(
            total_kv_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float32
        )
    v_cache = torch.randn_like(k_cache)
    k_scale = (k_cache.abs().amax() / 256).reshape(1, 1, 1, 1)
    v_scale = (v_cache.abs().amax() / 256).reshape(1, 1, 1, 1)
    # .item() rather than tensor division: broadcasting against the
    # (1, 1, 1, 1) scale would silently promote a 3-D non-paged cache to 4-D.
    k_cache_fp8 = (k_cache / k_scale.item()).to(torch.float8_e4m3fn)
    v_cache_fp8 = (v_cache / v_scale.item()).to(torch.float8_e4m3fn)

    batch_offsets_q = (
        torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
            ]
        )
        * num_qo_heads
        * head_dim
    ).int()

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    run_kwargs = dict(
        q=q_fp8,
        k_cache=k_cache_fp8,
        v_cache=v_cache_fp8,
        scale=float(1.0 / (head_dim**0.5)),
        workspace_buffer=workspace_buffer,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        return_lse=False,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        batch_offsets_q=batch_offsets_q,
        batch_offsets_o=batch_offsets_q,
    )
    if paged:
        run_kwargs["block_tables"] = torch.arange(
            total_num_pages, dtype=torch.int32, device=device
        ).reshape(batch_size, num_pages_per_seq)
    else:
        batch_offsets_kv = (
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64, device=device),
                    torch.cumsum(actual_seq_lens_kv.view(-1), dim=0),
                ]
            )
            * num_kv_heads
            * head_dim
        ).int()
        run_kwargs["batch_offsets_k"] = batch_offsets_kv
        run_kwargs["batch_offsets_v"] = batch_offsets_kv
    return run_kwargs


@pytest.mark.parametrize("o_scale_value", [4.0, 1.0 / 16.0])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("paged", [True, False])
def test_cudnn_prefill_fp8_output_scaling(o_scale_value, causal, paged):
    """FP8 output path: o_scale quantizes O and amax_s/amax_o are written."""
    device = "cuda:0"
    _skip_unless_fp8_prefill_supported(device)

    torch.manual_seed(2)
    run_kwargs = _make_fp8_prefill_inputs(
        batch_size=4,
        s_qo=32,
        s_kv=64,
        page_size=16,
        num_kv_heads=4,
        num_qo_heads=4,
        head_dim=128,
        device=device,
        paged=paged,
    )

    # Reference: identical fp8 compute, bf16 output (no output quantization).
    out_ref, _ = cudnn_batch_prefill_with_kv_cache(
        **run_kwargs, causal=causal, o_data_type=torch.bfloat16
    )

    o_scale = torch.tensor(o_scale_value, device=device, dtype=torch.float32).reshape(
        1, 1, 1, 1
    )
    amax_s = torch.full((1, 1, 1, 1), -1.0, device=device, dtype=torch.float32)
    amax_o = torch.full((1, 1, 1, 1), -1.0, device=device, dtype=torch.float32)
    out_fp8, _ = cudnn_batch_prefill_with_kv_cache(
        **run_kwargs,
        causal=causal,
        o_data_type=torch.float8_e4m3fn,
        o_scale=o_scale,
        amax_s=amax_s,
        amax_o=amax_o,
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

    # amax_o is the absolute maximum of O before o_scale quantization.
    torch.testing.assert_close(
        amax_o.reshape(()),
        out_ref.float().abs().amax(),
        atol=1e-3,
        rtol=5e-2,
    )
    # amax_s is the absolute maximum of the post-softmax matrix, in (0, 1].
    assert 0.0 < amax_s.item() <= 1.0 + 1e-3


def test_cudnn_prefill_fp8_output_scale_prevents_saturation():
    """o_scale < 1 keeps an O that overflows e4m3 (|O| > 448) representable."""
    device = "cuda:0"
    _skip_unless_fp8_prefill_supported(device)

    torch.manual_seed(3)
    run_kwargs = _make_fp8_prefill_inputs(
        batch_size=2,
        s_qo=8,
        s_kv=16,
        page_size=8,
        num_kv_heads=1,
        num_qo_heads=4,
        head_dim=128,
        device=device,
    )
    # Constant V = 512 makes every pre-quantization output element 512
    # (softmax weights sum to 1), which exceeds the e4m3 maximum of 448.
    # 512 / v_scale = 256, exactly representable in e4m3.
    run_kwargs["v_cache"] = torch.full_like(
        run_kwargs["v_cache"], 256.0, dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    run_kwargs["v_scale"] = torch.tensor(
        2.0, device=device, dtype=torch.float32
    ).reshape(1, 1, 1, 1)

    # Without o_scale the fp8 output is emitted unit-scaled (with a warning)
    # and saturates at the e4m3 maximum: the customer bug.
    with pytest.warns(UserWarning, match="o_scale"):
        out_sat, _ = cudnn_batch_prefill_with_kv_cache(
            **run_kwargs, causal=False, o_data_type=torch.float8_e4m3fn
        )
    sat_amax = out_sat.float().abs().amax().item()
    assert 447.0 <= sat_amax <= 448.0, (
        f"unit-scaled fp8 output should saturate at the e4m3 max, got {sat_amax}"
    )

    # With o_scale = 1/16 the stored values are 512/16 = 32 and dequantization
    # recovers magnitudes beyond the e4m3 maximum.
    o_scale_value = 1.0 / 16.0
    o_scale = torch.tensor(o_scale_value, device=device, dtype=torch.float32).reshape(
        1, 1, 1, 1
    )
    amax_o = torch.full((1, 1, 1, 1), -1.0, device=device, dtype=torch.float32)
    out_scaled, _ = cudnn_batch_prefill_with_kv_cache(
        **run_kwargs,
        causal=False,
        o_data_type=torch.float8_e4m3fn,
        o_scale=o_scale,
        amax_o=amax_o,
    )
    dequant = out_scaled.float() / o_scale_value
    assert dequant.abs().amax().item() > 448.0
    torch.testing.assert_close(
        dequant, torch.full_like(dequant, 512.0), atol=0.0, rtol=5e-2
    )
    torch.testing.assert_close(
        amax_o.reshape(()), torch.tensor(512.0, device=device), atol=0.0, rtol=5e-2
    )


def test_cudnn_prefill_fp8_args_rejected_on_non_fp8():
    """fp8-output arguments raise for non-fp8 q; q/k/v_scale only warn.

    ``o_scale``/``amax_s``/``amax_o`` (and an fp8 ``o_data_type``) are new and
    meaningless on a non-fp8 graph, so they hard-fail.  The long-standing
    ``q_scale``/``k_scale``/``v_scale`` arguments were always silently ignored
    on non-fp8 graphs and callers pass them unconditionally, so they must keep
    working (with a warning), not raise.
    """
    device = "cuda:0"
    s = 8
    num_heads = 4
    head_dim = 128
    q = torch.randn(s, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn_like(q)
    v_cache = torch.randn_like(q)
    seq_lens = torch.full((1, 1, 1, 1), s, dtype=torch.int32, device=device)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    scale_tensor = torch.ones(1, 1, 1, 1, device=device, dtype=torch.float32)

    run_kwargs = dict(
        scale=float(1.0 / (head_dim**0.5)),
        workspace_buffer=workspace_buffer,
        max_token_per_sequence=s,
        max_sequence_kv=s,
        actual_seq_lens_q=seq_lens,
        actual_seq_lens_kv=seq_lens,
        causal=False,
        return_lse=False,
    )

    for name in ("o_scale", "amax_s", "amax_o"):
        with pytest.raises(ValueError, match="fp8 query dtype"):
            cudnn_batch_prefill_with_kv_cache(
                q, k_cache, v_cache, **run_kwargs, **{name: scale_tensor}
            )

    with pytest.raises(ValueError, match="fp8 query dtype"):
        cudnn_batch_prefill_with_kv_cache(
            q, k_cache, v_cache, **run_kwargs, o_data_type=torch.float8_e4m3fn
        )

    with pytest.warns(UserWarning, match="ignored"):
        out, _ = cudnn_batch_prefill_with_kv_cache(
            q,
            k_cache,
            v_cache,
            **run_kwargs,
            q_scale=scale_tensor,
            k_scale=scale_tensor,
            v_scale=scale_tensor,
        )
    assert out.dtype == torch.bfloat16


def test_cudnn_prefill_fp8_missing_o_scale_warns_once():
    """The o_scale-missing warning fires exactly once per public call.

    The fp8 argument validation lives only in the public entry point; a
    second call site in _batch_prefill_with_kv_cache would emit the warning
    twice per call (the two frames defeat warnings' per-location dedup).
    """
    device = "cuda:0"
    _skip_unless_fp8_prefill_supported(device)

    torch.manual_seed(4)
    run_kwargs = _make_fp8_prefill_inputs(
        batch_size=2,
        s_qo=8,
        s_kv=16,
        page_size=8,
        num_kv_heads=1,
        num_qo_heads=4,
        head_dim=128,
        device=device,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cudnn_batch_prefill_with_kv_cache(
            **run_kwargs, causal=False, o_data_type=torch.float8_e4m3fn
        )
    o_scale_warnings = [w for w in caught if "o_scale" in str(w.message)]
    assert len(o_scale_warnings) == 1, o_scale_warnings


def test_sdpa_prefill_key_fn_discriminates_baked_attributes():
    """Attributes _build_prefill_graph bakes into a graph must key the cache.

    Uses meta tensors: the key fn is pure Python, so this needs no GPU and
    pins the graph-cache collisions (KV dtype, v_cache-derived d_vo, block
    table width, ragged-offset presence, aux int dtypes, strides) that would
    silently replay a structurally different graph.
    """
    from flashinfer.cudnn.prefill import _sdpa_prefill_key_fn

    b, h, s_q, s_kv, d = 4, 8, 32, 64, 128
    page_size = 16
    pages_per_seq = s_kv // page_size
    num_pages = b * pages_per_seq

    def meta(*shape, dtype=torch.bfloat16):
        return torch.empty(*shape, device="meta", dtype=dtype)

    def make_kwargs(**overrides):
        kwargs = dict(
            q=meta(b * s_q, h, d),
            k_cache=meta(num_pages, h, page_size, d),
            v_cache=meta(num_pages, h, page_size, d),
            scale=1.0 / (d**0.5),
            max_token_seq_q=s_q,
            max_sequence_kv=s_kv,
            actual_seq_lens_q=meta(b, 1, 1, 1, dtype=torch.int32),
            actual_seq_lens_kv=meta(b, 1, 1, 1, dtype=torch.int32),
            block_tables=meta(b, pages_per_seq, dtype=torch.int32),
        )
        kwargs.update(overrides)
        return kwargs

    base = _sdpa_prefill_key_fn(**make_kwargs())
    assert base == _sdpa_prefill_key_fn(**make_kwargs())

    variants = {
        "kv dtype": make_kwargs(
            k_cache=meta(num_pages, h, page_size, d, dtype=torch.float16),
            v_cache=meta(num_pages, h, page_size, d, dtype=torch.float16),
        ),
        "v_cache d_vo": make_kwargs(v_cache=meta(num_pages, h, page_size, d // 2)),
        "block table width": make_kwargs(
            block_tables=meta(b, 2 * pages_per_seq, dtype=torch.int32)
        ),
        "seq-lens dtype": make_kwargs(
            actual_seq_lens_kv=meta(b, 1, 1, 1, dtype=torch.int64)
        ),
        "ragged offsets": make_kwargs(
            batch_offsets_q=meta(b + 1, dtype=torch.int32),
            batch_offsets_o=meta(b + 1, dtype=torch.int32),
        ),
        "kv strides": make_kwargs(
            k_cache=meta(num_pages, page_size, h, d).permute(0, 2, 1, 3),
            v_cache=meta(num_pages, page_size, h, d).permute(0, 2, 1, 3),
        ),
    }
    for name, kwargs in variants.items():
        assert _sdpa_prefill_key_fn(**kwargs) != base, (
            f"cache key must change when {name} changes"
        )


def test_cudnn_prefill_fp8_out_args_require_graph_backend():
    """The cubin fallback cannot honor o_scale/amax_s/amax_o: it must raise
    NotImplementedError instead of silently ignoring them."""
    device = "cuda:0"
    s = 8
    num_heads = 4
    head_dim = 128
    q = torch.randn(s, num_heads, head_dim, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    k_cache = torch.randn_like(q, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    v_cache = torch.randn_like(q, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    seq_lens = torch.full((1, 1, 1, 1), s, dtype=torch.int32, device=device)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    o_scale = torch.ones(1, 1, 1, 1, device=device, dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="graph backend"):
        cudnn_batch_prefill_with_kv_cache(
            q,
            k_cache,
            v_cache,
            float(1.0 / (head_dim**0.5)),
            workspace_buffer,
            max_token_per_sequence=s,
            max_sequence_kv=s,
            actual_seq_lens_q=seq_lens,
            actual_seq_lens_kv=seq_lens,
            causal=False,
            return_lse=True,
            o_scale=o_scale,
            o_data_type=torch.float8_e4m3fn,
            backend="cubin",
        )
