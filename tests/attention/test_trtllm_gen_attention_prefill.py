"""Prefill tests extracted from ``test_trtllm_gen_attention_decode.py``.

Split out so the CI parallel runner (``scripts/task_run_unit_tests.sh``)
picks this up as a separate test file and schedules it on its own GPU in
parallel with the decode shard.  All shared helpers still live in
``test_trtllm_gen_attention_decode`` and are imported below; only the
prefill tests and the prefill-specific ``_test_trtllm_batch_prefill``
helper were moved here so the parametrize matrices remain byte-identical
to the originals.
"""

import pytest
import torch

import flashinfer
from flashinfer.mla import (
    MLAHeadDimensions,
    deepseek_mla_dimensions,
    smaller_mla_dimensions,
)
from flashinfer.utils import get_compute_capability
from tests.test_helpers.sink_attention_reference import sink_attention_unified
from tests.test_helpers.test_helpers import assert_close_with_mismatch_tolerance

from tests.attention.test_trtllm_gen_attention_decode import (
    DTYPE_MAP,
    GPU_DEVICE,
    TRTLLM_GEN_WORKSPACE_CHECK_BYTES,
    _skip_if_not_blackwell,
    create_kv_cache,
    create_output,
    create_page_table,
    create_query_tensor,
    create_workspace_buffers,
    flatten_paged_kv,
    flip_coin,
    generate_cumsum_lens,
    generate_seq_lens_prefill,
    get_last_page_len,
    make_query_non_contiguous,
    prepare_paged_kv_for_kernel,
    sdpa_paged_reference,
    trtllm_gen_workspace_softmax_end_bytes_context,
    unpack_compare_nvfp4,
    workspace_size,
)


def _test_trtllm_batch_prefill(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    causal: bool,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_q_len: int,
    max_kv_len: int,
    device_scale: float,
    head_dim: int,
    non_contiguous_query: bool = False,
    skips_softmax: bool = False,
    uses_shared_paged_kv_idx: bool = True,
    return_lse: bool | None = None,
    provide_lse: bool = False,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    if not causal and window_left >= 0:
        pytest.skip("Non-causal paged trtllm-gen tests only cover dense attention")

    if skips_softmax and q_dtype != kv_dtype:
        pytest.skip(
            "skips_softmax does not currently support Q and Kv types being different"
        )

    # NVFP4 KV cache constraints
    if kv_dtype == "nvfp4":
        if q_dtype != "fp8":
            pytest.skip("NVFP4 KV cache requires FP8 query")
        if o_dtype != "fp8":
            pytest.skip("NVFP4 KV cache only supports FP8 output")

    # Set up test parameters
    torch.manual_seed(0)

    # Generate random sequence lengths
    num_qo_heads = num_kv_heads * head_grp_size
    q_lens, _, seq_lens = generate_seq_lens_prefill(batch_size, max_q_len, max_kv_len)

    # Create query tensor and related data
    q, q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)
    q_indptr = generate_cumsum_lens(q_lens)

    # Create KV cache and related data
    kv_cache, k_scale, v_scale, ref_kv_cache, kv_cache_sf = create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        "bf16" if q_dtype == "fp8" or kv_dtype == "nvfp4" else q_dtype,
        kv_layout,
    )
    page_table, all_page_ids, page_per_seq = create_page_table(
        batch_size, seq_lens, page_size
    )
    kv_indptr = generate_cumsum_lens(page_per_seq)
    kv_last_page_len = get_last_page_len(seq_lens, page_size)

    kv_cache_kernel, page_table_kernel, kv_cache_sf_kernel = (
        prepare_paged_kv_for_kernel(
            kv_cache, page_table, uses_shared_paged_kv_idx, kv_cache_sf
        )
    )

    workspace_buffer, workspace_buffer_ref = create_workspace_buffers(GPU_DEVICE)

    # Create output tensor and related data
    create_out_tensor = flip_coin(
        batch_size, page_size, num_kv_heads, head_grp_size, o_dtype
    )
    can_infer_type = q.dtype == DTYPE_MAP[o_dtype] or create_out_tensor
    create_out_dtype = not can_infer_type or flip_coin(
        batch_size, page_size, num_kv_heads, head_grp_size, o_dtype, q_dtype
    )
    out, out_dtype, o_scale, o_sf_scale, o_sf_vec_size = create_output(
        q, o_dtype, create_out_tensor, create_out_dtype
    )

    sm_scale = float(1.0 / (head_dim**0.5))

    # Build reference output
    plan_params = {
        "qo_indptr": q_indptr,
        "paged_kv_indptr": kv_indptr,
        "paged_kv_indices": all_page_ids,
        "paged_kv_last_page_len": kv_last_page_len.to(GPU_DEVICE),
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim_qk": head_dim,
        "page_size": page_size,
        "causal": causal,
        "pos_encoding_mode": "NONE",
        "logits_soft_cap": 0.0,
        "q_data_type": ref_q.dtype,
        "kv_data_type": ref_kv_cache.dtype,
        "window_left": window_left,
    }
    lse_ref = None
    sink = torch.rand(num_qo_heads, device=GPU_DEVICE, dtype=torch.float32) * 5
    if head_dim > 256:
        # FlashInfer's own FA2/FA3 kernels don't support head_dim > 256;
        # fall back to a PyTorch SDPA reference (causal/windowed only, no sink support).
        assert not enable_sink, (
            "SDPA fallback does not model attention sinks; skip sink+head_dim>256 cases"
        )
        output_ref = sdpa_paged_reference(
            ref_q,
            ref_kv_cache,
            q_lens,
            seq_lens,
            page_table,
            page_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            kv_layout,
            window_left,
        )
    elif not enable_sink:
        wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer_ref, kv_layout
        )
        wrapper_ref.plan(**plan_params)
        output_ref, lse_ref = wrapper_ref.run(ref_q, ref_kv_cache, return_lse=True)
    else:
        # Construct flat K/V via helper
        k_flat, v_flat, kv_indptr_tokens = flatten_paged_kv(
            ref_kv_cache,
            page_table,
            seq_lens.to(GPU_DEVICE),
            page_size,
            kv_last_page_len,
            kv_layout,
        )
        output_ref = sink_attention_unified(
            ref_q,
            k_flat,
            v_flat,
            sink,
            window_left,
            causal,
            sm_scale,
            mode="varlen",
            batch_size=batch_size,
            qo_indptr=q_indptr,
            kv_indptr=kv_indptr_tokens,
        )

    # Run trtllm-gen function call
    bmm1_scale = q_scale * k_scale * sm_scale
    bmm2_scale = v_scale / o_scale
    if isinstance(bmm1_scale, torch.Tensor) and not device_scale:
        bmm1_scale = bmm1_scale.item()
    elif not isinstance(bmm1_scale, torch.Tensor) and device_scale:
        bmm1_scale = torch.tensor(bmm1_scale, device=GPU_DEVICE, dtype=torch.float32)
    if isinstance(bmm2_scale, torch.Tensor) and not device_scale:
        bmm2_scale = bmm2_scale.item()
    elif not isinstance(bmm2_scale, torch.Tensor) and device_scale:
        bmm2_scale = torch.tensor(bmm2_scale, device=GPU_DEVICE, dtype=torch.float32)

    # Optionally make query non-contiguous for testing stride support
    if non_contiguous_query:
        q_input = make_query_non_contiguous(q, num_qo_heads, head_dim)
    else:
        q_input = q.contiguous()

    # Using a tiny threshold should give the same result as normal attention.
    skip_softmax_threshold_scale_factor = 1e-30 if skips_softmax else None

    # Validate LSE only for paths where we have a reliable reference and a supported
    # (non-FP8 / non-FP4) numerical regime. Sink attention doesn't populate lse_ref.
    check_lse = (
        not enable_sink
        and not skips_softmax
        and o_dtype != "nvfp4"
        and kv_dtype != "nvfp4"
        and q_dtype != "fp8"
    )
    if (return_lse or provide_lse) and not check_lse:
        pytest.skip("LSE contract validation requires a reliable LSE reference")
    effective_return_lse = check_lse if return_lse is None else return_lse
    expects_lse = check_lse and (effective_return_lse or provide_lse)

    max_q_len_val = torch.max(q_lens).item()
    provided_lse = None
    if expects_lse:
        # Allocate LSE on the caller side so we can pre-populate it with NaNs
        # and catch missed writes. Shape is [total_qo_tokens, num_qo_heads].
        provided_lse = (
            torch.full(
                (ref_q.shape[0], num_qo_heads),
                float("nan"),
                device=GPU_DEVICE,
                dtype=torch.float32,
            )
            if provide_lse
            else None
        )
        # Zero out the guard region that sits immediately after the softmax
        # slab. If the kernel writes out of bounds we'll notice it flip to
        # non-zero below.
        softmax_end = trtllm_gen_workspace_softmax_end_bytes_context(
            num_qo_heads, batch_size, max_q_len_val
        )
        guard_end = min(softmax_end + TRTLLM_GEN_WORKSPACE_CHECK_BYTES, workspace_size)
        workspace_buffer[softmax_end:guard_end].zero_()

    output_and_lse = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        q_input,
        kv_cache_kernel,
        workspace_buffer,
        page_table_kernel,
        seq_lens.to(GPU_DEVICE),
        max_q_len_val,
        torch.max(seq_lens).item(),
        bmm1_scale,  # bmm1_scale
        bmm2_scale,  # bmm2_scale
        batch_size,
        q_indptr,
        kv_indptr,
        window_left,  # window_left
        causal=causal,
        out=out,
        out_dtype=out_dtype,
        o_sf_scale=o_sf_scale,
        o_sf_vec_size=o_sf_vec_size,
        kv_layout=kv_layout,
        enable_pdl=enable_pdl,
        sinks=(sink if enable_sink else None),
        kv_cache_sf=kv_cache_sf_kernel,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        lse=provided_lse,
        return_lse=effective_return_lse,
    )
    if expects_lse:
        if effective_return_lse:
            output, lse_out = output_and_lse
            if provide_lse:
                assert lse_out is provided_lse
        else:
            output = output_and_lse
            lse_out = provided_lse
            assert lse_out is not None
        assert lse_out.dtype == torch.float32
        assert lse_out.shape == (ref_q.shape[0], num_qo_heads)
        assert torch.isfinite(lse_out).all(), (
            "trtllm-gen context kernel produced non-finite LSE"
        )
        if lse_ref is not None:
            torch.testing.assert_close(lse_out, lse_ref.float(), rtol=1e-3, atol=1e-3)
        # Softmax slab and its guard region remain zero-initialized outside writes.
        softmax_end = trtllm_gen_workspace_softmax_end_bytes_context(
            num_qo_heads, batch_size, max_q_len_val
        )
        guard_end = min(softmax_end + TRTLLM_GEN_WORKSPACE_CHECK_BYTES, workspace_size)
        assert (workspace_buffer[softmax_end:guard_end].cpu().numpy() == 0).all(), (
            "trtllm-gen context kernel wrote past the softmax slab"
        )
        # Restore the head of the workspace so downstream wrapper runs can still assert
        # the counter region (first 8MB) remains zero-initialized.
        workspace_buffer[:softmax_end].zero_()
    else:
        output = output_and_lse
        # In context mode, with LSE disabled the softmax slab is never allocated, so the
        # head of the workspace stays zero.
        assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()

    if o_dtype == "nvfp4":
        output, output_ref = unpack_compare_nvfp4(
            output, output_ref, o_sf_scale, o_sf_vec_size
        )
        assert o_scale == 1.0
        rtol, atol = 4e-1, 1e0
    elif q_dtype == "fp8" and o_dtype == "fp8":
        rtol, atol = 5e-2, 7e-2
    elif q_dtype == "fp8" and o_dtype in ["bf16", "fp16"]:
        rtol, atol = 4e-2, 6e-2
    else:
        rtol, atol = 1e-2, 1e-2

    # NVFP4 KV cache has significant quantization error, especially with
    # outlier channels that create large per-block dynamic range.
    if kv_dtype == "nvfp4":
        rtol, atol = 5e-1, 5e-1

    # NVFP4 KV cache has higher mismatch rate due to 4-bit quantization noise,
    # especially with outlier channels that stress per-block scaling.
    allowed_mismatch_rate = 0.10 if kv_dtype == "nvfp4" else 1e-7
    # Calculate max allowed mismatched elements based on tensor size
    total_elements = (output.float() * o_scale).numel()
    max_mismatched_elements = int(allowed_mismatch_rate * total_elements)

    # convert to float32 for fp8 is not supported by assert_close
    assert_close_with_mismatch_tolerance(
        output.float() * o_scale,
        output_ref.float(),
        rtol=rtol,
        atol=atol,
        max_mismatched_elements=max_mismatched_elements,
    )

    # NVFP4 KV cache: use cosine similarity to catch block-scale mismatches
    # (e.g. wrong swizzling) that element-wise tolerances miss.
    if kv_dtype == "nvfp4":
        cos = torch.nn.functional.cosine_similarity(
            (output.float() * o_scale).reshape(-1),
            output_ref.float().reshape(-1),
            dim=0,
        )
        assert cos.item() > 0.86, (
            f"NVFP4 KV cache attention: cosine similarity {cos:.4f} < 0.86. "
            f"Block scale factors may be mismatched to FP4 data blocks."
        )

    if (
        o_dtype != "nvfp4" and kv_dtype != "nvfp4" and uses_shared_paged_kv_idx
    ):  # wrapper api does not support fp4 output/kv or separate KV page indices yet.
        # test wrapper with trtllm-gen backend
        wrapper_trtllm_gen = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout, backend="trtllm-gen"
        )
        plan_params["q_data_type"] = q.dtype
        plan_params["kv_data_type"] = kv_cache.dtype
        plan_params["o_data_type"] = DTYPE_MAP[o_dtype]
        wrapper_trtllm_gen.plan(**plan_params)
        output_wrapper = wrapper_trtllm_gen.run(
            q_input,
            kv_cache,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale / o_scale,
            enable_pdl=enable_pdl,
            sinks=(sink if enable_sink else None),
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        )
        # v_scale, o_scale in wrapper is emulated by multiplying output by v_scale instead of fused into kernel.
        if v_scale == o_scale == 1.0:
            assert (output_wrapper == output).all()
        else:
            torch.testing.assert_close(
                output.float(), output_wrapper.float(), rtol=1e-1, atol=1e-1
            )
        # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
        # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
        assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()


TRTLLM_BATCH_PREFILL_SHAPES = [
    (4, 16, 2, 1),
    (4, 32, 4, 5),
    (4, 64, 4, 8),
    (128, 16, 2, 5),
    (128, 32, 4, 1),
    (128, 64, 2, 8),
    (256, 16, 4, 8),
    (256, 32, 2, 8),
    (256, 64, 4, 1),
    (256, 64, 4, 5),
]


TRTLLM_BATCH_PREFILL_DTYPES = [
    ("bf16", "bf16", "bf16"),
    ("fp16", "fp16", "fp16"),
    ("fp8", "fp8", "bf16"),
    ("fp8", "fp8", "fp16"),
    ("fp8", "fp8", "fp8"),
    ("fp8", "fp8", "nvfp4"),
    ("fp8", "nvfp4", "fp8"),
]


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    TRTLLM_BATCH_PREFILL_SHAPES,
)
@pytest.mark.parametrize("window_left", [-1])  # todo(Siyuan): add 127 window_left
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    TRTLLM_BATCH_PREFILL_DTYPES,
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("max_q_len", [511])
@pytest.mark.parametrize("max_kv_len", [2047])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("non_contiguous_query", [False, True])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
@pytest.mark.parametrize("causal", [True, False])
def test_trtllm_batch_prefill(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    causal: bool,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_q_len: int,
    max_kv_len: int,
    head_dim: int,
    non_contiguous_query: bool,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
):
    _test_trtllm_batch_prefill(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        causal,
        window_left,
        q_dtype,
        o_dtype,
        kv_dtype,
        enable_pdl,
        enable_sink,
        max_q_len,
        max_kv_len,
        kv_dtype in ("fp8", "nvfp4"),
        head_dim,
        non_contiguous_query=non_contiguous_query,
        skips_softmax=skips_softmax,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )


@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("provide_lse", [False, True])
def test_trtllm_batch_prefill_lse_contract(return_lse, provide_lse):
    _test_trtllm_batch_prefill(
        "HND",
        2,
        16,
        2,
        2,
        True,
        -1,
        "fp16",
        "fp16",
        "fp16",
        False,
        False,
        64,
        128,
        False,
        128,
        uses_shared_paged_kv_idx=True,
        return_lse=return_lse,
        provide_lse=provide_lse,
    )


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    [
        (1, 16, 8, 8),
    ],
)
@pytest.mark.parametrize("window_left", [-1])  # todo(Siyuan): add 127 window_left
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [False])
@pytest.mark.parametrize("max_q_len", [8192])
@pytest.mark.parametrize("max_kv_len", [8192])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_prefill_bs1(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_q_len: int,
    max_kv_len: int,
    head_dim: int,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
):
    _test_trtllm_batch_prefill(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        True,
        window_left,
        q_dtype,
        o_dtype,
        kv_dtype,
        enable_pdl,
        enable_sink,
        max_q_len,
        max_kv_len,
        False,
        head_dim,
        skips_softmax=skips_softmax,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )


@pytest.mark.parametrize("page_size", [128, 256, 512, 1024])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_prefill_dynamic_page_size_gqa(
    page_size: int,
    uses_shared_paged_kv_idx: bool,
) -> None:
    _skip_if_not_blackwell()
    _test_trtllm_batch_prefill(
        "HND",
        batch_size=4,
        page_size=page_size,
        num_kv_heads=2,
        head_grp_size=5,
        causal=True,
        window_left=-1,
        q_dtype="bf16",
        o_dtype="bf16",
        kv_dtype="bf16",
        enable_pdl=None,
        enable_sink=False,
        max_q_len=257,
        max_kv_len=1024,
        device_scale=False,
        head_dim=128,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    [
        (4, 16, 2, 1),
        (4, 32, 4, 5),
        (128, 16, 2, 8),
    ],
)
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "bf16"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [False])
@pytest.mark.parametrize("max_q_len", [1, 255, 511])
@pytest.mark.parametrize("max_kv_len", [511, 2047])
@pytest.mark.parametrize("head_dim", [512])
@pytest.mark.parametrize("non_contiguous_query", [False])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_prefill_head_dim_512(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_q_len: int,
    max_kv_len: int,
    head_dim: int,
    non_contiguous_query: bool,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
):
    _test_trtllm_batch_prefill(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        True,
        window_left,
        q_dtype,
        o_dtype,
        kv_dtype,
        enable_pdl,
        enable_sink,
        max_q_len,
        max_kv_len,
        kv_dtype in ("fp8", "nvfp4"),
        head_dim,
        non_contiguous_query=non_contiguous_query,
        skips_softmax=skips_softmax,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )


@pytest.mark.parametrize("backend", ["trtllm-native", "cute-dsl"])
@pytest.mark.parametrize(
    "mla_dimensions", [deepseek_mla_dimensions, smaller_mla_dimensions]
)
@pytest.mark.parametrize("batch_size", [4, 128, 256])
@pytest.mark.parametrize("s_qo", [32, 64, 87])
@pytest.mark.parametrize("s_kv", [32, 64, 87])
@pytest.mark.parametrize("num_kv_heads", [16, 32])
@pytest.mark.parametrize("head_grp_size", [1, 5, 8])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("enable_sink", [False, True])
def test_trtllm_gen_prefill(
    backend: str,
    mla_dimensions: MLAHeadDimensions,
    batch_size: int,
    s_qo: int,
    s_kv: int,
    num_kv_heads: int,
    head_grp_size: int,
    causal: bool,
    skips_softmax: bool,
    enable_sink: bool,
) -> None:
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test as causal")

    head_dim_qk = mla_dimensions.qk_nope_head_dim + mla_dimensions.qk_rope_head_dim
    if backend == "cute-dsl":
        if head_dim_qk == 192:
            pytest.skip("cute-dsl does not support bf16 with head_dim=192")

    num_qo_heads = num_kv_heads * head_grp_size
    head_dim_vo = mla_dimensions.v_head_dim

    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    cumsum_s_qo = int(torch.sum(actual_seq_lens_q).item())
    cumsum_s_kv = int(torch.sum(actual_seq_lens_kv).item())

    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim_qk, device=device, dtype=torch.bfloat16
    )
    k_cache = torch.randn(
        (cumsum_s_kv, num_kv_heads, head_dim_qk),
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        (cumsum_s_kv, num_kv_heads, head_dim_vo),
        device=device,
        dtype=torch.bfloat16,
    )

    # Initialize scale
    scale = float(1.0 / (head_dim_qk**0.5))

    workspace_buffer, workspace_buffer_ref = create_workspace_buffers(device)

    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # kv_indptr = torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * s_kv

    # Create kv_indptr as cumulative sum of actual_seq_lens_kv
    kv_indptr = torch.cat(
        [
            torch.tensor(
                [0],
                device=device,
            ),
            torch.cumsum(actual_seq_lens_kv.view(-1), dim=0),
        ]
    ).int()

    sink = (
        torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5
        if enable_sink
        else None
    )
    lse_ref = None
    if enable_sink:
        output_ref = sink_attention_unified(
            q,
            k_cache,
            v_cache,
            sink,
            window_left=-1,
            causal=causal,
            sm_scale=scale,
            mode="varlen",
            batch_size=batch_size,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
        ).to(torch.bfloat16)
    else:
        wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer_ref,
            kv_layout="NHD",
            backend="cutlass",
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            head_dim_vo=head_dim_vo,
            causal=causal,
            sm_scale=scale,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        output_ref, lse_ref = wrapper.run(q, k_cache, v_cache, return_lse=True)
    output = torch.empty_like(output_ref)

    bmm1_scale = scale
    bmm2_scale = 1.0

    # Using a tiny threshold should give the same result as normal attention.
    skip_softmax_threshold_scale_factor = 1e-30 if skips_softmax else None

    output_trtllm, lse_trtllm = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        q,
        k_cache,
        v_cache,
        workspace_buffer,
        actual_seq_lens_kv,
        s_qo,
        s_kv,
        bmm1_scale,
        bmm2_scale,
        -1,
        batch_size,
        -1,
        qo_indptr,
        kv_indptr,
        False,
        causal,
        True,
        attention_sinks=sink,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        out=output,
        backend=backend,
    )
    torch.testing.assert_close(
        output_trtllm,
        output_ref,
        atol=1e-2,
        rtol=1e-2,
    )
    if lse_ref is not None:
        torch.testing.assert_close(
            lse_trtllm,
            lse_ref,
            atol=1e-3,
            rtol=1e-3,
        )
    # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
    # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
    if backend == "trtllm-native":
        assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()


@pytest.mark.parametrize("backend", ["cute-dsl"])
@pytest.mark.parametrize(
    "mla_dimensions", [deepseek_mla_dimensions, smaller_mla_dimensions]
)
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("s_qo", [8192])
@pytest.mark.parametrize("s_kv", [8192])
@pytest.mark.parametrize("num_kv_heads", [128])
@pytest.mark.parametrize("head_grp_size", [1])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("skips_softmax", [False, True])
def test_trtllm_gen_prefill_fp8(
    backend: str,
    mla_dimensions: MLAHeadDimensions,
    batch_size: int,
    s_qo: int,
    s_kv: int,
    num_kv_heads: int,
    head_grp_size: int,
    causal: bool,
    skips_softmax: bool,
) -> None:
    """Test cute-dsl prefill with FP8 (e4m3) input, bf16 output."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")

    head_dim_qk = mla_dimensions.qk_nope_head_dim + mla_dimensions.qk_rope_head_dim
    head_dim_vo = mla_dimensions.v_head_dim
    num_qo_heads = num_kv_heads * head_grp_size

    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    cumsum_s_qo = int(torch.sum(actual_seq_lens_q).item())
    cumsum_s_kv = int(torch.sum(actual_seq_lens_kv).item())

    # FP8 scales
    scale_q, scale_k, scale_v = 0.05, 0.04, 0.06

    # Generate in float32, quantize to FP8
    q_f32 = (
        torch.randn(
            cumsum_s_qo,
            num_qo_heads,
            head_dim_qk,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    )
    k_f32 = (
        torch.randn(
            cumsum_s_kv,
            num_kv_heads,
            head_dim_qk,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    )
    v_f32 = (
        torch.randn(
            cumsum_s_kv,
            num_kv_heads,
            head_dim_vo,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    )

    q = (q_f32 / scale_q).to(torch.float8_e4m3fn)
    k_cache = (k_f32 / scale_k).to(torch.float8_e4m3fn)
    v_cache = (v_f32 / scale_v).to(torch.float8_e4m3fn)

    # Reference: dequantize and run bf16 attention
    q_bf16 = (q.float() * scale_q).to(torch.bfloat16)
    k_bf16 = (k_cache.float() * scale_k).to(torch.bfloat16)
    v_bf16 = (v_cache.float() * scale_v).to(torch.bfloat16)

    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()
    kv_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_kv.view(-1), dim=0),
        ]
    ).int()

    workspace_buffer, workspace_buffer_ref = create_workspace_buffers(device)

    # Reference via cutlass backend (bf16)
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer_ref,
        kv_layout="NHD",
        backend="cutlass",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=1.0 / (head_dim_qk**0.5),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    output_ref, _ = wrapper.run(q_bf16, k_bf16, v_bf16, return_lse=True)

    output = torch.empty_like(output_ref)

    scale = 1.0 / (head_dim_qk**0.5)
    bmm1_scale = scale_q * scale_k * scale
    bmm2_scale = scale_v

    # Using a tiny threshold should give the same result as normal attention.
    skip_softmax_threshold_scale_factor = 1e-30 if skips_softmax else None

    output_fp8, _ = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        q,
        k_cache,
        v_cache,
        workspace_buffer,
        actual_seq_lens_kv,
        s_qo,
        s_kv,
        bmm1_scale,
        bmm2_scale,
        -1,
        batch_size,
        -1,
        qo_indptr,
        kv_indptr,
        False,
        causal,
        True,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        out=output,
        backend=backend,
    )

    torch.testing.assert_close(output_fp8, output_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("backend", ["trtllm-native", "cute-dsl"])
@pytest.mark.parametrize(
    "mla_dimensions", [deepseek_mla_dimensions, smaller_mla_dimensions]
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("s_qo", [1024])
@pytest.mark.parametrize("s_kv", [1024])
@pytest.mark.parametrize("num_kv_heads", [128])
@pytest.mark.parametrize("head_grp_size", [1])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("skips_softmax", [False, True])
def test_trtllm_gen_prefill_bs1(
    backend: str,
    mla_dimensions: MLAHeadDimensions,
    batch_size: int,
    s_qo: int,
    s_kv: int,
    num_kv_heads: int,
    head_grp_size: int,
    causal: bool,
    skips_softmax: bool,
):
    test_trtllm_gen_prefill(
        backend,
        mla_dimensions,
        batch_size,
        s_qo,
        s_kv,
        num_kv_heads,
        head_grp_size,
        causal,
        skips_softmax,
        enable_sink=False,
    )


def naive_ragged_attention(q, k, v, qo_indptr, kv_indptr, scale, causal):
    """Naive batched ragged attention in float32, head by head.

    Used as an independent reference to sanity-check other backends.
    q: [total_q, num_qo_heads, head_dim_qk]
    k: [total_kv, num_kv_heads, head_dim_qk]
    v: [total_kv, num_kv_heads, head_dim_vo]
    Returns output [total_q, num_qo_heads, head_dim_vo] in the same dtype as q.
    """
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_grp_size = num_qo_heads // num_kv_heads
    batch_size = len(qo_indptr) - 1
    out = torch.zeros(
        q.shape[0], num_qo_heads, v.shape[2], device=q.device, dtype=torch.float32
    )

    for b in range(batch_size):
        qs, qe = int(qo_indptr[b]), int(qo_indptr[b + 1])
        ks, ke = int(kv_indptr[b]), int(kv_indptr[b + 1])
        sq, skv = qe - qs, ke - ks
        q_b = q[qs:qe].float()  # [sq,  nqh, dqk]
        k_b = k[ks:ke].float()  # [skv, nkh, dqk]
        v_b = v[ks:ke].float()  # [skv, nkh, dvo]
        for h in range(num_qo_heads):
            kv_h = h // head_grp_size
            scores = q_b[:, h, :] @ k_b[:, kv_h, :].T * scale  # [sq, skv]
            if causal:
                # token at q-position i attends to kv positions 0 .. (skv - sq + i)
                offset = skv - sq
                mask = torch.arange(skv, device=q.device).unsqueeze(0) > (
                    torch.arange(sq, device=q.device).unsqueeze(1) + offset
                )
                scores = scores.masked_fill(mask, float("-inf"))
            out[qs:qe, h, :] = torch.softmax(scores, dim=-1) @ v_b[:, kv_h, :]

    return out.to(q.dtype)


# GLM-5 MHA form dimensions:
#   qk_nope=192, qk_rope=64  →  head_dim_qk=256
#   v_head_dim=256
#   num_heads=64 (MHA: q_heads == kv_heads, head_grp_size=1)
glm5_mla_dimensions = MLAHeadDimensions(
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    v_head_dim=256,
    kv_lora_rank=512,
)


@pytest.mark.cuda
@pytest.mark.parametrize("batch_size", [4, 16])
@pytest.mark.parametrize("s_qo", [32, 64])
@pytest.mark.parametrize("s_kv", [64, 256])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("skips_softmax", [False, True])
def test_trtllm_gen_prefill_glm5(
    batch_size: int,
    s_qo: int,
    s_kv: int,
    causal: bool,
    skips_softmax: bool,
) -> None:
    """Test trtllm_ragged_attention_deepseek with GLM-5 MHA shapes.

    GLM-5 MHA form: 64 heads, head_dim_qk=256 (192 nope + 64 rope), head_dim_vo=256.
    """
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] != 10:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv")

    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    num_kv_heads = 64
    num_qo_heads = 64
    head_dim_qk = (
        glm5_mla_dimensions.qk_nope_head_dim + glm5_mla_dimensions.qk_rope_head_dim
    )
    head_dim_vo = glm5_mla_dimensions.v_head_dim

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    cumsum_s_qo = int(torch.sum(actual_seq_lens_q).item())
    cumsum_s_kv = int(torch.sum(actual_seq_lens_kv).item())

    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim_qk, device=device, dtype=torch.bfloat16
    )
    k_cache = torch.randn(
        cumsum_s_kv, num_kv_heads, head_dim_qk, device=device, dtype=torch.bfloat16
    )
    v_cache = torch.randn(
        cumsum_s_kv, num_kv_heads, head_dim_vo, device=device, dtype=torch.bfloat16
    )

    scale = float(1.0 / (head_dim_qk**0.5))

    workspace_buffer, workspace_buffer_ref = create_workspace_buffers(device)

    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()
    kv_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_kv.view(-1), dim=0),
        ]
    ).int()

    # Reference: naive attention in float32
    output_naive = naive_ragged_attention(
        q, k_cache, v_cache, qo_indptr, kv_indptr, scale, causal
    )

    # TRT-LLM gen
    output = torch.empty_like(output_naive)

    skip_softmax_threshold_scale_factor = 1e-30 if skips_softmax else None

    output_trtllm, lse_trtllm = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        q,
        k_cache,
        v_cache,
        workspace_buffer,
        actual_seq_lens_kv,
        s_qo,
        s_kv,
        scale,
        1.0,
        -1,
        batch_size,
        -1,
        qo_indptr,
        kv_indptr,
        False,
        causal,
        True,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        out=output,
    )

    torch.testing.assert_close(output_trtllm, output_naive, atol=1e-2, rtol=1e-2)
