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
    unpack_compare_nvfp4,
)


def _test_trtllm_batch_prefill(
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
    device_scale: float,
    head_dim: int,
    non_contiguous_query: bool = False,
    skips_softmax: bool = False,
    uses_shared_paged_kv_idx: bool = True,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")

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
        "causal": True,
        "pos_encoding_mode": "NONE",
        "logits_soft_cap": 0.0,
        "q_data_type": ref_q.dtype,
        "kv_data_type": ref_kv_cache.dtype,
        "window_left": window_left,
    }
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
        output_ref = wrapper_ref.run(ref_q, ref_kv_cache)
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
            True,
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

    output = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        q_input,
        kv_cache_kernel,
        workspace_buffer,
        page_table_kernel,
        seq_lens.to(GPU_DEVICE),
        torch.max(q_lens).item(),
        torch.max(seq_lens).item(),
        bmm1_scale,  # bmm1_scale
        bmm2_scale,  # bmm2_scale
        batch_size,
        q_indptr,
        kv_indptr,
        window_left,  # window_left
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
    )
    # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
    # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
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


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    [
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
    ],
)
@pytest.mark.parametrize("window_left", [-1])  # todo(Siyuan): add 127 window_left
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("fp8", "fp8", "bf16"),
        ("fp8", "fp8", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
        ("fp8", "nvfp4", "fp8"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("max_q_len", [511])
@pytest.mark.parametrize("max_kv_len", [2047])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("non_contiguous_query", [False, True])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_prefill(
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
def test_trtllm_gen_prefill(
    mla_dimensions: MLAHeadDimensions,
    batch_size: int,
    s_qo: int,
    s_kv: int,
    num_kv_heads: int,
    head_grp_size: int,
    causal: bool,
    skips_softmax: bool,
) -> None:
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test as causal")

    num_qo_heads = num_kv_heads * head_grp_size
    head_dim_qk = mla_dimensions.qk_nope_head_dim + mla_dimensions.qk_rope_head_dim
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
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        out=output,
    )
    torch.testing.assert_close(
        output_trtllm,
        output_ref,
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        lse_trtllm,
        lse_ref,
        atol=1e-3,
        rtol=1e-3,
    )
    # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
    # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
    assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()


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
        mla_dimensions,
        batch_size,
        s_qo,
        s_kv,
        num_kv_heads,
        head_grp_size,
        causal,
        skips_softmax,
    )
