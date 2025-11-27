import math

import numpy as np
import pytest
import torch

from flashinfer import xqa, xqa_mla
from flashinfer.utils import get_compute_capability


def set_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def round_up(a, b):
    return math.ceil(a / b) * b


def div_up(a, b):
    return math.ceil(a / b)


props = torch.cuda.get_device_properties(0)
sm_count = props.multi_processor_count

beam_width = 1


def ref_attention(
    q,
    k_cache,
    v_cache,
    seq_len,
    q_scale,
    kv_scale,
    x_scale,
    attention_sinks,
    sliding_win_size,
    valid_elems_per_head,
    valid_elems_per_v_head=None,
):
    """
    Batched reference attention implementation.

    Args:
        q: [batch_size, nb_k_heads, head_grp_size, valid_elems_per_head]
        k_cache: [batch_size, nb_k_heads, seq_len, valid_elems_per_head]
        v_cache: [batch_size, nb_k_heads, seq_len, valid_elems_per_v_head]
        seq_len: scalar or [batch_size] tensor
        attention_sinks: [nb_k_heads, head_grp_size] or None

    Returns:
        out: [batch_size, nb_k_heads, head_grp_size, valid_elems_per_v_head]
    """
    batch_size, nb_k_heads, head_grp_size, _ = q.shape
    rcp_x_scale = 1.0 / x_scale
    qk_scale = q_scale * kv_scale / math.sqrt(valid_elems_per_head)

    # For MLA: V dimension may differ from K dimension
    if valid_elems_per_v_head is None:
        valid_elems_per_v_head = valid_elems_per_head

    # Convert to float32 for computation
    q_f32 = q.to(
        torch.float32
    )  # [batch_size, nb_k_heads, head_grp_size, valid_elems_per_head]
    k_cache_f32 = k_cache[:, :, :seq_len].to(
        torch.float32
    )  # [batch_size, nb_k_heads, seq_len, valid_elems_per_head]
    v_cache_f32 = v_cache[:, :, :seq_len, :valid_elems_per_v_head].to(
        torch.float32
    )  # [batch_size, nb_k_heads, seq_len, valid_elems_per_v_head]

    # Calculate sliding window start position
    if sliding_win_size == 0 or seq_len < sliding_win_size:
        seq_beg = 0
    else:
        seq_beg = seq_len - sliding_win_size

    # Q·K^T: [batch_size, nb_k_heads, head_grp_size, seq_len]
    gemm0_acc = torch.matmul(q_f32, k_cache_f32.transpose(-2, -1)) * qk_scale

    # Apply sliding window mask
    if seq_beg > 0:
        gemm0_acc[:, :, :, :seq_beg] = float("-inf")

    # Softmax
    row_max = torch.max(gemm0_acc, dim=-1, keepdim=True)[
        0
    ]  # [batch_size, nb_k_heads, head_grp_size, 1]
    x = torch.exp(
        gemm0_acc - row_max
    )  # [batch_size, nb_k_heads, head_grp_size, seq_len]

    row_sum = torch.sum(
        x, dim=-1, keepdim=True
    )  # [batch_size, nb_k_heads, head_grp_size, 1]

    # Add attention sinks contribution
    if attention_sinks is not None:
        # attention_sinks: [nb_k_heads, head_grp_size]
        # row_max: [batch_size, nb_k_heads, head_grp_size, 1]
        sink_weights = torch.exp(
            attention_sinks.unsqueeze(0).unsqueeze(-1) - row_max
        )  # [batch_size, nb_k_heads, head_grp_size, 1]
        row_sum = row_sum + sink_weights

    x = x * rcp_x_scale

    # Attention · V: [batch_size, nb_k_heads, head_grp_size, valid_elems_per_v_head]
    out = torch.matmul(x, v_cache_f32)

    out = out * (x_scale * kv_scale) / row_sum

    return out


@pytest.mark.skipif(
    get_compute_capability(torch.device(device="cuda"))[0] not in [9, 10, 12],
    reason="XQA is only supported on SM90, SM100, SM120 GPUs",
)
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("use_sliding_window", [True, False])
@pytest.mark.parametrize("input_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_attention_sinks", [True, False])
@pytest.mark.parametrize(
    "seq_len",
    [
        2,
        15,
        256,
        512,
        pytest.param(
            514,
            marks=pytest.mark.xfail(
                reason="seq_len=514 is known to fail in full test suite occasionally",
                strict=False,
            ),
        ),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("nb_k_heads", [2, 4])
@pytest.mark.parametrize("tokens_per_page", [16, 64])
@pytest.mark.parametrize("valid_elems_per_head", [32, 128])
@pytest.mark.parametrize("head_grp_size", [8, 16])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("q_scale", [1.0, 0.5])
@pytest.mark.parametrize(
    "fp8_kv_cache,kv_scale,use_fp8_output",
    [
        (False, 1.0, False),  # Non-FP8 KV cache: kv_scale=1.0, no FP8 output
        (True, 1.0, False),  # FP8 KV cache: kv_scale=1.0, no FP8 output
        (True, 1.0, True),  # FP8 KV cache: kv_scale=1.0, with FP8 output
        (True, 0.5, False),  # FP8 KV cache: kv_scale=0.5, no FP8 output
        (True, 0.5, True),  # FP8 KV cache: kv_scale=0.5, with FP8 output
    ],
)
def test_xqa(
    batch_size,
    nb_k_heads,
    seq_len,
    tokens_per_page,
    input_type,
    fp8_kv_cache,
    valid_elems_per_head,
    head_grp_size,
    use_attention_sinks,
    use_sliding_window,
    enable_pdl,
    kv_layout,
    kv_scale,
    q_scale,
    use_fp8_output,
):
    set_random_seed(0)

    nb_q_heads = nb_k_heads * head_grp_size

    output = torch.zeros(
        batch_size,
        beam_width,
        nb_q_heads,
        valid_elems_per_head,
        dtype=torch.float8_e4m3fn if use_fp8_output else input_type,
        device="cuda",
    )
    output.fill_(float("nan"))
    q_heads = torch.zeros(
        batch_size,
        beam_width,
        nb_q_heads,
        valid_elems_per_head,
        dtype=input_type,
        device="cuda",
    )
    q_heads.normal_(0, 1)
    if use_attention_sinks:
        # Vectorized creation of attention_sinks
        j_indices = torch.arange(head_grp_size, device="cuda")
        attention_sinks = 2.0 + (j_indices % 4).float()
        attention_sinks = (
            attention_sinks.unsqueeze(0).expand(nb_k_heads, head_grp_size).contiguous()
        )
    else:
        attention_sinks = None
    if use_sliding_window:
        sliding_win_size = 256
    else:
        sliding_win_size = 0

    max_seq_len = round_up(seq_len, tokens_per_page)
    nb_pages_per_seq = div_up(max_seq_len, tokens_per_page)
    # Total number of pages needed for all sequences
    total_num_pages = nb_pages_per_seq * batch_size

    # Create cache with specified layout
    if kv_layout == "NHD":
        # NHD layout: [num_pages, page_size, num_kv_heads, head_dim]
        cache_k_heads = torch.zeros(
            total_num_pages,
            tokens_per_page,
            nb_k_heads,
            valid_elems_per_head,
            dtype=input_type,
            device="cuda",
        )
        cache_v_heads = torch.zeros(
            total_num_pages,
            tokens_per_page,
            nb_k_heads,
            valid_elems_per_head,
            dtype=input_type,
            device="cuda",
        )
    else:  # HND layout
        # HND layout: [num_pages, num_kv_heads, page_size, head_dim]
        cache_k_heads = torch.zeros(
            total_num_pages,
            nb_k_heads,
            tokens_per_page,
            valid_elems_per_head,
            dtype=input_type,
            device="cuda",
        )
        cache_v_heads = torch.zeros(
            total_num_pages,
            nb_k_heads,
            tokens_per_page,
            valid_elems_per_head,
            dtype=input_type,
            device="cuda",
        )

    cache_k_heads.normal_(0, 1)
    cache_v_heads.normal_(0, 1)

    if fp8_kv_cache:
        # Scale down the cache heads to keep values within the representable range of FP8
        # and prevent overflow during computation. The factor 4.0 is chosen empirically.
        cache_k_heads /= 4.0
        cache_v_heads /= 4.0
    # Vectorized page list initialization
    total_pages = batch_size * nb_pages_per_seq
    page_list_arg = torch.arange(total_pages, dtype=torch.int32, device="cuda").view(
        batch_size, nb_pages_per_seq
    )

    # Shuffle page indices
    flattened = page_list_arg.flatten()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(42)
    indices = torch.randperm(flattened.numel(), generator=generator, device="cuda")
    shuffled_flat = flattened[indices]
    page_list_arg = shuffled_flat.view(batch_size, nb_pages_per_seq)

    # Vectorized zeroing of unused cache positions using advanced indexing
    if seq_len < max_seq_len:
        # Collect all (page_id, token_pos) pairs that need to be zeroed across all batches
        start_page = seq_len // tokens_per_page
        end_page = nb_pages_per_seq

        if start_page < end_page:
            # Get all page IDs that need partial/full zeroing: [batch_size, num_pages_to_zero]
            pages_to_zero = page_list_arg[
                :, start_page:end_page
            ]  # [batch_size, num_pages_to_zero]

            # For the first page (start_page), zero from [seq_len % tokens_per_page, tokens_per_page)
            # For subsequent pages, zero entirely [0, tokens_per_page)
            first_page_ids = pages_to_zero[:, 0]  # [batch_size]
            token_start_in_first_page = seq_len % tokens_per_page

            if token_start_in_first_page > 0:
                # Zero partial first page for all batches at once
                if kv_layout == "NHD":
                    cache_k_heads[first_page_ids, token_start_in_first_page:, :, :] = (
                        0.0
                    )
                    cache_v_heads[first_page_ids, token_start_in_first_page:, :, :] = (
                        0.0
                    )
                else:  # HND
                    cache_k_heads[first_page_ids, :, token_start_in_first_page:, :] = (
                        0.0
                    )
                    cache_v_heads[first_page_ids, :, token_start_in_first_page:, :] = (
                        0.0
                    )

            # Zero all subsequent full pages (if any) for all batches at once
            if pages_to_zero.shape[1] > 1:
                remaining_page_ids = pages_to_zero[
                    :, 1:
                ].flatten()  # Flatten all remaining pages
                if kv_layout == "NHD":
                    cache_k_heads[remaining_page_ids, :, :, :] = 0.0
                    cache_v_heads[remaining_page_ids, :, :, :] = 0.0
                else:  # HND
                    cache_k_heads[remaining_page_ids, :, :, :] = 0.0
                    cache_v_heads[remaining_page_ids, :, :, :] = 0.0

    seq_len_list = torch.zeros(
        batch_size, beam_width, dtype=torch.uint32, device="cuda"
    )
    seq_len_list.fill_(seq_len)

    kv_cache_scale = kv_scale

    nb_seq = nb_k_heads * batch_size
    nb_semaphores = round_up(nb_seq, 2) + 2 + nb_seq + 2

    semaphores = torch.zeros(nb_semaphores, dtype=torch.uint32, device="cuda")

    scratch_size = 256 << 20
    scratch_buf = torch.zeros(scratch_size, dtype=torch.uint8, device="cuda")

    rcp_out_scale = 4.0 if use_fp8_output else 1.0

    xqa(
        q_heads,
        cache_k_heads.to(torch.float8_e4m3fn) if fp8_kv_cache else cache_k_heads,
        cache_v_heads.to(torch.float8_e4m3fn) if fp8_kv_cache else cache_v_heads,
        page_list_arg,
        seq_len_list,
        output,
        scratch_buf,
        semaphores,
        nb_k_heads,
        tokens_per_page,
        sinks=attention_sinks,
        q_scale=torch.tensor(q_scale, device="cuda"),
        kv_scale=torch.tensor(kv_cache_scale, device="cuda"),
        sliding_win_size=sliding_win_size,
        kv_layout=kv_layout,
        sm_count=sm_count,
        enable_pdl=enable_pdl,
        rcp_out_scale=rcp_out_scale,
    )

    # Batch reconstruct all K/V caches from paged memory
    # [batch_size, nb_k_heads, max_seq_len, valid_elems_per_head]
    num_pages = (seq_len + tokens_per_page - 1) // tokens_per_page
    batch_k_cache = torch.zeros(
        batch_size,
        nb_k_heads,
        max_seq_len,
        valid_elems_per_head,
        dtype=input_type,
        device="cuda",
    )
    batch_v_cache = torch.zeros(
        batch_size,
        nb_k_heads,
        max_seq_len,
        valid_elems_per_head,
        dtype=input_type,
        device="cuda",
    )

    for req in range(batch_size):
        pages = page_list_arg[req, :num_pages]  # [num_pages]
        for idx_k_head in range(nb_k_heads):
            # Gather all pages at once
            if kv_layout == "NHD":
                k_pages = cache_k_heads[
                    pages, :, idx_k_head, :
                ]  # [num_pages, tokens_per_page, head_dim]
                v_pages = cache_v_heads[pages, :, idx_k_head, :]
            else:  # HND
                k_pages = cache_k_heads[
                    pages, idx_k_head, :, :
                ]  # [num_pages, tokens_per_page, head_dim]
                v_pages = cache_v_heads[pages, idx_k_head, :, :]

            # Reshape to contiguous sequence and store
            batch_k_cache[req, idx_k_head, : num_pages * tokens_per_page] = (
                k_pages.reshape(-1, valid_elems_per_head)
            )
            batch_v_cache[req, idx_k_head, : num_pages * tokens_per_page] = (
                v_pages.reshape(-1, valid_elems_per_head)
            )

    # Reshape q_heads: [batch_size, beam_width, nb_q_heads, dim] -> [batch_size, nb_k_heads, head_grp_size, dim]
    # Since beam_width = 1, we can squeeze it
    q_reshaped = q_heads.squeeze(1).reshape(
        batch_size, nb_k_heads, head_grp_size, valid_elems_per_head
    )

    # Batch compute reference attention
    ref_output_batch = ref_attention(
        q=q_reshaped,
        k_cache=batch_k_cache,
        v_cache=batch_v_cache,
        seq_len=seq_len,
        q_scale=q_scale,
        kv_scale=kv_cache_scale,
        x_scale=1.0,
        attention_sinks=attention_sinks if use_attention_sinks else None,
        sliding_win_size=sliding_win_size if use_sliding_window else 0,
        valid_elems_per_head=valid_elems_per_head,
    )  # [batch_size, nb_k_heads, head_grp_size, valid_elems_per_head]

    # Reshape kernel output to match: [batch_size, beam_width, nb_q_heads, dim] -> [batch_size, nb_k_heads, head_grp_size, dim]
    kernel_output_reshaped = (
        output.squeeze(1)
        .reshape(batch_size, nb_k_heads, head_grp_size, valid_elems_per_head)
        .to(torch.float32)
    )

    if use_fp8_output:
        ref_output_batch = ref_output_batch * rcp_out_scale

    # Set tolerances
    if fp8_kv_cache:
        atol = 0.05
        rtol = 0.05
    else:
        atol = 0.01
        rtol = 0.01
    if use_fp8_output:
        atol = 0.15
        rtol = 0.15

    # Compute differences for all elements at once
    diff_abs = torch.abs(ref_output_batch - kernel_output_reshaped)
    diff_rel = diff_abs / (torch.abs(ref_output_batch) + 1e-8)
    within_tolerance = (diff_abs <= atol) | (diff_rel <= rtol)

    # One-shot validation for all elements
    total_elements = ref_output_batch.numel()
    passing_elements = within_tolerance.sum().item()
    pass_ratio = passing_elements / total_elements
    required_ratio = 0.99

    assert pass_ratio >= required_ratio, (
        f"Batch validation failed: "
        f"Total {total_elements} elements, only {passing_elements} ({pass_ratio:.1%}) meet tolerance criteria, "
        f"require at least {required_ratio:.1%}"
    )


@pytest.mark.skipif(
    get_compute_capability(torch.device(device="cuda"))[0] not in [12],
    reason="XQA mla is only supported on SM120 GPUs",
)
@pytest.mark.parametrize("kv_scale", [1.0, 0.5])
@pytest.mark.parametrize("q_scale", [1.0, 0.5])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("seq_len", [2, 15, 256, 514, 2048])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens_per_page", [32, 64])
def test_xqa_mla(
    batch_size,
    seq_len,
    tokens_per_page,
    kv_scale,
    q_scale,
    enable_pdl,
):
    set_random_seed(0)

    # MLA specific constants (fixed, not parameterized)
    nb_k_heads = 1  # MLA only supports 1 K head
    head_grp_size = 128  # Fixed for MLA
    valid_elems_per_head_qk = 576  # Q and K dimension
    valid_elems_per_head_v = 512  # V dimension (output dimension)

    nb_q_heads = nb_k_heads * head_grp_size

    output = torch.zeros(
        batch_size,
        beam_width,
        nb_q_heads,
        valid_elems_per_head_v,  # Output dimension is 512 (V dimension)
        dtype=torch.bfloat16,
        device="cuda",
    )
    output.fill_(float("nan"))
    q_heads = torch.zeros(
        batch_size,
        beam_width,
        nb_q_heads,
        valid_elems_per_head_qk,  # Q dimension is 576
        dtype=torch.float32,
        device="cuda",
    )
    q_heads.normal_(0, 1)

    max_seq_len = round_up(seq_len, tokens_per_page)
    nb_pages_per_seq = div_up(max_seq_len, tokens_per_page)
    # Total number of pages needed for all sequences
    total_num_pages = nb_pages_per_seq * batch_size

    # NHD layout: [num_pages, page_size, num_kv_heads, head_dim]
    cache_k_heads = torch.zeros(
        total_num_pages,
        tokens_per_page,
        nb_k_heads,
        valid_elems_per_head_qk,  # K dimension is 576
        dtype=torch.float32,
        device="cuda",
    )
    cache_k_heads.normal_(0, 1)

    cache_v_heads = torch.zeros(
        total_num_pages,
        tokens_per_page,
        nb_k_heads,
        valid_elems_per_head_qk,  # V storage is 576 (but only 512 used)
        dtype=torch.float32,
        device="cuda",
    )
    cache_v_heads.normal_(0, 1)

    cache_k_heads /= 4.0
    cache_v_heads /= 4.0

    # Vectorized page list initialization
    total_pages = batch_size * nb_pages_per_seq
    page_list_arg = torch.arange(total_pages, dtype=torch.int32, device="cuda").view(
        batch_size, nb_pages_per_seq
    )

    # Shuffle page indices
    flattened = page_list_arg.flatten()
    indices = torch.randperm(flattened.numel(), device="cuda")
    shuffled_flat = flattened[indices]
    page_list_arg = shuffled_flat.view(batch_size, nb_pages_per_seq)

    # Vectorized zeroing of unused cache positions (NHD layout only for MLA)
    if seq_len < max_seq_len:
        start_page = seq_len // tokens_per_page
        end_page = nb_pages_per_seq

        if start_page < end_page:
            pages_to_zero = page_list_arg[
                :, start_page:end_page
            ]  # [batch_size, num_pages_to_zero]

            first_page_ids = pages_to_zero[:, 0]  # [batch_size]
            token_start_in_first_page = seq_len % tokens_per_page

            if token_start_in_first_page > 0:
                # Zero partial first page for all batches at once (NHD layout)
                cache_k_heads[first_page_ids, token_start_in_first_page:, :, :] = 0.0
                cache_v_heads[first_page_ids, token_start_in_first_page:, :, :] = 0.0

            # Zero all subsequent full pages (if any) for all batches at once
            if pages_to_zero.shape[1] > 1:
                remaining_page_ids = pages_to_zero[:, 1:].flatten()
                cache_k_heads[remaining_page_ids, :, :, :] = 0.0
                cache_v_heads[remaining_page_ids, :, :, :] = 0.0

    seq_len_list = torch.zeros(
        batch_size, beam_width, dtype=torch.uint32, device="cuda"
    )
    seq_len_list.fill_(seq_len)

    kv_cache_scale = kv_scale

    nb_seq = nb_k_heads * batch_size
    nb_semaphores = round_up(nb_seq, 2) + 2 + nb_seq + 2

    semaphores = torch.zeros(nb_semaphores, dtype=torch.uint32, device="cuda")

    scratch_size = 256 << 20
    scratch_buf = torch.zeros(scratch_size, dtype=torch.uint8, device="cuda")

    xqa_mla(
        q_heads.to(torch.float8_e4m3fn),
        cache_k_heads.to(torch.float8_e4m3fn),
        cache_v_heads.to(torch.float8_e4m3fn),
        page_list_arg,
        seq_len_list,
        output,
        scratch_buf,
        semaphores,
        tokens_per_page,
        q_scale=q_scale,
        kv_scale=kv_cache_scale,
        sm_count=sm_count,
        enable_pdl=enable_pdl,
    )

    # Batch reconstruct all K/V caches from paged memory
    # [batch_size, nb_k_heads, max_seq_len, valid_elems_per_head_qk]
    num_pages = (seq_len + tokens_per_page - 1) // tokens_per_page
    batch_k_cache = torch.zeros(
        batch_size,
        nb_k_heads,
        max_seq_len,
        valid_elems_per_head_qk,
        dtype=torch.float32,
        device="cuda",
    )
    batch_v_cache = torch.zeros(
        batch_size,
        nb_k_heads,
        max_seq_len,
        valid_elems_per_head_qk,
        dtype=torch.float32,
        device="cuda",
    )

    for req in range(batch_size):
        pages = page_list_arg[req, :num_pages]  # [num_pages]
        for idx_k_head in range(nb_k_heads):
            # NHD layout: [num_pages, tokens_per_page, nb_k_heads, head_dim]
            k_pages = cache_k_heads[
                pages, :, idx_k_head, :
            ]  # [num_pages, tokens_per_page, head_dim]
            v_pages = cache_v_heads[pages, :, idx_k_head, :]

            # Reshape to contiguous sequence and store
            batch_k_cache[req, idx_k_head, : num_pages * tokens_per_page] = (
                k_pages.reshape(-1, valid_elems_per_head_qk)
            )
            batch_v_cache[req, idx_k_head, : num_pages * tokens_per_page] = (
                v_pages.reshape(-1, valid_elems_per_head_qk)
            )

    # Reshape q_heads: [batch_size, beam_width, nb_q_heads, dim] -> [batch_size, nb_k_heads, head_grp_size, dim]
    # Since beam_width = 1, we can squeeze it
    q_reshaped = q_heads.squeeze(1).reshape(
        batch_size, nb_k_heads, head_grp_size, valid_elems_per_head_qk
    )

    # Batch compute reference attention
    ref_output_batch = ref_attention(
        q=q_reshaped,
        k_cache=batch_k_cache,
        v_cache=batch_v_cache,
        seq_len=seq_len,
        q_scale=q_scale * math.sqrt(576),
        kv_scale=kv_cache_scale,
        x_scale=1.0,
        attention_sinks=None,
        sliding_win_size=0,
        valid_elems_per_head=valid_elems_per_head_qk,  # Q/K dimension (576)
        valid_elems_per_v_head=valid_elems_per_head_v,  # V dimension (512)
    )  # [batch_size, nb_k_heads, head_grp_size, valid_elems_per_v_head]

    # Reshape kernel output to match: [batch_size, beam_width, nb_q_heads, valid_elems_per_v_head] -> [batch_size, nb_k_heads, head_grp_size, valid_elems_per_v_head]
    kernel_output_reshaped = (
        output.squeeze(1)
        .reshape(batch_size, nb_k_heads, head_grp_size, valid_elems_per_head_v)
        .to(torch.float32)
    )

    # Set tolerances
    atol = 0.05
    rtol = 0.05

    # Compute differences for all elements at once
    diff_abs = torch.abs(ref_output_batch - kernel_output_reshaped)
    diff_rel = diff_abs / (torch.abs(ref_output_batch) + 1e-8)
    within_tolerance = (diff_abs <= atol) | (diff_rel <= rtol)

    # One-shot validation for all elements
    total_elements = ref_output_batch.numel()
    passing_elements = within_tolerance.sum().item()
    pass_ratio = passing_elements / total_elements
    required_ratio = 0.95

    assert pass_ratio >= required_ratio, (
        f"Batch validation failed: "
        f"Total {total_elements} elements, only {passing_elements} ({pass_ratio:.1%}) meet tolerance criteria, "
        f"require at least {required_ratio:.1%}"
    )
