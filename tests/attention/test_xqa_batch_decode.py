import pytest
import torch
from tests.test_helpers.sink_attention_reference import sink_attention_unified

import flashinfer
from flashinfer.utils import get_compute_capability

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}

GPU_DEVICE = "cuda:0"

global_workspace_buffer = None  # can be empty initialized
global_xqa_workspace_buffer = None  # must be zero initialized
workspace_size = 256 * 1024 * 1024


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def generate_seq_lens_decode(batch_size, q_len_per_req, max_in_kv_len):
    q_lens = torch.full((batch_size,), q_len_per_req, dtype=torch.int32)
    in_kv_lens = torch.randint(0, max_in_kv_len + 1, (batch_size,), dtype=torch.int)
    in_kv_lens[-1] = max_in_kv_len
    seq_lens = q_lens + in_kv_lens
    return q_lens, in_kv_lens, seq_lens


def generate_cumsum_lens(lens):
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=GPU_DEVICE),
            torch.cumsum(lens.to(GPU_DEVICE), dim=0, dtype=torch.int32),
        ]
    )


def create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype):
    q = torch.randn(
        torch.sum(q_lens).item(),
        num_qo_heads,
        head_dim,
        dtype=torch.bfloat16 if q_dtype == "fp8" else DTYPE_MAP[q_dtype],
        device=GPU_DEVICE,
    )
    if q_dtype == "fp8":
        q, q_scale = to_float8(q)
        # Reference implementation have functional issue or low precision with fp8, use bfloat16 and fake-quantization instead.
        ref_q = q.bfloat16() * q_scale
    else:
        q_scale = 1.0
        ref_q = q

    return q, q_scale, ref_q


def create_kv_cache(
    batch_size, seq_lens, page_size, num_kv_heads, head_dim, kv_dtype, ref_kv_dtype
):
    # Create separate K and V caches with NHD layout
    max_seq_len = torch.max(seq_lens).item()
    num_tokens = max_seq_len * batch_size
    num_pages = (num_tokens + page_size - 1) // page_size
    ref_kv_dtype_torch = DTYPE_MAP[ref_kv_dtype]
    if kv_dtype != "fp8":
        assert kv_dtype == ref_kv_dtype, (
            "kv_dtype and ref_kv_dtype must be the same for non-fp8 kv_cache"
        )

    # NHD layout: [num_pages, page_size, num_kv_heads, head_dim]
    k_cache = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=ref_kv_dtype_torch,
        device=GPU_DEVICE,
    )
    v_cache = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=ref_kv_dtype_torch,
        device=GPU_DEVICE,
    )

    # Convert K and V separately to fp8 if needed
    if kv_dtype == "fp8":
        k_cache, k_scale = to_float8(k_cache / 4.0)
        v_cache, v_scale = to_float8(v_cache / 4.0)
        # use high precision and fake-quantization for reference to avoid precision/functional issue
        ref_kv_cache = torch.stack(
            [
                k_cache.to(ref_kv_dtype_torch) * k_scale,
                v_cache.to(ref_kv_dtype_torch) * v_scale,
            ],
            dim=1,
        )
    else:
        k_scale = v_scale = 1.0
        ref_kv_cache = torch.stack([k_cache, v_cache], dim=1)
    # Combine K and V into interleaved format for the API
    kv_cache = torch.stack([k_cache, v_cache], dim=1)

    return kv_cache, k_scale, v_scale, ref_kv_cache


def create_page_table(batch_size, seq_lens, page_size):
    page_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_pages_per_seq = torch.max(page_per_seq).item()

    # Generate random but unique page IDs for all sequences
    total_pages_needed = torch.sum(page_per_seq).item()
    all_page_ids = torch.randperm(
        total_pages_needed, dtype=torch.int32, device=GPU_DEVICE
    )

    # Generate unique page IDs for all sequences
    page_tables = torch.zeros(
        (batch_size, max_num_pages_per_seq), dtype=torch.int32, device=GPU_DEVICE
    )

    # Populate page tables and track page assignments
    page_id = 0
    for i in range(batch_size):
        num_pages_needed = page_per_seq[i]
        page_tables[i, :num_pages_needed] = all_page_ids[
            page_id : page_id + num_pages_needed
        ]
        page_id += num_pages_needed
    return page_tables, all_page_ids, page_per_seq


def flatten_paged_kv(
    ref_kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    kv_last_page_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build flat K/V and token-level indptr from paged KV cache and page table.

    This version is specifically for NHD layout.
    """
    device = ref_kv_cache.device
    batch_size = int(page_table.shape[0])

    # Move loop-control tensors to CPU to avoid GPU sync in loops
    page_table_cpu = page_table.cpu()
    seq_lens_cpu = seq_lens.cpu()
    kv_last_page_len_cpu = kv_last_page_len.cpu()
    page_per_seq = (seq_lens_cpu + page_size - 1) // page_size
    k_list = []
    v_list = []
    for i in range(batch_size):
        pages_i = int(page_per_seq[i].item())
        last_len_i = int(kv_last_page_len_cpu[i].item())
        for j in range(pages_i):
            page_id = int(page_table_cpu[i, j].item())
            k_page = ref_kv_cache[page_id, 0]  # NHD: [page_size, num_heads, head_dim]
            v_page = ref_kv_cache[page_id, 1]
            if j == pages_i - 1:
                # NHD layout: truncate first dimension
                k_page = k_page[:last_len_i, :, :]
                v_page = v_page[:last_len_i, :, :]
            # NHD layout: already in "p h d" format, no need to rearrange
            k_list.append(k_page)
            v_list.append(v_page)
    k_flat = torch.cat(k_list, dim=0)
    v_flat = torch.cat(v_list, dim=0)
    kv_indptr_tokens = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ]
    )
    return k_flat, v_flat, kv_indptr_tokens


def create_workspace_buffers(device):
    # Lazily initialize and reuse global workspace buffers
    global global_workspace_buffer, global_xqa_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.empty(
            workspace_size, dtype=torch.int8, device=device
        )
    if global_xqa_workspace_buffer is None:
        global_xqa_workspace_buffer = torch.zeros(
            workspace_size, dtype=torch.int8, device=device
        )
    return global_xqa_workspace_buffer, global_workspace_buffer


def create_output(q, o_dtype):
    """Create output tensor for the given query and output dtype."""
    if o_dtype == "fp8":
        o_scale = torch.rand(1).item() * 0.5 + 0.5  # Scale range: 0.5 ~ 1.0
        out = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    else:
        o_scale = 1.0
        out = torch.empty(q.shape, dtype=DTYPE_MAP[o_dtype], device=q.device)

    return out, o_scale


def get_last_page_len(seq_lens, page_size):
    """Get the valid token count in the last page for each sequence"""
    last_page_len = seq_lens % page_size
    # If the sequence length is a multiple of page_size, the last page is full
    last_page_len = torch.where(last_page_len == 0, page_size, last_page_len)
    return last_page_len


@pytest.mark.skipif(
    get_compute_capability(torch.device(device="cuda"))[0] not in [9, 10, 12],
    reason="XQA is only supported on SM90, SM100, SM120 GPUs",
)
@pytest.mark.parametrize(
    "batch_size,q_len_per_req,page_size,num_kv_heads,head_grp_size",
    [
        (4, 1, 16, 2, 2),
        (4, 1, 32, 2, 4),
        (128, 1, 64, 2, 6),
        (256, 1, 64, 4, 8),
    ],
)
@pytest.mark.parametrize("window_left", [-1, 127])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("bf16", "fp8", "bf16"),
        ("fp16", "fp8", "fp16"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [True, False, None])
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("max_in_kv_len", [110])
def test_xqa_batch_decode(
    batch_size,
    q_len_per_req,
    page_size,
    num_kv_heads,
    head_grp_size,
    window_left,
    q_dtype,
    o_dtype,
    kv_dtype,
    enable_pdl,
    enable_sink,
    max_in_kv_len,
):
    """Test xqa_batch_decode_with_kv_cache function.

    This test is specifically for xqa which only supports NHD layout.
    """
    if q_len_per_req > 1:
        pytest.skip("xqa does not support speculative decoding yet")

    # Set up test parameters
    torch.manual_seed(0)
    head_dim = 128

    # Generate random sequence lengths
    num_qo_heads = num_kv_heads * head_grp_size
    q_lens, in_kv_lens, seq_lens = generate_seq_lens_decode(
        batch_size, q_len_per_req, max_in_kv_len
    )

    # Create query tensor and related data
    q, q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)
    q_indptr = generate_cumsum_lens(q_lens)

    # Create KV cache and related data (NHD layout)
    kv_cache, k_scale, v_scale, ref_kv_cache = create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        "bf16" if q_dtype == "fp8" else q_dtype,
    )
    page_table, all_page_ids, page_per_seq = create_page_table(
        batch_size, seq_lens, page_size
    )
    kv_indptr = generate_cumsum_lens(page_per_seq)
    kv_last_page_len = get_last_page_len(seq_lens, page_size)

    workspace_buffer, workspace_buffer_ref = create_workspace_buffers(GPU_DEVICE)

    # Create output tensor and related data
    out, o_scale = create_output(q, o_dtype)

    sm_scale = float(1.0 / (head_dim**0.5))

    # Build reference output
    kv_layout = "NHD"  # xqa only supports NHD
    plan_params = {
        "indptr": kv_indptr,
        "indices": all_page_ids,
        "last_page_len": kv_last_page_len.to(GPU_DEVICE),
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "page_size": page_size,
        "pos_encoding_mode": "NONE",
        "kv_data_type": ref_kv_cache.dtype,
        "q_data_type": ref_q.dtype,
        "window_left": window_left,
    }
    if not enable_sink:
        if q_len_per_req == 1:
            wrapper_ref = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer_ref, kv_layout, use_tensor_cores=True
            )
            wrapper_ref.plan(**plan_params)
            output_ref = wrapper_ref.run(ref_q, ref_kv_cache)
        else:
            # speculative decoding test
            wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer_ref, kv_layout
            )
            plan_params_prefill = plan_params.copy()
            plan_params_prefill.update(
                {
                    "qo_indptr": q_indptr,
                    "paged_kv_indptr": plan_params_prefill.pop("indptr"),
                    "paged_kv_indices": plan_params_prefill.pop("indices"),
                    "paged_kv_last_page_len": plan_params_prefill.pop("last_page_len"),
                    "head_dim_qk": plan_params_prefill.pop("head_dim"),
                    "causal": True,
                    "logits_soft_cap": 0.0,
                }
            )
            wrapper_ref.plan(**plan_params_prefill)
            output_ref = wrapper_ref.run(ref_q, ref_kv_cache)
    else:
        # Construct flat K/V via helper (NHD layout)
        k_flat, v_flat, kv_indptr_tokens = flatten_paged_kv(
            ref_kv_cache,
            page_table,
            seq_lens.to(GPU_DEVICE),
            page_size,
            kv_last_page_len,
        )
        sink = torch.rand(num_qo_heads, device=GPU_DEVICE, dtype=torch.float32) * 5
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

    # Run xqa_batch_decode_with_kv_cache function
    output = flashinfer.decode.xqa_batch_decode_with_kv_cache(
        q.contiguous(),
        kv_cache,
        workspace_buffer,
        page_table,
        seq_lens.to(GPU_DEVICE),
        torch.max(seq_lens).item(),
        q_scale * k_scale * sm_scale,  # bmm1_scale
        v_scale / o_scale,  # bmm2_scale
        window_left,  # window_left
        out=out,
        enable_pdl=enable_pdl,
        sinks=(sink if enable_sink else None),
        q_len_per_req=q_len_per_req,
    )

    # Verification
    torch.testing.assert_close(
        output,
        output_ref,
        rtol=1e-1 if kv_dtype == "fp8" else 1e-2,
        atol=1e-1 if kv_dtype == "fp8" else 1e-2,
    )


if __name__ == "__main__":
    # Run a simple test case
    test_xqa_batch_decode(
        batch_size=4,
        q_len_per_req=1,
        page_size=16,
        num_kv_heads=2,
        head_grp_size=1,
        window_left=-1,
        q_dtype="bf16",
        kv_dtype="bf16",
        o_dtype="bf16",
        enable_pdl=True,
        enable_sink=True,
        max_in_kv_len=110,
    )
