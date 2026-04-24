import math
from typing import Union

import pytest
import torch
from flashinfer.mla import (
    MLAHeadDimensions,
    deepseek_mla_dimensions,
    smaller_mla_dimensions,
)
from tests.test_helpers.utils_fp4 import (
    cast_from_fp4,
    recover_swizzled_scales,
    ref_fp4_quant,
)
from tests.test_helpers.test_helpers import assert_close_with_mismatch_tolerance
import einops
from tests.test_helpers.sink_attention_reference import sink_attention_unified
from flashinfer.fp4_quantization import nvfp4_quantize_paged_kv_cache

import flashinfer
from flashinfer.utils import FP4Tensor, ceil_div, round_up, get_compute_capability

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
    "nvfp4": "nvfp4",
}

GPU_DEVICE = "cuda:0"

global_workspace_buffer = None  # can.be empty initialized
global_trtllm_gen_fmha_workspace_buffer = None  # must be zero initialized
workspace_size = 256 * 1024 * 1024


def flip_coin(*args, **kwargs):
    # Use any test parameters to deterministically decide branch
    # This makes test configurations go through different paths
    param_tuple = args + tuple(sorted(kwargs.items()))
    hash_value = hash(param_tuple)
    return (hash_value % 2) == 0


def to_float8(x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def generate_seq_lens_prefill(batch_size: int, max_q_len: int, max_in_kv_len: int):
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    q_lens[-1] = max_q_len
    in_kv_lens = torch.randint(0, max_in_kv_len + 1, (batch_size,), dtype=torch.int)
    in_kv_lens[-1] = max_in_kv_len
    seq_lens = q_lens + in_kv_lens
    return q_lens, in_kv_lens, seq_lens


def generate_seq_lens_decode(
    batch_size: int,
    q_len_per_req: int | None,
    max_in_kv_len: int,
    max_q_len: int | None,
):
    if q_len_per_req is not None:
        assert max_q_len is None, "Can not specify both q_len_per_req and max_q_len."
        q_lens = torch.full((batch_size,), q_len_per_req, dtype=torch.int32)
    else:
        assert max_q_len is not None, "Must specify either q_len_per_req or max_q_len."
        q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    in_kv_lens = torch.randint(0, max_in_kv_len + 1, (batch_size,), dtype=torch.int)
    in_kv_lens[-1] = max_in_kv_len
    seq_lens = q_lens + in_kv_lens
    return q_lens, in_kv_lens, seq_lens


def generate_cumsum_lens(lens: torch.Tensor):
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=GPU_DEVICE),
            torch.cumsum(lens.to(GPU_DEVICE), dim=0, dtype=torch.int32),
        ]
    )


def create_query_tensor(
    q_lens: torch.Tensor, num_qo_heads: int, head_dim: int, q_dtype: str
):
    q = torch.randn(
        int(torch.sum(q_lens).item()),
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
    batch_size: int,
    seq_lens: torch.Tensor,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype: str,
    ref_kv_dtype: str,
    kv_layout: str = "HND",
):
    # Create separate K and V caches
    max_seq_len = torch.max(seq_lens).item()
    num_pages_per_seq = (max_seq_len + page_size - 1) // page_size
    num_pages = num_pages_per_seq * batch_size
    ref_kv_dtype_torch = DTYPE_MAP[ref_kv_dtype]
    if kv_dtype not in ("fp8", "nvfp4"):
        assert kv_dtype == ref_kv_dtype, (
            "kv_dtype and ref_kv_dtype must be the same for non-fp8/nvfp4 kv_cache"
        )

    # Create cache with appropriate layout
    if kv_layout == "HND":
        # HND layout: [num_pages, num_kv_heads, page_size, head_dim]
        k_cache = torch.randn(
            num_pages,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=ref_kv_dtype_torch,
            device=GPU_DEVICE,
        )
        v_cache = torch.randn(
            num_pages,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=ref_kv_dtype_torch,
            device=GPU_DEVICE,
        )
    else:  # NHD layout
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

    kv_cache_sf = None
    # Convert K and V separately to fp8 if needed
    if kv_dtype == "fp8":
        k_cache, k_scale = to_float8(k_cache)
        v_cache, v_scale = to_float8(v_cache)
        # use high precision and fake-quantization for reference to avoid precision/functional issue
        ref_kv_cache = torch.stack(
            [
                k_cache.to(ref_kv_dtype_torch) * k_scale,
                v_cache.to(ref_kv_dtype_torch) * v_scale,
            ],
            dim=1,
        )
        kv_cache = torch.stack([k_cache, v_cache], dim=1)
    elif kv_dtype == "nvfp4":
        # Add outlier channels to stress per-block scaling.
        outlier_start = head_dim // 4
        outlier_end = outlier_start + min(16, head_dim // 8)
        k_cache[..., outlier_start:outlier_end] *= 30.0
        v_cache[..., outlier_start:outlier_end] *= 2.0
        # Reference is the unquantized BF16 data (with outliers applied)
        ref_kv_cache = torch.stack([k_cache, v_cache], dim=1)
        kv_cache, kv_cache_sf, k_scale, v_scale = nvfp4_quantize_paged_kv_cache(
            k_cache, v_cache, kv_layout=kv_layout
        )
    else:
        k_scale = v_scale = 1.0
        ref_kv_cache = torch.stack([k_cache, v_cache], dim=1)
        kv_cache = torch.stack([k_cache, v_cache], dim=1)

    return kv_cache, k_scale, v_scale, ref_kv_cache, kv_cache_sf


def create_page_table(batch_size: int, seq_lens: torch.Tensor, page_size: int):
    page_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_pages_per_seq = int(torch.max(page_per_seq).item())

    # Generate random but unique page IDs for all sequences
    total_pages_needed = int(torch.sum(page_per_seq).item())
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


def prepare_paged_kv_for_kernel(
    kv_cache: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    page_table: torch.Tensor,
    uses_shared_paged_kv_idx: bool,
    kv_cache_sf: Union[tuple[torch.Tensor, torch.Tensor], None] = None,
) -> tuple[
    Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    torch.Tensor,
    Union[tuple[torch.Tensor, torch.Tensor], None],
]:
    """Convert shared-page KV cache layout to separate-page layout for TRT-LLM.

    When uses_shared_paged_kv_idx is True (FlashInfer/vLLM style), returns the
    original tensors unchanged.

    When False (TRT-LLM style), interleaves K and V pages so original page p
    becomes K at index 2*p and V at 2*p+1, and builds a
    [batch_size, 2, maxPages] page table where dim 1 distinguishes K (0) and V (1).
    Returns the reshaped cache as a (cache, cache) tuple so both K and V share
    the same base pointer.  Block scales, if provided, are interleaved the same
    way since the kernel uses the same page indices to access them.

    Returns:
        (kv_cache_arg, page_table, kv_cache_sf) ready to pass to the kernel.
    """
    if uses_shared_paged_kv_idx:
        return kv_cache, page_table, kv_cache_sf

    def _interleave_kv(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Stack [num_pages,...] K and V along dim-1 then flatten to [2*num_pages,...]."""
        return torch.stack([k, v], dim=1).reshape(k.shape[0] * 2, *k.shape[1:])

    if isinstance(kv_cache, tuple):
        k_cache, v_cache = kv_cache
        interleaved = _interleave_kv(k_cache, v_cache)
    else:
        num_pages = kv_cache.shape[0]
        interleaved = kv_cache.reshape(num_pages * 2, *kv_cache.shape[2:])

    trtllm_page_table = torch.stack([2 * page_table, 2 * page_table + 1], dim=1)

    if kv_cache_sf is not None:
        k_sf, v_sf = kv_cache_sf
        interleaved_sf = _interleave_kv(k_sf, v_sf)
        kv_cache_sf = (interleaved_sf, interleaved_sf)

    return (interleaved, interleaved), trtllm_page_table, kv_cache_sf


def flatten_paged_kv(
    ref_kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    kv_last_page_len: torch.Tensor,
    kv_layout: str = "HND",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build flat K/V and token-level indptr from paged KV cache and page table."""
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
            k_page = ref_kv_cache[page_id, 0]
            v_page = ref_kv_cache[page_id, 1]
            if kv_layout == "HND":
                # HND layout: [num_kv_heads, page_size, head_dim]
                if j == pages_i - 1:
                    k_page = k_page[:, :last_len_i, :]
                    v_page = v_page[:, :last_len_i, :]
                k_list.append(einops.rearrange(k_page, "h p d -> p h d"))
                v_list.append(einops.rearrange(v_page, "h p d -> p h d"))
            else:  # NHD layout
                # NHD layout: [page_size, num_kv_heads, head_dim]
                if j == pages_i - 1:
                    k_page = k_page[:last_len_i, :, :]
                    v_page = v_page[:last_len_i, :, :]
                k_list.append(einops.rearrange(k_page, "p h d -> p h d"))
                v_list.append(einops.rearrange(v_page, "p h d -> p h d"))
    k_flat = torch.cat(k_list, dim=0)
    v_flat = torch.cat(v_list, dim=0)
    kv_indptr_tokens = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ]
    )
    return k_flat, v_flat, kv_indptr_tokens


def create_workspace_buffers(device: torch.device):
    # Lazily initialize and reuse global workspace buffers
    global global_workspace_buffer, global_trtllm_gen_fmha_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.empty(
            workspace_size, dtype=torch.int8, device=device
        )
    if global_trtllm_gen_fmha_workspace_buffer is None:
        global_trtllm_gen_fmha_workspace_buffer = torch.zeros(
            workspace_size, dtype=torch.int8, device=device
        )
    return global_trtllm_gen_fmha_workspace_buffer, global_workspace_buffer


def create_output(
    q: torch.Tensor, o_dtype: str, create_out_tensor: bool, create_out_dtype: bool
):
    if o_dtype == "fp8":
        o_scale = torch.rand(1).item() * 0.5 + 0.5  # Scale range: 0.5 ~ 1.0
    else:
        o_scale = 1.0
    o_sf_scale = (
        300 if o_dtype == "nvfp4" else None
    )  # choose a value to make error smaller by testing.
    o_sf_vec_size = 16 if o_dtype == "nvfp4" else None

    if create_out_tensor:
        if o_dtype == "nvfp4":
            fp4_out_shape = q.shape[:-1] + (ceil_div(q.shape[-1], 2),)

            extra_size = torch.randint(0, 256, (1,)).item()

            fp4_out_scale_shape = (
                round_up(q.shape[0] + extra_size, 128),
                round_up(q.shape[1] * q.shape[2] // o_sf_vec_size, 4),
            )

            out_scale_factor = torch.empty(
                fp4_out_scale_shape, dtype=torch.float8_e4m3fn, device=q.device
            )
            rounded_extra_size = fp4_out_scale_shape[0] - q.shape[0]
            o_sf_start_index = (
                torch.randint(0, rounded_extra_size, (1,)).item()
                if rounded_extra_size > 0
                else 0
            )
            out_data = torch.empty(fp4_out_shape, dtype=torch.uint8, device=q.device)
            out = FP4Tensor(out_data, out_scale_factor, o_sf_start_index)
        else:
            out = torch.empty_like(q, dtype=DTYPE_MAP[o_dtype])
    else:
        out = None
    out_dtype = DTYPE_MAP[o_dtype] if create_out_dtype else None
    return out, out_dtype, o_scale, o_sf_scale, o_sf_vec_size


def get_last_page_len(seq_lens, page_size):
    kv_last_page_len = seq_lens % page_size
    kv_last_page_len[kv_last_page_len == 0] = page_size
    return kv_last_page_len


def unpack_compare_nvfp4(
    output: FP4Tensor,
    output_ref: torch.Tensor,
    o_sf_scale: float,
    o_sf_vec_size: int,
    sf_rtol: float = 2e-1,
    sf_atol: float = 2e-1,
    rmse_tol: float = 0.3,
):
    output_ref, out_scale_factor_ref = ref_fp4_quant(
        output_ref, o_sf_scale, o_sf_vec_size
    )

    output_unpacked = cast_from_fp4(output.data)
    out_scale_factor = recover_swizzled_scales(
        output.scale,
        output_unpacked.shape[0],
        math.prod(list(output_unpacked.shape[1:])),
        o_sf_vec_size,
        output.scale_start_index,
    )

    torch.testing.assert_close(
        out_scale_factor.float().reshape(out_scale_factor_ref.shape),
        out_scale_factor_ref.float(),
        rtol=sf_rtol,
        atol=sf_atol,
    )
    rmse = torch.sqrt(torch.mean((output_unpacked.float() - output_ref.float()) ** 2))
    assert rmse.item() < rmse_tol
    return output_unpacked, output_ref


def generate_causal_mask(
    batch_size: int,
    q_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate causal attention mask for speculative decoding.

    Parameters
    ----------
    batch_size : int
        Batch size
    q_seq_len : int
        Query sequence length (number of speculative decoding tokens)
    device : torch.device
        Target device for the mask tensor

    Returns
    -------
    torch.Tensor
        Causal mask with shape [batch_size, q_seq_len, mask_size_per_row]
        where mask_size_per_row = divUp(q_seq_len, 32) * 2 (in uint16_t units).
        Data type: torch.uint16

    """
    num_packed_masks_per_token = (q_seq_len + 31) // 32

    q_indices = torch.arange(q_seq_len, device=device, dtype=torch.int32).unsqueeze(1)
    kv_indices = torch.arange(q_seq_len, device=device, dtype=torch.int32).unsqueeze(0)

    causal_bool_mask = kv_indices <= q_indices

    padded_seq_len = num_packed_masks_per_token * 32
    if padded_seq_len > q_seq_len:
        padding = torch.zeros(
            q_seq_len, padded_seq_len - q_seq_len, device=device, dtype=torch.bool
        )
        causal_bool_mask = torch.cat([causal_bool_mask, padding], dim=1)

    causal_bool_mask = causal_bool_mask.view(q_seq_len, num_packed_masks_per_token, 32)

    bit_positions = torch.tensor(
        [1 << i for i in range(32)], device=device, dtype=torch.int64
    )

    mask_uint32 = (
        (causal_bool_mask.to(torch.int64) * bit_positions).sum(dim=-1).to(torch.uint32)
    )

    mask_uint32 = (
        mask_uint32.unsqueeze(0)
        .expand(batch_size, q_seq_len, num_packed_masks_per_token)
        .contiguous()
    )

    mask_uint16 = mask_uint32.view(torch.uint16)

    return mask_uint16


def sdpa_paged_reference(
    ref_q: torch.Tensor,
    ref_kv_cache: torch.Tensor,
    q_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_layout: str,
    window_left: int,
):
    """Pure PyTorch SDPA reference for head dims unsupported by FlashInfer kernels.

    ref_kv_cache layout:
      HND: [num_pages, 2, num_kv_heads, page_size, head_dim]
      NHD: [num_pages, 2, page_size, num_kv_heads, head_dim]
    """
    sm_scale = 1.0 / (head_dim**0.5)
    batch_size = q_lens.shape[0]
    q_indptr = torch.cat(
        [
            torch.zeros(1, dtype=q_lens.dtype, device=q_lens.device),
            torch.cumsum(q_lens, dim=0),
        ]
    )
    outputs = []
    for b in range(batch_size):
        q_start = q_indptr[b].item()
        q_end = q_indptr[b + 1].item()
        q_b = ref_q[q_start:q_end]  # [q_len, num_qo_heads, head_dim]
        s_len = seq_lens[b].item()
        num_pages = (s_len + page_size - 1) // page_size

        # Gather KV from paged cache
        page_ids = page_table[b, :num_pages]
        kv_pages = ref_kv_cache[page_ids]
        k_pages = kv_pages[:, 0]  # K half
        v_pages = kv_pages[:, 1]  # V half

        if kv_layout == "HND":
            # k_pages: [num_pages, num_kv_heads, page_size, head_dim]
            # transpose to [num_pages, page_size, num_kv_heads, head_dim] then flatten pages
            k_flat = k_pages.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)[
                :s_len
            ]
            v_flat = v_pages.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)[
                :s_len
            ]
        else:  # NHD
            # k_pages: [num_pages, page_size, num_kv_heads, head_dim]
            k_flat = k_pages.reshape(-1, num_kv_heads, head_dim)[:s_len]
            v_flat = v_pages.reshape(-1, num_kv_heads, head_dim)[:s_len]

        # k_flat, v_flat: [s_len, num_kv_heads, head_dim]
        q_len = q_b.shape[0]
        head_grp = num_qo_heads // num_kv_heads

        # Expand KV for GQA: [s_len, num_qo_heads, head_dim]
        k_exp = (
            k_flat.unsqueeze(2)
            .expand(-1, num_kv_heads, head_grp, -1)
            .reshape(s_len, num_qo_heads, head_dim)
        )
        v_exp = (
            v_flat.unsqueeze(2)
            .expand(-1, num_kv_heads, head_grp, -1)
            .reshape(s_len, num_qo_heads, head_dim)
        )

        # Transpose to [num_qo_heads, seq_len, head_dim] for SDPA
        q_t = q_b.transpose(0, 1).float()  # [num_qo_heads, q_len, head_dim]
        k_t = k_exp.transpose(0, 1).float()  # [num_qo_heads, s_len, head_dim]
        v_t = v_exp.transpose(0, 1).float()  # [num_qo_heads, s_len, head_dim]

        # Build causal mask: query position i can attend to kv position j if j <= (s_len - q_len) + i
        kv_offset = s_len - q_len
        q_pos = torch.arange(q_len, device=q_b.device).unsqueeze(1) + kv_offset
        k_pos = torch.arange(s_len, device=q_b.device).unsqueeze(0)
        causal_mask = k_pos <= q_pos  # [q_len, s_len]
        if window_left >= 0:
            causal_mask = causal_mask & (q_pos - k_pos <= window_left)
        attn_mask = causal_mask.unsqueeze(0).expand(num_qo_heads, -1, -1)

        out_b = torch.nn.functional.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            scale=sm_scale,
        )
        outputs.append(
            out_b.transpose(0, 1).to(ref_q.dtype)
        )  # [q_len, num_qo_heads, head_dim]

    return torch.cat(outputs, dim=0)


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


def _test_trtllm_batch_decode(
    backend: str,
    kv_layout: str,
    batch_size: int,
    q_len_per_req: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_in_kv_len: int,
    head_dim: int,
    device_scale: bool = False,
    max_q_len: int | None = None,
    non_contiguous_query: bool = False,
    skips_softmax: bool = False,
    uses_shared_paged_kv_idx: bool = True,
) -> None:
    """
    Common function for testing trtllm-gen decode.

    Combinations of parameters are tested in test_trtllm_batch_decode() and test_trtllm_batch_decode_...()
    """
    compute_capability = get_compute_capability(torch.device(device="cuda"))

    # Check GPU architecture requirements for different backends
    if backend == "trtllm-gen" and compute_capability[0] != 10:
        pytest.skip("trtllm-gen backend requires SM100 and SM103 GPUs.")
    if backend == "xqa" and compute_capability[0] < 9:
        pytest.skip("xqa backend requires SM90+ GPUs.")

    if backend == "xqa" and skips_softmax:
        pytest.skip("xqa backend does not support skips_softmax")

    if skips_softmax and q_dtype != kv_dtype:
        pytest.skip(
            "skips_softmax does not currently support Q and Kv types being different"
        )

    # xqa backend doesn't support nvfp4 output
    if backend == "xqa" and o_dtype == "nvfp4":
        pytest.skip("xqa backend does not support nvfp4 output")

    if backend == "xqa" and q_dtype == "fp8":
        pytest.skip("xqa backend only supports fp16 and bf16 query")

    # XQA backend doesn't support non-shared page indices
    if backend == "xqa" and not uses_shared_paged_kv_idx:
        pytest.skip("xqa backend does not support non-shared page indices")

    if backend == "trtllm-gen" and o_dtype == "fp8" and q_dtype != "fp8":
        pytest.skip("trtllm-gen backend only supports fp8 output for fp8 query")

    # NVFP4 KV cache constraints
    if kv_dtype == "nvfp4":
        if backend != "trtllm-gen":
            pytest.skip("NVFP4 KV cache only supported by trtllm-gen backend")
        if q_dtype != "fp8":
            pytest.skip("NVFP4 KV cache requires FP8 query")
        if o_dtype != "fp8":
            pytest.skip("NVFP4 KV cache only supports FP8 output")
        pass

    # Set up test parameters
    torch.manual_seed(0)

    # Generate random sequence lengths
    num_qo_heads = num_kv_heads * head_grp_size
    q_lens, in_kv_lens, seq_lens = generate_seq_lens_decode(
        batch_size, q_len_per_req, max_in_kv_len, max_q_len
    )

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

    kv_cache_arg, page_table_kernel, kv_cache_sf_kernel = prepare_paged_kv_for_kernel(
        kv_cache, page_table, uses_shared_paged_kv_idx, kv_cache_sf
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
        if q_len_per_req is not None and q_len_per_req == 1:
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

    if q_len_per_req and q_len_per_req > 1:
        # only used for xqa speculative decoding
        mask = generate_causal_mask(batch_size, q_len_per_req, GPU_DEVICE)
    else:
        mask = None

    # Run decode function call with specified backend
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

    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        q_input,
        kv_cache_arg,
        workspace_buffer,
        page_table_kernel,
        seq_lens.to(GPU_DEVICE),
        torch.max(seq_lens).item(),
        bmm1_scale,
        bmm2_scale,
        window_left,  # window_left
        out=out,
        out_dtype=out_dtype,
        o_sf_scale=o_sf_scale,
        o_sf_vec_size=o_sf_vec_size,
        sinks=(sink if enable_sink else None),
        kv_layout=kv_layout,
        enable_pdl=enable_pdl,
        backend=backend,
        q_len_per_req=q_len_per_req,
        o_scale=o_scale,
        mask=mask,
        max_q_len=max_q_len if max_q_len is not None else None,
        cum_seq_lens_q=q_indptr if max_q_len is not None else None,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        kv_cache_sf=kv_cache_sf_kernel,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )
    if backend == "trtllm-gen":
        # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
        # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
        assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()

    if o_dtype == "nvfp4":
        output, output_ref = unpack_compare_nvfp4(
            output, output_ref, o_sf_scale, o_sf_vec_size
        )
        assert o_scale == 1.0
        rtol, atol = 3e-1, 1e0
    elif q_dtype == "fp8" and o_dtype == "fp8":
        rtol, atol = 5e-2, 7e-2
    elif q_dtype == "fp8" and o_dtype in ["bf16", "fp16"]:
        rtol, atol = 4e-2, 7e-2
    else:
        rtol, atol = 1e-2, 1e-2

    if backend == "xqa" and kv_dtype == "fp8":
        atol = 1e-1
        rtol = 1e-1

    # NVFP4 KV cache has significant quantization error, especially with
    # outlier channels that create large per-block dynamic range.
    if kv_dtype == "nvfp4":
        rtol, atol = 5e-1, 5e-1

    # convert to float32 for fp8 is not supported by assert_close
    # relax rtol and atol for speculative decoding test
    if (q_len_per_req and q_len_per_req > 1) or (max_q_len and max_q_len > 1):
        rtol, atol = rtol * 2, atol * 2

    # NVFP4 KV cache has higher mismatch rate due to 4-bit quantization noise,
    # especially with outlier channels that stress per-block scaling.
    allowed_mismatch_rate = 0.10 if kv_dtype == "nvfp4" else 5e-5
    # Calculate max allowed mismatched elements based on tensor size
    total_elements = (output.float() * o_scale).numel()
    max_mismatched_elements = int(allowed_mismatch_rate * total_elements)

    assert_close_with_mismatch_tolerance(
        output.float() * o_scale,
        output_ref.float(),
        rtol=rtol,
        atol=atol,
        max_mismatched_elements=max_mismatched_elements,
    )

    # NVFP4 KV cache: use cosine similarity instead of element-wise comparison.
    # Cosine similarity is scale-invariant, which is important because the
    # FP4→FP8 dequant path introduces a global scale factor.  More critically,
    # it reliably catches block-scale mismatches (e.g. wrong swizzling) that
    # element-wise tolerances miss when values happen to be small.
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

    # Only test wrapper with trtllm-gen backend
    if (
        o_dtype != "nvfp4"
        and kv_dtype != "nvfp4"
        and backend == "trtllm-gen"
        and q_len_per_req
        is not None  # only test for the case all requests have the same q_len
        and uses_shared_paged_kv_idx
    ):  # wrapper api does not support fp4 output/kv or separate KV page indices yet.
        # test wrapper with trtllm-gen backend
        wrapper_trtllm_gen = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
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
            q_len_per_req=q_len_per_req,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        )
        # v_scale, o_scale in wrapper is emulated by multiplying output by v_scale instead of fused into kernel.
        if v_scale == o_scale == 1.0:
            assert (output_wrapper == output).all()
        else:
            # todo(Yingyi): fix precision issue with this test
            if not (
                q_dtype == "fp8"
                and kv_dtype == "fp8"
                and o_dtype == "fp8"
                and batch_size == 256
                and q_len_per_req == 3
                and page_size == 64
                and num_kv_heads == 4
                and head_grp_size == 5
            ):
                torch.testing.assert_close(
                    output.float(),
                    output_wrapper.float(),
                    rtol=1e-1,
                    atol=1e-1,
                )
            else:
                assert_close_with_mismatch_tolerance(
                    output.float(),
                    output_wrapper.float(),
                    rtol=1e-1,
                    atol=1e-1,
                    max_mismatched_elements=5,
                )
        # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
        # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
        assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()


@pytest.mark.parametrize("backend", ["trtllm-gen", "xqa"])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,q_len_per_req,page_size,num_kv_heads,head_grp_size",
    [
        (4, 1, 16, 2, 1),
        (4, 1, 32, 2, 5),
        (4, 2, 64, 2, 5),
        (4, 3, 32, 2, 5),
        (4, 3, 64, 2, 1),
        (4, 4, 64, 4, 1),
        (4, 5, 64, 4, 8),
        (128, 1, 64, 2, 5),
        (128, 2, 32, 4, 1),
        (128, 3, 16, 4, 8),
        (128, 4, 16, 2, 5),
        (128, 5, 16, 2, 5),
        (256, 1, 64, 4, 8),
        (256, 2, 16, 2, 8),
        (256, 3, 64, 4, 5),
        (256, 4, 32, 2, 8),
        (256, 5, 32, 2, 1),
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
        ("bf16", "fp8", "fp8"),
        ("fp16", "fp8", "fp8"),
        ("fp8", "fp8", "bf16"),
        ("fp8", "fp8", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
        ("fp8", "nvfp4", "fp8"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [True, False, None])
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("max_in_kv_len", [110])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("non_contiguous_query", [False, True])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_decode(
    backend: str,
    kv_layout: str,
    batch_size: int,
    q_len_per_req: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_in_kv_len: int,
    head_dim: int,
    non_contiguous_query: bool,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
):
    # xqa backend does not support non-contiguous query yet
    if backend == "xqa" and non_contiguous_query:
        pytest.skip("xqa backend does not support non-contiguous query")

    # fixme(qsang-nv): failing tests for xqa + head dim 256.
    if backend == "xqa" and head_dim == 256:
        pytest.skip("xqa backend + head dim 256 cases have precision issues")

    # General set of tests for trtllm-gen decode
    _test_trtllm_batch_decode(
        backend,
        kv_layout,
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
        head_dim,
        kv_dtype in ("fp8", "nvfp4"),
        non_contiguous_query=non_contiguous_query,
        skips_softmax=skips_softmax,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )


@pytest.mark.parametrize("kv_layout", ["HND"])  # trtllm-gen only support HND
@pytest.mark.parametrize(
    "batch_size,q_len_per_req,page_size,num_kv_heads,head_grp_size",
    [
        (1, 1, 16, 8, 8),
        (1, 1, 32, 8, 8),
    ],
)
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("fp8", "fp8", "fp8"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [False])
@pytest.mark.parametrize("max_in_kv_len", [4096, 8192])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("device_scale", [True, False])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_decode_bs1(
    kv_layout: str,
    batch_size: int,
    q_len_per_req: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_in_kv_len: int,
    head_dim: int,
    device_scale: bool,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
) -> None:
    # Small number of test cases for batch size 1
    _test_trtllm_batch_decode(
        "trtllm-gen",
        kv_layout,
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
        head_dim,
        device_scale,
        skips_softmax=skips_softmax,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )


@pytest.mark.parametrize("kv_layout", ["HND"])  # trtllm-gen only support HND
@pytest.mark.parametrize(
    "batch_size,q_len_per_req,page_size,num_kv_heads,head_grp_size",
    [
        (4, 1, 16, 2, 1),
        (4, 1, 32, 2, 5),
        (4, 3, 64, 2, 1),
        (4, 4, 64, 4, 1),
        (128, 3, 16, 4, 8),
        (128, 4, 16, 2, 5),
        (256, 4, 32, 2, 8),
        (256, 5, 32, 2, 1),
    ],
)
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("fp8", "fp8", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
        ("fp8", "nvfp4", "fp8"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [False])
@pytest.mark.parametrize("max_in_kv_len", [110])
@pytest.mark.parametrize("head_dim", [256])
@pytest.mark.parametrize("device_scale", [True, False])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_decode_head_dim_256(
    kv_layout: str,
    batch_size: int,
    q_len_per_req: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_in_kv_len: int,
    head_dim: int,
    device_scale: bool,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
):
    # Small number of test cases for head_dim = 256
    _test_trtllm_batch_decode(
        "trtllm-gen",
        kv_layout,
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
        head_dim,
        device_scale,
        skips_softmax=skips_softmax,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )


@pytest.mark.parametrize("kv_layout", ["HND"])  # trtllm-gen only support HND
@pytest.mark.parametrize(
    "batch_size,q_len_per_req,page_size,num_kv_heads,head_grp_size",
    [
        (1, 1, 16, 2, 1),
        (1, 1, 32, 2, 5),
        (1, 3, 64, 2, 1),
        (1, 4, 64, 4, 1),
        (32, 4, 16, 2, 8),
        (32, 8, 16, 2, 8),
        (32, 16, 16, 2, 8),
    ],
)
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp8", "fp8", "fp8"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [False])
@pytest.mark.parametrize("max_in_kv_len", [4096, 8192, 16384, 32768, 65536, 131072])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("device_scale", [True, False])
@pytest.mark.parametrize("skips_softmax", [False])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_decode_long_sequence_length(
    kv_layout: str,
    batch_size: int,
    q_len_per_req: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_in_kv_len: int,
    head_dim: int,
    device_scale: bool,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
) -> None:
    # Small number of test cases for long sequence length
    _test_trtllm_batch_decode(
        "trtllm-gen",
        kv_layout,
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
        head_dim,
        device_scale,
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


@pytest.mark.parametrize("kv_layout", ["HND"])  # trtllm-gen only support HND
@pytest.mark.parametrize(
    "batch_size,q_len_per_req,page_size,num_kv_heads,head_grp_size",
    [
        (4, 1, 16, 2, 1),
        (4, 1, 32, 2, 5),
        (4, 3, 64, 4, 1),
        (128, 3, 16, 4, 8),
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
@pytest.mark.parametrize("max_in_kv_len", [110, 4096, 8192])
@pytest.mark.parametrize("head_dim", [512])
@pytest.mark.parametrize("device_scale", [True, False])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_decode_head_dim_512(
    kv_layout: str,
    batch_size: int,
    q_len_per_req: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_in_kv_len: int,
    head_dim: int,
    device_scale: bool,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
):
    _test_trtllm_batch_decode(
        "trtllm-gen",
        kv_layout,
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
        head_dim,
        device_scale,
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

    # DSL FMHA varlen kernel uses negative pointer offsets, so tensors need
    # front-padding of max_s elements to ensure valid GPU memory before data.
    if backend == "cute-dsl":
        q_full = torch.randn(
            s_qo + cumsum_s_qo,
            num_qo_heads,
            head_dim_qk,
            device=device,
            dtype=torch.bfloat16,
        )
        q = q_full[s_qo:]
        k_full = torch.randn(
            s_kv + cumsum_s_kv,
            num_kv_heads,
            head_dim_qk,
            device=device,
            dtype=torch.bfloat16,
        )
        k_cache = k_full[s_kv:]
        v_full = torch.randn(
            s_kv + cumsum_s_kv,
            num_kv_heads,
            head_dim_vo,
            device=device,
            dtype=torch.bfloat16,
        )
        v_cache = v_full[s_kv:]
    else:
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
    if backend == "cute-dsl":
        output_full = torch.empty(
            s_qo + cumsum_s_qo,
            num_qo_heads,
            head_dim_vo,
            device=device,
            dtype=output_ref.dtype,
        )
        output = output_full[s_qo:]
    else:
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
        backend=backend,
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

    # Generate in float32, quantize to FP8 with front-padding
    q_f32 = (
        torch.randn(
            s_qo + cumsum_s_qo,
            num_qo_heads,
            head_dim_qk,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    )
    k_f32 = (
        torch.randn(
            s_kv + cumsum_s_kv,
            num_kv_heads,
            head_dim_qk,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    )
    v_f32 = (
        torch.randn(
            s_kv + cumsum_s_kv,
            num_kv_heads,
            head_dim_vo,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    )

    q = (q_f32 / scale_q).to(torch.float8_e4m3fn)[s_qo:]
    k_cache = (k_f32 / scale_k).to(torch.float8_e4m3fn)[s_kv:]
    v_cache = (v_f32 / scale_v).to(torch.float8_e4m3fn)[s_kv:]

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

    # Output with front-padding
    output_full = torch.empty(
        s_qo + cumsum_s_qo,
        num_qo_heads,
        head_dim_vo,
        device=device,
        dtype=torch.bfloat16,
    )
    output = output_full[s_qo:]

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
    )


def make_query_non_contiguous(
    q: torch.Tensor, num_qo_heads: int, head_dim: int
) -> torch.Tensor:
    """
    Create a non-contiguous version of the query tensor.
    Create a (N, H, 2*D) tensor and slice the first D dimensions: x[..., :D]
    This produces a non-contiguous view with the same data.
    """
    n, h, d = q.shape
    # Create a larger tensor with 2*D in the last dimension
    large_tensor = torch.zeros(n, h, 2 * d, dtype=q.dtype, device=q.device)
    large_tensor[..., :d] = q
    # Slice to get non-contiguous query (only last dim is contiguous)
    q_non_contiguous = large_tensor[..., :d]
    assert not q_non_contiguous.is_contiguous(), "Query should be non-contiguous"
    return q_non_contiguous


@pytest.mark.parametrize("backend", ["trtllm-gen"])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,max_q_len,page_size,num_kv_heads,head_grp_size,head_dim",
    [
        (4, 1, 16, 2, 1, 128),
        (4, 1, 32, 2, 5, 128),
        (4, 2, 64, 2, 5, 128),
        (4, 3, 32, 2, 5, 128),
        (4, 3, 64, 2, 1, 128),
        (4, 4, 64, 4, 1, 128),
        (4, 5, 64, 4, 8, 128),
        # Iterate over head_dim 128, 256 for these configs to simplify
        *[(bs, 4, 64, 4, 16, hd) for bs in [4, 8, 16, 32] for hd in [128, 256]],
        (128, 1, 64, 2, 5, 128),
        (128, 2, 32, 4, 1, 128),
        (128, 3, 16, 4, 8, 128),
        (128, 4, 16, 2, 5, 128),
        (128, 5, 16, 2, 5, 128),
        (256, 1, 64, 4, 8, 256),
        (256, 2, 16, 2, 8, 256),
        (256, 3, 64, 4, 5, 256),
        (256, 4, 32, 2, 8, 256),
        (256, 16, 32, 2, 8, 256),
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
        ("bf16", "fp8", "fp8"),
        ("fp16", "fp8", "fp8"),
        ("fp8", "fp8", "bf16"),
        ("fp8", "fp8", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [True, False, None])
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("max_in_kv_len", [110])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [False, True])
def test_trtllm_batch_decode_spec(
    backend: str,
    kv_layout: str,
    batch_size: int,
    max_q_len: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_in_kv_len: int,
    head_dim: int,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
) -> None:
    _test_trtllm_batch_decode(
        backend,
        kv_layout,
        batch_size,
        None,  # q_len_per_req
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
        head_dim,
        max_q_len=max_q_len,
        skips_softmax=skips_softmax,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )
