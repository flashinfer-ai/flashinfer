# Adapted from Block-Sparse-Attention/bsa_attn_interface.py for FlashInfer integration.
#
# This module vendors the JIT-path dispatch helpers needed by the SM100/SM110
# blk64 CuTe-DSL forward kernel (int32 bound checks, split-KV workspace sizing,
# scheduler heuristics, and the KV-bucketed combine kernel wrapper). The
# upstream AOT (ahead-of-time precompiled artifact) branches are intentionally
# omitted -- FlashInfer's blk64 integration is JIT-only, matching the existing
# blk128 CuTe-DSL integration (see bsa_attn_sm100_blk128.py).

from functools import lru_cache
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from ..bsa_utils.cache_utils import get_jit_cache
from ..bsa_utils.testing import is_fake_mode
from ..bsa_utils.cute_tensor_utils import to_cute_tensor
from .bsa_fwd_combine import BlockSparseAttnForwardCombine

# Baseline KV-split combine kernel geometry (shared across SM90/SM100/SM110;
# upstream keeps a single configuration to simplify maintenance).
_COMBINE_TILE_M = 16
_COMBINE_K_BLOCK_SIZE = 64
_COMBINE_NUM_THREADS = 128
_COMBINE_STAGES = 4

_SM100_BLK64_INT32_MAX = torch.iinfo(torch.int32).max


@lru_cache(maxsize=None)
def _get_device_arch() -> int:
    """Current CUDA device's compute-capability arch, e.g. 100 for SM100.

    Duplicated (not imported) from bsa_attn_sm100_blk128.py intentionally: the
    blk128 module has a hard dependency on the separate `quack-kernels`
    package, and blk64 must remain importable even when that package (or the
    blk128 backend) is unavailable.
    """
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}


def _sm100_blk64_require_int32(name: str, value: int) -> int:
    value = int(value)
    if value < 0 or value > _SM100_BLK64_INT32_MAX:
        raise ValueError(
            f"SM100 blk64 {name}={value} must fit in int32 "
            f"(<= {_SM100_BLK64_INT32_MAX})"
        )
    return value


def _sm100_blk64_round_up_to_block(name: str, value: int, block: int = 64) -> int:
    value = _sm100_blk64_require_int32(name, value)
    rounded = ((value + block - 1) // block) * block
    return _sm100_blk64_require_int32(f"{name}_rounded", rounded)


def validate_sm100_blk64_int32_bounds(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor],
    q2k_block_nums: Optional[torch.Tensor],
) -> None:
    """Guard values that the SM100 blk64 path stores or casts as int32."""
    batch, num_heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[2]
    num_m_blocks = (seqlen_q + 63) // 64

    for tensor_name, tensor in (("q", q), ("k", k), ("v", v)):
        _sm100_blk64_require_int32(f"{tensor_name}.shape[0]", tensor.shape[0])
        _sm100_blk64_require_int32(f"{tensor_name}.shape[1]", tensor.shape[1])
        _sm100_blk64_require_int32(f"{tensor_name}.shape[2]", tensor.shape[2])
        _sm100_blk64_require_int32(f"{tensor_name}.shape[3]", tensor.shape[3])
        _sm100_blk64_require_int32(f"{tensor_name}.stride(1)", tensor.stride(1))
        _sm100_blk64_require_int32(f"{tensor_name}.stride(2)", tensor.stride(2))
        _sm100_blk64_require_int32(f"{tensor_name}.stride(3)", tensor.stride(3))

    _sm100_blk64_round_up_to_block("seqlen_k", seqlen_k)
    _sm100_blk64_require_int32("num_m_blocks", num_m_blocks)
    _sm100_blk64_require_int32("block_indices_stride", q2k_block_index.shape[-1])
    _sm100_blk64_require_int32("total_q_tiles", batch * num_heads * num_m_blocks)
    _sm100_blk64_require_int32("block_sparse_num", block_sparse_num)

    if block_sizes is not None and block_sizes.numel() > 0:
        _sm100_blk64_require_int32("block_sizes.numel", block_sizes.numel())
    if q2k_block_nums is not None and q2k_block_nums.numel() > 0:
        _sm100_blk64_require_int32("q2k_block_nums.numel", q2k_block_nums.numel())


def sm100_blk64_requires_int64_kv_strides(
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    """Return whether the rank-6 Int32 TMA coordinate basis is unsafe."""
    coord_stride_limit = 1 << 27
    for tensor in (k, v):
        batch, heads, seqlen_k, _ = tensor.shape
        stride_b, stride_h, stride_s, stride_d = map(int, tensor.stride())
        rank6_shape = (64, 64, 2, heads, (seqlen_k + 63) // 64, batch)
        rank6_stride = (
            stride_s,
            stride_d,
            64 * stride_d,
            stride_h,
            64 * stride_s,
            stride_b,
        )
        if any(
            stride < 0 or stride > _SM100_BLK64_INT32_MAX
            for stride in rank6_stride
        ):
            return True
        block_stride = rank6_stride[4]
        batch_stride = rank6_stride[5]
        if rank6_shape[4] > 1 and block_stride >= coord_stride_limit:
            return True
        if rank6_shape[5] > 1 and batch_stride >= coord_stride_limit:
            return True
    return False


def _tensor_dynamic_layout_compile_key(t: torch.Tensor, leading_dim: int = -1):
    """Match the static rank/dtype/broadcast parts of mark_layout_dynamic()."""
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return (
        t.dtype,
        t.ndim,
        int(leading_dim),
        int(t.stride(leading_dim)),
        tuple(s == 0 for s in t.stride()),
    )


def dynamic_tensors_compile_key(
    namespace: str,
    config: tuple,
    tensors: Tuple[Optional[torch.Tensor], ...],
    leading_dims: Optional[Tuple[int, ...]] = None,
):
    """Build a compile-cache key from static config plus per-tensor layout facts.

    Unlike a purely-static key, this also captures each tensor's dtype/rank/
    leading-dim stride/broadcast pattern -- the same facts CuTe-DSL's
    ``mark_layout_dynamic()`` bakes into a compiled specialization. Omitting
    this would risk cache hits against a specialization compiled for an
    incompatible tensor layout (e.g. a broadcast scale tensor), which is a
    correctness bug, not just a perf one.
    """
    if leading_dims is None:
        leading_dims = tuple(-1 for _ in tensors)
    assert len(tensors) == len(leading_dims)
    return (
        namespace,
        *config,
        *(
            _tensor_dynamic_layout_compile_key(tensor, leading_dim)
            if tensor is not None
            else None
            for tensor, leading_dim in zip(tensors, leading_dims)
        ),
    )


def ceil_div_int(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def ceil_log2_int(x: int) -> int:
    x = int(x)
    assert x >= 1
    return (x - 1).bit_length()


def sm100_blk64_kv_splits_from_count(kv_blocks: int, max_kv_splits: int = 16) -> int:
    """Choose the long-Q split count from a uniform sparse-block count."""
    kv_blocks = int(kv_blocks)
    if kv_blocks >= 900:
        splits = 8
    elif kv_blocks >= 450:
        splits = 4
    elif kv_blocks >= 256:
        splits = 2
    else:
        splits = 1
    return max(1, min(int(splits), int(max_kv_splits), kv_blocks))


def sm100_blk64_auto_kv_splits(
    q: torch.Tensor,
    q2k_block_index: torch.Tensor,
    fixed_block_sparse_num: int,
    max_kv_splits: int = 16,
) -> int:
    """Choose KV splits for the SM100 blk64 KV-bucketed target cases."""
    if is_fake_mode() or not q.is_cuda:
        return 1

    kv_blocks = int(fixed_block_sparse_num)
    if kv_blocks <= 0:
        kv_blocks = int(q2k_block_index.shape[-1])
    if kv_blocks <= 1:
        return 1

    return sm100_blk64_kv_splits_from_count(kv_blocks, max_kv_splits)


def build_sm100_blk64_kv_split_offsets(
    q2k_block_nums: Optional[torch.Tensor],
    uniform_block_sparse_num: int,
    batch_size: int,
    num_heads: int,
    num_q_blocks: int,
    kv_splits: int,
    device: torch.device,
) -> torch.Tensor:
    """Build 8-block-aligned split offsets for the blk64 forward kernels."""
    assert 1 <= kv_splits <= 256, "kv_splits must be in [1, 256]"
    if q2k_block_nums is not None and q2k_block_nums.numel() > 0:
        valid_kv = q2k_block_nums.to(torch.int32).contiguous().clamp_min(0)
    else:
        valid_kv = torch.full(
            (batch_size, num_heads, num_q_blocks),
            int(uniform_block_sparse_num),
            dtype=torch.int32,
            device=device,
        )

    split_ids = torch.arange(kv_splits + 1, dtype=torch.int64, device=device)
    valid_kv_i64 = valid_kv.to(torch.int64)
    avg_blocks = valid_kv_i64 // kv_splits
    aligned_base = (avg_blocks // 8) * 8
    use_even_split = aligned_base == 0
    remainder = valid_kv_i64 - aligned_base * kv_splits

    even_offsets = (
        valid_kv_i64[..., None] * split_ids + kv_splits - 1
    ) // kv_splits
    aligned_offsets = aligned_base[..., None] * split_ids + torch.minimum(
        remainder[..., None], split_ids * 8
    )
    aligned_offsets = torch.minimum(aligned_offsets, valid_kv_i64[..., None])
    return (
        torch.where(use_even_split[..., None], even_offsets, aligned_offsets)
        .to(torch.int32)
        .contiguous()
    )


def _blk64_split_workspace_bytes(q: torch.Tensor, value_dim: int, kv_splits: int) -> int:
    """Estimate live split-KV partial, combine-output, and offset storage."""
    batch, num_heads, seqlen_q, _ = q.shape
    num_q_blocks = ceil_div_int(seqlen_q, 64)
    rows = batch * num_heads * seqlen_q
    partial_bytes = kv_splits * rows * (value_dim + 1) * 4
    final_bytes = rows * (value_dim * q.element_size() + 4)
    offset_bytes = batch * num_heads * num_q_blocks * (kv_splits + 1) * 4
    return int(partial_bytes + final_bytes + offset_bytes)


def resolve_sm100_blk64_split_workspace(
    q: torch.Tensor,
    value_dim: int,
    kv_splits: int,
    allow_fallback: bool,
) -> int:
    """Fit split-KV workspace to currently available CUDA allocator capacity."""
    kv_splits = int(kv_splits)
    if kv_splits <= 1 or is_fake_mode() or not q.is_cuda:
        return kv_splits

    free_bytes, total_bytes = torch.cuda.mem_get_info(q.device)
    reclaimable_bytes = max(
        0,
        torch.cuda.memory_reserved(q.device) - torch.cuda.memory_allocated(q.device),
    )
    reserve_bytes = max(512 << 20, int(total_bytes * 0.05))
    budget_bytes = max(0, free_bytes + reclaimable_bytes - reserve_bytes)

    candidate = kv_splits
    while candidate > 1:
        required_bytes = _blk64_split_workspace_bytes(q, value_dim, candidate)
        if required_bytes <= budget_bytes:
            return candidate
        if not allow_fallback:
            required_gib = required_bytes / (1 << 30)
            budget_gib = budget_bytes / (1 << 30)
            raise RuntimeError(
                f"blk64 split-KV kv_splits={kv_splits} requires about "
                f"{required_gib:.2f} GiB of live workspace, but only "
                f"{budget_gib:.2f} GiB is available after the safety reserve; "
                "lower kv_splits"
            )
        candidate //= 2
    return 1


def choose_sm100_blk64_use_clc(
    q: torch.Tensor,
    block_sparse_num: int,
    q2k_block_nums: Optional[torch.Tensor] = None,
    layout: str = "bhsd",
) -> bool:
    """Select the measured-fastest blk64 scheduler for the common wrapper path.

    Callers can pass ``use_clc=True`` or ``False`` to force a path, or leave it
    as ``None`` to use this shape-based policy.
    """
    if q2k_block_nums is not None and q2k_block_nums.numel() > 0:
        return True

    if layout == "bshd":
        batch, seqlen_q, h, _ = q.shape
    else:
        assert layout == "bhsd", f"layout must be 'bhsd' or 'bshd', got {layout!r}"
        batch, h, seqlen_q, _ = q.shape

    num_m_blocks = (seqlen_q + 63) // 64
    large_long_topk = num_m_blocks >= 8192 and block_sparse_num >= 512
    if large_long_topk:
        return True

    if h == 1:
        return False

    total_tiles = batch * h * num_m_blocks
    enough_tiles = num_m_blocks >= 128 and total_tiles >= 512
    light_tile = block_sparse_num <= (64 if h == 2 else 128)
    return enough_tiles and light_tile


def make_sm100_blk64_cute_args(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
    softmax_scale: float,
    q2k_block_index: torch.Tensor,
    block_sizes: Optional[torch.Tensor],
    block_sparse_num: int,
    q2k_block_nums: Optional[torch.Tensor],
    split_offsets: Optional[torch.Tensor],
    current_stream,
    *,
    enable_tvm_ffi: bool,
) -> tuple:
    """Build the SM100 blk64 CuTe ABI without changing optional arguments."""
    return (
        to_cute_tensor(q, enable_tvm_ffi=enable_tvm_ffi),
        to_cute_tensor(k, enable_tvm_ffi=enable_tvm_ffi),
        to_cute_tensor(v, enable_tvm_ffi=enable_tvm_ffi),
        to_cute_tensor(out, enable_tvm_ffi=enable_tvm_ffi),
        to_cute_tensor(lse, assumed_align=4, enable_tvm_ffi=enable_tvm_ffi),
        (
            to_cute_tensor(q_scale, assumed_align=4, enable_tvm_ffi=enable_tvm_ffi)
            if q_scale is not None
            else None
        ),
        (
            to_cute_tensor(k_scale, assumed_align=4, enable_tvm_ffi=enable_tvm_ffi)
            if k_scale is not None
            else None
        ),
        (
            to_cute_tensor(v_scale, assumed_align=4, enable_tvm_ffi=enable_tvm_ffi)
            if v_scale is not None
            else None
        ),
        softmax_scale,
        to_cute_tensor(q2k_block_index, enable_tvm_ffi=enable_tvm_ffi),
        (
            to_cute_tensor(block_sizes, enable_tvm_ffi=enable_tvm_ffi)
            if block_sizes is not None
            else None
        ),
        block_sparse_num,
        (
            to_cute_tensor(q2k_block_nums, enable_tvm_ffi=enable_tvm_ffi)
            if q2k_block_nums is not None
            else None
        ),
        (
            to_cute_tensor(split_offsets, enable_tvm_ffi=enable_tvm_ffi)
            if split_offsets is not None
            else None
        ),
        current_stream,
    )


def _to_cute_tensor_dynamic_compact_shape(
    t: torch.Tensor,
    mode,
    assumed_align: int = 16,
    leading_dim: int = -1,
    divisibility: int = 1,
    stride_order=None,
    enable_tvm_ffi: bool = True,
) -> cute.Tensor:
    tensor = to_cute_tensor(
        t,
        assumed_align=assumed_align,
        leading_dim=leading_dim,
        enable_tvm_ffi=enable_tvm_ffi,
    )
    if isinstance(mode, int):
        mode = (mode,)
    stride_order = t.dim_order() if stride_order is None else stride_order
    for mode_i in mode:
        tensor = tensor.mark_compact_shape_dynamic(
            mode=mode_i,
            stride_order=stride_order,
            divisibility=divisibility,
        )
    return tensor


def _make_blk64_combine_cute_args(
    o_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out_bshd: torch.Tensor,
    lse_bsh: torch.Tensor,
    current_stream,
    *,
    enable_tvm_ffi: bool,
) -> tuple:
    return (
        _to_cute_tensor_dynamic_compact_shape(
            o_partial,
            mode=(0, 1, 2, 3),
            stride_order=(1, 0, 3, 2, 4),
            enable_tvm_ffi=enable_tvm_ffi,
        ),
        _to_cute_tensor_dynamic_compact_shape(
            lse_partial,
            mode=(0, 1, 2, 3),
            assumed_align=4,
            leading_dim=2,
            stride_order=(1, 0, 3, 2),
            enable_tvm_ffi=enable_tvm_ffi,
        ),
        _to_cute_tensor_dynamic_compact_shape(
            out_bshd,
            mode=(0, 1, 2),
            stride_order=(0, 1, 2, 3),
            enable_tvm_ffi=enable_tvm_ffi,
        ),
        _to_cute_tensor_dynamic_compact_shape(
            lse_bsh,
            mode=(0, 1, 2),
            assumed_align=4,
            stride_order=(0, 1, 2),
            enable_tvm_ffi=enable_tvm_ffi,
        ),
        None,
        None,
        None,
        None,
        None,
        current_stream,
    )


def _bsa_fwd_blk64_kv_bucketed_combine_compile_key(
    arch: int,
    dtype,
    head_dim: int,
    combine_tile_m: int,
    combine_k_block_size: int,
    log_max_splits: int,
    combine_num_threads: int,
    combine_stages: int,
):
    return (
        int(arch),
        dtype,
        cutlass.Float32,
        int(head_dim),
        int(combine_tile_m),
        int(combine_k_block_size),
        int(log_max_splits),
        int(combine_num_threads),
        int(combine_stages),
        "bshd_nonvarlen_seqlen_dynamic_env_stream_v1",
    )


_combine_compile_cache = get_jit_cache("bsa_fwd_blk64_kv_bucket_combine")


def combine_blk64_kv_bucketed_partials(
    q: torch.Tensor,
    o_partial_phys: torch.Tensor,
    lse_partial_phys: torch.Tensor,
    kv_splits: int,
    device_arch: int,
    output_dtype: Optional[torch.dtype] = None,
    use_fast_16_split: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine KV-bucketed partial outputs using the shared CuTeDSL combine kernel.

    JIT-only: always compiles (or reuses a cached compile) of
    ``BlockSparseAttnForwardCombine``; there is no AOT precompiled-artifact
    fast path here (see module docstring).
    """
    kv_splits = int(kv_splits)
    assert 1 <= kv_splits <= 256, "kv_splits must be in [1, 256]"
    assert not use_fast_16_split or kv_splits == 16, (
        "the 256-thread combine specialization requires exactly 16 splits"
    )

    batch, num_heads, seqlen_q, _ = q.shape
    head_dim = o_partial_phys.shape[-1]
    if o_partial_phys.dtype != torch.float32:
        raise TypeError("KV-bucketed blk64 fwd requires fp32 O partial")

    split_heads = kv_splits * num_heads
    o_partial = o_partial_phys.as_strided(
        (kv_splits, batch, seqlen_q, num_heads, head_dim),
        (
            num_heads * seqlen_q * head_dim,
            seqlen_q * split_heads * head_dim,
            head_dim,
            seqlen_q * head_dim,
            1,
        ),
    )
    lse_partial = lse_partial_phys.as_strided(
        (kv_splits, batch, seqlen_q, num_heads),
        (
            num_heads * seqlen_q,
            seqlen_q * split_heads,
            1,
            seqlen_q,
        ),
    )
    if output_dtype is None:
        output_dtype = q.dtype
    out_bshd = torch.empty(
        (batch, seqlen_q, num_heads, head_dim), dtype=output_dtype, device=q.device
    )
    lse_bsh = torch.empty(
        (batch, seqlen_q, num_heads), dtype=torch.float32, device=q.device
    )
    dtype = torch2cute_dtype_map[output_dtype]
    log_max_splits = ceil_log2_int(kv_splits)
    combine_num_threads = 256 if use_fast_16_split else _COMBINE_NUM_THREADS

    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    )

    compile_key = _bsa_fwd_blk64_kv_bucketed_combine_compile_key(
        device_arch,
        dtype,
        head_dim,
        _COMBINE_TILE_M,
        _COMBINE_K_BLOCK_SIZE,
        log_max_splits,
        combine_num_threads,
        _COMBINE_STAGES,
    )
    if compile_key not in _combine_compile_cache:
        combine_kernel = BlockSparseAttnForwardCombine(
            dtype=dtype,
            head_dim=head_dim,
            tile_m=_COMBINE_TILE_M,
            k_block_size=_COMBINE_K_BLOCK_SIZE,
            log_max_splits=log_max_splits,
            num_threads=combine_num_threads,
            stages=_COMBINE_STAGES,
        )
        jit_args = _make_blk64_combine_cute_args(
            o_partial, lse_partial, out_bshd, lse_bsh, current_stream, enable_tvm_ffi=True
        )
        _combine_compile_cache[compile_key] = cute.compile(
            combine_kernel, *jit_args, options="--enable-tvm-ffi"
        )

    if not is_fake_mode():
        _combine_compile_cache[compile_key](
            o_partial, lse_partial, out_bshd, lse_bsh, None, None, None, None, None, current_stream
        )

    # Combine writes its native BSHD/BSH layout. Return BHSD/BHS views without D2D.
    out = out_bshd.transpose(1, 2)
    lse = lse_bsh.transpose(1, 2)
    return out, lse


def workaround_cutlass_hash_import_bug():
    """Avoid optional generated dialect imports that are broken in this environment."""
    import importlib
    import sys
    import types

    for suffix in ("arith", "dialect_proxy", "gpu", "lru_cache_ir", "op"):
        canonical = f"cutlass._mlir_helpers.{suffix}"
        alias = f"cutlass.base_dsl._mlir_helpers.{suffix}"
        try:
            sys.modules.setdefault(alias, importlib.import_module(canonical))
        except Exception:
            pass

    for name in (
        "cutlass._mlir.dialects._iket_ops_gen",
        "cutlass._mlir.dialects._bitfield_ops_gen",
        "cutlass._mlir.dialects._pyir_ops_gen",
        "cutlass._mlir.dialects._ub_ops_gen",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))


def validate_sm100_blk64_fp8_sage(
    batch_size: int,
    num_head: int,
    q2k_block_nums: Optional[torch.Tensor],
    block_sizes: Optional[torch.Tensor],
    seqlen_q: int,
    seqlen_k: int,
    head_dim_v: int,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Enforce the Sage FP8 v1 contract's hard limits (fixed top-k, full blocks).

    These limits come from the upstream kernel itself, not from this
    integration -- surfaced early (at plan()/dispatch time) with a clear
    message so they are not mistaken for a migration regression.
    """
    if batch_size != 1:
        raise ValueError(
            "Sage FP8 blk64 requires batch_size == 1 (upstream kernel limit)"
        )
    if num_head not in (4, 8):
        raise ValueError(
            f"Sage FP8 blk64 requires num_head in (4, 8), got {num_head} "
            "(upstream kernel limit)"
        )
    if q2k_block_nums is not None and q2k_block_nums.numel() > 0:
        raise ValueError(
            "Sage FP8 blk64 requires a uniform top-k (q2k_block_nums must be "
            "None); variable per-row KV counts are not supported yet"
        )
    if block_sizes is not None and block_sizes.numel() > 0:
        raise ValueError(
            "Sage FP8 blk64 requires full 64-token KV blocks (block_sizes must "
            "be None); partial/padded KV blocks are not supported yet"
        )
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32 or v_scale.dtype != torch.float32:
        raise ValueError("Sage FP8 q_scale/k_scale/v_scale must be float32")
    expected_q_scale_shape = (batch_size, num_head, seqlen_q)
    if tuple(q_scale.shape) != expected_q_scale_shape:
        raise ValueError(
            f"q_scale must be [B,H,Sq]={expected_q_scale_shape}, got {tuple(q_scale.shape)}"
        )
    expected_k_scale_shape = (batch_size, num_head, (seqlen_k + 15) // 16)
    if tuple(k_scale.shape) != expected_k_scale_shape:
        raise ValueError(
            f"k_scale must be [B,H,ceil(Sk/16)]={expected_k_scale_shape}, got {tuple(k_scale.shape)}"
        )
    expected_v_scale_shape = (num_head, head_dim_v)
    if tuple(v_scale.shape) != expected_v_scale_shape:
        raise ValueError(
            f"v_scale must be [H,Dv]={expected_v_scale_shape}, got {tuple(v_scale.shape)}"
        )
