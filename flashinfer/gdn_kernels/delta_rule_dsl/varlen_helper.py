import math

import torch
import cutlass
import cutlass.cute as cute


BLK = 64
CP_CHUNK_LEN_GRANULARITY = 512
CP_DEFAULT_SHORT_FIXUP_TO_PREFILL_WORKLOAD_RATIO_NUMERATOR = 1
CP_DEFAULT_SHORT_FIXUP_TO_PREFILL_WORKLOAD_RATIO_DENOMINATOR = 1
CP_SM120_SHORT_FIXUP_TO_PREFILL_WORKLOAD_RATIO_NUMERATOR = 1
CP_SM120_SHORT_FIXUP_TO_PREFILL_WORKLOAD_RATIO_DENOMINATOR = 2
CP_SM120_SHORT_HEURISTIC_MAX_HEADS = 16
CP_SM100_PARALLELISM_THRESHOLD_DENOMINATOR = 4
CP_HBM_PARALLELISM_THRESHOLD_NUMERATOR = 1
CP_HBM_PARALLELISM_THRESHOLD_DENOMINATOR = 2
CP_GDDR_PARALLELISM_THRESHOLD_NUMERATOR = 1
CP_GDDR_PARALLELISM_THRESHOLD_DENOMINATOR = 3


_INTEGER_DTYPES = (
    torch.int32,
    torch.int64,
)


def is_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in _INTEGER_DTYPES


def check_slot_args(
    num_seqs: int,
    num_sab_heads: int,
    D: int,
    device: torch.device,
    use_state_indices: bool,
    needs_checkpointing: bool,
    beta: torch.Tensor | None,
    state_checkpoints: torch.Tensor | None,
    output_state_indices: torch.Tensor | None,
    checkpoint_state_indices: torch.Tensor | None,
    prefix_lens: torch.Tensor | None,
) -> None:
    """Validate the slot-addressing arguments of the non-CP prefill kernels."""
    for name, tensor, shape in (
        ("output_state_indices", output_state_indices, (num_seqs,)),
        ("checkpoint_state_indices", checkpoint_state_indices, None),
        ("prefix_lens", prefix_lens, (num_seqs,)),
    ):
        if tensor is None:
            continue
        if not is_integer_dtype(tensor.dtype):
            raise RuntimeError(f"{name} must have an integer dtype, got {tensor.dtype}")
        if tensor.ndim != 1 or (shape is not None and tuple(tensor.shape) != shape):
            raise RuntimeError(
                f"{name} must be 1D{'' if shape is None else f' of shape {shape}'}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.device != device:
            raise RuntimeError(f"{name} must be on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    if output_state_indices is not None and not use_state_indices:
        raise RuntimeError("output_state_indices requires state_indices")
    if (
        checkpoint_state_indices is not None or prefix_lens is not None
    ) and not needs_checkpointing:
        raise RuntimeError(
            "checkpoint_state_indices and prefix_lens require "
            "checkpoint_every_n_tokens > 0"
        )
    if prefix_lens is not None and beta is None:
        # Padding rows before an unaligned prefix are identity rows built from
        # a zero beta; pass ones for an ungated beta.
        raise RuntimeError("prefix_lens requires beta")
    if checkpoint_state_indices is not None and (
        state_checkpoints.ndim != 4
        or tuple(state_checkpoints.shape[1:]) != (num_sab_heads, D, D)
        or state_checkpoints.stride()[1:] != (D * D, D, 1)
        or state_checkpoints.stride(0) < num_sab_heads * D * D
    ):
        raise RuntimeError(
            "state_checkpoints must be a pool [C, H, D, D] of dense [H, D, D] "
            f"rows with checkpoint_state_indices, got shape "
            f"{tuple(state_checkpoints.shape)} stride {state_checkpoints.stride()}"
        )


def checkpoint_row_align(state_checkpoints: torch.Tensor) -> int:
    """Alignment, in elements, of every row of a checkpoint pool: the largest
    power of two up to 16 bytes that divides both the base address and the
    row stride. Part of the compile key, so the row stores can be vectorized
    for any pool without assuming more than the pool guarantees."""
    size = state_checkpoints.element_size()
    return math.gcd(
        16 // size, state_checkpoints.stride(0), state_checkpoints.data_ptr() // size
    )


def check_checkpoint_key_range(
    num_entries: int, every_n_tokens: int, num_tokens: int
) -> None:
    """The slot checkpoint paths find a checkpoint's entry through the 32-bit
    key ``(first entry - 1) * N + offset`` plus the block end (see
    ``checkpoint_entry_key`` in the kernels); keep it in range."""
    if (num_entries + 2) * every_n_tokens + num_tokens >= 2**31:
        raise RuntimeError(
            "checkpoint indexing exceeds 32 bits: "
            f"({num_entries} entries + 2) * checkpoint_every_n_tokens "
            f"{every_n_tokens} + {num_tokens} tokens >= 2**31"
        )


def integer_dtype_to_cutlass(dtype: torch.dtype) -> type[cutlass.Numeric]:
    try:
        return {
            torch.int32: cutlass.Int32,
            torch.int64: cutlass.Int64,
        }[dtype]
    except KeyError as err:
        raise RuntimeError(f"expected an integer dtype, got {dtype}") from err


def _ceil_div(a, b):
    return (a + b - 1) // b


def _round_up(a, b):
    return _ceil_div(a, b) * b


def chunk_bound_host(num_items: int, total: int, chunk_size: int) -> int:
    if chunk_size <= 0:
        raise RuntimeError(f"chunk_size must be positive, got {chunk_size}")
    m = min(num_items, total)
    return m + (total - m) // chunk_size


def workspace_num_chunks_host(
    cu_seqlens: torch.Tensor, chunk_size: int, total_seqlen: int
) -> int:
    if cu_seqlens.ndim != 1:
        raise RuntimeError(f"cu_seqlens must be 1D, got {tuple(cu_seqlens.shape)}")
    num_seqs = cu_seqlens.numel() - 1
    return chunk_bound_host(num_seqs, total_seqlen, chunk_size)


def max_num_chunks_host(max_seqlen: int, chunk_size: int) -> int:
    return (max_seqlen + chunk_size - 1) // chunk_size


def is_gddr_device_host(device_name: str) -> bool:
    """Best-effort device-class check for host-side CP dispatch.

    Unknown datacenter names default to the HBM threshold. Consumer/workstation
    names default to the GDDR threshold.
    """
    lowered = device_name.lower()
    gddr_markers = ("geforce", "rtx", "workstation")
    return any(marker in lowered for marker in gddr_markers)


def cp_parallelism_threshold_host(device_name: str) -> tuple[int, int]:
    if is_gddr_device_host(device_name):
        return (
            CP_GDDR_PARALLELISM_THRESHOLD_NUMERATOR,
            CP_GDDR_PARALLELISM_THRESHOLD_DENOMINATOR,
        )
    return (
        CP_HBM_PARALLELISM_THRESHOLD_NUMERATOR,
        CP_HBM_PARALLELISM_THRESHOLD_DENOMINATOR,
    )


def cp_short_workload_ratio_host(
    device_capability: tuple[int, int] | None = None,
    num_heads: int | None = None,
) -> tuple[int, int] | None:
    if device_capability is not None and device_capability[0] == 12:
        if num_heads is None or num_heads > CP_SM120_SHORT_HEURISTIC_MAX_HEADS:
            return None
        return (
            CP_SM120_SHORT_FIXUP_TO_PREFILL_WORKLOAD_RATIO_NUMERATOR,
            CP_SM120_SHORT_FIXUP_TO_PREFILL_WORKLOAD_RATIO_DENOMINATOR,
        )
    return (
        CP_DEFAULT_SHORT_FIXUP_TO_PREFILL_WORKLOAD_RATIO_NUMERATOR,
        CP_DEFAULT_SHORT_FIXUP_TO_PREFILL_WORKLOAD_RATIO_DENOMINATOR,
    )


def should_use_cp_host(
    num_parallel_work: int,
    num_sms: int,
    device_name: str,
    device_capability: tuple[int, int] | None = None,
) -> bool:
    """Return whether a public wrapper should dispatch to the CP path.

    `num_parallel_work` is the non-CP kernel parallelism, typically batch times
    output/state heads. CP is selected only when that parallelism is strictly
    below the card-specific threshold.
    """
    if device_capability is not None and device_capability[0] == 10:
        return num_parallel_work * CP_SM100_PARALLELISM_THRESHOLD_DENOMINATOR < num_sms

    threshold_num, threshold_den = cp_parallelism_threshold_host(device_name)
    return num_parallel_work * threshold_den < num_sms * threshold_num


#: SM80 auto-CP: exact longest sequence at least this, parallel work at most this.
CP_SM80_MIN_MAX_SEQ_LEN = 8192
CP_SM80_MAX_PARALLEL_WORK = 16


def should_use_cp_sm80_host(
    max_seq_len: int | None,
    num_parallel_work: int,
    device_capability: tuple[int, int],
) -> bool:
    """Whether SM80 `use_cp="auto"` takes CP; an unknown longest sequence means no."""
    # Measured on compute capability 8.0 only.
    if device_capability != (8, 0) or max_seq_len is None:
        return False
    return (
        max_seq_len >= CP_SM80_MIN_MAX_SEQ_LEN
        and num_parallel_work <= CP_SM80_MAX_PARALLEL_WORK
    )


#: The one shape the V32 fused specialization was measured on.
V32_SM80_MAX_SEQ_LEN = 8192
V32_SM80_NUM_SEQS = 1
V32_SM80_NUM_Q_HEADS = 4
V32_SM80_NUM_V_HEADS = 16


def should_use_v32_sm80_host(
    max_seq_len: int | None,
    num_seqs: int,
    num_q_heads: int,
    num_v_heads: int,
    device_capability: tuple[int, int],
) -> bool:
    """Whether auto takes the V32 fused specialization (exactly its measured shape)."""
    if device_capability != (8, 0) or max_seq_len is None:
        return False
    return (
        max_seq_len == V32_SM80_MAX_SEQ_LEN
        and num_seqs == V32_SM80_NUM_SEQS
        and num_q_heads == V32_SM80_NUM_Q_HEADS
        and num_v_heads == V32_SM80_NUM_V_HEADS
    )


def choose_cp_chunk_len_host(
    max_seqlen: int,
    num_heads: int,
    num_sms: int,
    chunk_len_granularity: int = CP_CHUNK_LEN_GRANULARITY,
    device_capability: tuple[int, int] | None = None,
    total_seqlen: int | None = None,
    num_seqs: int = 1,
    device_name: str = "",
) -> int:
    """Choose a CP chunk length for the CP workspace kernels.

    MN precompute launches one CTA per sequence chunk and state head. Pick the
    smallest granularity-aligned chunk length whose safely bounded CTA count is
    at most one wave.
    """
    assert chunk_len_granularity % 64 == 0
    if total_seqlen is None:
        total_seqlen = max_seqlen

    # Short sequences are dominated by the fixup recurrence and
    # prefill recurrence. Balance S / C * F against C / BLK * P, with tunable
    # F/P measured from fixed-iteration profiles.
    # S / C: Number of chunks per sequence
    # C / BLK: Number of prefill iterations per chunk
    # F: Fixup recurrence cost per iteration
    # P: Prefill recurrence cost per iteration
    # Then S / C * F = C / BLK * P => C = sqrt(S * BLK * F / P)
    ratio = cp_short_workload_ratio_host(device_capability, num_heads)
    if ratio is not None:
        ratio_num, ratio_den = ratio
        threshold_num, threshold_den = cp_parallelism_threshold_host(device_name)

        approx_ctas = _ceil_div(total_seqlen, chunk_len_granularity) * num_heads
        if approx_ctas * threshold_den < num_sms * threshold_num:
            square = _ceil_div(max_seqlen * BLK * ratio_num, ratio_den)
            balanced_chunk_len = math.isqrt(square)
            if balanced_chunk_len * balanced_chunk_len < square:
                balanced_chunk_len += 1
            return max(BLK, _round_up(balanced_chunk_len, BLK))

    # Target one wave of MN CTAs. Account for the known longest sequence, then
    # safely bound the chunks contributed by all remaining uneven sequences.
    target_chunks = max(1, num_sms // num_heads)
    remaining_seqlen = max(0, total_seqlen - max_seqlen)
    remaining_seqs = max(0, num_seqs - 1)

    def chunk_bound_for_len(chunk_len: int) -> int:
        return _ceil_div(max_seqlen, chunk_len) + chunk_bound_host(
            remaining_seqs, remaining_seqlen, chunk_len
        )

    lo = 1
    hi = max(1, _ceil_div(max_seqlen, chunk_len_granularity))
    while lo < hi:
        mid = (lo + hi) // 2
        if chunk_bound_for_len(mid * chunk_len_granularity) <= target_chunks:
            hi = mid
        else:
            lo = mid + 1
    return lo * chunk_len_granularity


@cute.jit
def chunk_bound(seq_idx: cutlass.Int32, total, chunk_size) -> cutlass.Int32:
    m = seq_idx
    if total < m:
        m = cutlass.Int32(total)
    return cutlass.Int32(m + (total - m) // chunk_size)


@cute.jit
def chunks_for_len(seq_len: cutlass.Int32, chunk_size) -> cutlass.Int32:
    chunk_size_value = chunk_size
    if cutlass.const_expr(isinstance(chunk_size, cute.FastDivmodDivisorV2)):
        chunk_size_value = chunk_size.divisor
    return (seq_len + chunk_size_value - cutlass.Int32(1)) // chunk_size


@cute.jit
def logical_chunk_to_work_desc(
    cu_seqlens: cute.Tensor,
    logical_chunk_idx: cutlass.Int32,
    chunk_size: cutlass.Int32,
    num_seqs: cutlass.Int32,
):
    seq_idx = cutlass.Int32(0)
    chunk_idx_in_seq = logical_chunk_idx
    running = cutlass.Int32(0)
    for candidate_seq in cutlass.range(num_seqs, unroll=1):
        seq_start = cu_seqlens[candidate_seq]
        seq_len = cutlass.Int32(
            cu_seqlens[candidate_seq + cutlass.Int32(1)] - seq_start
        )
        seq_chunks = chunks_for_len(seq_len, chunk_size)
        next_running = running + seq_chunks
        if logical_chunk_idx >= running and logical_chunk_idx < next_running:
            seq_idx = candidate_seq
            chunk_idx_in_seq = logical_chunk_idx - running
        running = next_running
    return seq_idx, chunk_idx_in_seq


@cute.jit
def varlen_chunk_idx(
    seq_idx: cutlass.Int32,
    tok_idx_start,
    chunk_idx_in_seq: cutlass.Int32,
    chunk_size,
) -> cutlass.Int32:
    return chunk_bound(seq_idx, tok_idx_start, chunk_size) + chunk_idx_in_seq


@cute.jit
def varlen_chunk_valid_len(
    seq_len: cutlass.Int32,
    chunk_idx_in_seq: cutlass.Int32,
    chunk_size: cutlass.Int32,
) -> cutlass.Int32:
    remaining = seq_len - chunk_idx_in_seq * chunk_size
    if remaining > chunk_size:
        remaining = chunk_size
    return remaining
