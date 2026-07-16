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
CP_HBM_PARALLELISM_THRESHOLD_NUMERATOR = 1
CP_HBM_PARALLELISM_THRESHOLD_DENOMINATOR = 2
CP_GDDR_PARALLELISM_THRESHOLD_NUMERATOR = 1
CP_GDDR_PARALLELISM_THRESHOLD_DENOMINATOR = 3


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


def should_use_cp_host(num_parallel_work: int, num_sms: int, device_name: str) -> bool:
    """Return whether a public wrapper should dispatch to the CP path.

    `num_parallel_work` is the non-CP kernel parallelism, typically batch times
    output/state heads. CP is selected only when that parallelism is strictly
    below the card-specific threshold.
    """
    threshold_num, threshold_den = cp_parallelism_threshold_host(device_name)
    return num_parallel_work * threshold_den < num_sms * threshold_num


def choose_cp_chunk_len_host(
    max_seqlen: int,
    num_heads: int,
    num_sms: int,
    chunk_len_granularity: int = CP_CHUNK_LEN_GRANULARITY,
    device_capability: tuple[int, int] | None = None,
    total_seqlen: int | None = None,
    device_name: str = "",
) -> int:
    """Choose a CP chunk length for the CP workspace kernels.

    The TTFT path is usually one long sequence. For that case, MN precompute
    launches `ceil_div(max_seqlen, chunk_len) * num_heads` CTAs. Pick the
    smallest granularity-aligned chunk length whose CTA count is at most one wave.
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

    # target for one wave of CTAs
    target_chunks = max(1, num_sms // num_heads)
    min_chunk_len = _ceil_div(max_seqlen, target_chunks)
    return _round_up(min_chunk_len, chunk_len_granularity)


@cute.jit
def chunk_bound(
    seq_idx: cutlass.Int32, total: cutlass.Int32, chunk_size: cutlass.Int32
) -> cutlass.Int32:
    m = seq_idx
    if total < m:
        m = total
    return m + (total - m) // chunk_size


@cute.jit
def chunks_for_len(seq_len: cutlass.Int32, chunk_size: cutlass.Int32) -> cutlass.Int32:
    return (seq_len + chunk_size - cutlass.Int32(1)) // chunk_size


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
        seq_start = cutlass.Int32(cu_seqlens[candidate_seq])
        seq_end = cutlass.Int32(cu_seqlens[candidate_seq + cutlass.Int32(1)])
        seq_chunks = chunks_for_len(seq_end - seq_start, chunk_size)
        next_running = running + seq_chunks
        if logical_chunk_idx >= running and logical_chunk_idx < next_running:
            seq_idx = candidate_seq
            chunk_idx_in_seq = logical_chunk_idx - running
        running = next_running
    return seq_idx, chunk_idx_in_seq


@cute.jit
def varlen_chunk_idx(
    seq_idx: cutlass.Int32,
    tok_idx_start: cutlass.Int32,
    chunk_idx_in_seq: cutlass.Int32,
    chunk_size: cutlass.Int32,
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
