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


#: The exact longest sequence at or above which SM80 CP is worth selecting,
#: and the parallelism at or below which it is. Fitted over a 32-cell sweep
#: that separated every cell at both boundaries, and re-measured on the frozen
#: pointer entries with the shared invocation context. Both are needed: CP wins
#: on long sequences whose head count leaves the fused kernel's grid too small
#: to fill the device, and loses as soon as either side of that stops holding.
CP_SM80_MIN_MAX_SEQ_LEN = 8192
CP_SM80_MAX_PARALLEL_WORK = 8


def should_use_cp_sm80_host(
    max_seq_len: int | None,
    num_parallel_work: int,
    device_capability: tuple[int, int],
) -> bool:
    """Whether SM80 `use_cp="auto"` should dispatch to CP.

    `max_seq_len` is the *exact* longest sequence, which only the caller knows:
    `cu_seqlens` lives on the device and reading it here would synchronize on
    every call. Given `None` this returns False -- the fused path is the
    fallback, not a guess from `total_seq_len`, which over-states the maximum
    for every multi-sequence batch and would select CP for batches it loses on.

    Restricted to compute capability 8.0. SM86 and SM89 have a different SM
    count, a different L2 and a smaller register file per SM, and the two
    thresholds were fitted on neither; they stay on the fused path until they
    are measured.
    """
    if device_capability != (8, 0):
        return False
    if max_seq_len is None:
        return False
    return (
        max_seq_len >= CP_SM80_MIN_MAX_SEQ_LEN
        and num_parallel_work <= CP_SM80_MAX_PARALLEL_WORK
    )


#: The V32 fused specialization's measured contract. Everything here was
#: measured on `1x8192 gva4x16`, in three independent processes, with the
#: caller handing its bf16 state pool rows in directly. Against the shipped
#: V64 fused path it is 1.066x-1.093x on all four items -- bare and with the
#: shipped norm, at the backend and integration boundaries -- which is why it
#: is selected here. Against Triton it is ahead at the bare backend boundary
#: (1.010x-1.025x), at parity at the normalized backend boundary
#: (0.993x-1.007x), and behind at both integration boundaries
#: (0.975x-0.991x): it narrows the fused path's gap rather than closing it.
#: Nothing outside this contract has been measured, so nothing outside it is
#: selected.
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
    """Whether auto should take the V32 fused specialization.

    Host-side only, and deliberately exact rather than a region: this is one
    validated shape, not a heuristic. An unknown longest sequence means no --
    the value cannot be read off the device without a synchronization, which
    is what the caller passes it to avoid.

    Compute capability 8.0 only. 8.6 and 8.9 have a different shared-memory
    budget per SM and have never been run, and the specialization adds four
    shared buffers, so they are excluded until they are.
    """
    if device_capability != (8, 0):
        return False
    if max_seq_len is None:
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
def chunk_bound(
    seq_idx: cutlass.Int32, total, chunk_size: cutlass.Int32
) -> cutlass.Int32:
    m = seq_idx
    if total < m:
        m = cutlass.Int32(total)
    return cutlass.Int32(m + (total - m) // chunk_size)


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
