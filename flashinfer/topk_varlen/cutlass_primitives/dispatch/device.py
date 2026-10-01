"""Device facts, queried once per device.  The only place that looks at the hardware.

Everything above this module receives these facts as values; nothing else calls into torch or
the driver for capabilities.  Co-resident cluster counts come from the driver's occupancy
query (per-GPC placement makes ``SMs // size`` wrong: B200 fits 45 clusters of 3, not 49).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

__all__ = ["DeviceFacts", "device_facts", "max_active_clusters", "cluster_capacity"]


@dataclass(frozen=True)
class DeviceFacts:
    name: str
    index: int
    capability: tuple
    sm_count: int
    shared_memory_optin: int  # bytes per block with the opt-in attribute
    supports_clusters: bool  # thread-block clusters and DSMEM (SM90+)
    max_fast_cluster: int  # largest cluster whose DSMEM merge beats the slab merge here (0: none; see _facts)
    cheap_small_clusters: bool  # a small cluster costs little over one CTA (see _facts)
    filter_unroll: (
        int  # 16-byte vectors per thread in flight per filter iteration (see _facts)
    )
    supports_pdl: bool  # programmatic dependent launch (SM90+)
    packed_bf16_compare: bool  # setp.le.bf16x2 (SM90+); fp16x2 is available everywhere
    staggered_count: bool  # register kernels enter the histogram count warp by warp, not in a burst (see _facts)
    bin_cursors: bool  # the streaming resolution emits through a cursor per histogram bin rather than one shared cursor (see _facts)


@functools.cache
def _facts(index: int) -> DeviceFacts:
    props = torch.cuda.get_device_properties(index)
    cc = (props.major, props.minor)
    return DeviceFacts(
        name=props.name,
        index=index,
        capability=cc,
        sm_count=props.multi_processor_count,
        shared_memory_optin=getattr(props, "shared_memory_per_block_optin", 48 * 1024),
        supports_clusters=cc[0] >= 9,
        # Measured on the RTX 5080 (SM120, 84 SMs): clusters of 2 beat the slab (64K b=32: 12.4
        # vs 15.1 us; 256K b=32: 21.5 vs 23.9) but clusters of 6 and 8 lose badly (256K b=8:
        # 16.5 vs 12.7; 1M b=8: 45.6 vs 23.7).  The data-center parts (SM90, SM100, SM107) win
        # with clusters up to 8.
        max_fast_cluster=0 if cc[0] < 9 else (4 if cc[0] == 12 else 8),
        # Whether a 4-CTA cluster over a 64K fp32 row beats one CTA streaming it: measured with
        # the clustered register kernel against the streaming kernel at b=8.  Rubin 6.25 vs 6.42
        # us (k=1024), 5.86 vs 6.18 (k=512); RTX 5080 8.94 vs 11.0, 8.14 vs 10.6.  H100 loses
        # 13.7 vs 12.5, 13.0 vs 12.4; B200 8.28 vs 7.87 at k=1024 and ties at k=512.
        cheap_small_clusters=cc[0] >= 9 and cc not in ((9, 0), (10, 0), (10, 3)),
        # The filter pass is bound by bytes in flight, not issue: eight vectors per thread per
        # iteration (128 B) beat four on H100 (64K b=132 k=2048 27.2 -> 25.0 us; unroll 2:
        # 34.9), B200 (64K b=256 k=1024 17.3 -> 15.3, 1M b=148 k=2048 110.6 -> 105.7, 1M b=8
        # 12.0 -> 11.5) and Rubin (64K b=208 k=512 8.70 -> 8.38, 64K b=256 k=1024 13.2 ->
        # 11.8, 1M b=8 9.39 -> 9.03), and lost on L40S (64K b=142 k=512 25.9 -> 28.1, 64K
        # b=256 k=1024 43.8 -> 50.7) and the RTX 5080 (256K b=64 59.1 -> 63.3); on the A100
        # it loses at 1024 threads (64K b=108 21.5 -> 23.7) and wins at 512 (64K b=256 k=1024
        # 89.5 -> 86.2, ragged k=2048 100 -> 87), which the streaming policy special-cases.
        filter_unroll=8 if cc[0] in (9, 10) else 4,
        supports_pdl=cc[0] >= 9,
        packed_bf16_compare=cc[0] >= 9,
        # The register kernels' histogram count is 16 shared atomics per thread.  Left to the
        # compiler, the first conversion is hoisted above the zeroing barrier, so the barrier
        # becomes a rendezvous after the load latency and all warps issue their atomics in one
        # burst; pinning the count below the barrier (register_row.words_after_barrier) lets
        # each warp start as its own loads land.  Stall-sampled and graph-timed at 16K b=64
        # (burst -> pinned): A100 10.51 -> 10.06 us (16K b=256: 34.0 -> 27.5), L40S 11.57 ->
        # 10.57, RTX 5080 6.36 -> 5.93; the parts with faster shared atomics lose the overlap
        # of conversions with the load tail instead: B200 5.73 -> 5.92, Rubin 4.25 -> 4.42,
        # H100 8.98 -> 9.09.
        staggered_count=cc[0] in (8, 12),
        # One shared cursor for every emitted winner serializes as same-address atomics when
        # two 512-thread CTAs share an SM (B200 64K b=256 k=2048: 16.94 -> 16.40 us with a
        # cursor per bin; RTX 5080 equal).  On the A100 the per-bin form measured 64K b=108
        # k=2048 (one 1024-thread CTA per SM) 35.9 -> 36.4..38.4 us against the single cursor,
        # and 256K b=64 k=1024 79.3 -> 78.5, in a session shared with other jobs; the single
        # cursor reproduces v0.1.27 there exactly, so SM80 keeps it.
        bin_cursors=cc[0] != 8,
    )


def device_facts(device=None) -> DeviceFacts:
    """Facts for a torch device (default: the current one)."""
    index = torch.device(device or torch.cuda.current_device()).index
    if index is None:
        index = torch.cuda.current_device()
    return _facts(index)


@functools.cache
def max_active_clusters(index: int, cluster_size: int) -> int:
    """Clusters of ``cluster_size`` CTAs the device can run at once (driver occupancy query).

    Falls back to seven eighths of ``SMs // size`` if the query is unavailable.
    """
    facts = _facts(index)
    try:
        from cutlass.utils.hardware_info import HardwareInfo

        return int(HardwareInfo(index).get_max_active_clusters(cluster_size))
    except Exception:  # noqa: BLE001 - the query is an optimization, never a requirement
        return (facts.sm_count // cluster_size) * 7 // 8


def cluster_capacity(index: int, cluster_size: int) -> int:
    """Clusters of ``cluster_size`` CTAs that run in one wave, as measured rather than as queried.

    The driver's query is exact for clusters of 8 (a batch one row past it doubles the time:
    B200 15 -> 16 rows 17.1 -> 30.6 us, Rubin 22 -> 24 13.9 -> 27.8, H100 14 -> 16 35.2 ->
    60.4) but pessimistic for small clusters: H100 runs 66 pairs where it says 57 (64 rows 95.7
    us, 70 rows 179), Rubin 66 triples where it says 60 (66 rows 38.8, 69 rows 63.4), B200 45
    triples where it says 42 (45 rows 41.5, 47 rows 72.0).  Small clusters pack up to
    ``SMs // size`` less a few, so the capacity for sizes up to 4 is the larger of the query and
    ``SMs // size - 5``; the 5 is the tightest measured loss (H100 triples: 39 rows 67.4 us,
    40 rows 108.3, where SMs // 3 is 44).  Every other measured boundary sits at or above the
    formula (B200 quads 33-34 of 37 // formula 32; Rubin quads > 50 of 52; Rubin pairs > 104
    of 104; RTX 5080 pairs > 42 of 42, triples > 26 of 28, quads > 19 of 21).
    """
    queried = max_active_clusters(index, cluster_size)
    if cluster_size > 4:
        return queried
    return max(queried, _facts(index).sm_count // cluster_size - 5)
