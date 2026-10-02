"""Default streaming configuration for a device and a problem: the routing that reads the facts.

Rules, each with its measurement:

* Stage 16K candidates only where the aim plus three sigma of the survivor count would cross
  the 8K stage's cap (``_needs_big_stage``: 1M fp32 rows and up at one CTA per row).  At 1M
  fp32 elements the tight aim is 6144 and its spread reaches 10K survivors on some rows; with
  an 8K stage one row in a batch of 148 overflowed, re-walked (75 us) and set the batch time:
  174 us against 105 us with the larger stage.  Everywhere else the larger carveout only costs
  L1 for the walk (A100 256K b=64 75.2 -> 70.3 us at 8K; B200: 1M b=8 slab 16 13.47 -> 13.02,
  1M b=12 cluster 8 15.16 -> 14.95, 256K b=8 slab 16 9.77 -> 9.53, 256K b=64 cluster 2 22.3
  -> 21.8).
* The larger stage needs about 150 KB of shared memory; parts with a 99 KB budget (L40S,
  RTX 5080) keep 8K and sample four adjacent vectors per thread instead (one 64-byte fetch, a
  16K sample): the survivor count's spread around the aim halves, and with the balanced aim
  (``phases/aim.py``) both stage tails sit beyond five sigma, so the 8K stage stops overflowing
  at 1M rows.  Measured: L40S 1M b=142 k=2048 had one row in ten re-walking (median 1006 us
  against an 847 us phase sum); with two vectors one outlier row of 142 still overflowed at
  3.5 sigma; RTX 5080 1M b=84 k=2048 1297 -> 1195 us.  The extra bin increments cost about
  0.8 us per vector per row, which only long rows can afford.
* Programmatic dependent launch and the packed bf16 compare on SM90 and newer; the packed
  fp16 compare everywhere.
* Row splits, cluster merge (SM90+): when one CTA per row leaves SMs idle, the row is shared
  by the largest cluster shape whose clusters all fit one wave (``cluster_capacity``: the
  driver's occupancy query, corrected for small shapes where it under-reports; never plain
  ``SMs // size``: B200 fits 45 clusters of 3, not 49) and whose slices keep at least 4096
  elements.  Shapes above 4 only pay off from 4 MB rows (the wider merge outweighs the shorter
  slice below that; measured on Rubin 256K b=32: 4 -> 6 went 8.6 -> 8.9 us), and below 1 MB
  shape 3 already loses (64K b=64: 7.0 -> 7.6 us), so the allowed set narrows with the row
  size.  When no shape fits one wave, the largest shape whose CTAs fit the SM count is taken
  anyway: its second partial wave costs at most twice a wave, which is never worse than one
  CTA per row (H100 1M b=64: cluster 2 95.7 us, one CTA 155; Rubin 1M b=64: cluster 3 38.0,
  cluster 2 43.8).
* Row splits, slab merge, small batches of long rows on any part: 16 CTAs per row from 1 MB
  and 32 from 4 MB when the grid fits one wave.  The cluster merge caps at 8 CTAs and the slab
  slices are shorter; measured on B200 1M b=8: cluster 8 15.7 us, slab 16 13.8; b=1: cluster 8
  15.6, slab 32 12.2; 256K b=8: cluster 6 10.3, slab 16 10.1.  Between 16-way slabs and 8-way
  slabs sits the 8-CTA cluster where it fits one wave (1M b=12: B200 16.5 vs slab 8 18.3, H100
  29.9 vs 33.4; Rubin 1M b=16: 13.0 vs 15.3).  The slab form cannot re-walk an undershoot, so
  it takes the wide aim.
* Row splits, slab merge, parts without clusters: the largest of 8/4/2 CTAs per row that
  keeps the grid within one wave and slices of at least 4096 elements.  Cluster shapes are
  capped by the device fact ``max_fast_cluster`` (4 on consumer Blackwell, where clusters of 6
  and 8 measured far slower than the slab; 8 on the data-center parts).  Two waves of half
  rows cost more than one wave of whole rows (A100 64K b=64: 2 CTAs per row 38.5 us against
  one CTA 23.3).  Exception from FlashInfer's A100 measurements: rows of 4 MB or more in
  batches up to 64 split four ways even into a second wave, because one CTA streams a 4 MB row
  at ~12 GB/s against ~20 GB/s for shorter slices (1M fp32 b=64: 364 -> 249 us).
* Wide batches (more rows than SMs) run 512-thread CTAs two per SM when the row is at most
  256 KB (512 KB up to two rows per SM).  A 512-thread CTA solves a row 10-60% slower alone,
  so this only pays when the batch runs in waves: measured on B200 at b=256, 64K k=1024
  21.8 -> 18.1 us, 128K 38.2 -> 34.0, ragged 64K k=2048 24.4 -> 20.6.  The stage is 4K for
  k <= 1024 and 8K for k=2048: at k=2048 the tight aim (3072) sits at a 4K stage's cap and
  the survivor spread (up to 3.9K on a 256-row batch) overflowed one row into a re-walk that
  set the batch to 29.5 us.  Above the byte bound the longer per-CTA pass outweighs the
  second wave saved.
"""

from __future__ import annotations

import torch

from ...dispatch.device import DeviceFacts, cluster_capacity

from ..kernels.streaming import EXACT_ONLY, StreamingConfig

__all__ = ["streaming_config_for", "splits_for"]

BIG_ROW_BYTES = 1 << 20
MIN_SLICE = 4096


def splits_for(
    facts: DeviceFacts, rows: int, n: int, elem_bytes: int
) -> tuple[int, str]:
    """(CTAs per row, merge medium) for a batch of ``rows`` rows of ``n`` elements."""
    if n < 16384 or rows >= facts.sm_count:
        return 1, "cluster"
    row_bytes = n * elem_bytes
    # long rows in small batches: wide slab splits beat the widest cluster (B200 1M b=8: slab
    # 16 13.8 us, cluster 8 15.7; RTX 5080 1M b=8: slab 8 23.7, cluster 8 45.6)
    if row_bytes >= BIG_ROW_BYTES:
        for s in (32, 16) if row_bytes >= 4 * BIG_ROW_BYTES else (16,):
            if n // s >= MIN_SLICE and rows * s <= facts.sm_count:
                return s, "slab"
        # below 16 CTAs per row the widest cluster beats the 8-way slab (1M b=12: B200 16.5 vs
        # 18.3 us, H100 29.9 vs 33.4; Rubin 1M b=16: 13.0 vs 15.3), and the slab beats a cluster
        # that would need a second wave (B200 1M b=16: slab 8 18.3, cluster 8 30.6)
        if facts.max_fast_cluster >= 8 and rows <= cluster_capacity(facts.index, 8):
            return 8, "cluster"
        if n // 8 >= MIN_SLICE and rows * 8 <= facts.sm_count:
            return 8, "slab"
    if facts.max_fast_cluster >= 2:
        shapes: tuple
        if row_bytes >= 4 * BIG_ROW_BYTES:
            shapes = (8, 6, 4, 3, 2)
        elif row_bytes >= BIG_ROW_BYTES:
            shapes = (6, 4, 3, 2)
        else:
            shapes = (4, 2)
        for s in shapes:
            if (
                s <= facts.max_fast_cluster
                and n // s >= MIN_SLICE
                and rows <= cluster_capacity(facts.index, s)
            ):
                return s, "cluster"
        # no shape fits one wave: a second wave of a few small clusters still beats one CTA per
        # row on half the SMs (H100 1M b=64: cluster 2 95.7 us, one CTA 155; the split rows
        # take at most twice a wave, which is never worse than the unsplit row)
        for s in shapes:
            if (
                s <= facts.max_fast_cluster
                and n // s >= MIN_SLICE
                and rows * s <= facts.sm_count
            ):
                return s, "cluster"
        return 1, "cluster"
    for s in (8, 4, 2):
        if n // s >= MIN_SLICE and rows * s <= facts.sm_count:
            return s, "slab"
    if row_bytes >= 4 * BIG_ROW_BYTES and rows <= 64:
        return 4, "slab"
    return 1, "slab"


def _unroll_for(
    facts: DeviceFacts, threads: int, slice_elems: int, per_vector: int
) -> int:
    """Vectors per thread per filter iteration: the device's fact (8 where bytes in flight pay,
    see ``device.py``; 8 for 512-thread CTAs on the A100 too, where the register headroom
    lets it pay), never more than the dead mask holds (32 elements) nor more than the slice
    gives each thread (a slice of one iteration is walked as a boundary iteration otherwise:
    B200 64K b=8 in 4-CTA clusters 7.37 -> 8.21 us at unroll 8)."""
    unroll = facts.filter_unroll
    if threads == 512 and facts.capability == (8, 0):
        unroll = 8
    unroll = min(unroll, 32 // per_vector)
    per_thread = max(1, (slice_elems // per_vector) // threads)
    while unroll > 1 and unroll > per_thread:
        unroll //= 2
    return unroll


def _walk_width_for(splits: int, n: int) -> int:
    """Survivor reloads issued together in the filter's bit-walk (``phases/filter_pass.py``).
    Whole-row CTAs stage every survivor of the row and pay the reload chain in full: four in
    flight on rows up to 256K (six parts, 64K b=256 k=2048: B200 18.7 -> 16.7 us, H100 49.1 ->
    39.9, A100 91.9 -> 83.0, L40S 52.7 -> 44.4; ragged rows gain more, H100 44.4 -> 33.8), two on
    1M whole rows and on every split (A100 1M b=8 39.8 -> 36.4; on B200/Rubin/H100 splits two
    is within noise of one and four costs 2-3%)."""
    return 4 if splits == 1 and n <= (1 << 18) else 2


def streaming_config_for(
    facts: DeviceFacts, dtype: torch.dtype, k: int, n: int, rows: int = 0
) -> StreamingConfig:
    """The configuration the dispatcher would choose for (device, dtype, k, row length, batch)."""
    elem_bytes = 4 if dtype == torch.float32 else 2
    row_bytes = n * elem_bytes
    threads = 1024
    # a multiple of threads (the radix select's slots per thread), holding k where eight slots
    # per thread suffice; a wider crossing bin takes the exact select
    tie_capacity = min(8 * threads, max(2 * threads, -(-k // threads) * threads))
    packed = dtype == torch.float16 or (
        dtype == torch.bfloat16 and facts.packed_bf16_compare
    )
    splits, merge = splits_for(facts, rows, n, elem_bytes) if rows else (1, "cluster")
    aim = "wide" if (merge == "slab" and splits > 1) else "tight"
    per_vector = 16 // elem_bytes
    cfg = StreamingConfig(
        threads=threads,
        tie_capacity=tie_capacity,
        pdl=facts.supports_pdl,
        packed_compare=packed,
        splits=splits,
        merge=merge,
        aim=aim,
        unroll=_unroll_for(facts, threads, n // splits, per_vector),
        walk_width=_walk_width_for(splits, n),
        # whole-row configurations: rows that fit the registers (16K fp32 at 1024 threads, 8K
        # at 512) take the register-resident phases instead of the census
        register_arm=splits == 1,
        count_after_barrier=facts.staggered_count,
        bin_cursors=facts.bin_cursors,
        # row_order stays off here: the ordered kernel wins 7-15% on ragged wide batches but a
        # standalone prepare launch that ranks the rows costs 1.3 us, more than it saves on the
        # ledger's ragged cell and 12% on uniform batches (docs/measured-worse.md).  A caller
        # that supplies the order (topk_streaming(row_order=)) gets the ordered kernel.
    )
    row_kb = row_bytes >> 10
    wide_batch = rows > facts.sm_count and (
        row_kb <= 256 or (row_kb <= 512 and rows <= 2 * facts.sm_count)
    )
    if wide_batch and k <= 2048:
        wide_stage = 4096 if k <= 1024 else 8192
        # The census arm's cutoff drops to 8K at 512 threads: the sampled arm is cheaper from
        # 12K up there (k=2048: 10.7 vs 12.3 us at 12288, 11.4 vs 14.7 at 16384; k=512: 7.4 vs
        # 11.3).  lpt_order (longest rows first, ranked inside every CTA) stays off: its serial
        # prologue costs every batch 3-6% and pays back only on some parts' ragged batches
        # (docs/measured-worse.md); a caller who knows the lengths can order the rows itself.
        wide = StreamingConfig(
            **{
                **cfg.__dict__,
                "threads": 512,
                "ctas_per_sm": 2,
                "stage": wide_stage,
                "tie_capacity": max(1024, -(-k // 512) * 512),
                "unroll": _unroll_for(facts, 512, n, per_vector),
                "short_cutoff": 8192,
            }
        )
        if 2 * wide.shared_memory_bytes() <= facts.shared_memory_optin:
            return wide
        if k == 2048:
            # 99 KB parts: the 8K stage does not fit two CTAs per SM; a 3584-entry stage does,
            # and a second sample vector keeps its tails at 3.6 sigma with the balanced aim
            # (the tight aim would sit at the cap: docs/measured-worse.md, 4K stage at k=2048)
            small = StreamingConfig(
                **{**wide.__dict__, "stage": 3584, "sample_vectors": 2}
            )
            if 2 * small.shared_memory_bytes() <= facts.shared_memory_optin:
                return small
    # large k: k must sit below three quarters of the stage x splits (the balanced aim needs
    # room on both sides).  The stage grows first, to what the part's shared memory holds; then
    # the row is split across more CTAs through the slab; a k that no stage x splits holds
    # takes the exact select for every row.
    if k >= 3 * cfg.stage * splits // 4:
        cfg = _fit_large_k(facts, cfg, k, n, per_vector)
    if (
        not cfg.exact_only
        and cfg.stage < 16384
        and _needs_big_stage(k, n, rows, threads, cfg.stage, cfg.splits)
    ):
        big = StreamingConfig(**{**cfg.__dict__, "stage": 16384})
        if big.shared_memory_bytes() * big.ctas_per_sm <= facts.shared_memory_optin:
            cfg = big
        else:  # the stage cannot grow: shrink the survivor spread instead (a 16K sample)
            cfg = StreamingConfig(**{**cfg.__dict__, "sample_vectors": 4})
    return _fit_shared_memory(facts, cfg)


def _fit_shared_memory(facts: DeviceFacts, cfg: StreamingConfig) -> StreamingConfig:
    """Shrink the tie stage, a thread's worth at a time down to two slots per thread, until the
    configuration fits the part's shared memory: on 99 KB parts an 8K survivor stage and an 8K
    tie stage do not both fit, and the survivor stage is the one the sampled path needs (an
    overflowing crossing bin only sends the row to the exact select)."""
    while (
        cfg.shared_memory_bytes() * cfg.ctas_per_sm > facts.shared_memory_optin
        and cfg.tie_capacity > 2 * cfg.threads
    ):
        cfg = StreamingConfig(
            **{**cfg.__dict__, "tie_capacity": cfg.tie_capacity - cfg.threads}
        )
    return cfg


def _largest_stage(facts: DeviceFacts, cfg: StreamingConfig) -> int:
    """The largest stage (multiple of 256) whose configuration fits the part's shared memory."""
    other = cfg.shared_memory_bytes() - 2 * (cfg.stage + 4) * 4
    budget = facts.shared_memory_optin // cfg.ctas_per_sm - other
    return max(2048, (budget // 8 - 4) // 256 * 256)


def _fit_large_k(
    facts: DeviceFacts, cfg: StreamingConfig, k: int, n: int, per_vector: int
) -> StreamingConfig:
    """A configuration whose stage x splits holds k with the balanced aim's margin, or the
    exact-only configuration.

    Two stages compete for shared memory.  The survivor stage must hold 2k / splits per CTA
    (the cap, 3/4 of stage x splits, then sits at 1.5k and the balanced aim has k/4 of room on
    each side; at a third of margin the cap sat within 1.4% of k at 1M k=200000 and every row
    fell back).  The tie stage must hold the crossing bin among the survivors, which at these k
    is survivors / 256 times the density peak near the bar: a few thousand entries at k=200000
    (with 2048 slots every row fell back to the exact select, 772 us on B200).  So: the widest
    tie stage first, the fewest extra splits that let the survivor stage fit beside it; splits
    beyond one wave are accepted (a second wave of split rows beats one CTA streaming the row
    several times).  When nothing fits, every row takes the exact select."""

    def stage_for(splits: int) -> int:
        return -(-(2 * k + 256) // (256 * splits)) * 256

    candidates = [cfg.splits] + [
        s for s in (8, 16, 32) if s > cfg.splits and n // s >= MIN_SLICE
    ]
    for tie in (8 * cfg.threads, 4 * cfg.threads, 2 * cfg.threads):
        if tie < k // 32:
            # the crossing bin among ~1.5k survivors spread over 256 bins, at the density peak:
            # a tie stage below k / 32 overflowed on every row at 1M k=400000 (2048 slots, 679 us
            # against 300 us for the exact select)
            break
        base = StreamingConfig(**{**cfg.__dict__, "tie_capacity": tie})
        limit = _largest_stage(facts, base)
        for splits in candidates:
            stage = max(
                2048, 256 * splits, stage_for(splits)
            )  # the slab merge's scratch needs 256 x splits
            if stage > limit:
                continue
            if splits == cfg.splits:
                return StreamingConfig(
                    **{**base.__dict__, "stage": max(cfg.stage, stage)}
                )
            return StreamingConfig(
                **{
                    **base.__dict__,
                    "splits": splits,
                    "merge": "slab",
                    "aim": "wide",
                    "stage": stage,
                    "unroll": _unroll_for(facts, cfg.threads, n // splits, per_vector),
                    "walk_width": _walk_width_for(splits, n),
                    "register_arm": False,  # a split row is never whole in one CTA (the whole-row base carried the arm)
                }
            )
    # the exact select's tie stage is its census's: eight slots per thread
    return StreamingConfig(
        **{
            **cfg.__dict__,
            "splits": 1,
            "merge": "cluster",
            "aim": "tight",
            "stage": 2048,
            "short_cutoff": EXACT_ONLY,
            "tie_capacity": 8 * cfg.threads,
        }
    )


def _needs_big_stage(
    k: int, n: int, rows: int, threads: int, stage: int, splits: int
) -> bool:
    """Whether the tight aim plus three sigma of the survivor count would cross three quarters
    of ``stage`` (the aim's cap), so that a row's survivors would overflow it and re-walk.

    The survivor count spreads with sigma ``sqrt(aim * length / samples)``; at 1M fp32 rows
    (aim 6144, sigma 1250) an 8K stage overflows one row in ten, at 256K rows (aim about 2400,
    sigma 390) never.  The larger stage is not free: it takes the L1 the filter's survivor
    re-reads use (A100 256K b=64: 75.2 us with the 16K stage, 70.3 with 8K), so it is taken
    only where the spread demands it.  Mirrors ``phases/aim.py``: the fixed margin, the
    3.5-sigma floor for batches of 32 rows or more.
    """
    import math

    samples = (
        threads * 4
    )  # 16-byte vectors of fp32; 16-bit rows sample twice as many, which only helps
    per_sample = n / samples
    aim = k + max(k // 2, n // 256)
    if rows >= 32:
        zq = 3.5 * math.sqrt(per_sample)
        root = (zq + math.sqrt(zq * zq + 4.0 * k)) * 0.5
        aim = max(aim, int(root * root) + 1)
    sigma = math.sqrt(aim * per_sample)
    return aim + 3.0 * sigma > 0.75 * stage * splits
