# Cake non-deterministic GDN chunked training backend (SM100 / SM103 / SM107)

Experimental backend for the forward and backward of packed GDN (scalar-gated
DeltaNet) layers in training: head dimension 128, chunk 64, grouped Q/K heads
(`HV` a multiple of `H`), no context parallelism.  The public entry point is
`cake_gdn_nondet` in `flashinfer.cake_gdn_nondet_train`.  Every launch is
fixed-order today, so two runs of one call on one device give the same bits;
the contract does not promise run-to-run bits.  The
results match the deterministic package
`flashinfer.experimental.cake.cake_gdn_det_train` within that package's accuracy
budgets, not bitwise: where the backward runs the package tail or a cluster
tail of a value-head group of two or three (see Chain), `dq` and `dk` are
rounded to BF16 once, after the FP32 sum of the value heads' partials, where the
deterministic package rounds each partial.  The output and the final state have
the deterministic package's bits.  `dv`, `dg`, `dbeta` and the initial-state
gradient have them too, except where the backward runs the tcgen05 adjoint
scan, which rounds `dv2` once on `sm_100a`, `sm_103a` and `sm_107a`, or takes
the adjoint table priced for `dhu_tc2` (see Chain); they then match within the
same budgets.
On `sm_107a` a call may also take a deeper split table (see Chain); the
outputs that depend on that table then match within the same budgets too.

The facade `flashinfer.cake_gdn_nondet_train` is a stable FlashInfer API: its
entry points are `@flashinfer_api` functions.  This package, under
`flashinfer.experimental.cake`, is its backend.
SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the supported
architectures; another device raises `RuntimeError`.  The SM107 libraries need
CUDA 13.4 or newer.  The feature does not
participate in automatic backend selection, autotuning or trace-apply.  Its
libraries are loaded prebuilt from `aot/` when a builder compiled them and
JIT-built by FlashInfer from the sources and flags `build.json` records
otherwise.  The package reads no environment variable.

## Mode rules

The mode is in the name.  No entry takes a `deterministic` keyword: passing
one raises `TypeError`.

| call | runs | under `torch.use_deterministic_algorithms(True)` (strict or `warn_only=True`) |
|---|---|---|
| `flashinfer.cake_gdn_nondet_train.cake_gdn_nondet` and this package's `cake_gdn_nondet`, `CakeGDNNondetFunction.apply`, `cake_gdn_nondet_train_forward`, `cake_gdn_nondet_train_backward` | this package | runs unchanged: no alert, no warning, no fallback (every launch is fixed-order today) |
| `flashinfer.cake_gdn_det_train.cake_gdn_det` | the deterministic package | unchanged |

* This package is the non-deterministic route only.  Nothing routes a call
  to the deterministic package and nothing falls back to it: a caller who
  wants the deterministic route calls `flashinfer.cake_gdn_det_train`.
* PyTorch's deterministic algorithms do not change the route.  Every launch
  of this version is fixed-order, so no call raises PyTorch's alert or warns.
  The contract does not promise run-to-run bits: a later version may add an
  unordered launch, which then raises PyTorch's alert before it runs
  (`RuntimeError`; a `UserWarning` and then the launch under `warn_only=True`).
* The forward tape carries the tag `"mode": "nondet"`.
  `cake_gdn_nondet_train_backward` refuses (`ValueError`) a tape without it,
  such as a tape of the deterministic package.
* The context-parallel packages follow the same rules
  (`flashinfer.cake_gdn_nondet_train_kcp` and `flashinfer.cake_gdn_det_train_kcp`).

## Opt-in

Import the entry point explicitly; nothing in `flashinfer` imports this
package.

```python
from flashinfer.cake_gdn_nondet_train import cake_gdn_nondet

o, final_state = cake_gdn_nondet(
    q, k, v, g, beta,                    # [B, T, H, 128] / [B, T, HV, 128] bf16; g, beta [B, T, HV]
    scale=None,                          # 1 / sqrt(128) when None
    initial_state=None,                  # [N, HV, 128, 128] fp32 or None
    output_final_state=False,
    cu_seqlens=cu_seqlens,               # packed batch (B == 1): int32 / int64 offsets on the device
    cu_seqlens_cpu=cu_seqlens_cpu,       # the same offsets on the host: host-table route; None: device route
    use_qk_l2norm_in_kernel=True,
)
```

`is_cake_gdn_nondet_train_supported(device=None)` in the same module returns
`True` when the device is an exported target.

Without `cu_seqlens` the `B` sequences have length `T` each.  `g` is
computed in FP32; `beta` is FP32 or BF16 and its gradient comes back in its
dtype.  For a packed batch, `cu_seqlens_cpu` selects the route that builds the
chunk and segment tables on the host; without it the `meta` kernel builds
them on the device.  The device route plans its scans for a bound layout, so
the two routes can pick different plans and agree to rounding, not bitwise.

The entry point checks its inputs before the first launch and raises
`ValueError`: every tensor on the device of `q`, `k` of the shape of `q`,
`v`, `g` and `beta` over the tokens of `q`, `HV` a positive multiple of `H`,
`initial_state` of shape `[N, HV, 128, 128]`, `cu_seqlens` a 1-D int32 / int64
tensor with a batch of 1, and `cu_seqlens_cpu` with as many offsets, starting
at 0, not decreasing and ending at the tokens of `q` (an empty sequence is
allowed).  These checks read tensor metadata only; the host offsets are
checked when their layout is first planned.  The device route trusts the values
of `cu_seqlens`, since reading them would synchronize the device.

## Deterministic and non-deterministic routes

Packed GDN training has four Cake modules, one per (mode, parallelism).  Each
facade sits at the top level of `flashinfer`; its backend package sits under
`flashinfer.experimental.cake`:

| mode | one rank (this API, FLA signature) | context parallel (KCP) |
|---|---|---|
| `det` (deterministic) | `flashinfer.cake_gdn_det_train` (`cake_gdn_det`) | `flashinfer.cake_gdn_det_train_kcp` (`cake_gdn_det_kcp`) |
| `nondet` (within the det budgets) | `flashinfer.cake_gdn_nondet_train` (`cake_gdn_nondet`) | `flashinfer.cake_gdn_nondet_train_kcp` (`cake_gdn_nondet_kcp`) |

Each backend package is standalone: it imports nothing else from `flashinfer`
and ships its own copy of every kernel it shares with another.  Nothing
selects a package implicitly.

This package exports four names at its root:

| name | what it is |
|---|---|
| `cake_gdn_nondet` | the autograd entry point (FLA signature) |
| `CakeGDNNondetFunction` | its autograd Function |
| `cake_gdn_nondet_train_forward` | the forward chain over flattened inputs and one chunk table; returns the tape |
| `cake_gdn_nondet_train_backward` | the backward chain over one forward tape |

The planner and its helpers (`make_options`, `scan_plans`,
`build_chunk_meta`, ...) and the packed-row Function
(`CakeGDNNondetPackedFunction`, see Chain) stay in `cake_chain`.

## Migration

The old names fail at import; there are no aliases.

| old | new |
|---|---|
| `flashinfer.experimental.cake.cake_gdn_train` | `flashinfer.cake_gdn_nondet_train` (facade), `flashinfer.experimental.cake.cake_gdn_nondet_train` (backend) |
| `chunk_gated_delta_rule(..., deterministic=False)` | `cake_gdn_nondet(...)` |
| `ChunkGatedDeltaRuleFunction` | `CakeGDNNondetFunction` |
| `gdn_chunk_forward(..., deterministic=False)` | `cake_gdn_nondet_train_forward(...)` |
| `gdn_chunk_backward(..., deterministic=False)` | `cake_gdn_nondet_train_backward(...)` |
| package-root re-exports of `cake_chain` internals: `CHUNK`, `FINALIZE_STAGES`, `FULL`, `HEAD_DIM`, `PRESCAN_CONFIGS`, `REG_SCAN_CONFIGS`, `ChainOptions`, `ChunkMeta`, `ScanPlan`, `SegmentTable`, `StageSet`, `StreamGroups`, `build_chunk_meta`, `build_chunk_meta_from_cu_seqlens`, `choose_scan_plan`, `dhu_tc_pricing`, `finalize_stage`, `lpt_sequence_order`, `lpt_walk`, `make_options`, `scan_plans`, `sdt_pricing`, `sequence_groups`, `stage_set`, `two_stream_split`, `use_prep_tma` | `cake_chain.<name>` (same name) |
| `benchmarks/bench_cake_gdn_train.py` | `benchmarks/bench_cake_gdn_nondet_train.py` |

## Chain

The forward runs the prep (Q/K L2 norm, gate cumsum), the tcgen05 WY stage and
the forward scan fused with the output (`fwd_ho`).  The prep stages the Q/K
tiles by TMA with one CTA per (chunk, key head) (`prep_tma`), except where that
grid needs a second, short wave: with `slots = 6 x SMs` and
`CTAs = chunks x H`, the plain prep runs for `slots < CTAs <= 2.5 x slots`
(`sm_100a`, `sm_103a`; `sm_107a` has no plain-prep band and runs `prep_tma`
for every grid), and
for Q or K not 16-byte aligned.  Both preps give the same bits.  The WY stage
runs `wy_tc2`: the chains and outputs of `wy_tc` (the same bits) with a
64-column TMEM accumulator and packed shared-memory operands, so five CTAs fit
on an SM where `wy_tc` fits four.  A long sequence can be
split into segments: a pre-scan and a stitch then give every segment its start
state.  On `sm_107a`, a forward scan of one launch whose `segments x HV` CTAs
take at most half the SMs runs `fwd_ho_vh2`: two CTAs per segment and value
head, each the scan and output of one half of the value columns (the same
bits).  The backward runs the adjoint scan, either `dv_local` plus a register
scan or, on a serial plan, the tcgen05 scan with `dv_local` fused (`dhu_tc2`),
which a host-side cost rule selects on the host-table route.  On `sm_100a`,
`sm_103a` and `sm_107a` (`DVL32_ARCHS`) the tcgen05 scan runs as
`dhu_tc2_dvl32`: it adds the FP32 `dv_local` values to `dv2`, so `dv2` rounds
to BF16 once, where `dhu_tc2` rounds `dv_local` first (B200: 0.990-0.995 x
`dhu_tc2`).  On that route a
split `(64, 1)` plan runs the second pass of its adjoint scan on `dhu_tc2`,
from the stitched segment states, where a second cost rule prices it at most
0.95 of `dv_local` plus the register pass and, on `sm_100a`, at least 40 us
cheaper; its pre-scan then computes the `dv_local` tiles of its chunks itself,
so no `dv_local` stage runs.  That pre-scan (`dhu_pre_dvl_r64k1`) keeps its
`dv_local` tile in BF16 in the segment map: an operand rounding, not a rounded
partial sum of an output.  Its FP32-tile form `dhu_pre_dvl32_r64k1` ships but
is not routed (`PRE_DVL32_ARCHS` is empty): on B200 it cost up to 1.2 % of the
backward on the split plans of slow-decay gates.  The two pass-2 forms give the
same bits on `dhu_tc2` and agree within the budgets on `dhu_tc2_dvl32`.  On a host-built
table without plan pins, this package may also replace the planner's adjoint
table (priced for the register scans) by the table priced for `dhu_tc2`
itself: the serial table or a `(64, 1)` split table of any segment length,
priced as the makespan of one CTA per (segment, value head) times the
`dhu_tc2` step time plus the pre-scan and stitch terms.  It takes that table
when it saves at least 10 us and 5 % of the priced adjoint pass (on `sm_107a`
at least 2 us, with no percentage floor, `TC_PLAN_SELECT`), and runs it on
`dhu_tc2`; a serial table that the two-stream rule groups keeps its form.
On `sm_107a` (`CHAIN_PLAN`) each direction may then take a deeper `(64, 1)`
split table (more segments than the chosen table), priced for the chain that
runs it (`fwd_ho` for the forward, `dhu_tc2` for the adjoint) with R200 step
times and fitted split costs, when it saves at least 2 us and prices at most
0.95 of the chosen table; a forward table that runs `fwd_ho_vh2` is priced at
0.75 of its `fwd_ho` chain time (measured on R200), so such a table can keep
its form; a serial table that
the two-stream rule groups keeps its form, and `sm_100a` / `sm_103a` keep their
tables.  On `sm_100a`, `sm_103a` and `sm_107a` (`SPLIT_R_FORM`) a value-head
group of three with its cluster tail also keeps the planner's `(64, 1)` split
adjoint table when the `dhu_tc2` pricing runs its pass 2 and the two-stream rule
groups its segments (the R form below; the deterministic package's table and
groups).
Elsewhere the forward table is the planner's.  A different table
changes the bits of the outputs that depend on it (other segment boundaries),
so the deterministic package's bits need its tables: pin `scan_config` /
`segment_chunks`.  Then one of six
tails computes `dq`, `dk`, `dv`, `dg` and `dbeta`.  The device's architecture,
the call's value-head group `G = HV / H` and its (chunk, value head) CTAs
select the fastest one measured on B200, GB300 and R200 (the production rows, packs,
source head pairs and group shapes of `benchmarks/bench_cake_gdn_nondet_train.py`),
and the `tail` keyword pins one:

| tail | selected for | `dq` / `dk` rounding | run to run today (not a contract) |
|---|---|---|---|
| `"tail"`: one tcgen05 kernel per (chunk, value head), a workspace slot per value head, the last CTA of a (chunk, key head) sums the slots | `G` other than 2 and 3 with at least the architecture's `TAIL_MIN_CTAS` chunks x `HV` | the value heads' FP32 partials summed in value-head order, one BF16 rounding after the FP32 L2-norm pullback | bitwise |
| `"tail_g2_f32x"`: one two-CTA cluster kernel per (chunk, key head), the CTAs exchange their FP32 partials over distributed shared memory | `G = 2` | as `"tail"` (the same bits) | bitwise |
| `"tail_g3_f32x"`: as `"tail_g2_f32x"` with three-CTA clusters | `G = 3` | as `"tail"` (the same bits) | bitwise |
| `"tail_g2"`: the deterministic package's two-CTA cluster kernel | a pin at `G = 2` | each value head's partial rounded to BF16, summed in a fixed order (the deterministic package's bits) | bitwise |
| `"tail_g3"`: the deterministic package's three-CTA cluster kernel | a pin at `G = 3` | as `"tail_g2"` | bitwise |
| `"unfused"`: `dqkwg`, the WY pullback and the finalize | `G` other than 2 and 3 below the architecture's `TAIL_MIN_CTAS` chunks x `HV` | as `"tail_g2"` | bitwise |

The cluster tails are the same on `sm_100a`, `sm_103a` and `sm_107a`
(`CLUSTER_TAILS` holds them per architecture; `sm_103a` and `sm_107a` take
those of `sm_100a` until they are measured there).  On B200 (the tail kernels
alone) `tail_g3_f32x` is 1.024-1.034 x `tail_g3` on the `(2, 6)` production
rows and 1.034-1.051 x on the smaller head pairs, where `"tail"` is
1.073-1.081 x `tail_g3` from 3,782 (chunk, value head) CTAs on and 0.99-1.006 x
at 1,539-3,078 CTAs; `tail_g2_f32x` is 1.014-1.036 x `tail_g2`, which measured
16-19 % faster than `"tail"` at `G = 2` (B200 and GB300).

The cluster tails `tail_g2_f32x` and `tail_g3_f32x` read `v` and write `dq`,
`dk` and `dv` through token-row strides.  When `q`, `k` and `v` are the head
views of one packed BF16 row that requires grad (`row.split((H * 128, H * 128,
HV * 128), -1)`, each viewed as `[B, T, heads, 128]`) and the call's tail is
one of them, the row is the autograd input (`CakeGDNNondetPackedFunction`):
the backward allocates one gradient row and the tail writes `dq`, `dk` and
`dv` into its head views, so no concatenation runs.  Any other layout, another
tail, or a row that records no gradient takes `CakeGDNNondetFunction`, with
the same bits.  The forward still copies `v` into a contiguous tensor (the WY
stage reads it contiguous).

`TAIL_MIN_CTAS` holds the (chunk, value head) CTAs from which the package
tail runs, per architecture and group class (`tail_min_ctas(arch, G)`):

| architecture | `G = 1` | `G` of 4 and more |
|---|---|---|
| `sm_100a` (B200) | 2048 | 2048 |
| `sm_103a` (GB300) | 2048 | 1024 |
| `sm_107a` (R200) | 1024 | 1024 |

Each floor is the smallest size where `"tail"` measured no slower than
`"unfused"` in the backward.  On B200 the two tie at 2048 CTAs, and `"tail"`
is at least 16 % faster from 3072 CTAs on.  The backward time ratio `"tail"` /
`"unfused"` at 1024 CTAs is 1.02 (`G = 1`) and 0.89 (`G = 4`) on GB300, and
0.87 (`G = 1`) and 0.77-0.79 (`G = 4`) on R200 (216 SMs).

Every tail meets the deterministic package's accuracy budgets.  `dv`, `dg`
and `dbeta` have one writer each and the same bits in every tail; they are
the deterministic package's bits where the adjoint rounds `dv2` as that
package does (see above).

The workspace of `"tail"` holds one 64 KiB FP32 slot per (chunk, value head)
of a launch, written before it is read, and one 32-bit arrival counter per
(chunk, key head).  The counters are created zero (`torch.zeros` on the
launching stream) and every launch leaves them zero, so a later launch needs
no memset.  The chain keeps one workspace per (device, stream of the launch)
with `host_cache=True` (a fresh one per call otherwise), and launches the
chunk table in segments of at most `TAIL_WORKSPACE_TILES // HV` chunks (4096
slots, 256 MiB; at least one chunk), in stream order.  Calls on different streams take different workspaces.  A
CUDA graph captured on its own stream takes a workspace of that stream (zeroed
inside the capture when the capture creates it); replay one captured graph at
a time, not concurrently with itself or with eager calls on its capture
stream.  A value-head group of one uses no workspace.

On the host-table route the long sequences start first.  The tables list the
sequences by chunk count, longest first (ties in the caller's order); every
chunk keeps its token offset, the initial-state rows are gathered into that
order and the final state and the initial-state gradient come back in the
caller's order.  The forward scan also walks its segments longest first.  Each
CTA computes what it computes in the caller's order, so `lpt=False` changes
the order of the work, not the results.  The device route keeps the caller's order.

A serial scan runs one CTA per (sequence, value head) with one CTA per SM, so
its last round can leave SMs idle.  On the host-table route such a scan can
run as two contiguous groups of sequences.  With `n` sequences, `HV` value
heads and `SMs` multiprocessors, `idle(c) = ceil(c / SMs) x SMs - c` counts the
idle slots of `c` CTAs; `k` in `[1, n)` minimizes `idle(k x HV)` (ties to the
larger `k`).  The scan is split only if `idle(k x HV) < idle(n x HV)`; a scan
with `n < 2` or `n x HV <= SMs` is never split.  The groups follow the table
order (longest first unless `lpt=False`).  The adjoint `dhu_tc2` scan of a
value-head group of two runs the first `k` sequences on the caller's stream,
then their fused tail while a side stream scans the rest; so does the scan
beside the cluster tail of a group of three on `sm_100a` and `sm_103a`
(`TWO_STREAM_SERIAL_G3_ARCHS`, the deterministic package's rule).  The forward
runs `wy_tc` and `fwd_ho` of the first `n - k` sequences on the side stream;
once that `wy_tc` is done, the caller's stream runs those of the last `k`.  On
`sm_100a`, `sm_103a` and `sm_107a` (`SPLIT_R_FORM`) the `dhu_tc2` pass 2 of a
split adjoint table beside that tail runs the same way over its segments (`n`
segments, the first `k` on the caller's stream).  The side
stream is a stream of PyTorch's pool with the device's greatest priority, one
per caller stream; it is forked from the caller's stream with events and joined
before the stage returns (also when a launch raises), so the call can be
captured in a CUDA graph (warm it up first, like any captured call).  Every
tail runs on the caller's stream.
`streams=1` keeps every launch on the caller's stream.

## Speed against the deterministic package

`benchmarks/bench_cake_gdn_nondet_train.py` times this package against
`flashinfer.experimental.cake.cake_gdn_det_train` (both package entries, no
facade) on its rows, packs, head pairs
and group shapes (ABBA order, CUDA events, the L2 flushed before every call),
prints the stages each backward launched, and lists at the end every shape and
boundary where this package is slower than the other arm by more than the sum
of the two spreads.  Backward time of the deterministic package over this
package's (above 1: this package is faster; the median over 12 ABBA groups of
the two packages' time ratio inside a group, for the cells whose groups agree).
All figures in this section were measured before the rename of FlashInfer
#8928, with the same programs and tables:

| shapes | what this package runs differently | B200 | GB300 |
|---|---|---|---|
| `g1_*`, `g4_*` (`G` 1 or 4, at least 2048 chunks x `HV`) | the `"tail"` tail (and the priced table on the `4x4096` ones) | 1.24-1.46 | 1.28-1.42 |
| `r01`, `r02`, `r07`, `r10`, the `(2, 6)` packs `issue`, `P3`, `P1`, the pairs `(8, 24)`, `(4, 12)`, `(2, 6)`; `r04` and `(1, 3)` on B200 | the adjoint table priced for `dhu_tc2` | 1.02-1.19 (without `r07_1x16384_h8`, `p7074_P3_259174x2_h2`, `p7074_P1_241817x6_h2`) | 1.04-1.15 |
| the other shapes, among them `(16, 48)`, `A16`, `64x4096` and `g3_*` | nothing: the deterministic package's programs and tables | 1.00 (without `g3_16x4096_h1`, `r03_8x4096_h8`, `r09_16x4096_h16`; power-capped: `r05_16x4096_h8`, `r06_real15x4.3k_h8`) | 0.99-1.00 |

A shape named power-capped is left out of that cell: the GPU ran at its power
limit for 5 % or more of the samples of one of the two packages there, so the
cell waits for a sustained measurement (cool-down, warm-up, then at least 300
steady samples per block).

On the re-planned head pairs `(8, 24)`, `(4, 12)` and `(2, 6)` the forward +
backward measures 1.10 (without `(4, 12)`, `(2, 6)`) on B200 and 1.06-1.11 on GB300, the checkpointed
step (two forwards and the backward) 1.07 (without `(4, 12)`, `(2, 6)`) and 1.03-1.08.  This package is slower than the
deterministic package beyond the sum of the two spreads only in the forward of `g4_16x4096_h4` (0.997) and the forward of `r03_8x4096_h8` (0.995) on B200.  The rows that run the same programs measure down to
0.989 (`(1, 3)`, GB300) in the backward, and the forward of every
shape down to 0.991 (`(2, 6)`, GB300): small calls whose issue time
is about their device time, where the entry's mode check and option checks
(about 0.5 us a call; the rename removed the mode check) and the backward's
tail rule show.

## Plan pins (keyword-only)

| keyword | default | effect |
|---|---|---|
| `scan_config` | planner | register-scan `(value block, key split)` of both scans: `(32, 2)`, `(32, 1)` or `(64, 1)` |
| `segment_chunks` | planner | segment length in chunks of the segment-parallel scans; `0` forces the serial scan; lengths above 128 (the chunks one pre-scan CTA covers) are not supported: a call whose plan then splits a sequence raises `ValueError` |
| `finalize` | by size | finalize of the `"unfused"` tail: `"finalize"` or `"finalize_mlp"` (the latter needs `HV == 3 * H`) |
| `warmup_log2` | `-10 / ln 2` | warm-up threshold of the pre-scans |
| `dhu_scan` | rule | adjoint scan of a serial plan: `"reg"` (`dv_local` + register scan) or `"tc"` (`dhu_tc2`); also the second pass of a split `(64, 1)` plan on the host-table route (`"tc"` with the fused pre-scan; the same bits as `"reg"` on `dhu_tc2`, within the budgets on `dhu_tc2_dvl32`) |
| `prep` | rule | forward prep: `"plain"` or `"tma"` (`prep_tma`; Q or K not 16-byte aligned still take the plain prep) |
| `save_state` | `True` | keep the forward state of the architecture's tape level for the backward (`T_full`: `w`, `h`, `v_new`; `u`, read only to rebuild `h` / `v_new`, is freed when the forward returns); `False` recomputes them |
| `host_cache` | `True` | keep the chunk tables, the `dhu_tc2` selections and the per-stream workspaces between calls |
| `lpt` | `True` | host-table route: dispatch the sequences and the forward segments longest first; `False` keeps the caller's order |
| `streams` | `2` | host-table route: `2` lets a serial scan run as two sequence groups on two streams where the rule above splits it; `1` keeps one stream |
| `tail` | rule | backward tail (see Chain): `"tail"`, `"tail_g2_f32x"` or `"tail_g2"` (`HV == 2 * H` only), `"tail_g3_f32x"` or `"tail_g3"` (`HV == 3 * H` only) or `"unfused"` |

## Limitations

- Head dimension 128 and chunk 64 only; `q`, `k`, `v` BF16.
- No context parallelism and no cross-rank state passing.
- `sm_100a`, `sm_103a` and `sm_107a` only (`sm_107a` with CUDA 13.4 or newer).
