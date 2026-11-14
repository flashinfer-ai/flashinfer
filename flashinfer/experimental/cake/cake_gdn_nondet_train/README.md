# Cake non-deterministic GDN chunked training backend (SM100 / SM103 / SM107)

Experimental backend for the forward and backward of packed GDN (scalar-gated
DeltaNet) layers in training: head dimension 128, chunk 64, grouped Q/K heads
(`HV` a multiple of `H`), no context parallelism.  The public entry point is
`cake_gdn_nondet` in `flashinfer.cake_gdn_nondet_train`.  Every launch is
fixed-order, so every output repeats bit for bit (see Outputs).  The
results match the deterministic package
`flashinfer.experimental.cake.cake_gdn_det_train` within that package's accuracy
budgets, not bitwise: where the backward runs the package tail or a cluster
tail of a value-head group of two or three (see Chain), `dq` and `dk` are
rounded to BF16 once, after the FP32 sum of the value heads' partials, where the
deterministic package rounds each partial.  The output and the final state have
the deterministic package's bits where the forward runs that package's
kernels (the prep forward, see Chain); where it runs a forward of its own, on
the WY stage's `A` from `wy_a` (the split forward or the serial A form; a
`prep` pin turns both off), they match within the same budgets.  `dv`, `dg`,
`dbeta` and the initial-state gradient have the deterministic package's bits
too after the prep forward, except where the backward runs the tcgen05 adjoint
scan, which rounds `dv2` once on `sm_100a`, `sm_103a` and `sm_107a`, or takes
the adjoint table priced for `dhu_tc2` or the one-wave split of the adjoint
walk (see Chain); after a forward on `A`, or with such a backward, they match
within the same budgets.
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

## Outputs

Every output repeats bit for bit: two identical calls on one device (the same
inputs, arguments, layout and route, plain or under activation checkpointing)
give the same bits, and a second backward into the same leaves adds exactly
the first one's gradients.  No output of this version may vary.  The outputs
need not be bitwise equal to the deterministic package's (see above).  The tables
give the error of each output against an FP64 reference next to the error of
the deterministic package as of FlashInfer commit `2e989556` (then
`flashinfer.experimental.cake_gdn_det_train`, entry
`chunk_gated_delta_rule_det`) on the same inputs.

| output | run to run | FP64 error at the `(2, 6)` pack of 131,071 + 87,531 + 43,385 tokens on B200 (`sm_100a`, 148 SMs), this package / the deterministic package: rel-L2; worst | largest ratio to the deterministic package over 66 cases on B200: rel-L2; worst |
|---|---|---|---|
| `o` | repeats bit for bit | 5.149e-3 / 5.095e-3; 5.706e-3 / 5.783e-3 | 1.012; 1.042 |
| final state | repeats bit for bit | 4.278e-3 / 4.211e-3; 4.599e-3 / 4.591e-3 | 1.018; 1.052 |
| `dq` | repeats bit for bit | 4.879e-3 / 5.100e-3; 5.238e-3 / 5.800e-3 | 1.003; 1.040 |
| `dk` | repeats bit for bit | 5.939e-3 / 6.201e-3; 6.124e-3 / 6.745e-3 | 1.000; 1.028 |
| `dv` | repeats bit for bit | 4.789e-3 / 4.826e-3; 6.202e-3 / 6.272e-3 | 1.009; 1.061 |
| `dg` | repeats bit for bit | 5.954e-3 / 5.888e-3; 6.105e-3 / 5.924e-3 | 1.029; 1.059 |
| `dbeta` | repeats bit for bit | 5.162e-3 / 5.182e-3; 5.439e-3 / 5.375e-3 | 1.015; 1.069 |
| initial-state gradient | repeats bit for bit | 3.531e-3 / 3.518e-3; 3.780e-3 / 3.798e-3 (the pack from a nonzero initial state) | 1.009; 1.036 (over the 15 cases with a nonzero initial state) |

The same measurement on GB300 (`sm_103a`, 152 SMs), GB200 (`sm_100a`, 152 SMs)
and R200 (`sm_107a`, 212 SMs).  GB300 and GB200 give the same figures in every
cell of their columns:

| output | GB300 and GB200: the pack, this package / the deterministic package: rel-L2; worst | GB300 and GB200: largest ratio over 66 cases | R200: the pack | R200: largest ratio over 66 cases |
|---|---|---|---|---|
| `o` | 5.148e-3 / 5.095e-3; 5.847e-3 / 5.969e-3 | 1.012; 1.080 | 5.095e-3 / 5.095e-3; 5.796e-3 / 5.796e-3 | 1.010; 1.080 |
| final state | 4.276e-3 / 4.212e-3; 4.751e-3 / 4.541e-3 | 1.018; 1.086 | 4.212e-3 / 4.212e-3; 4.460e-3 / 4.460e-3 | 1.016; 1.066 |
| `dq` | 4.879e-3 / 5.100e-3; 5.256e-3 / 5.938e-3 | 1.003; 1.014 | 4.823e-3 / 5.101e-3; 5.084e-3 / 5.874e-3 | 1.000; 1.000 |
| `dk` | 5.939e-3 / 6.200e-3; 6.205e-3 / 6.816e-3 | 1.000; 1.067 | 5.904e-3 / 6.200e-3; 6.246e-3 / 6.768e-3 | 0.999; 1.017 |
| `dv` | 4.789e-3 / 4.826e-3; 6.203e-3 / 6.194e-3 | 1.009; 1.078 | 4.731e-3 / 4.825e-3; 5.873e-3 / 6.280e-3 | 1.009; 1.044 |
| `dg` | 5.973e-3 / 5.897e-3; 5.975e-3 / 5.926e-3 | 1.043; 1.188 (see below) | 5.886e-3 / 5.893e-3; 5.759e-3 / 6.199e-3 | 1.043; 1.188 (see below) |
| `dbeta` | 5.164e-3 / 5.181e-3; 5.458e-3 / 5.422e-3 | 1.012; 1.066 | 5.098e-3 / 5.182e-3; 5.428e-3 / 5.587e-3 | 1.012; 1.041 |
| initial-state gradient | 3.533e-3 / 3.516e-3; 3.798e-3 / 3.849e-3 | 1.008; 1.043 (15 cases) | 3.485e-3 / 3.517e-3; 3.782e-3 / 3.782e-3 | 1.008; 1.020 (15 cases) |

rel-L2 is `||out - ref|| / ||ref||` and worst is `max |out - ref| / max |ref|`;
`ref` is an FP64 token-by-token recurrence (FP64 Q/K L2 norm; the gradients
from a hand-written FP64 adjoint, and from autograd on the small layouts,
where the two agree to at most `5.0e-15`), not chunked.  Each figure is the
median over 25 input seeds on the GPU of its column (packed q / k / v head
views of one row, the host-table route, in-kernel L2 norm, scale
`1 / sqrt(128)`, zero initial state except in the initial-state row).  The pack
columns use decays log-spaced per value head from `1e-4` to `1` per token
(times `U[0.5, 1.5)`) and `beta = sigmoid(N(0, 1))`.  The 66 cases are the
three `(2, 6)` packs of the source issue, a 64 x 4,096-token pack and the head
pairs `(16, 48)`, `(8, 24)`, `(4, 12)`, `(2, 6)` and `(1, 3)` at 32,768 tokens
in three sequences, each under three gate regimes (FLA's GatedDeltaNet
initialization, the log-spaced decays, and `logsigmoid(N(0, 1))`); five small
layouts with chunk tails and one-token sequences; adversarial inputs: a
131,072-token sum without decay, slow decay, near-cancellation (repeated and
teacher-forced keys and values with `beta` near 1), heavy-tailed inputs and
concentrated keys, large-norm and zero keys, `beta` near 0, near 1 and near 2
with near-parallel keys, every chunk tail, one-token, empty and 1,500 short
sequences, value bursts, mirrored heads whose `dq` / `dk` are a small
difference, and the unnormalized path; and thirteen of these adversarial
inputs again on the sequence lengths of the pack columns, which B200, GB200
and GB300 run on the split forward (see Chain).  The initial-state row takes
15 more cases that start every sequence from an FP32 state of `0.1 N(0, 1)`:
the three `(2, 6)` packs and the `(8, 24)` pair under the three gate regimes,
and three of the small layouts; its pack figures are the pack columns' layout
and decays from that state.  A ratio is the median of this package's error
over the seeds divided by the median of the deterministic package's error over
the same seeds.  Every case was measured plain and under activation
checkpointing (`torch.utils.checkpoint`, non-reentrant), with the same
figures.  R200 runs the prep forward on the pack (see Chain), so its `o` and
final-state errors there are the deterministic package's.  The split forward
adds no measurable error to the serial one: on B200, over the `(8, 24)` pair
and two of the `(2, 6)` packs (2 to 15 pieces per sequence) under the
log-spaced decays, `logsigmoid(N(0, 1))` and no decay, from a nonzero initial
state (forward only, one seed per case), the split forward's rel-L2 error is
1.000 x that of the serial A form on `o` and 0.998-1.003 x on the final state.

On B200 every ratio is at most 1.069.  On GB300, GB200 and R200 every ratio is
at most 1.1 but one: the ratio of the worst `dg` element on the small `(1, 3)`
layout (seven sequences of 1 to 700 tokens, log-spaced decays), 1.188 over
seeds 0-24.  Over 100 seeds that ratio is 0.981 on GB300 and on GB200, which
give the same bits there (bootstrap 95 % interval 0.88-1.07).  The inputs are
drawn on the host CPU, and the aarch64 hosts of GB300, GB200 and R200 round
the drawn values differently from the x86_64 host of B200, so the same seed
draws other inputs on B200; on the same inputs B200, GB200, GB300 and R200
give the same bits on this layout, and on B200's inputs the ratio is 0.884.
The layout runs the serial A form (see Chain); a `prep` pin, which runs the
prep forward, gives it the deterministic package's `dg` bits.  From a nonzero
initial state every ratio is at most 1.081 on B200.  On GB300, GB200 and R200
one is above 1.1: the rel-L2 of `dg` on the same small layout, 1.130 over
seeds 0-24.  It follows the input draw as the 1.188 does: B200 gives 1.130 on
the aarch64-drawn inputs, and B200, GB300 and R200 give 1.060 on the
x86_64-drawn ones.  Over 100 seeds on GB300 it is 1.011 (bootstrap 95 %
interval 0.93-1.10).

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
  A later version may add an unordered launch for an output that then may
  vary (see Outputs); that launch raises PyTorch's alert before it runs
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
Under saved-tensor hooks (non-reentrant activation checkpointing, offload)
the device route saves its tables with the forward's other tensors, so
checkpointing drops them with the rest and its replayed forward builds them
again; a plain call keeps them for its backward.

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

The forward takes one of three routes, chosen on the host from the call's
layout, dtypes and alignment, never from tensor contents.  A call on a
host-built table (the host-table route or an equal-length call) with
normalized Q/K at 16-byte aligned bases, FP32 `beta`, a value-head group of at
most three (`WY_A_MAX_GROUP`) and no `prep` pin can take one of the first two,
both on the WY stage's `A` from `wy_a`: one persistent CTA per SM computes
`A = bf16(X)`, `X = (I + L)^-1` of every chunk and value head (a tcgen05
product and TF32x3 merges, within half a BF16 ulp of the FP64 `X`), beside
the prep's bits of the Q/K norms, the gate cumsum and the normalized Q/K.  A
call that neither of the two takes runs the third, the prep forward (below).

* The split forward (`FWD_SPLIT_ARCHS`: `sm_100a`, `sm_103a`), where the
  forward table splits a sequence, no `scan_config` / `segment_chunks` pin
  fixes it and the pieces of the host piece table are at least 48 chunks long
  (`cake_plan.FORWARD_SPLIT_MIN_PIECE`).  The piece table (`cake_plan.pieces`)
  cuts every long sequence into pieces of at most `L` chunks, the smallest `L`
  whose first wave fits one CTA per SM; the one-wave split of the adjoint walk
  (below) reads the same table.  Wave 1 runs the first piece of every split
  sequence from its initial state beside the affine maps `[B | M]` of the
  middle pieces (on a side stream; `streams=1` keeps one stream), the stitch
  gives every piece its start state, and wave 2 runs every other piece.  On
  `sm_100a`, and on `sm_103a` for pieces of at least 96 chunks
  (`FWD_SPLIT_A_MIN_PIECE`), the waves run `fwd_ho_a` and the maps
  `fwd_summary_a`, which form `u` and `w` in the CTA; on shorter `sm_103a`
  pieces `wy_tc2` forms `w` and `u` from `A`, and the waves run `fwd_fused` and
  the maps `fwd_summary`.  Where the backward splits its adjoint walk on the
  same pieces and the device has a cost entry that prices it
  (`cake_plan.M_SHARE_COSTS`: B200) at a priced saving of at least
  `cake_plan.M_SHARE_MIN_SAVING` (15 us), the taped forward also keeps the middle
  pieces' maps for the backward's first wave (M-share, `M_SHARE_ARCHS`; see
  below).
* The serial A form (`FWD_A_ARCHS`: `sm_100a`, `sm_103a`, `sm_107a`), where
  the forward table splits no sequence: `fwd_ho_a` runs the scan and the
  output on `A`, `qn` and `kn` with one CTA per (segment, value head), forming
  `u = bf16(A bf16(beta v))` and `w = bf16(A bf16(beta 2^g kn))` in the CTA
  at the deterministic forward's rounding points: two launches where the prep
  forward runs three.  On `sm_107a` (`FWD_A_SERIAL_KEEP`) a host table without
  plan pins keeps the planner's cheapest serial forward table where the
  planner takes a split one, if that serial table prices at most 1.10 x the
  split one and its sequences x `HV` CTAs fill at least 0.65 of the SMs; this
  route then runs it.

On the shapes of the speed table below, `sm_100a` (B200, GB200) runs the
split forward on `fwd_ho_a` on the three `(2, 6)` packs and the `(8, 24)`
pair, and `sm_103a` (GB300) on the packs, with the `wy_tc2` form on the
`(8, 24)` pair (pieces of 63 chunks); on all four GPUs the 64 x 4,096-token
pack and the `(16, 48)` pair run the serial A form.  The pairs `(4, 12)`,
`(2, 6)` and `(1, 3)`, whose pieces are shorter than 48 chunks, every other
timed shape on R200, and every call outside the two routes above run the prep
forward: the prep (Q/K L2 norm, gate cumsum), the tcgen05 WY stage and the
forward scan fused with the output (`fwd_ho`).  The prep stages the Q/K
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
which a host-side cost rule selects on the host-table route; a host-built
table can instead take the one-wave split of the adjoint walk (below).  On
`sm_100a`, `sm_103a` and `sm_107a` (`DVL32_ARCHS`) the tcgen05 scan runs as
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
Elsewhere the forward table is the planner's, except the split forward's
piece table and the serial table of `FWD_A_SERIAL_KEEP` (see above).  A
different table changes the bits of the outputs that depend on it (other
segment boundaries), and a forward on `wy_a`'s `A` changes the bits of every
output, so the deterministic package's bits need its tables and its forward:
pin `scan_config` / `segment_chunks` and `prep`.  Then one of six
tails computes `dq`, `dk`, `dv`, `dg` and `dbeta`.  The device's architecture,
and the call's value-head group `G = HV / H` select one that rounds `dq` /
`dk` once (the cluster tail of `G = 2` and `G = 3`, the package tail of every
other group, at every size), and the `tail` keyword pins any of them:

| tail | selected for | `dq` / `dk` rounding | run to run |
|---|---|---|---|
| `"tail"`: one tcgen05 kernel per (chunk, value head), a workspace slot per value head, the last CTA of a (chunk, key head) sums the slots | `G` other than 2 and 3 | the value heads' FP32 partials summed in value-head order, one BF16 rounding after the FP32 L2-norm pullback | bitwise |
| `"tail_g2_f32x"`: one two-CTA cluster kernel per (chunk, key head), the CTAs exchange their FP32 partials over distributed shared memory | `G = 2` | as `"tail"` (the same bits) | bitwise |
| `"tail_g3_f32x"`: as `"tail_g2_f32x"` with three-CTA clusters | `G = 3` | as `"tail"` (the same bits) | bitwise |
| `"tail_g2"`: the deterministic package's two-CTA cluster kernel | a pin at `G = 2` | each value head's partial rounded to BF16, summed in a fixed order (the deterministic package's bits) | bitwise |
| `"tail_g3"`: the deterministic package's three-CTA cluster kernel | a pin at `G = 3` | as `"tail_g2"` | bitwise |
| `"unfused"`: `dqkwg`, the WY pullback and the finalize | a pin | as `"tail_g2"` | bitwise |

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
the same bits.  The forward reads `v` in place there too: `wy_tc2` and
`fwd_ho_a` take its token pitch (`_v_rows`, the deterministic package's
rule); a `v` whose pitch they cannot take is copied once into contiguous rows.

The package tail runs at every size, so `dq` / `dk` round once on small
calls too.  There it can be slower than `"unfused"` in the backward: the two
tie at 2048 (chunk, value head) CTAs on B200; on GB300 the backward time ratio
`"tail"` / `"unfused"` is 1.02 at 1024 CTAs (`G = 1`) and at 512 (`G = 4`); on
an R200 board with 216 SMs (not the 212-SM R200 of the tables in this file)
`"tail"` is faster from 1024 CTAs on.

Every tail meets the deterministic package's accuracy budgets.  `dv`, `dg`
and `dbeta` have one writer each and the same bits in every tail; they are
the deterministic package's bits where the forward and the adjoint run as that
package's do (see above).

The workspace of `"tail"` holds one 64 KiB FP32 slot per (chunk, value head)
of a launch, written before it is read, and one 32-bit arrival counter per
(chunk, key head).  The counters are created zero (`torch.zeros` on the
launching stream) and every launch leaves them zero, so a later launch needs
no memset.  The chain keeps one workspace per (device, stream of the launch)
with `host_cache=True` (a fresh one per call otherwise), and launches the
chunk table in segments of at most `TAIL_WORKSPACE_TILES // HV` chunks (4096
slots, 256 MiB; at least one chunk), in stream order.  Calls on different streams take different workspaces.  A
CUDA graph capture takes its own workspace (fresh buffers from the graph's
memory pool, the counters zeroed inside the capture) instead of the pooled one
of its stream; replay one captured graph at a time, not concurrently with
itself.  A value-head group of one uses no workspace.

The autograd backward of a host-table or equal-length call with
`host_cache=True` and the kept tape is planned before it launches.  Its first
call for a layout, stream and set of arguments (the q / k and value head
counts among them) records its launches, the gradients it allocates and its
stream switches.  Later calls replay that record: the saved tensors, `do` and
the new gradients fill its slots, and every other argument is a pooled
workspace of the stream, a chunk table of the layout or a scalar.  The record
holds no tensor of a call.  A replay needs the same stream, saved tensors and
a `do` of the recorded shapes, strides, dtypes and 16-byte base alignment (a
saved-tensor hook such as `torch.autograd.graph.save_on_cpu` can change them;
a strided or unaligned `do` is first copied, as the general path copies it),
no final-state gradient, the stream's pool tables and the module tables the
record was made with; otherwise the call runs the general path and records
again.  A replay makes the same launches as the general path, so the bits
are the same.  A backward inside a CUDA graph capture neither replays nor
records a plan (a plan's launches read the pooled workspaces of its stream):
it runs the general path on the capture's own workspaces.
`cake_chain.BWD_PLAN = False` keeps every call on the general path.

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
(`TWO_STREAM_SERIAL_G3_ARCHS`, the deterministic package's rule).  The prep
forward runs `wy_tc` and `fwd_ho` of the first `n - k` sequences on the side
stream; once that `wy_tc` is done, the caller's stream runs those of the last
`k`.  The serial A form does not group: it runs its two launches on the
caller's stream.  On
`sm_100a`, `sm_103a` and `sm_107a` (`SPLIT_R_FORM`) the `dhu_tc2` pass 2 of a
split adjoint table beside that tail runs the same way over its segments (`n`
segments, the first `k` on the caller's stream).  On B200
(`ONE_ROUND_TAIL_DEVICES`: `sm_100a` with 148 SMs) a serial adjoint scan of
one round that fills more than half of the SMs, whose sequences end at
different steps, can also fork before the walks: the shortest sequences walk
on the side stream and run their tail there, on the SMs that the longer walks
leave idle (the same CTAs and bits).  The side
stream is a stream of PyTorch's pool with the device's greatest priority, one
per caller stream; it is forked from the caller's stream with events and joined
before the stage returns (also when a launch raises), so the call can be
captured in a CUDA graph (warm it up first, like any captured call).  Every
other tail runs on the caller's stream.
`streams=1` keeps every launch on the caller's stream.

The host-table route caches device tables between calls: a layout's chunk and
segment tables, its split tables (the split forward's and the adjoint walk's)
and the identity walk of the forward scan.  A table is written on the stream
of the call that builds it, and an eager call on another stream waits for
that write (an event recorded after it) before its launches read the table;
a hit on a layout's tables covers its adjoint split tables too, which only the
backward (and its replayed plan) reads.  A CUDA graph capture does not cache
the tables it builds, since they are written only when the graph replays.  It
reads the cached tables without waiting, so their writes must be done when
the capture starts (`torch.cuda.graph` synchronizes the device first), and
keeps them for the life of the process, since every replay reads them.  The
device synchronization covers only work queued before the capture: while one
thread captures calls of this package, another thread's call on the same
device that builds a table must have its stream synchronized before the
capture reads that layout.  A capture also takes its own workspaces instead of
the pooled per-stream ones, and its backward neither replays nor records a
plan.

The one-wave split of the adjoint walk runs on `sm_100a`, `sm_103a` and
`sm_107a` (`ADJOINT_SPLIT_ARCHS`), on a host-built table without plan or
`dhu_scan` pins:

* Its pieces are those of the split forward's piece table
  (`cake_plan.pieces`).  It runs where `adjoint_split_pricing` prices it below
  the serial table, both at their full cost with the cost entry of the device
  (`adjoint_split_costs`: the entry of `ADJOINT_SPLIT_DEVICE_COSTS` for the
  architecture and SM count, else that of `ADJOINT_SPLIT_COSTS`); the price
  does not depend on the gates.  `cake_plan.ADJOINT_FORMS` fixes the backward
  form of a device and head shape where a measurement decided it: `nosplit`
  keeps the planner's tables without the split, whose pre-scans skip the steps
  the warm-up cut allows (GB200, H 1 / HV 3).
* Wave 1 walks the last piece of every sequence from the final-state gradient
  (`dhu_tc2_dvl32_modes`) beside the affine maps `[B | M]` of every piece but
  each sequence's first (`dhu_tc2_dvl32_gm`, under the pre-scans' warm-up
  threshold: where a piece's decay from its start falls below it, the map
  keeps the chunks up to there and its `B` alone).  The stitch then composes
  the end adjoint of every other piece, and wave 2 walks those pieces from
  them, the first piece of each sequence storing the initial-state gradient.
  Every chunk's walk is the serial walk's from its end adjoint, which the
  split computes up to the stitch's rounding (3xBF16 split products, about 16
  mantissa bits per step)
  and the warm-up bound.
* M-share: where the taped split forward kept the middle pieces' maps
  (`M_SHARE_ARCHS`, priced by `cake_plan.M_SHARE_COSTS`, which has an entry
  for B200 only), wave 1 computes no maps: it walks the middle pieces from
  zero for their start adjoints on the side stream beside the last pieces'
  walks, which store theirs; after the join the stitch composes the end
  adjoints with the forward's `M^T` (`stitch_mt`) and wave 2 runs, both on the
  caller's stream.
* Streams without M-share: the maps, the stitch and wave 2 run on the side
  stream beside the last pieces' walks, every launch bound before the fork.
  Where the shorter of wave 1's two launches costs at most the entry's `fork`
  (the host's fork and join: 65 us on GB200, `sm_100a` with 152 SMs, and on
  every `sm_103a` device, measured on GB300; 0 elsewhere), or with
  `streams=1`, every launch runs on the caller's stream, maps first.  Where
  every piece fits one wave beside the others (pieces x `HV` at most the SMs,
  `ONE_WAVE_SPLIT_ARCHS`), the maps, the stitch and one walk of every piece
  run on the caller's stream.  The three stream forms give the same bits.

## Speed against the deterministic package

`benchmarks/bench_cake_gdn_nondet_train.py` times this package against
`flashinfer.experimental.cake.cake_gdn_det_train` (both package entries, no
facade) on its rows, packs, head pairs
and group shapes (ABBA order, CUDA events, the L2 flushed before every call),
prints the stages each backward launched, and lists at the end every shape and
boundary where this package is slower than the other arm by more than the sum
of the two spreads.

The table gives the time of the deterministic package over this package's
(above 1: this package is faster), from a separate measurement: CUPTI GPU time
with a cold L2, the median of each process's samples, the mean over four fresh
processes in ABBA order (on R200 the four processes ran at once, one on each of
four GPUs of one node), PyTorch 2.14.1 on its CUDA 13.0 runtime inside a CUDA
13.4 development container, every arm on the same GPU (on GB200 every process
bound to the GPU-local Grace socket).  B200, GB200 and GB300 ran a build with
this version's kernels, routes and bits; R200 ran an earlier build whose
routes and plans on these rows, and bits under two of the three gate regimes,
are this version's.  The deterministic
package is that of FlashInfer commit `18c327b4` (on R200 `c8a4d8a2`, with the
same deterministic package).  It is not the accuracy baseline of Outputs: it
carries that package's later kernel and table changes, and the deterministic
package that ships beside this version differs from it in host code only.  A
cell gives the geometric mean over 30 rows, the lowest and the highest row,
and the rows below 1.  The rows are the three `(2, 6)` packs and the 64 x
4,096-token pack of Outputs, the `(16, 48)` pair (twice, in two rows) and the
pairs `(8, 24)`, `(4, 12)`, `(2, 6)` and `(1, 3)` at 32,768 tokens, each under
the three gate regimes of Outputs.

| boundary | B200 (`sm_100a`, 148 SMs) | GB200 (`sm_100a`, 152 SMs) | GB300 (`sm_103a`, 152 SMs) | R200 (`sm_107a`, 212 SMs) |
|---|---|---|---|---|
| forward | 1.237 (0.998-1.480; 1 row below 1) | 1.340 (1.000-1.795; none) | 1.320 (0.986-1.807; 3 rows) | 1.116 (1.002-1.465; none) |
| backward | 1.076 (0.955-1.218; 8 rows) | 1.079 (0.987-1.201; 5 rows) | 1.068 (0.979-1.191; 5 rows) | 1.129 (0.977-1.383; 2 rows) |
| forward + backward | 1.132 (0.959-1.271; 5 rows) | 1.172 (0.978-1.332; 3 rows) | 1.158 (0.966-1.327; 3 rows) | 1.140 (0.995-1.353; 2 rows) |
| checkpointed step (two forwards and the backward) | 1.138 (1.005-1.336; none) | 1.171 (0.982-1.301; 1 row) | 1.166 (1.011-1.290; none) | 1.088 (0.978-1.252; 6 rows) |

Every row below 1 is under the `logsigmoid(N(0, 1))` gate or FLA's
initialization, on the three `(2, 6)` packs or the pairs `(8, 24)`, `(4, 12)`
and `(2, 6)`; the lowest, 0.955, is the backward of the two-sequence pack
(128,147 + 131,027 tokens) on B200 under both of them.  Under these
fast-decaying gates the deterministic package's pre-scans, in its forward and
in its backward, skip the steps whose decay has fallen below the warm-up
threshold.  This package's split forward and the walks of its one-wave adjoint
split run every step, so the deterministic package gains more there: on B200
the backward of the 261,987-token pack takes 2.948 / 2.390 / 2.399 ms in the
deterministic package and 2.488 / 2.491 / 2.495 ms in this one, and the
forward of the two-sequence pack 1.465 / 1.122 / 1.135 ms and 1.126 / 1.124 /
1.125 ms, under the log-spaced decays / `logsigmoid(N(0, 1))` / FLA's
initialization.  The forward rows below 1 (B200, GB300) run the split
forward.  At the `(8, 24)` pair the checkpointed step costs more than the
forward and the forward + backward measured alone: 0.25-0.45 ms more in this
package and at most 0.14 ms more in the deterministic package, so the row of GB200
under `logsigmoid(N(0, 1))` measures 0.982 although its forward and its
backward measure 1.101 and 1.150.  The checkpointed rows below 1 on R200
(0.978-0.999) are its packs under the two gates, where it runs the prep
forward (1.002-1.005) and its backward alone is below 1 on two of them; there
the checkpointed step's extra cost over the forward and the forward + backward
is 0.05-0.06 ms larger in this package than in the deterministic package.

## Plan pins (keyword-only)

| keyword | default | effect |
|---|---|---|
| `scan_config` | planner | register-scan `(value block, key split)` of both scans: `(32, 2)`, `(32, 1)` or `(64, 1)` |
| `segment_chunks` | planner | segment length in chunks of the segment-parallel scans; `0` forces the serial scan; lengths above 128 (the chunks one pre-scan CTA covers) are not supported: a call whose plan then splits a sequence raises `ValueError` |
| `finalize` | by size | finalize of the `"unfused"` tail: `"finalize"` or `"finalize_mlp"` (the latter needs `HV == 3 * H`) |
| `warmup_log2` | `-10 / ln 2` | warm-up threshold of the pre-scans |
| `dhu_scan` | rule | adjoint scan of a serial plan: `"reg"` (`dv_local` + register scan) or `"tc"` (`dhu_tc2`); also the second pass of a split `(64, 1)` plan on the host-table route (`"tc"` with the fused pre-scan; the same bits as `"reg"` on `dhu_tc2`, within the budgets on `dhu_tc2_dvl32`) |
| `prep` | rule | forward prep: `"plain"` or `"tma"` (`prep_tma`; Q or K not 16-byte aligned still take the plain prep); a pin also keeps the call off the forwards on `wy_a`'s `A` (see Chain) |
| `save_state` | `True` | keep the forward state of the architecture's tape level for the backward (`T_full`: `w`, `h`, `v_new`; `u`, read only to rebuild `h` / `v_new`, is freed when the forward returns); `False` recomputes them |
| `host_cache` | `True` | keep the chunk tables, the `dhu_tc2` selections, the per-stream workspaces and the backward's launch record between calls (a CUDA graph capture takes its own workspaces and no launch record) |
| `lpt` | `True` | host-table route: dispatch the sequences and the forward segments longest first; `False` keeps the caller's order |
| `streams` | `2` | host-table route: `2` lets a serial scan run as two sequence groups on two streams where the rule above splits it (the adjoint scan, and the scan of the prep forward; the serial A form keeps one stream), and lets the split forward and the one-wave split of the adjoint walk run their first wave beside a side stream, and, on B200, lets a one-round serial adjoint scan fork its shortest sequences' walk and tail onto the side stream; `1` keeps one stream |
| `tail` | rule | backward tail (see Chain): `"tail"`, `"tail_g2_f32x"` or `"tail_g2"` (`HV == 2 * H` only), `"tail_g3_f32x"` or `"tail_g3"` (`HV == 3 * H` only) or `"unfused"` |

## Limitations

- Head dimension 128 and chunk 64 only; `q`, `k`, `v` BF16.
- No context parallelism and no cross-rank state passing.
- `sm_100a`, `sm_103a` and `sm_107a` only (`sm_107a` with CUDA 13.4 or newer).
