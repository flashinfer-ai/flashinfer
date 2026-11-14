# Cake deterministic GDN chunked training backend (SM100 / SM103 / SM107)

## Mode and dispatch rules

- This package is the deterministic (`det`) route only.  Its public names
  carry the mode word: `cake_gdn_det` (the autograd entry),
  `CakeGDNDetFunction` (its autograd class), `cake_gdn_det_train_forward` and
  `cake_gdn_det_train_backward` (the low-level chain over a chunk table and a
  tape).  The package exports no other name; the planner and table helpers
  stay in `cake_chain`.  The facade `flashinfer.cake_gdn_det_train` exports
  `cake_gdn_det`, `CakeGDNDetFunction` and the support query
  `is_cake_gdn_det_train_supported`; the low-level chain stays in this
  package because it takes the package's chunk table and tape.
- The entries take no `deterministic` keyword.  Passing one, with any value,
  raises `TypeError`.
- `torch.use_deterministic_algorithms(True)` changes nothing: every launch is
  fixed-order, so no alert is raised and the outputs and gradients are the
  same bits as without it.
- The non-deterministic route is a separate package.  Its results match this
  package within this package's accuracy budgets, not bitwise.  Nothing here
  routes to it.
- The forward tape carries this package's tag, `"mode": "det"`.
  `cake_gdn_det_train_backward` refuses any other tape with `ValueError`; a
  tape of the non-deterministic package carries `"mode": "nondet"`.

Experimental backend for the forward and backward of packed GDN (scalar-gated
DeltaNet) layers in training: head dimension 128, chunk 64, grouped Q/K heads
(`HV` a multiple of `H`), no context parallelism.  Every launch is a plain
stream-ordered kernel: no atomic on a value (the wavefront pair below counts
stored tiles with atomic counters, whose order no value depends on), no
data-dependent ordering, so the outputs and every gradient repeat bit for bit
from run to run for one layout and one plan.  Some stages run as two groups of sequences on the caller's stream and a
side stream; the groups write disjoint outputs, so this holds with one or two
streams.

The facade `flashinfer.cake_gdn_det_train` is a stable FlashInfer API: its
entry points are `@flashinfer_api` functions.  This package, under
`flashinfer.experimental.cake`, is its backend.
SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the supported
architectures; another device raises `RuntimeError`.  The SM107 libraries need
CUDA 13.4 or newer.  The feature does not
participate in automatic backend selection, autotuning or trace-apply.  Its
libraries are loaded prebuilt from `aot/` when a builder compiled them and
JIT-built by FlashInfer from the sources and flags `build.json` records
otherwise.  The package reads no environment variable.

## Opt-in

Import the entry point explicitly from the facade
`flashinfer.cake_gdn_det_train` (it exports `cake_gdn_det`,
`CakeGDNDetFunction` and `is_cake_gdn_det_train_supported`, and forwards every
argument to this package); nothing in `flashinfer` imports this package
otherwise.

```python
from flashinfer.cake_gdn_det_train import cake_gdn_det

o, final_state = cake_gdn_det(
    q, k, v, g, beta,                    # [B, T, H, 128] / [B, T, HV, 128] bf16; g, beta [B, T, HV]
    scale=None,                          # 1 / sqrt(128) when None
    initial_state=None,                  # [N, HV, 128, 128] fp32 or None
    output_final_state=False,
    cu_seqlens=cu_seqlens,               # packed batch (B == 1): int32 / int64 offsets on the device
    cu_seqlens_cpu=cu_seqlens_cpu,       # the same offsets on the host: host-table route; None: device route
    use_qk_l2norm_in_kernel=True,
)
```

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
which a host-side cost rule selects on the host-table route.  On that route a
split `(64, 1)` plan runs the second pass of its adjoint scan on `dhu_tc2`,
from the stitched segment states, where a second cost rule prices it at most
0.95 of `dv_local` plus the register pass and, on `sm_100a`, at least 40 us
cheaper; its pre-scan then computes the `dv_local` tiles of its chunks itself,
so no `dv_local` stage runs.  Both forms give the same bits on `dhu_tc2`; a
value-head group of three on the architectures of `DVL32_ARCHS` runs
`dhu_tc2_dvl32` instead (below).  For a value-head
group of two (`HV == 2 * H`) or of three (`HV == 3 * H`) one fused tcgen05
cluster kernel then computes `dq`, `dk`, `dv`, `dg` and `dbeta`; other groups
run `dqkwg`, the WY pullback and the finalize.

For a value-head group of three, on `sm_100a`, `sm_103a` and `sm_107a`, the
fused tail rounds `dq` and `dk` once: its three CTAs exchange the group's FP32
`dq` / `dk` partials over distributed shared memory and sum them in value-head
order before one BF16 rounding (`tail_g3_f32x`; `tail_g3` rounds each value
head's partials before their sum).  The tail writes `dq`, `dk` and `dv`
through their token-row strides, so when `q`, `k` and `v` are the head views of
one packed BF16 row that requires grad (`row.split((H * 128, H * 128, HV *
128), -1)`, each viewed as `[B, T, heads, 128]`), the row is the autograd input
and the backward writes one packed gradient row, with no concatenation and the
same bits as on the views.  The tcgen05 adjoint scan of such a call adds the
FP32 `dv_local` values to `dv2` before its one rounding (`dhu_tc2_dvl32`), on
serial plans and in the second pass of split ones.
Both are fixed-order with one writer per element, so the gradients repeat bit
for bit; they differ from the previous release of this package (within its
accuracy budgets).  The register adjoint forms still round `dv_local` to BF16
first, so for such a call they give the same `o` and final state as
`dhu_tc2_dvl32`, but other bits of `dk`, `dv`, `dg`, `dbeta` and the
initial-state gradient (within the budgets): the cost rules and the `dhu_scan`
pin pick the bits, and on a split `(64, 1)` plan the device-table route always
runs the register pass.  Groups of two keep `tail_g2` and `dhu_tc2`.

On the host-table route, a call with FP32 `beta`, `use_qk_l2norm_in_kernel`,
`HV <= 3 * H` and no `prep`, `scan_config` or `segment_chunks` pin runs one of
two forwards on the A of `wy_a`, the WY A producer (A, the Q/K norms, the gate
cumsum and the normalized Q/K in one persistent kernel, one CTA per SM over
the (chunk, key head) items in a fixed order):

- On `sm_100a`, `sm_103a` and `sm_107a`, a call whose forward segment table is
  serial runs `wy_a` and `fwd_ho_a`, the forward scan and output on that A
  (u and w formed in the CTA, never in memory).  On `sm_107a`, such a call
  without plan pins keeps a serial forward table where the planner splits one,
  if the serial table prices at most 1.10 times the split table and its
  `sequences x HV` CTAs fill at least 0.65 of the SMs.
- On `sm_100a` and `sm_103a`, a call whose forward segment table splits a
  sequence, with pieces of at least 48 chunks in the host plan's one-wave piece
  table, runs the split forward: `wy_a`, then wave 1 (the first piece of every
  split sequence) beside the affine maps of the middle pieces on a side stream
  (joined before the stitch), the stitch, and wave 2 (every other piece from
  its stitched start state).  The waves run `fwd_ho_a` with `fwd_summary_a`
  (on `sm_103a` for pieces of at least 96 chunks), else the WY stage on the A
  of `wy_a` with `fwd_fused` and `fwd_summary`.  The piece length is the
  smallest one whose first wave fits one CTA per SM; it depends on the
  sequence lengths, `HV` and the SM count only.

The route is chosen from shapes, dtypes, the architecture, the SM count and the
pins, never from an address or a value: a Q, K or V that a kernel cannot read
in place, or a `g` / `beta` not at a 16-byte aligned base, is copied first.
Both forwards are fixed-order (no atomic, wave 1 and the maps write disjoint
segments), so they are bitwise run to run with one or two streams.  Their
bits differ from the prep forward's, so the outputs and gradients of these
FP32-`beta` host-table calls change from the previous release of this package
(within its accuracy budgets).  Every other call (a BF16 `beta`, Q/K not
normalized, the device route, a larger value-head group, a `prep`,
`scan_config` or `segment_chunks` pin) runs the prep forward with the same
launches and bits as before.  The other options (`dhu_scan`, `finalize`,
`warmup_log2`, `lpt`, `streams`, `save_state`, `host_cache`) do not change the
forward route.

The forward keeps for the backward only the state of the architecture's tape
level (`T_full`: `w`, `h` and `v_new`; `u` is read only to rebuild `h` and
`v_new`, so the backward runs the WY stage again only where those are
missing).  A call that records no backward (grad mode off, or no input that
requires grad) and the first pass of a non-reentrant activation checkpoint
(`torch.utils.checkpoint` with `use_reentrant=False`) keep no tape: the forwards
on the A of `wy_a` then store no `w`, `h` or `v_new`.  The saved list keeps its
13 entries with placeholders of the same shape, dtype and device in place of
the state not kept, which the backward never reads.

On the host-table route, where the `dhu_tc2` adjoint scan (serial, or the second
pass of a split plan) feeds the fused tail of a value-head group of two with a
BF16 `beta` on the caller's stream, and its `segments x HV` CTAs take at most
half the SMs, the backward runs the two as a wavefront pair: the scan counts
each (chunk, value head) tile in a workspace once it has stored that tile's
`dh` and `dv2`, and the tail visits the chunks in the order the scan finishes
them and waits for each count before it reads the tile.  The tail runs on the
side stream once every scan CTA has started.  The pair runs on `sm_103a` and
`sm_107a`, and on `sm_100a` with at least 32 value heads.  On `sm_100a` and
`sm_103a` the pair also runs the second pass of a split plan whose `segments x HV` CTAs
outnumber the SMs, at any value-head count, on a call of at least 393216
(3 x 2^17) tokens x value heads (below that the pair's host cost outlasts the
GPU time it saves): the pre-scan and the stitch run before the fork, and the
tail starts once the last scan CTA of the second round has started.  That
scan is `dhu_tc2_wavefront_split`, whose store warps meet at a barrier every
four chunks and let one thread fence and add the counts of those chunks;
the scan of at most half the SMs keeps `dhu_tc2_wavefront`, whose store
threads each fence every two chunks and add one per chunk (on `sm_107a` its
store warps meet every eight chunks and one thread fences and adds the counts,
as the split pass 2's scan does every four).  A
CUDA graph does not order its two branches, so under a capture the backward
records the separate launches.  The tail reads the values it reads without the pair: the
same bits.

On the host-table route the long sequences start first.  The tables list the
sequences by chunk count, longest first (ties in the caller's order); every
chunk keeps its token offset, the initial-state rows are gathered into that
order and the final state and the initial-state gradient come back in the
caller's order.  The forward scan also walks its segments longest first.  Each
CTA computes what it computes in the caller's order, so the outputs and the
gradients are bitwise the same with `lpt=False`.  The device route keeps the
caller's order.

A serial scan runs one CTA per (sequence, value head) with one CTA per SM, so
its last round can leave SMs idle.  On the host-table route such a scan can
run as two contiguous groups of sequences.  With `n` sequences, `HV` value
heads and `SMs` multiprocessors, `idle(c) = ceil(c / SMs) x SMs - c` counts the
idle slots of `c` CTAs; `k` in `[1, n)` minimizes `idle(k x HV)` (ties to the
larger `k`).  The scan is split only if `idle(k x HV) < idle(n x HV)`; a scan
with `n < 2` or `n x HV <= SMs` is never split.  The groups follow the table
order (longest first unless `lpt=False`).  The adjoint `dhu_tc2` scan of a
value-head group of two runs the first `k` sequences on the caller's stream,
then their fused tail while a side stream scans the rest.  The forward runs
`wy_tc` and `fwd_ho` of the first `n - k` sequences on the side stream; once
that `wy_tc` is done, the caller's stream runs those of the last `k`.  The side
stream is a stream of PyTorch's pool with the device's greatest priority, one
per caller stream; it is forked from the caller's stream with events and joined
before the stage returns (also when a launch raises), so the call can be
captured in a CUDA graph (warm it up first, like any captured call).
`streams=1` keeps every launch on the caller's stream; the bits are the same.

The host-table route caches device tables between calls: a layout's chunk and
segment tables, the split forward's tables, the wavefront pair's tile order and
the identity walk of the forward scan.  A table is written on the stream of the
call that builds it, and
an eager call on another stream waits for that write (an event recorded after
it) before its launches read the table.  A CUDA graph capture does not cache the
tables it builds, since they are written only when the graph replays.  It reads
the cached tables without waiting, so their writes must be done when the
capture starts (`torch.cuda.graph` synchronizes the device first), and keeps
them for the life of the process, since every replay reads them.  The device
synchronization covers only work queued before the capture: while one thread
captures calls of this package, another thread's call on the same device that
builds a table must have its stream synchronized before the capture reads that
layout.  A capture also takes its own workspaces instead of the pooled
per-stream ones.

## Plan pins (keyword-only)

| keyword | default | effect |
|---|---|---|
| `scan_config` | planner | register-scan `(value block, key split)` of both scans: `(32, 2)`, `(32, 1)` or `(64, 1)`; keeps the call on the prep forward |
| `segment_chunks` | planner | segment length in chunks of the segment-parallel scans; `0` forces the serial scan; lengths above 128 (the chunks one pre-scan CTA covers) are not supported: a call whose plan then splits a sequence raises `ValueError`; keeps the call on the prep forward |
| `finalize` | by size | finalize of the unfused tail (`HV != 2 * H`): `"finalize"` or `"finalize_mlp"` (the latter needs `HV == 3 * H`); for `HV == 3 * H` a pin runs the unfused tail instead of the fused kernel (`dv`, `dg` and `dbeta` the same bits; `dq` / `dk` rounded per value head) |
| `warmup_log2` | `-10 / ln 2` | warm-up threshold of the pre-scans |
| `dhu_scan` | rule | adjoint scan of a serial plan: `"reg"` (`dv_local` + register scan) or `"tc"` (`dhu_tc2`; `dhu_tc2_dvl32` for `HV == 3 * H` on `DVL32_ARCHS`); also the second pass of a split `(64, 1)` plan on the host-table route (`"tc"` with the fused pre-scan); `"tc"` gives the bits of `"reg"` on `dhu_tc2`, and on `dhu_tc2_dvl32` the same `o` and final state with other gradient bits within the budgets |
| `prep` | rule | forward prep: `"plain"` or `"tma"` (`prep_tma`; Q or K not 16-byte aligned still take the plain prep); keeps the call on the prep forward |
| `save_state` | `True` | keep the forward state of the tape level (`w`, `h`, `v_new`) for the backward; `False` recomputes them |
| `host_cache` | `True` | keep the chunk tables, the `dhu_tc2` selections and the per-stream workspaces between calls (a CUDA graph capture takes its own workspaces) |
| `lpt` | `True` | host-table route: dispatch the sequences and the forward segments longest first; `False` keeps the caller's order (the same bits) |
| `streams` | `2` | host-table route: `2` lets a serial scan run as two sequence groups on two streams where the rule above splits it; `1` keeps one stream (the same bits) |

## Limitations

- Head dimension 128 and chunk 64 only; `q`, `k`, `v` BF16.
- No context parallelism and no cross-rank state passing.
- `sm_100a`, `sm_103a` and `sm_107a` only (`sm_107a` with CUDA 13.4 or newer).

## Names (FlashInfer #8928)

The package moved from `flashinfer.experimental.cake_gdn_det_train` and its
entries took the mode-word names below.  The former names and the former path
were removed with no alias: importing them raises `ImportError`.  The library
names, the program names and the kernels are unchanged.

| former name | name |
|---|---|
| `flashinfer.experimental.cake_gdn_det_train` | `flashinfer.experimental.cake.cake_gdn_det_train` |
| `chunk_gated_delta_rule_det`, `chunk_gated_delta_rule` | `cake_gdn_det` |
| `ChunkGatedDeltaRuleDetFunction`, `ChunkGatedDeltaRuleFunction` | `CakeGDNDetFunction` |
| `gdn_chunk_det_forward`, `gdn_chunk_forward` | `cake_gdn_det_train_forward` |
| `gdn_chunk_det_backward`, `gdn_chunk_backward` | `cake_gdn_det_train_backward` |
