# Cake deterministic GDN chunked training backend (SM100 / SM103)

Experimental backend for the forward and backward of packed GDN (scalar-gated
DeltaNet) layers in training: head dimension 128, chunk 64, grouped Q/K heads
(`HV` a multiple of `H`), no context parallelism.  Every launch is a plain
stream-ordered kernel: no atomics, no data-dependent ordering, so the outputs
and every gradient are bitwise the same run to run for one layout and one
plan.  Some stages run as two groups of sequences on the caller's stream and a
side stream; the groups write disjoint outputs, so this holds with one or two
streams.

Both the API and the backend are experimental: no compatibility guarantee.
SM100 (B200 / GB200) and SM103 (B300 / GB300) are the supported
architectures; another device raises `RuntimeError`.  The feature does not
participate in automatic backend selection, autotuning or trace-apply.  Its
libraries are loaded prebuilt from `aot/` when a builder compiled them and
JIT-built by FlashInfer from the sources and flags `build.json` records
otherwise.  The package reads no environment variable.

## Opt-in

Import the entry point explicitly; nothing in `flashinfer` imports this
package.

```python
from flashinfer.experimental.cake_gdn_det_train import chunk_gated_delta_rule

o, final_state = chunk_gated_delta_rule(
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
`CTAs = chunks x H`, the plain prep runs for `slots < CTAs <= 2.5 x slots`, and
for Q or K not 16-byte aligned.  Both preps give the same bits.  A long sequence can be
split into segments: a pre-scan and a stitch then give every segment its start
state.  The backward runs the adjoint scan, either `dv_local` plus a register
scan or, on a serial plan, the tcgen05 scan with `dv_local` fused (`dhu_tc2`),
which a host-side cost rule selects on the host-table route.  For a value-head
group of two (`HV == 2 * H`) one fused tcgen05 kernel then computes `dq`, `dk`,
`dv`, `dg` and `dbeta`; other groups run `dqkwg`, the WY pullback and the
finalize.

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

## Plan pins (keyword-only)

| keyword | default | effect |
|---|---|---|
| `scan_config` | planner | register-scan `(value block, key split)` of both scans: `(32, 2)`, `(32, 1)` or `(64, 1)` |
| `segment_chunks` | planner | segment length in chunks of the segment-parallel scans; `0` forces the serial scan; lengths above 128 (the chunks one pre-scan CTA covers) are not supported: a call whose plan then splits a sequence raises `ValueError` |
| `finalize` | by size | finalize of the unfused tail (`HV != 2 * H`): `"finalize"` or `"finalize_mlp"` (the latter needs `HV == 3 * H`) |
| `warmup_log2` | `-10 / ln 2` | warm-up threshold of the pre-scans |
| `dhu_scan` | rule | adjoint scan of a serial plan: `"reg"` (`dv_local` + register scan) or `"tc"` (`dhu_tc2`) |
| `prep` | rule | forward prep: `"plain"` or `"tma"` (`prep_tma`; Q or K not 16-byte aligned still take the plain prep) |
| `save_state` | `True` | keep the forward's `w`, `u`, `h`, `v_new` for the backward; `False` recomputes them |
| `host_cache` | `True` | keep the chunk tables, the `dhu_tc2` selections and the per-stream workspaces between calls |
| `lpt` | `True` | host-table route: dispatch the sequences and the forward segments longest first; `False` keeps the caller's order (the same bits) |
| `streams` | `2` | host-table route: `2` lets a serial scan run as two sequence groups on two streams where the rule above splits it; `1` keeps one stream (the same bits) |

## Limitations

- Head dimension 128 and chunk 64 only; `q`, `k`, `v` BF16.
- No context parallelism and no cross-rank state passing.
- `sm_100a` and `sm_103a` only.
