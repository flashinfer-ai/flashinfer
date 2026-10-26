# Cake deterministic GDN chunked training backend (SM100 / SM103)

Experimental backend for the forward and backward of packed GDN (scalar-gated
DeltaNet) layers in training: head dimension 128, chunk 64, grouped Q/K heads
(`HV` a multiple of `H`), no context parallelism.  Every launch is a plain
stream-ordered kernel: no atomics, no data-dependent ordering, so the outputs
and every gradient are bitwise the same run to run for one layout and one
plan.

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
them on the device, with the same results.

## Plan pins (keyword-only)

| keyword | default | effect |
|---|---|---|
| `scan_config` | planner | register-scan `(value block, key split)` of both scans: `(32, 2)`, `(32, 1)` or `(64, 1)` |
| `segment_chunks` | planner | segment length in chunks of the segment-parallel scans; `0` forces the serial scan |
| `finalize` | by size | `"finalize"` or `"finalize_mlp"` (the latter needs `HV == 3 * H`) |
| `warmup_log2` | `-10 / ln 2` | warm-up threshold of the pre-scans |
| `save_state` | `True` | keep the forward's `w`, `u`, `h`, `v_new` for the backward; `False` recomputes them |
| `host_cache` | `True` | keep the chunk tables and the per-stream workspaces between calls |

## Limitations

- Head dimension 128 and chunk 64 only; `q`, `k`, `v` BF16.
- No context parallelism and no cross-rank state passing.
- `sm_100a` and `sm_103a` only.
