# Maintaining the Blackwell next/ GenPhase fork

Read [VENDOR.md](VENDOR.md) for the exact integration baseline, namespace mapping
and local device differences. This package is independently owned by SM100/103;
keep the original Rubin tree and frontend unchanged.

## Layout and imports

```text
sm100/next_cutedsl_megamoe/
├── src/blackwell_genphase_sources/
│   ├── communication/, helpers/, quant_def.py, api.py
│   └── kernel_src/{schedulers/, blackwell/inference/mega/}
├── shim/{_paths.py, block_scaled.py, genphase.py, comm.py,
│         kernel_helpers.py, correctness.py}
├── __init__.py
├── VENDOR.md
└── SKILL.md
```

Only this package's `shim/` imports `blackwell_genphase_sources`. Consumers use
the public package API. Preserve the private namespace and its bootstrap guard;
never alias or replace Rubin's `sources` modules in `sys.modules`.
Raw `src/` changes must match the explicit port inventory, and stay excluded
from formatters. Do not update the legacy Blackwell or Rubin snapshots as a
side effect of a GenPhase change.

## Updating the port

1. Record a complete upstream/integration revision. Recreate the starting
   export and namespace/directory mapping described in `VENDOR.md`.
2. Reapply and review the Blackwell-only device changes. Preserve native K32
   MMA, static B-row ownership in both loaders and the 512-column TMEM budget.
3. Compare unchanged dependencies byte-for-byte. Update the source record for
   every deliberate local difference and keep raw imports inside the shim.
4. Check config, operand layouts, compile/launch signatures and reset bounds
   against the selected source revision. Keep the current admission limits
   unless broader configurations have native qualification.
5. Reuse `tests/moe_ep/test_mega_workspace_knobs.py` for ownership, admission
   and workspace plans. Run the GenPhase cases in the existing
   `test_nvfp4_cutedsl_kernel_vs_reference.py` on native Blackwell, and the
   corresponding `test_mega_cuda_graph_multirank.py` cases at EP2 and EP4.
   Verify early/late weights, empty/last/holes/asymmetric/dense-restored routes,
   poisoned combine storage and first/65th captured replays.
6. Verify that Rubin source/frontend remain unchanged and Blackwell imports do
   not load the Rubin package. Run applicable pre-commit checks.

Use `Sm100GenphaseMoeConfig` and `Sm100GenphaseSymmBuffer`, transform weights
with `preprocess_genphase_weights`, and construct a relauncher with
`sm100_genphase_mega_launch_thunk`. Native checks select `sm_100a` or `sm_103a`
before the DSL imports. Do not change default backend selection as part of a
source-layout edit.

## Configuration selection

The explicit GenPhase backend resolves tile, cluster and grouped schedule from
an exact workload cache or host-side heuristics. `knobs="auto"` runs collective
tuning before capture; explicit fields constrain the candidates. Keep selection
in the backend and `shim/{tuning,knob_cache,autotune}.py`, without inspecting
device routing during forward. Bump the cache implementation revision when the
kernel or tuning behavior changes. Retuning must preserve staging/output
pointers, and auto workspaces must remain unpooled.
Caller-supplied candidates must contain exactly all GenPhase knob keys; reject
incomplete or unknown key sets on all ranks before starting any trial, so cache
publication cannot fail on rank 0 while peers wait at the final barrier.

Reuse `tests/moe_ep/test_mega_workspace_knobs.py` for candidate, constraint,
cache and host API contracts, and the existing MXFP8 multirank test for native
auto/cache/graph validation. See `docs/design_docs/moe_ep_runbook.md` for the
offline CLI and live-token profile workflow.
