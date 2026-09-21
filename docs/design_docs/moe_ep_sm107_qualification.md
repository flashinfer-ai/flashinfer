# SM107 MegaMoE validation

The Rubin backends support NVFP4 and MXFP8 E4M3/E5M2 inference through
`MoEEpLayer`, with BF16 combine and output. The kernel comes from CuTe DSL
MegaMoE `1667b47a`; [VENDOR.md](../../flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/VENDOR.md)
records the full source and exporter revisions.

## Correctness coverage

At FlashInfer `5bd5aeef60c44a99341e6b6a183968d73bf582e7`, native tests passed
50 single-GPU cases and 16 EP4 cases on each rank.

## Setup

Use one GPU per rank in a single SM107 NVLink domain:

```bash
git submodule update --init --recursive
export CUTE_DSL_ARCH=sm_107a
python -m pip install --no-build-isolation -e '.[sm107]'
```

Set the target before importing CuTe DSL. The shim uses native Rubin APIs
available in CuTe DSL 4.8.0.dev0 and compatible newer builds. The minimum
public DSL build has not been tested on Rubin.

For each run, save the source revision and any local diff, `python -m
flashinfer.collect_env`, `nvidia-smi -q`, and `nvidia-smi topo -m`. Record
power limits, clocks, other GPU activity, and `NVSHMEM_SYMMETRIC_SIZE`.

## Supported contracts

| Contract | Requirement |
|---|---|
| Device | Exact compute capability 10.7; all tensors and the current stream on the owning device |
| Distributed execution | One GPU per EP rank, same NVLink domain, matching NVSHMEM PE and EP identities |
| Experts and routes | Positive expert count, evenly sharded; top-k in `[1, E]`; at most 16,384 experts and 2,097,152 padded routes/rank |
| Public layer geometry | H multiple of 64 for NVFP4, 128 for MXFP8; I multiple of 64 |
| Canonical weights | Floating `[E_local, 2I, H]` and `[E_local, H, I]`; canonical gate rows precede up rows |
| Transformed weights | K-major physical storage, 16-byte aligned, exact typed scale planes; preserve physical layout when concatenating experts |
| BF16 inputs | `quantize_input=True`; staging quantizes on the caller's stream |
| Prequantized inputs | Matching data format; scales are `[T, H/16]` E4M3 for NVFP4 or `[T, H/32]` E8M0 for MXFP8; uint8 scale storage is interpreted as raw bytes. Pass the logical scale columns, excluding internal workspace communication padding. |
| Normalization | Unit normalization; `fc1_alpha`, `fc2_alpha`, and `fc1_norm_const` must be omitted |
| Routing values | Unique valid expert IDs per token or `-1` masked slots; finite scores; repeated masked slots are allowed |
| Output | BF16; owned tensor by default; workspace views expire on the next workspace use or destruction |
| Graphs and pooling | Warm up eagerly on every rank before capture; sequential, stream-ordered use of a shared workspace |
| Determinism | In-kernel FC2 reduction is opt-in and requires early routing weights; repeatability must be characterized separately |

Routing value checks use device assertions so they remain active on graph
replay without a host synchronization. Invalid routing can invalidate the
worker's CUDA context, as with other invalid CUDA indexing operations.
Use separate worker processes for negative device-assertion tests. Valid
masked routes are supported; their combine slots are cleared each launch.
This clearing and routing validation have a performance cost that belongs
in the measured kernel and full-forward spans respectively.

The shim rounds physical token capacity up by at most three rows to keep
four-int router loads in bounds. The public live-token limit is unchanged.
Workspace reuse across simultaneous streams is unsupported: serialize
access with stream ordering/events, or use distinct workspaces. External
NVSHMEM initialization must use the same EP membership; PE rank/count
checks cannot reconstruct a foreign initializer's membership list.

Graph replay keeps captured shapes, pointers, and live-token count fixed.
Change tensor contents in place; recapture for a new shape/count, or mask
inactive rows while retaining the captured shape.

Pre-quantized **weight packs**, non-unit scaling, BF16 Rubin kernels,
training, local fused-routing MegaMoE, MXFP4, mixed W4A8, and cross-node
communication are not part of this implementation. BF16 input staging currently uses Torch operations; its cost is included in
full-forward measurements.

## Correctness tests

The strict runner checks the GPU and compiler, rejects skips and empty suites,
reports OOMs as failures, and terminates all workers on timeout. Run it from
the repository root and keep each run in a separate persistent directory:

```bash
export CUTE_DSL_ARCH=sm_107a
: "${FI_RESULTS:?Set a fresh persistent results directory}"
python tests/moe_ep/qualify_sm107.py --suite all --world-size 4 \
    --output-dir "$FI_RESULTS/ep4"
```

`--suite all` includes portable checks, single-GPU tests, and distributed tests.
Use `--suite single` on a one-GPU host or `--suite multi` for distributed tests
alone. `--world-size` accepts 2, 4, or 8. Logs and JUnit results are saved for
every rank.

The shell runner provides the same tests:

```bash
bash tests/moe_ep/run_tests.sh oracle_sm107
NPROC_MULTIRANK=4 bash tests/moe_ep/run_tests.sh mega_sm107
```

`oracle_sm107` runs with `MEGA_NO_DIST=1`; `mega_sm107` uses `torchrun`.
`PYTHON` and `TORCHRUN` can override the executables. The strict runner covers
these files, so the shell commands are useful for debugging individual suites.

Coverage includes all three formats, early/late routing weights, separate and
in-kernel reduction, partial tiles, routing-vector tails, masked routes, idle
ranks, workspace reuse, and pooled layers with different weights captured on a
new stream. Boundary tests exercise one-CTA and two-CTA instructions, deeper K,
mixed clusters, bulk TMA stages, token-back modes, and clamp. Metadata tests
reject unsupported scalars, scale layouts, and unsafe geometry.

Run negative device-assertion cases in separate workers. After a CUDA failure,
terminate the whole job: collective free/finalize can hang if a peer has failed.

## Packaging and sanitizer checks

For an installed-wheel check, add `--installed-package`. The runner removes
the checkout from import resolution and records the imported package path.
Native installed-wheel and sanitizer results have not been collected for this
export. The following commands can be used when investigating memory or
synchronization errors:

```bash
python tests/moe_ep/qualify_sm107.py --suite single --filter router_vector_tails \
    --sanitizer-tool memcheck --output-dir "$FI_RESULTS/memcheck"
python tests/moe_ep/qualify_sm107.py --suite single --sanitizer-tool initcheck \
    --output-dir "$FI_RESULTS/initcheck"
python tests/moe_ep/qualify_sm107.py --suite single --sanitizer-tool racecheck \
    --output-dir "$FI_RESULTS/racecheck"
python tests/moe_ep/qualify_sm107.py --suite single --sanitizer-tool synccheck \
    --output-dir "$FI_RESULTS/synccheck"
python tests/moe_ep/qualify_sm107.py --suite multi --world-size 4 \
    --filter pooled_layers_graph --sanitizer-tool memcheck \
    --output-dir "$FI_RESULTS/ep4-memcheck"
```

Keep the raw reports, including NVSHMEM diagnostics. Longer stress runs,
disjoint EP groups, failure injection, and other routing distributions can be
added for the specific code or deployment being validated.

## Benchmarks and CI

[TUNING.md](../../flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/TUNING.md)
defines the two geometries, timing scopes, cache policies, and rank statistics.
Use compute/eager with L2 flushing for historical
comparisons and kernel/forward × eager/graph without flushing. Each point has
20 warmups, 50 timed iterations, and three process repetitions.

The manual `.github/workflows/moe-ep-sm107.yml` workflow accepts native runner
labels. Adding it to required CI needs a provisioned Rubin runner and an owner.
`run_tests.sh all` selects the Blackwell suites; use the SM107 targets for Rubin.
Engine integration and whole-model serving benchmarks are separate follow-ups.

Device-kernel fixes belong in the upstream kernel repository. Re-export them
and update `VENDOR.md`; keep `kernel_src/**/src/` byte-identical to that export.
