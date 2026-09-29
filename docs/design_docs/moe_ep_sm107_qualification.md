# SM107 MegaMoE validation

**Scope**: Validation requirements, correctness tests, and qualification procedures for the SM107 MegaMoE backends.

The Rubin backends support NVFP4, MXFP8 E4M3/E5M2, and MXFP4 weights with
MXFP8 E4M3 activations through
`MoEEpLayer`, with BF16 combine and output. The kernel comes from CuTe DSL
MegaMoE `1667b47a`; [VENDOR.md](../../flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/VENDOR.md)
records the full source and exporter revisions.

## Correctness coverage

At FlashInfer `0f710df8e7e1843a91009d679a9f4deed47352f3`, native tests passed
187 host/CUDA checks, 72 single-GPU cases, and 34 distributed cases per rank
at EP2, EP4, and EP8, with no skips. Coverage includes SiTU, prequantized
weight ingestion, non-unit NVFP4 scaling, pooled workspaces, and graph replay.

The subsequent E8M0 reference-decoder fix has a separate regression covering
all 256 encodings, typed and raw-byte scales, and strided storage. The native
qualification counts above refer to the stated commit.

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
| Public layer geometry | H multiple of 64 for NVFP4, 128 for MX formats; I multiple of 64, or 128 for MXFP4/MXFP8 |
| Canonical weights | Floating `[E_local, 2I, H]` and `[E_local, H, I]`; canonical gate rows precede up rows |
| Prequantized weights | Canonical gate-then-up packed data and typed, unswizzled block scales; see the ingestion contract below |
| Transformed weights | K-major physical storage, 16-byte aligned, exact typed scale planes; preserve physical layout when concatenating experts |
| BF16 inputs | `quantize_input=True`; staging quantizes on the caller's stream |
| Prequantized inputs | Matching data format; scales are `[T, H/16]` E4M3 for NVFP4 or `[T, H/32]` E8M0 for MXFP8; uint8 scale storage is interpreted as raw bytes. Pass the logical scale columns, excluding internal workspace communication padding. |
| Activation | SwiGLU by default; SiTU requires both positive, finite beta parameters and excludes gate/up clamps |
| Normalization | NVFP4 accepts positive, finite local-expert FP32 vectors for `fc1_alpha`, `fc2_alpha`, and `fc1_norm_const`. Per-call values override config defaults; if both omit a value, staging uses one. MXFP8 and MXFP4/MXFP8 do not accept these scalars. |
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

BF16 Rubin kernels, training, local fused-routing MegaMoE, FP4 activations
with E8M0 scales, and cross-node communication are not part of this implementation. BF16 input
staging uses Torch operations; its cost is included in full-forward measurements.

## SiTU and prequantized weights

The Rubin configs accept `activation="situ"`, `situ_beta`, and
`situ_linear_beta`. Both betas are required, matching the vendored kernel:

```python
gate_act = situ_beta * tanh(gate / situ_beta) * sigmoid(gate)
up_act = situ_linear_beta * tanh(up / situ_linear_beta)
intermediate = gate_act * up_act
```

These constants are uniform across experts. They are part of workspace and
kernel identities. `activation="swiglu"` remains the default and does not
accept SiTU betas. SiTU excludes `gate_up_clamp` and the MXFP8
`activation_clamp` alias.

Use `PrequantizedMoEWeights(w13, w2, w13_scale, w2_scale)` as the layer's
`weights` argument, with `MegaConfig.preprocess_weights=True` (the default).
Here E is the number of local experts:

| Format | w13 / w2 | w13_scale / w2_scale |
|---|---|---|
| NVFP4 | `[E, 2I, H/2]` / `[E, H, I/2]`, uint8 or float4_e2m1fn_x2 | `[E, 2I, H/16]` / `[E, H, I/16]`, float8_e4m3fn |
| MXFP4/MXFP8 | `[E, 2I, H/2]` / `[E, H, I/2]`, uint8 or float4_e2m1fn_x2 | `[E, 2I, H/32]` / `[E, H, I/32]`, float8_e8m0fnu |
| MXFP8 | `[E, 2I, H]` / `[E, H, I]`, float8_e4m3fn or float8_e5m2 matching `kind` | `[E, 2I, H/32]` / `[E, H, I/32]`, float8_e8m0fnu |

FC1 gate rows precede up rows in both data and scales. FP4 packs the even K
element into the low nibble. The backend interleaves gate/up rows, creates
K-major views, and pads/swizzles scales without numerical conversion. Scale
tensors must have the stated dtype; explicitly reinterpret raw scale bytes
with `.view(dtype)` first. Do not pass already interleaved/swizzled tensors
through this canonical ingestion path; those belong in `transformed_weights`.

Select the mixed format with `Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig`.
The backend name lists activation, weight, and output formats in that order;
the upstream quantization tag is `mxfp4_mxfp8`. Both FC1 inputs and the
FC1-to-FC2 intermediate use MXFP8 E4M3 with block-32 E8M0 scales. Weights
remain packed E2M1 throughout ingestion. Floating-point weight preprocessing
also produces this layout, using power-of-two scales and nearest-even E2M1
rounding.

For [K3 routed experts](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json),
use H=3584, I=3072, 896 total experts, top-k=16,
`activation="situ"`, `situ_beta=4.0`, and `situ_linear_beta=25.0`.
Set `apply_topk_in_fc1=False` to apply routing weights after FC2; the default
applies them before intermediate quantization and can round differently.
The geometry regression uses synthetic checkpoint-format tensors. It does
not cover the model's latent projections, shared experts, or engine integration.

## NVFP4 normalization

`input_norm_const` controls BF16 input staging. If a quantizer uses
normalization n, its packed data multiplied by its block scales approximates
`n * original`. For input normalization nx, per-expert weight normalizations
nw1/nw2, and intermediate normalization nh, the corresponding settings are:

```python
cfg = Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
    intermediate_size=I,
    top_k=top_k,
    activation="situ",
    situ_beta=1.25,
    situ_linear_beta=0.75,
    input_norm_const=nx,
    fc1_alpha=(nx * nw1).reciprocal(),  # local-expert CUDA FP32 vectors
    fc1_norm_const=nh,
    fc2_alpha=(nh * nw2).reciprocal(),
)
layer = MoEEpLayer(
    bootstrap=bootstrap,
    fleet_params=fleet_params,
    weights=PrequantizedMoEWeights(w13, w2, sf13, sf2),
    backend=MegaConfig(megakernel=cfg),
)
```

FC1 alpha is applied before activation; intermediate normalization is applied
during FC1-to-FC2 quantization; FC2 alpha is applied before the BF16 output
conversion. Prequantized weight block scales remain unchanged. BF16 weight
preprocessing uses unit normalization, so nw1 and nw2 are one for that path.

Per-call tensors on `MoEEpTensors` override the matching config tensor. An
omitted override restores the config default, or one if the config also omits
it. Scalars have shape `[E_local]`, are contiguous FP32 tensors on the input
device, and follow local expert order. Their device assertions remain active
in graph replay. Values are copied into stable workspace buffers on every
forward, including empty batches, so sequential pooled layers do not share
stale values. Update captured source tensor contents in place for graph replay.

For `quantize_input=False`, input bytes and scales are copied unchanged;
`input_norm_const` does not requantize them. The caller must supply alphas
matching the producer's normalization. nx must be common across source ranks:
one destination-expert alpha cannot correct different source-rank global
normalizations. These scalars do not represent independent gate/up global
scales; a checkpoint adapter must account for that distinction.

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

Coverage includes the block-scaled formats, early/late routing weights, separate and
in-kernel reduction, partial tiles, routing-vector tails, masked routes, idle
ranks, workspace reuse, and pooled layers with different weights captured on a
new stream. Boundary tests exercise one-CTA and two-CTA instructions, deeper K,
mixed clusters, bulk TMA stages, token-back modes, and clamp. Metadata tests
reject unsupported scalars, scale layouts, and unsafe geometry. Activation and
weight-ingestion tests compare against canonical-layout Torch math, including
non-unit per-expert scaling, BF16/prequantized inputs, and pooled SiTU graphs.

Output comparisons require relative L2 error below 0.02 for MXFP8 and
MXFP4/MXFP8, and 0.06 for NVFP4. The test metric is `norm(output - reference) / max(norm(reference),
1e-6)`. These are aggregate acceptance limits, not per-element error bounds.
The references share quantization and unpacking helpers with preprocessing;
the canonical-weight reference independently checks the physical weight
transform. Activation evaluation, accumulation, intermediate requantization,
and expert-output reduction can round differently from the kernel. These
tests do not establish bitwise parity with upstream quantizers or model quality.

Layout and prequantized-ingestion tests compare bytes. The E8M0 decoder test
checks every finite encoding against its exact FP32 value and checks NaN
separately. In particular, byte 0 decodes to `2**-127` and byte 255 to NaN.

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
