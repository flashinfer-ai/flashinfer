# Experimental Ulysses architecture follow-ups

Owner: @tiffany940107. Tracking: [#5316](https://github.com/flashinfer-ai/flashinfer/issues/5316).
This is a follow-up to the stable head-chunk primitives in
[#5027](https://github.com/flashinfer-ai/flashinfer/pull/5027), not a replacement.
Initial base: `0ef005a2cdbd9cbcb70a818f66655205c8a7558e` (main).

## Included versus deferred

Architecture labels below describe the experiments' origin, **not** a claim that
every architecture has been validated with this port.

| Origin | Included implementation | Not included / acceptance still needed |
| --- | --- | --- |
| SM100 | BF16 distributed-Q/O FA4 runner, peer-address checks, remote Q reads, owner-O writeback, O-TMA and padding-query scheduler extent; pinned external kernel patch | Not FP8 FA4; not a general mask/varlen backend; B200 U2/U4/U8 correctness, lifetime and paired performance reruns required |
| SM90 | BF16-to-E4M3 per-head quantization fused with destination-major QKV packing; new complete examples using existing FlashInfer FA3 BF16/FP8 | No new FA3 kernel implementation, automatic global-scale reduction or FP8 output communication; Hopper tests required |
| SM120 | Coarse producer and K-mean primitives; new complete examples using existing FlashInfer BF16 BSA, FP8 FMHA and Sage kernels | No new Sage/BSA kernel implementation, SGLang-specific norm/RoPE-to-wire or CE transport; multi-GPU and model-quality gates remain |

The [native-kernel examples](ulysses_native_attention_examples.md) now connect
ordinary/whole-QKV/head-chunk Ulysses to **existing in-tree** attention kernels
on SM90, SM100 and SM120. These examples use BF16 communication and do not need
the external FA4 patch. The SM100 remote-Q/owner-O experiment remains separate;
the native SM100 example uses TRT-LLM Gen, not that distributed FA4 kernel.

These are explicit experimental APIs under
`flashinfer.comm.ulysses_experimental`. Implementations live under
`flashinfer.experimental.ulysses`. Importing the stable package does not load the
optional FA4 dependency. No AOT registration, automatic backend dispatch, default
communicator changes or process-group creation is added to the library.

## Safety and integration boundaries

- The framework owns TP/EP/CP/SP membership and passes its explicit Ulysses group.
  Do not pass a tensor-parallel or expert-parallel group merely because its size
  matches. Nothing here composes or changes those parallelism policies.
- All ranks must agree on schedules, geometry, scales, backend and metadata.
  FA4 preparation performs a collective preflight before symmetric allocations.
  Hot-path invalid inputs or asynchronous failures cannot safely trigger a
  rank-local fallback. Frameworks must agree on fallback **before** launch.
- No reentrancy or overlapping requests on one prepared object's storage.
  Producers and K-mean bind to their first calling CUDA stream. Consumers on
  other streams must record/wait on events and finish before storage is reused.
  FA4 outputs, producer views and K-mean views alias reusable workspaces.
- The APIs are inference-only building blocks. Frozen projection weights are
  copied/repacked at preparation; rebuild if weights change. Grouped GEMM may
  change rounding or lose GEMM efficiency. It is not bitwise equivalent to a
  whole QKV GEMM, and is not assumed faster.
- A schedule partitions **H/U heads per destination**, not all H heads.
  Each producer group contains the selected band for **every** destination,
  and all three Q/K/V planes. Uneven bands are supported. With H/U=1, use one
  group and do not expect head-pipeline overlap. When H is not divisible by U,
  reject instead of silently repartitioning.
- The benchmark creates its own input/output process groups and streams; it is
  reference scheduling code, not a new public distributed runtime.

## SM100: what is different from ordinary Ulysses?

Ordinary Ulysses packs and exchanges Q/K/V, runs attention on a rank's heads over
the global sequence, then exchanges O back to sequence owners. The distributed
FA4 runner stages Q at sequence owners, pushes K/V to head owners, and lets the
patched persistent attention kernel read Q remotely and write O directly to the
sequence owner's window. The head owner computes; the sequence owner stores
the corresponding input/output rows. K/V communication still exists. This is
not simply a head-chunk stream scheduler and does not eliminate all movement.

The optional `sched_used_q` extent avoids work for disposable padding-query rows;
it does not shorten valid KV context. O-TMA is a separate output-write mechanism.

Admitted contract is intentionally narrow:

- SM100, full NVLink, NVSHMEM symmetric-memory backend; full NVLink is an explicit
  caller attestation, not an automatic topology probe.
- BF16, physical batch 1, H=56, D=128, non-causal prefix attention.
- U2/U4 physical S=37888; U8 physical S=38912, query scheduling extent 37888.
- `0 < used_seqlen <= 37888`; **only** the valid global prefix output is promised.
  Query rows at or after `used_seqlen` must be discarded, including possibly
  stale unscheduled rows. This is not complete `cu=[0,used,S]` output equivalence.
- Input Q/K are already normalized and rotated by the framework. No QK norm,
  RoPE, output projection, CFG, packed multiple samples or arbitrary masks.
- Inputs must match the prepared dtype/device/shape on every rank. Do not alias
  internal staging windows. New-device/new-stream lifecycle and CUDA Graph
  capture of this port remain explicit acceptance items, not advertised features.

### External dependency setup

The runner needs a **dedicated** FlashAttention checkout at
`c2006099f3ff03de187f4e1b27e756fe6df482ba`, plus the bundled cumulative patch.
The patch includes the original distributed kernel addition and subsequent
O-TMA/peer-layout/scheduler changes; it does not require a private intermediate
commit. It never edits
the installed dependency at import or runtime. Check/apply explicitly:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git /path/to/flash-attention-ulysses
git -C /path/to/flash-attention-ulysses checkout c2006099f3ff03de187f4e1b27e756fe6df482ba
python scripts/apply_ulysses_fa4_patch.py /path/to/flash-attention-ulysses
python scripts/apply_ulysses_fa4_patch.py /path/to/flash-attention-ulysses --apply
export PYTHONPATH="$PWD:/path/to/flash-attention-ulysses:$PYTHONPATH"
```

Install the pinned checkout's CuTe/CUDA Python and NVSHMEM prerequisites in the
test environment. The script checks HEAD and cleanliness and defaults to a
read-only patch check. The copied patch modifies FlashAttention's BSD-3-Clause
code; its license is retained at
`flashinfer/experimental/ulysses/patches/FLASH_ATTENTION_LICENSE`.
The cumulative patch uses zero-context hunks (applied with `--unidiff-zero`) to
avoid whitespace-only context lines being rewritten by repository hooks.
Application to a clean pinned checkout was checked locally; all three resulting
source files matched the historical implementation byte-for-byte. That is a
source reproducibility check, not a successful B200 kernel run.
The external patch needs an upstream ownership decision before graduation.

On an explicitly verified homogeneous full-NVLink B200 node:

```bash
NVSHMEM_DISABLE_NVLS=1 torchrun --standalone --nproc-per-node=2 -m pytest \
  tests/experimental/ulysses/test_sm100_distributed.py -q
```

Repeat with 4 and 8 ranks. This validates valid-prefix output against ordinary
Ulysses plus stock FA4 and reuses the prepared runner with new inputs. The test
is skipped in ordinary single-process CI; such a skip is not SM100 validation.

## FP8 quantization/packing contract

`pack_ulysses_qkv_fp8(q,k,v,scales,world_size=U,out=out)` accepts BF16
`[S_local,H,D]`, including positive row/head strides and contiguous D, and writes
E4M3 `[U,S_local,H/U,3D]` into caller-owned contiguous storage.

Each of the three FP32 `[H]` scales is a **dequantization** factor. The caller
must agree finite positive scales globally, e.g. MAX-all-reduce each per-head
amax across the Ulysses group, clamp to `1e-12`, then divide by 448. Neither the
reduction nor its cost is hidden in this primitive. FP32 rounded division and
clamp precede E4M3 conversion; approximate division failed the long-shape
byte-exact reference check and is deliberately not used.

Transport may reinterpret E4M3 storage as bytes if its backend lacks FP8 dtype
support. The consumer must interpret scales/head order consistently and supply
them to its FP8 attention backend. The new native examples quantize **after BF16
communication**, using a fixed scalar recipe; they do not demonstrate transport
of this per-head FP8 payload or guarantee model quality. Quantization remains lossy
relative to BF16 even when pack output matches the quantized reference exactly.

## SM120 producer and K-mean

`prepare_ulysses_producer` snapshots NK BF16 `[3*H*D,K]` weights in Q/K/V plane
order. `produce(x, group_index)` runs one prepared GEMM and returns three
`[1,S_local,U*head_count,D]` views. A framework may communicate a ready group
while subsequent projection/attention work proceeds. Repacking/setup is outside
steady-state timing; storage footprint includes the packed weight copy and
per-group output buffers.

The K-mean helper copies each incoming chunk into persistent full-H scratch and
runs the same whole-head reduction geometry. This avoids changing the reduction
geometry solely because a Sage head schedule changes. It does not guarantee
identical reductions across PyTorch versions, add a Sage backend, or by itself
reduce latency. Extra memory and repeated reductions are explicit tradeoffs.

Runnable projection/head-pipeline comparison (choose schedule for H/U):

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
torchrun --standalone --nproc-per-node=2 benchmarks/comm/bench_ulysses_grouped_producer.py --schedule 14,14
torchrun --standalone --nproc-per-node=4 benchmarks/comm/bench_ulysses_grouped_producer.py --schedule 7,7
torchrun --standalone --nproc-per-node=8 benchmarks/comm/bench_ulysses_grouped_producer.py --schedule 3,4
```

The baseline and candidate use the same head-pipeline attention/communication;
only whole versus grouped QKV projection changes. Timing is X to local all-head
O, **excluding** QK norm, RoPE and output projection. This is not the historical
SGLang full-sublayer benchmark. Samples alternate AB/BA and use max-rank latency.
No small-head or bandwidth/compute tradeoff policy is enabled automatically.
The producer benchmark also accepts `--attention sm120-sage` (or any native
example backend), replacing its default SDPA compute with the same in-tree
kernel adapter used by the complete QKV-to-O examples.

## Current validation, not historical performance

Current hardware: **one RTX PRO 6000 Blackwell Workstation Edition**, PyTorch
`2.15.0.dev20260915+cu134`. It is not the historical multi-GPU Server Edition
setup. Rank-shaped layout tests simulate destinations on one GPU; the Gloo
preflight rejection test uses two CPU processes. Neither is a multi-GPU test.

```bash
python -m pytest tests/experimental/ulysses tests/comm/test_ulysses_head_chunk.py -q
PYTHONPATH="$PWD:$PYTHONPATH" python benchmarks/comm/bench_ulysses_fp8_pack.py --rows 18944 --heads 56 --world 2 --iterations 50
```

| Current check | Baseline median | Candidate median | Interpretation |
| --- | ---: | ---: | --- |
| Quant+pack, local S=18944,H=56,D=128, simulated U=2, 50 AB/BA pairs | 8.688432 ms | 0.861968 ms | 10.080x **versus multi-op PyTorch reference with temporaries**, precomputed scales; not an attention/communication/E2E speedup |
| Producer pipeline smoke, U=1,S=128,H=8,D=32,K=64, groups=[4,4], 5 pairs | 0.544640 ms | 0.653216 ms | Candidate slower; smoke only, no inter-GPU overlap; not a performance recommendation |

No current SM90, SM100 or multi-GPU speedup is claimed. Historical local results
motivated this port but cannot replace rerunning the new code and dependencies.
Ready-for-review gates include those reruns, numerical checks, stream/reuse
stress, a realistic projection workload, and no regressions for selected shapes.

## Graduation

Target the next minor release **after** acceptance gates pass. First review
checkpoint: 2026-10-15; no promised stable release until ownership and hardware
validation are resolved. The tracking issue records the gates. Framework-level
SGLang/TRT-LLM policy, model quality and external Sage/BSA integration are separate
workstreams, not implicit promises of these operator primitives.
