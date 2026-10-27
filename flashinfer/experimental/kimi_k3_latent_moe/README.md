# Experimental Kimi-K3 Stable LatentMoE front / tail projections (generated programs)

Both this API and its generated-program backend are experimental and may
change without backward compatibility. Calling the API explicitly opts into
the experimental feature and emits FlashInfer's experimental API warning.
There is no automatic backend selection.

`flashinfer.kimi_k3_latent_moe` computes, per rank and layer, the two dense
projection groups that surround the routed experts of `nvidia/Kimi-K3-NVFP4`
(`KimiSparseMoeBlock`, latent width 3584; flashinfer-ai/flashinfer#4568):

```
x [T, 7168] bf16
front:  logits     = FP32(x @ gate_weight.T)                       [T, 896]      router logits
        latent     = BF16(x @ down_weight.T)                       [T, 3584]     routed-expert input
        shared_act = SiTU(x @ shared_gate.T, x @ shared_up.T)      [T, 6144/TP]  shared-expert intermediate
tail:   y   = KimiRMSNorm(sum of the P routed partials)            [T, 3584]     FP32 normalise -> BF16 -> * bf16 weight, eps 1e-5
        out = BF16(y[:, cols] @ up_weight[:, cols].T + shared_act @ shared_down.T)   [T, 7168], one rounding of the fp32 sum
SiTU(g, u) = 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25)   (fp32 math, one bf16 rounding)
```

`cols` is this rank's `3584 / TP` latent slice (row-parallel up-projection);
with TP > 1 the caller all-reduces `out` afterwards.  Weights are the checkpoint's
`nn.Linear` `[out, in]` BF16 tensors in the serving layout (vLLM `KimiMoE` /
SGLang `KimiK3MoE`): `gate_weight`, `down_weight`, `norm_weight`, `up_weight`
replicated; the shared experts `MergedColumnParallel` (gate/up rows sharded,
passed as one `[2 * 6144/TP, 7168]` concatenation) + `RowParallel`
(`shared_down_weight` `[7168, 6144/TP]`).  Nothing is copied or re-packed.

## Route (mirrors the Cake production launcher)

| stage | tokens | program | launches |
| --- | --- | --- | --- |
| front | `T <= 128` | weight-streaming swapped-AB tcgen05 kernel (`decode:*`): 128-row weight tiles are the MMA A operand, the tokens the padded B operand (N in {8, 16, 32, 64, 128}), router / latent / shared SiTU tiles in one unified tile space; aligned 2-CTA cluster pairs when `2 x tiles` fits one wave (TP8; from N 16 both CTAs of a pair finalize half the values, `_sf`), one CTA per tile otherwise (TP1) | 1 |
| front | `T > 128` | persistent 2-CTA tcgen05 GEMM (`front:i<6144/TP>[e1]`; the `e1` instance streams the weights `evict_first` for `T <= 512`, at TP8 for `T <= 768`, and on 152-SM parts at TP8 for `T <= 16384` and at TP1 for `T <= 2048`), 256x256 pair tiles over the K-concatenated `[gate; down; shared_gate; shared_up]` rows with class-specific epilogues; the trailing wave's tiles run as two aligned K halves with fp32 partials and a fixed-order fix-up when that fits one wave (`front_split_plan`); on 152-SM parts the TP8 rows of 16 row tiles (`T` 1793-2048) and, on `sm_103a`, of 8 row tiles (`T` 897-1024) run the fold form (`front:i768e1fold`, `front_fold_plan`): one item per resident cluster, lanes of 8 (4) clusters splitting the flattened K iterations evenly, the raw fp32 accumulator carried across a lane boundary and the trailing-wave plan's split tiles kept, so the outputs are bitwise those of `front_split_plan`; it launches cooperatively, and a device that holds fewer co-resident clusters than its grid keeps the trailing-wave plan | 1 |
| tail | `T <= 128` | the same streaming kernel with the KimiRMSNorm fused into the launch (`decode:*_f*`): the epilogue warps normalise the routed rows while the TMA warp streams the shared-down segment; for the smallest T the normalised rows are written straight into a resident smem B operand (`_sb`, staged rows `_rs`), larger T use the global `y_workspace` protocol; from N 32 with one routed partial both CTAs of a pair finalize half the tokens (`_sf`), the TP1 tail arriving on the `y_workspace` counter as soon as its rows are written (`_ea`) and the TP8 tail at N 32 / 64 prefetching its first up-projection weight tiles into L2 during the wait (`_upf`) | 1 |
| tail | `T > 128` | one-pass RMSNorm kernel (`tail_norm:e<0|1>[pk]`, late / early PDL trigger chosen from the GEMM grid; the `pk` instance, for the TP8 rows of one routed partial from 2048 tokens, keeps the row as packed bf16x2 words) + persistent 2-CTA GEMM (`tail_gemm:tp<1|8>`) over `[up slice | shared down]` with trailing-wave stream-K, launched programmatic-dependent | 2 |

The host planner (`cake_backend.decode_front_plan`, `decode_tail_plan`,
`prefill_front_plan`, `front_split_plan`, `front_fold_plan`, `prefill_tail_plan`, `split_plan`) is the Cake planner re-implemented; every
plan names its physical kernel through a logical key registered in
`cake_jit.KERNELS` (one program per key, compiled for both SM100a and SM103a
from the same source; `cake_jit.SPECIALIZATIONS` adds the compile-line
constants of keys that share a program, e.g. the TP1 / TP8 front and tail
GEMMs).  Every token count from 1 to 16384, both tensor-parallel degrees and
any number of routed partials resolve to a registered program.  Every plan is
derived from the device's own architecture and SM count (`torch.cuda.get_device_properties(...).multi_processor_count`):
the decode grids are tile-bound (94 / 112 / 131 co-resident CTAs on every
qualified part) and the prefill GEMMs receive their resident-pair windows
(`num_items`, `full_items`, `sk_*`) as launch arguments, so one program serves
every qualified count.  `prepare_*` admits the SM counts listed under "Tested
physical configurations" (`cake_backend.QUALIFIED_SM_COUNTS`, 148 and 152) and
raises `NotImplementedError` naming the device's count otherwise: the decode
programs rely on every CTA of their grid being resident and the plan rules were
validated on those parts only.  `generated_program_available(device, stage, tp,
num_tokens, num_partials)` answers for the exact route `prepare_*` would
require (the decode tail instance and the TP8 prefill norm instance depend on the routed-partial count `P`).
Plans are memoised per shape and nothing is planned per launch, so a prepared
runner (or a CUDA Graph capturing it) replays for new values written into the
bound buffers (see "Concurrency" for the per-runner device state).

```python
import torch
from flashinfer.kimi_k3_latent_moe import (
    kimi_k3_latent_moe_front,
    kimi_k3_latent_moe_tail,
    prepare_kimi_k3_latent_moe_front,
    prepare_kimi_k3_latent_moe_tail,
)

T, tp, rank = 16, 8, 0
i_local = 6144 // tp
dev = torch.device("cuda")
x = torch.randn(T, 7168, device=dev, dtype=torch.bfloat16)
logits = torch.empty(T, 896, device=dev, dtype=torch.float32)
latent = torch.empty(T, 3584, device=dev, dtype=torch.bfloat16)
shared_act = torch.empty(T, i_local, device=dev, dtype=torch.bfloat16)
# gate_weight [896, 7168], down_weight [3584, 7168], shared_gate_up [2 * i_local, 7168] (gate rows then up rows)
runner = prepare_kimi_k3_latent_moe_front(x, gate_weight, down_weight, shared_gate_up, logits, latent, shared_act)
runner()  # or capture into a CUDA Graph

# tail: routed [P, T, 3584] partials, norm_weight [3584], up_weight [7168, 3584], shared_down [7168, i_local]
out = torch.empty(T, 7168, device=dev, dtype=torch.bfloat16)
y = torch.empty(T, 3584, device=dev, dtype=torch.bfloat16)
kimi_k3_latent_moe_tail(routed, norm_weight, up_weight, shared_act, shared_down, out, tp=tp, rank=rank, y_workspace=y)
```

## Concurrency

The programs keep two kinds of mutable device scratch beside the caller's
buffers, and the package owns them so that launches on distinct streams or
graphs never share any:

- **Control words** (the decode grid counter, the prefill stream-K arrival
  counters and the fused-norm semaphores / claim ticket / retirement count) must
  read zero before a launch, and every launch returns them to zero.  Each
  `prepare_*` call takes its own words (at most 1.25 KiB) from a per-device slab
  that is zeroed and synchronised once; they are never freed or reused, so a
  CUDA Graph that captured a runner stays valid after the runner is dropped.
  Prepare once per layer and shape, before capture: preparing under capture
  takes a free slab slot (no graph node), or, with the slab exhausted, zeroes
  new words inside the capture (one zero-fill kernel node in that graph).
- **Partials** (the FP32 stream-K workspace of split prefill plans: one
  `128 x block_n` slot per contributor CTA of every split tile, and two
  `128 x 256` slots per CTA of the front's fold form, written before they are
  read inside one launch).  An eager launch uses one workspace per (device,
  stream), allocated on that stream at its first split launch and grown to the
  largest split plan launched there; later launches on the stream reuse it in
  stream order (as PyTorch's caching allocator reuses a block on its stream),
  and distinct streams never share one.  The workspaces are held for the life
  of the process: one per stream that ran a split plan eagerly (TP1 `T = 1024`
  needs 29 MiB on a 148-SM part).  A launch under CUDA Graph capture takes a
  block of the graph's memory pool instead, once, at capture.  Whole plans and
  the decode route use no workspace; an out-of-memory error can surface at the
  first eager split launch on a stream, when its workspace grows, or at capture.

Contract:

- Distinct runners never share control words; runners launched on one
  stream share that stream's partial workspace in stream order, and launches on
  distinct streams, host threads, green contexts or CUDA Graphs never mix their
  control words or partials.
- A runner must not run concurrently with itself (its outputs, `y_workspace` and
  control words are its own): use one runner per concurrent stream.
- CUDA Graphs that share one PyTorch memory pool (`pool=`) must not replay
  concurrently (PyTorch's rule for any graph that allocates).
- `runner()` launches on the current stream of the runner's device, whichever
  device is current.
- The one-shot `kimi_k3_latent_moe_front` / `kimi_k3_latent_moe_tail` calls
  prepare per call: eager calls share one set of control words per device,
  stream and size (calls on one stream are stream-ordered), and a call under
  capture takes fresh words, which are never freed.  Prefer prepared runners
  under capture and where a PDL kernel that never waits on its predecessor can
  sit between two calls on one stream.
- A cooperative route (see "Residency" below) needs its whole grid resident at
  once, one CTA per SM.  Launched eagerly on a stream that
  offers fewer SMs (a smaller green context), it runs nothing and raises a
  `RuntimeError` naming the route and the SMs it needs; the CUDA runtime's last
  error is cleared, so later work in the process is unaffected.  Captured on
  such a stream, the launch fails the capture, again with no lasting error;
  note that `torch.cuda.graph` leaves its capture stream current when the
  capture fails, so restore the caller's stream before launching further work.

### Residency

Grid, launch and partition behaviour per route (`P` routed partials; on a
partition, such as a green context, that offers fewer SMs than a cooperative
grid, the launch is refused as above; an ordinary launch runs there in more
waves):

| route | keys | grid (CTAs) | launch | waits for other CTAs |
| --- | --- | --- | --- | --- |
| front, `T <= 128` | `decode:*_o3584_*` | 131 (TP1), 94 (TP8) | ordinary | no |
| front, `T > 128` | `front:i<6144/TP>[e1][n224]` | one CTA pair per work item; cluster launch control hands out the rest | ordinary | no unbounded wait: whichever CTA arrives last sums a split tile's halves |
| front, TP8 fold rows (152-SM parts) | `front:i768e1fold` | 144; 152 for `T` 897-1024 on `sm_103a` | cooperative | yes: the carries and partial runs of other lanes |
| tail, `T <= 128` | `decode:*_o7168_*_f*` | 112 | cooperative, except the staged-rows instances (`_sb`) | yes: the fused norm's grid counter (`_sb`: no) |
| tail, `T > 128`, unfused | `tail_norm:*` + `tail_gemm:*f0*` | norm row blocks; GEMM one CTA pair per work item, cluster launch control | ordinary; the GEMM programmatic-dependent | no |
| tail, `T > 128`, fused single wave (TP1) | `tail_gemm:tp1e1f1*` | at most the SM count, one work item per CTA pair | cooperative, programmatic-dependent | yes: latent rows that other CTAs normalise |
| tail, `T > 128`, fused claimed (TP1) | `tail_gemm:*f1*ci` | one CTA pair per resident cluster, at most one per work item: up to every SM | cooperative, programmatic-dependent | yes: rows, the item ticket and the retirement count |

A cooperative grid starts only with all of its CTAs resident, one per SM, so
it does not overlap work that keeps those SMs busy: the claimed tail takes up
to every SM, the fold rows 144 or 152 of them and the decode tail 112.  None of
the cooperative programs uses cluster launch control.

## Tested physical configurations

The programs are compiled per architecture; the host plan is derived per device.
Qualification (GPU correctness at the contract tolerances, bit-exact
re-launch and CUDA-graph replay, complete-call CUPTI cold-L2 benchmarks of the
exported route against the production launcher and the contract baseline) was
run on the configurations below.  Other SM counts are refused by `prepare_*`.

| device | architecture | SMs | driver | status |
| --- | --- | ---: | --- | --- |
| NVIDIA B200 | `sm_100a` | 148 | 580.82.07 | qualified (export protocol, every denominator row) |
| NVIDIA B300 SXM6 | `sm_103a` | 148 | 580.126.09 | qualified (export protocol, every denominator row) |
| NVIDIA GB300 (NVL72) | `sm_103a` | 152 | 580.159.03 | qualified (export protocol, every denominator row of the 152-SM configuration; complete-call speedup over the torch / cuBLAS chain measured per row) |
| NVIDIA GB200 (NVL72) | `sm_100a` | 152 | not recorded (no export stage) | admitted: host plan and program registry verified for every route on 152 SMs (the same programs as GB300); on GB200 hardware the package GPU tests pass and the complete-call contract rows were timed against the torch / cuBLAS chain (front: every row faster; tail: see the delivery summary); no export stage of this delivery |

## Limits

- SM100 (B200 / GB200) and SM103 (B300 / GB300) only: tcgen05 / TMEM / TMA / 2-CTA MMA programs,
  on the SM counts of the table above.
- TP 1 and TP 8 (no expert parallelism); TP 12 (7168 / 12 is not a multiple of
  64 for the tail slice) is out of scope (flashinfer-ai/flashinfer#4542).
- Model-layout weights only; the chunk-major packed-weight variant of the
  Cake kernels (+3-8 % on decode rows, one extra weight copy per rank) is not
  exported.
- Exactly the shapes of the validated row set are supported: `T` in
  `{1, ..., 128}` for the decode programs (any count, padded to the next N)
  and `T > 128` for the prefill programs (any count; validated up to 16384).

## Files

- `cake_backend.py`: host planner, operand validation, launch binding, runners.
- `cake_jit.py`: `MODULES` / `KERNELS` registry populated by the Cake export;
  JIT specs of the generated programs (`gen_jit_spec`, per-arch nvcc flags).
- `csrc/cake_kimi_k3_latent_moe/<arch>/*.cu`: generated device + host binding
  translation units (do not edit; `csrc/.clang-format` disables formatting).
- `tests/experimental/test_cake_kimi_k3_latent_moe.py`: host-plan unit tests
  and GPU correctness against the FP32/BF16 torch reference for both stages,
  TP 1 / 8, `T in {1, 8, 16, 128, 256, 4096}` (bit-identical re-launch and
  CUDA-graph replay; `y` byte-exact).
