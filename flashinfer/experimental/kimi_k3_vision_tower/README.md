# Experimental Kimi-K3 vision tower (Cake backend)

Both the API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection. Tracking: flashinfer-ai/flashinfer#4568 (Kimi-K3 vision
tower + PatchMergerV2 kernels).

`flashinfer.kimi_k3_vision.kimi_k3_vision_tower(pixel_values, grid_thws,
weights, out=None)` runs the complete vision path of `nvidia/Kimi-K3-NVFP4`
(`modeling_kimi_k3.py`: `MoonViT3dPretrainedModel` with `merge_type =
sd2_tpool` + `PatchMergerMLPV2`) on SM100 (B200) and SM103 (B300/GB300) as a
sequence of generated tcgen05 programs:

| Stage | Program | Math (BF16 tensors, FP32 accumulation) |
| --- | --- | --- |
| `patch_embed` | `gemm:pos_sqxw:*` | `x = bf16(pixels[T, 588] @ Wpe^T) + pos_rows` (Conv2d 14x14/14 as a GEMM, bilinear-resized 64x64 positional table + sincos time rows) + handoff `xw = bf16(x * norm0[0])`, `stats = rowsumsq(x)` |
| `layer_norm_qkv_rope` (x27) | `gemm:norm_qkv_rope:*_cs` | `q, k, v = RoPE2D(bf16((xw @ Wqkv^T) * rstd(x)))`: the RMSNorm weight applied on the activation side by the producer (`xw`), row `rstd` from the FP32 `stats` handoff, original `Wqkv`, interleaved-pair 2-D RoPE in the epilogue from the packed f16x2 `(cos, sin)` table of the plan (`pack_rope_table`; FP32 rotation), head-split outputs |
| `layer_attention` (x27) | `attention:<layout>[:wide][:split]`, layout = `tiles2` / `tiles1` / `ring3` | packed-varlen noncausal attention per `grid_thw` segment, 12 heads x 128, FP32 softmax, one BF16 rounding; `ring3` = the one-tile layout with the shared-O three-deep score ring (`cake_backend.ring3_selected`: every one-tile row on SM100, 576..10764-token segments on SM103); `:wide` = the wide unit table with the next-unit prefetch on short rows (`unit_prefetch_selected`: T <= 1656 on SM100, T <= 2552 on SM103); `:split` = the KV-split build that writes the tail-round units' K/V parts as FP32 partials (`cake_backend.select_layout_and_split`) |
| `layer_attention_merge` (x27, split rows only) | `attention_merge` | exact fixed-order merge of the K/V parts of every split unit (`softmax(m, l)` rescale in FP32, one BF16 rounding into `attn_out`); launched right after the attention kernel when the plan has `num_merge_units > 0`, on SM100 only (the SM103 split builds merge the parts in the kernel) |
| `layer_out_proj` (x27) | `gemm:residual_wo_sqxw:*` | `x += bf16(a @ Wo^T)` + handoff `xw = bf16(x * norm1)`, `stats` |
| `layer_norm_fc0_gelu` (x27) | `gemm:norm_gelu:*` | `f = bf16(gelu_tanh(bf16((xw @ Wfc0^T) * rstd(x))))`, original `Wfc0` |
| `layer_fc1` (x26) / `layer_fc1_last` (x1) | `gemm:residual_fc1_sqxw:*` / `gemm:residual_fc1:*` | `x += bf16(f @ Wfc1^T)` + handoff `xw = bf16(x * norm0[l+1])`, `stats`; the last layer runs the plain form (the merge reads `x` directly) |
| `final_norm_merge` | `merge` | `m = mean_t(bf16(RMSNorm(x, final_norm)))` over 2x2 spatial windows, `[N, 4096]` |
| `merger_gemm0` | `gemm:gelu_erf:*` | `h = bf16(gelu_erf(bf16(m @ Wp0^T)))` |
| `merger_gemm1` + `merger_rmsnorm_apply` | `gemm:rmsnorm:*`, `rmsnorm_apply` | `out = bf16(RMSNorm(bf16(h @ Wp1^T), post_norm, eps = 1e-5))` |

`RMSNorm` uses `eps = 2^-7` (`nn.RMSNorm(eps=None)` on BF16) except the
projector's post norm (`1e-5`). `*` is the production tile configuration the
host selects per token count (`cake_backend.select_tile_config`, the Cake
`kimi_k3_vision_gemm` policy: 128x64 / 128x128 single-CTA tiles and a split-K
cluster for small `T`, the single-CTA 128x128 eight-warp tile `s_e8_pf` for
the residual GEMMs while the pair grid underfills the machine (out-proj to
6144 rows, FC1 to 2304), 256x256 / 256x128 2-CTA pair tiles above (the
out-proj pair tile is the TMA-epilogue twin `m_tma1` up to 8191 rows and the
256x256 eight-warp TMA-epilogue tile `l_e8_tma1x` from 8192 rows,
`cake_backend.WO_PAIR_BANDS`; the FC1 forms take that tile with its half-N
tail, `l_e8_tma1x_h`, at the token counts of `cake_backend.FC1_TMA_WINDOWS`);
the residual / pos
forms run their packed bf16x2-epilogue twins `*_p` / `*_pf` with the
pre-mainloop residual prefetch, the QKV GEMM its packed-RoPE-table twins
`*_cs`; the half-N tail twins `*_h` (`cake_backend.HALF_TWIN`,
`half_tail_split`) run the tiles past the last full persistent round as
half-width items whenever they fit one half-round with margin, bitwise
identical per output element to the base tile; on the architectures of
`cake_backend.WEIGHT_PF_LEAD_ARCHES` the PDL_EARLY QKV, FC0 and FC1 forms up
to 256 rows run the `*_wpl` twin of their tile, which prefetches the weight
boxes past the first ring fill into L2 before the grid-dependency wait (on
`cake_backend.QKV_T576_ARCHES`, SM103, the QKV GEMM of a 576-token batch runs
the same twin of its half-N tile, `s_cs_h_wpl`), and on
those of `cake_backend.A_BOX_ARCHES` the merger GEMMs up to 64 merged rows run
the `xs_ab` / `s_ab` twin, which loads a 64-row A box: same launch, same stored
rows; on `cake_backend.READY_COUNTER_ARCHES` (SM103) a batch inside
`cake_backend.READY_COUNTER_WINDOWS` (8192 tokens) whose schedule model lets a
consumer start early (`cake_backend.ready_counter_edges`) runs the encoder's
GEMM -> GEMM edges (FC1 -> next QKV, out-proj -> FC0, FC0 -> FC1) on the
ready-counter twins `*_rc*`, which replace the grid-dependency wait of the A
loads by per 128-row block counters, so a consumer's first items start while
its producer finishes its last ones: same launch, same stored rows); the
attention unit layout
and the automatic tail split are chosen per `grid_thws` batch from the
split-aware LPT makespans of both layouts
(`cake_backend.select_layout_and_split`: the `units mod G` cheapest units of
the tail round are cut into `k <= 8` K/V parts when that shortens the
makespan), the persistent grid is trimmed to the smallest cluster count with
the same makespan where that frees at least 10 clusters (SM103;
`cake_backend.shape_grid`), and the one-tile layout's kernel form follows the
architecture and the longest segment (`cake_backend.ring3_selected`).

Registered programs and the contract denominator: `REQUIRED_KERNEL_KEYS`
lists every logical kernel the plan rule can select per architecture (the
GEMM tile / PDL census, the attention-form census over single segments up to
70016 tokens and the exported rows, the merge kernels).  The export registers
the programs its rows launch: the 22 rows of the evaluation contract (the
acceptance denominator) plus twelve export-only *coverage* rows
(`COVERAGE_ROW_GRIDS`; validated bitwise against the source tower, not part
of the acceptance geomean) that reach the keys no contract row exercises
(for example half-N twins at token counts outside the rows, the out-proj
`m_tma1` tile below 8192 rows, the plain two-tile form on SM100, the plain /
split one-tile forms on SM103) -- so every reachable key is
registered and `UNCOVERED_KERNEL_KEYS` is empty.  The check stays: a
`grid_thws` batch whose plan resolves to an unregistered key raises
`NotImplementedError` naming the missing keys -- it is never served by another
binary.

Programmatic dependent launch: the registered programs follow the Cake
production PDL default (`KIMI_K3_VISION_TOWER_PDL`, on since the round-2
route), so the shared launcher launches every kernel with the
programmatic-stream-serialization attribute (owned by the generated launcher
source, not by this host code): the kernel
runs its mbarrier / TMEM prologue while the previous launch drains, waits on
`griddepcontrol.wait` before touching a route buffer and signals
`launch_dependents` before its epilogue. The host launch sequence is the same
either way, the attribute is captured into the CUDA graph as a programmatic
dependency edge, and the numerics are identical.

| Tensor | Shape | dtype |
| --- | --- | --- |
| `pixel_values` | `[T, 3, 14, 14]` normalized patches in packed `grid_thws` order (`t` slow, then `y`, then `x`), contiguous, 16-byte aligned | bfloat16 |
| `grid_thws` | host list of `(t, h, w)`; `1 <= t <= 4`, `h`, `w` even and `<= 512`; `T = sum t*h*w` | int |
| `weights` | `nn.Linear` `[out, in]` BF16 parameters without biases (below) | bfloat16 |
| `out` | `[N, 7168]`, `N = sum (h/2)*(w/2)` (optional, caller-owned) | bfloat16 |

`weights` is a dict: `patch_proj [1024, 588]` (the Conv2d weight flattened),
`pos_emb [64, 64, 1024]`, `time_weight [4, 1024]`
(`cake_backend.sincos_time_table()`), `final_norm [1024]`, `merger_proj0
[4096, 4096]`, `merger_proj1 [7168, 4096]`, `post_norm [7168]` and `layers`
= 27 x `{"norm0": [1024], "wqkv": [4608, 1024], "wo": [1024, 1536], "norm1":
[1024], "fc0": [4096, 1024], "fc1": [1024, 4096]}`. Prepare it once per model
with `prepare_kimi_k3_vision_weights` (patch-projection padding; every
other parameter must be a contiguous bf16 tensor and is referenced, not copied; the RMSNorm weights are not folded into the GEMM weights: a folded
`bf16(W * w_norm)` is a fixed weight perturbation the HF chain does not have,
so the norm weight is applied on the activation side, `xw = bf16(x * w_next)`,
by the residual epilogues); the dict form is prepared on every call.

```python
import torch
from flashinfer.kimi_k3_vision import kimi_k3_vision_tower, prepare_kimi_k3_vision_tower
from flashinfer.experimental.kimi_k3_vision_tower.cake_backend import prepare_kimi_k3_vision_weights

prepared = prepare_kimi_k3_vision_weights(weights)          # once per model
grid_thws = [(1, 56, 74), (4, 52, 92)]                       # one 1024x768 image + one 4-frame 720p group
pixels = ...                                                 # bf16 [T, 3, 14, 14], T = 56*74 + 4*52*92
out = kimi_k3_vision_tower(pixels, grid_thws, prepared)      # bf16 [N, 7168], N = 28*37 + 26*46

# Serving form: plan once per grid_thws batch, launch without allocation, capture into a CUDA graph.
runner = prepare_kimi_k3_vision_tower(pixels, grid_thws, prepared, out)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    runner.launch()
graph.replay()                                               # new pixel values in the same buffer
```

`prepare_kimi_k3_vision_tower` derives everything a serving runtime keeps per
`grid_thws` batch (`cu_seqlens`, RoPE `cos`/`sin` and their packed f16x2
table, positional rows, the merge table, the attention segment plan, unit
table and merge table, the tile selection and all workspaces incl. the FP32
partial workspace of a split plan) and binds the 27 x 5 + 5 launches (27 x 6
+ 5 on an SM100 split plan: the attention merge follows every attention
launch);
`launch()` performs no
allocation and no host synchronization; when it raises between two launches
(an interrupt or an error on the host), it first enqueues the zeroing of the
ready counters of a counter-gated plan on the same stream, so the next forward
starts from clear counters. Pass a cached
`build_kimi_k3_vision_plan(...)` / `pos_emb_rows(...)` to skip the derivation.
Per-stage launches for tests and profiling: `runner.stages[name]()`
(`layer_attention` runs the attention kernel and, on an SM100 split plan, its
merge; `layer_attention_merge` the merge alone).

Correctness (tests in `tests/experimental/test_cake_kimi_k3_vision_tower.py`):
every stage against the FP32 oracle of its operator on BF16 inputs at
`atol = rtol = 1e-2`; the complete call against the FP32 tower oracle with the
oracle-fairness gate of the Cake evaluation contract (error no worse than the
HF BF16 chain's); CUDA-graph replay bit-identical to the eager launch; no
allocation in `launch()`. Benchmark against the HF torch chain with the
FlashInfer ragged BF16 attention route:
`benchmarks/bench_cake_kimi_k3_vision_tower.py`.

The sources under `csrc/` are Cake exporter output (`.clang-format` disables
formatting there so they stay byte-faithful to the export): one kernel source
per program under `csrc/cake_kimi_k3_vision_tower/` (`*_kernel.cu`), the
TVM-FFI launch bindings (`*_binding.cu`, 23 sources shared by the 81
programs), the shared device header
`cake_kimi_k3_vision_tower_device_common.cuh`, the device helper headers under
`csrc/cake_device_helpers/` and `csrc/cake_ffi_checks.h`. `build.json` lists
one library per program and architecture (its sources with their SHA-256,
nvcc flags and the architecture's gencode); `cake_build.py` serves a library
from `aot/` when a builder has compiled it from the manifest and otherwise
builds it through FlashInfer's JIT from the same sources and flags.
`cake_jit.py` registers the programs: one `_PROGRAMS` row per program
(architectures, argument plan, launch values), the `ARG_PLANS`, and the
`KERNELS` table both architectures share plus the per-architecture entries (4
SM103-only keys), expanded into `MODULES` / `KERNELS` at import.
A program built for both architectures whose SM100 and SM103 text differ is
one source with compile-time architecture arms (`#if __CUDA_ARCH__ == 1000`:
SM100 text first). The KV-split ring3 attention form is one program per
architecture (its SM103 build merges the K/V parts in the kernel), and the
plain one-tile / two-tile-split attention forms exist on SM103 only because
the plan rule selects them there alone. On a device without a registered
program the entry points raise `NotImplementedError`.
