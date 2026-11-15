# Experimental NVFP4 attention

Both this API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection.

## Contract

```python
from flashinfer.prefill import prepare_nvfp4_attention, nvfp4_attention_workspace_bytes

runner = prepare_nvfp4_attention(q, k, v, out, backend="cake")  # softmax_scale=None, workspace=None
runner.launch()  # or runner(): quantizes q, k, v on the device and writes out
```

Supported on SM100, SM103 and SM120 GPUs. `q`, `k`, `v` and `out` are BF16
tensors of the logical shape `[B, H, S, 128]` with any strides: the channel
stride is 1, the batch, head and token strides are multiples of 16 elements
(32 bytes), no two rows alias. An `[B, S, H, 128]` buffer is passed as its
`[B, H, S, 128]` view (`x.transpose(1, 2)`) and is read or written in place;
the view is the contract (strides alone cannot tell the two storage orders
apart when `S == H`). Any `B, H, S >= 1`. `softmax_scale` defaults to
`128 ** -0.5` and must be positive. The route computes dense, non-causal
multi-head attention with `Hq == Hkv`:
`out = bf16(softmax(q k^T * softmax_scale) v)`.

Not supported: causal or windowed masks, GQA/MQA, paged KV, returning the
log-sum-exp, backward, head dimensions other than 128, dtypes other than BF16.

Preparation validates the tensors, carves the workspace and binds the
arguments of the programs of the device's architecture (four on SM100 / SM103,
three on SM120). Nothing is compiled, allocated or synchronised at launch.
Every launch re-reads `q`, `k` and `v` -- quantization is part of the launch --
so their values are free between launches while their identities, shapes and
strides are bound. Prepare a new runner when a tensor, a shape or a stride
changes.

### Workspace

`workspace` is a caller-owned contiguous `uint8` tensor on the inputs' device,
256-byte aligned, of at least `nvfp4_attention_workspace_bytes(B, H, S, device)`
bytes; without one the runner allocates it at prepare (never at launch). The
workspace holds the quantized operands, the statistics and, on SM100 / SM103,
the FP32 operands and BF16 image of the block-mean compensation table and the
attention's split-tile partials of one launch, in 256-byte slots. With
`Sq = ceil(S / 512) * 512`, `Skv = ceil(S / 128) * 128`, `N = ceil(S / 128)`,
`T = B * Skv`, `G = ceil(Sq / 1024)` (groups of 8 Q blocks) and the `n_split`,
`n_chunks` of the split plan below:

| Region | SM100 / SM103 (bytes) | SM120 (bytes) |
| -- | -- | -- |
| `q4`, `k4` (E2M1 codes; on SM100 / SM103 the first level) | `B*H*Sq*64`, `B*H*Skv*64` | -- |
| `q8`, `k8` (INT8 codes of the centred K and of the offset Q rows) | -- | `T*H*128` each |
| `q_sf`, `k_sf` (UE4M3 scales; on SM100 / SM103 the first level) | `B*H*Sq*8`, `B*H*Skv*8` | -- |
| `q4l`, `k4l` (E2M1 codes of the second, residual level) | `B*H*Sq*64`, `B*H*Skv*64` | -- |
| `q_sf_l`, `k_sf_l` (UE4M3 scales of the second level) | `B*H*Sq*8`, `B*H*Skv*8` | -- |
| `qm8` (INT8 offset row of every 128-row Q block) | -- | `B*H*N*128` |
| `q_amax`, `qm_amax`, `k_amax` (FP32 block amax of the centred Q block, its mean row and the 128-key K block) | -- | `B*H*N*4` each |
| `k_minmax` (FP32 per-tile channel maxima and minima of K) | -- | `B*H*N*1024` |
| `qm32` (FP32 Q block means, one 512-byte row per 128-row block) | `B*H*Sq*4` | -- |
| `kc32` (FP32 centred K rows) | `B*H*Skv*512` | -- |
| `tab` (BF16 compensation image, (hi, lo) pairs) | `B*H*32*G*Skv` | -- |
| `vt4` (transposed V codes) | `B*H*Skv*64` | `H*128*T/2` |
| V scales | `v_sf_lo` + `v_sf_hi`: `B*H*Skv*4` each | `v_sf`: `T*H*8` |
| `mean_k` (FP32) | `B*H*512` | `B*H*512` |
| `q_mean` (FP32) | -- | `B*N*H*512` |
| `partials` (FP32) | `B*H*ceil(S/512)*512` | `B*N*H*512` |
| `counters` (u32, self-resetting) | `4*B*H` | `4*B*H` |
| `part_o` (FP32, split-tile partial outputs) | `max(1, n_split*n_chunks)*512*128*4` | -- |
| `part_ml` (FP32, split-tile row max and sum) | `max(1, n_split*n_chunks)*512*2*4` | -- |
| `split_counters` (u32, self-resetting) | `4*max(1, 2*n_split)` | -- |
| `unit_table` (i32) | -- | `16*B*H*N` |

Every region starts on its own 256-byte slot. On SM100 / SM103 the operands,
table and statistics of the `B*H` heads take
`B*H*(148*Sq + 728*Skv + 32*G*Skv + 512*ceil(S/512) + 512)` bytes before the
counters' slot and the split regions; `kc32` (512 bytes per padded key) and
`tab` (32 bytes per padded key and group of 8 Q blocks) dominate at long `S`.
For `(B, H, S) = (1, 40, 32760)` that is `Sq = Skv = 32768`, `G = 32`:
`40*(148*32768 + 728*32768 + 32*32*32768 + 512*64 + 512)` bytes, plus the
counters' slot and, when nothing splits, one row of `part_o` and `part_ml` and
one split counter's slot.

On SM100 / SM103 the attention kernel runs `n_clusters = sm_count // 2`
persistent two-CTA clusters over `total_tiles = B * H * ceil(S / 512)` tiles of
512 query rows, whole tiles first. The `n_split = total_tiles % n_clusters`
tiles of its partial last wave are each split into
`n_chunks = min(4, N, n_clusters // n_split)` key chunks (no split when that is
below 2: `n_split = 0`, `n_chunks = 1`). The chunk items follow the whole tiles,
write their partial outputs to `part_o` / `part_ml`, and the last-arriving chunk
of a tile combines them into `out`; the split counters self-reset. The
workspace is therefore sized for the SM count of the device it was carved for:
`nvfp4_attention_workspace_bytes(B, H, S, device)` is that device's figure, and
a caller's `workspace` is checked against it at prepare.

`runner.workspace_bytes` and `runner.route_metadata["workspace_bytes"]` report
the total. A caller pooling workspaces sizes the pool with the largest
problem's figure on the device the pool serves.

### Launch, CUDA graphs, determinism

A launch is the chain of the device's architecture on the current torch
stream of the inputs' device, in order. On SM100 / SM103 it is four kernels:
`stats` (per-(batch, head) K means), `quantize` (two-level residual E2M1 codes
of the centred Q and K with UE4M3 block scales -- the second level holds the
residual of the first -- the transposed V with its scales, and the FP32 block
means `qm32` and centred K rows `kc32`, written to the workspace), `delta`
(the exact FP32 compensation table `qm32 . kc32` per (batch, head, Q block,
key) on the tensor core, written as the attention's (hi, lo) BF16 seed image
`tab`; its grid is `min(items, sm_count)` over the (batch, head, 128-block tile
of Q blocks, 128-key tile) items) and `attention` (seeds every 128-key score
tile with `tab` by one BF16 MMA before the three block-scaled QK MMAs
`Qh Kh^T + Ql Kh^T + Qh Kl^T` -- the fourth product `Ql Kl^T` is dropped by
definition -- then the quantized attention, BF16 output). On SM120 it is three
kernels: `stats` (K means and the per-128-row-block Q means), `quantize`
(single-level E2M1 codes with UE4M3 block scales of the centred Q and K, the
block-mean tile `qm4` / `qm_sf` and the transposed V) and `attention`
(compensates the block mean with one block-scaled MMA per 128-key tile). On
SM100 / SM103 the attention grid is `2 * min(sm_count // 2, n_items)` CTAs in
two-CTA clusters over the `n_items = n_full_tiles + n_split * n_chunks` work
items of the split plan above. `quantize`, `delta` and `attention` launch with
programmatic dependent launch and wait for their producer inside the kernel;
`stats` launches plainly. The statistics kernel's counters and the SM100 /
SM103 attention's split counters self-reset, so a launch leaves the workspace
ready for the next one.

The runner is CUDA-graph capturable; capture belongs to the caller. Replays are
bitwise repeatable: every reduction has a fixed order and the chain carries no
device-side state across launches. `runner.launch_stage(name)` launches one
stage alone (`"stats"`, `"quantize"`, `"delta"` on SM100 / SM103,
`"attention"`), for diagnostics. `runner.route_metadata` reports the
architecture, the recipe id, the QK operand format, the tile order, the key
grouping, the registry record of each stage, the launch grids, on SM100 /
SM103 the split plan (`split_plan`: `n_full_tiles`, `n_chunks`, `n_items`,
`grid`, `sm_count`), the softmax scale and the workspace size; `runner.out` is
the caller's `out`.

### Numerics

<!-- Pending from the validation run: the measured error figures. -->

| | SM100 / SM103 | SM120 |
| -- | -- | -- |
| Recipe id | `nvfp4-sage3-v1` (placeholder) | same |
| QK operands | `nvfp4_res2d`: two-level residual NVFP4 Q and K -- the first level E2M1 with one UE4M3 scale per 16 channels (`max(amax / 6, 2^-9)`) of the centred operand, the second level the same rule on the FP32 residual of the first; the scores accumulate `Qh Kh^T + Ql Kh^T + Qh Kl^T` in FP32 (`Ql Kl^T` dropped, bounded by 1/36 of the first level's magnitude bound) | `int8_blk`: INT8 Q and K -- K centred per (batch, head) and scaled per 128-key block (`k8 = clamp(rint(K' * 127 / k_amax))`); Q scaled per 128-row block by `amax_s = max(q_amax, (q_amax + qm_amax) / 2)` with the block mean kept as an INT8 offset row (`o8 = clamp(rint(qm * 127 / amax_s))`, `q8 = clamp(rint(Q * 127 / amax_s - o8))` of the uncentred rows); the score `(q8 + o8) . k8` is an exact integer and `amax_s * k_amax / 127^2` folds into the tile's softmax constant |
| PV operands | NVFP4: E2M1 codes with one UE4M3 scale per (channel, 16 keys) block of the transposed V | same |
| K | centred per (batch, head); the two-level residual form above (no rotation) | centred per (batch, head); INT8 with one FP32 scale `k_amax / 127` per 128-key block of the packed token axis (no rotation) |
| Q | centred per 128-row block and quantized as the two-level residual form above; the removed block mean is kept exactly in FP32 (TF32-rounded rows, `qm32`) and its compensation `qm K'^T` over the FP32 centred K rows (`kc32`) is the exact FP32 tensor-core table of the `delta` program, seeded into the score accumulator as a (hi, lo) BF16 pair (`hi + lo = table` to 2^-17 relative) | scaled per 128-row block (`amax_s / 127`); the block mean is an INT8 offset row `qm8` whose compensation `o8 K^T` is one more INT8 MMA row per 128-key tile, shared one tile ahead and seeded into the score accumulator (the exact integer form of Sage3's `delta_s`) |
| Softmax | FP32 online softmax in the log2 domain over 128-key tiles (the row maximum adopted whenever it moves); two-level `P`: the tile's maximal `p` brought to `2688 = 448 * 6`, one UE4M3 scale per 16 keys (block max / 6, no floor), block-relative E2M1 codes; the FP32 row sum is taken over the unquantized `p` | same |
| Tensor core | `tcgen05` `kind::mxf4nvf4` block-scaled MMA with TMEM accumulators (three per 128-key tile for the residual QK form, after one BF16 MMA seeding the tile from `tab`); a persistent 2-CTA cluster per 512-row Q tile; the `delta` table is an FP32 tensor-core product per (Q-block tile, 128-key tile) | `mma.sync.m16n8k32.s32.s8.s8` for `Q K^T` over 128B-swizzled INT8 tiles; `mma.sync kind::mxf4nvf4.block_scale.scale_vec::4X` m16n8k64 for `P V`; one CTA of 8 warps per 128-row unit |
| Tile order | descending (padded keys of the last tile masked) | ascending |
| Key groups | consecutive 16-key blocks | the 16-key blocks of a 32-key chunk interleaved (the register layout of the `P` fragments) |

Error bands (filled in from the validation run): against the recipe emulation
(the same quantization in FP32) the route agrees within `atol 1e-2 + rtol 1e-2`;
against exact BF16 attention on real activations the validation reports the
cosine similarity and relative L2 error per shape (**pending**). The package
tests check every launch against a dequantized-operand oracle at
`atol 1.0, rtol 0.1` (the FP4 contract tolerance) over the regimes `randn`,
`v_x8`, `onehot`, `outlier32` and `outlier32_q`, tails (`S % 128 != 0`),
`S < 128`, `S == 1`, `B > 1` and both storage orders.

## Package layout

The backend is a self-contained package. `cake_jit.py` is the registry: a role
table of `MODULES` records, one per stage and architecture group -- eight
records: `stats_sm100`, `quantize_sm100`, `delta_sm100` for SM100 and SM103
with `attention_sm100` on SM100 and `attention_sm103` on SM103; `stats_sm120`,
`quantize_sm120`, `attention_sm120` for SM120 (`records_for(arch) -> {stage:
record}`; the SM120 chain has no `delta` stage) -- each carrying
the host plan (the program's library, the architectures it is delivered for,
the source module and build keywords it was rendered from, its role and stage,
the launch record, the argument plan of its `run` entry, the kernel-parameter
to logical-argument map, the element type of every buffer parameter, the grid
rule, the selection rule and the workspace rule) and the loader glue
(`records_for`, `launch_grid`, `launch_args`, `load`, `run_module`). What to
compile is recorded once in `build.json`: the generated CUDA sources under
`csrc/`, their digests and nvcc flags, one entry per (program, architecture).
`cake_build.py` loads a prebuilt library from `aot/` when a builder compiled
one and otherwise JIT-builds the manifest entry through FlashInfer from the
same sources and flags.

`cake_backend.py` holds the host: tensor validation, the workspace layout
(`workspace_layout`), the logical argument values of every record and the
runner. `tests/experimental/test_cake_nvfp4_attention.py` holds the oracle, the
validated shape set, the graph-replay and workspace contracts and the manifest
contract; `tests/experimental/test_cake_nvfp4_attention_api.py` the public
entry; `benchmarks/bench_cake_nvfp4_attention.py` times the complete call per
shape (CUPTI, cold L2, graph replay); `examples/cake_nvfp4_attention.py` is a
complete example.

## Migration from the previous experimental route

| | Previous route | This route |
| -- | -- | -- |
| GPUs | SM103 only | SM100, SM103, SM120 |
| Shapes | `S % 512 == 0`, contiguous `out` | any `S >= 1`; any strides on `q`, `k`, `v` and `out` |
| Entry | `prepare_nvfp4_attention(q, k, v, out, backend="cake")` | same, plus `softmax_scale=` and `workspace=`; `nvfp4_attention_workspace_bytes(B, H, S, device)` |
| Quantization | at prepare (`pack_nvfp4_attention_inputs`; values bound) | inside every launch (values free between launches) |
| Runner | `runner()`, `runner.main_kwargs` | `runner()` / `runner.launch()`, `runner.launch_stage(name)`, `runner.route_metadata`, `runner.workspace_bytes` |
| Registry | one record, `record_for(arch)` | eight records, `records_for(arch) -> {stage: record}` (the SM120 chain has no `delta` stage) |
| Numerics | per-tensor E2M1 with E4M3 block scales of the raw operands | the centred recipe above; outputs differ from the previous route by design |
