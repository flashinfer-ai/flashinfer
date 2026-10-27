# Task-Scheduled MLA Decode

Prims-TS MLA provides paged dense attention and sparse attention over a primary
pool plus an optional extra pool. Query/key rotation and Q quantization happen
upstream. K/V use one shared latent head; output has 512 channels.

## Target models and cache recipes

| Target / use | QK / V | Primary storage | Extra storage | Q / MMA | Output |
|---|---|---|---|---|---|
| DeepSeek V2/V3-style absorbed dense MLA | 576 / 512 | Native BF16 or E4M3 | None | Matching BF16 or FP8 | BF16 |
| DeepSeek V3.2-style sparse MLA | 576 / 512 | NVFP4 NoPE512 + FP8 RoPE64, 352 B/token | Same format when present | E4M3 / FP8 | BF16 |
| DeepSeek V4.1-style sparse MLA | 512 / 512 | NVFP4, 288 B/token | NVFP4, 288 B/token | E4M3 / FP8 | BF16 or fused output RoPE + FP8 quantization |
| DeepSeek V4.1-style SWA + compressed KV | 512 / 512 | Per-tensor E4M3, 512 B/token | NVFP4, 288 B/token | E4M3 / FP8 | BF16 or fused output RoPE + FP8 quantization |
| Native sparse MLA | 512 or 576 / 512 | Native BF16 or E4M3 | Same dtype | Matching BF16 or FP8 | BF16; fused output RoPE + FP8 quantization for D512 |

These are architectural/model recipes; the API takes capability and format
parameters. NVFP4 has embedded block16 E4M3 scales and no global storage scale.
Only the primary pool may use per-tensor FP8 in a mixed recipe. Its
`kv_cache_scale` is a positive Python number passed as a runtime FP32 kernel
argument, as are `softmax_scale` and `output_scale`. Scale values do not create
JIT variants. CUDA graphs capture these values; use graph parameter updates or
recapture when changing them for replay.
D512 includes rotated channels in the ordinary 512-channel storage; D576 has a
separate QK-only 64-channel RoPE tail. Record diagrams and scale-byte ordering
are documented in [helpers/nvfp4_kv.py](helpers/nvfp4_kv.py).

## Supported execution

| Feature | Support |
|---|---|
| GPU families | SM100/SM103 Blackwell; NVFP4 paths qualified locally on GB300 |
| Query shape | Fixed `[B,Sq,H,D]` or packed `[total_q,H,D]` with query offsets |
| Query heads | Native automatic policy admits supported positive counts; NVFP4 qualification covers H8/H16/H32/H64/H128 |
| Dense cache pages | 16, 32, 64 or 128 tokens; dense or bottom-right causal masking |
| Sparse cache pages | Compact rows with independent physical page strides; indices address storage rows directly |
| Sparse selection | Separate int32 primary/extra indices and live lengths; `-1` holes, tails and empty rows supported |
| Valid-prefix option | Every index before its live length is valid; it does not imply physical contiguity |
| Output quantization | D512: conjugate output RoPE on channels 448:512, E4M3 output, block32 or block128 scales in UE8M0 or FP32 |
| Execution | Automatic 1CTA/2CTA and split-KV selection; FP32 accumulation; planned CUDA-graph replay with stable buffers |

Use `BatchMLADecodePagedTSWrapper` for dense pages and
`BatchSparseMLADecodePagedTSWrapper` for sparse pools. Their `plan()` and `run()`
docstrings define tensor layouts, workspace ownership, validation and capture
requirements. The one-shot helpers are eager interfaces; plan before capture.

Native dense 2CTA loaders cache 32 page IDs per warp. FP8 K/V loaders also
prefetch the next window before waiting for a free data stage; the first
window is loaded synchronously. Sparse NVFP4 loaders use source-tagged row
indices instead of these dense page-table windows.

`MLAAttentionTuning(softmax_rescale_threshold_log2=T)` enables the optional
softmax correction skip in both families. `None` preserves the schedule's
default; explicit zero uses exact row maxima. Positive T freezes the anchor
until it rises by more than T binades and scales FP8 probabilities by
`448 * 2**-T`, trading tail precision for fewer output corrections. Bounds are
0–8 for E4M3 and 0–32 for BF16. Each distinct T specializes a kernel; see the
`plan()` docstring for the precision implications.
