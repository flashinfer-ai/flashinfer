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

Native D576 FP8 attention shares the sparse wrapper with packed caches. Small
query/head grids can use the q64 Keep-AB reload schedule: four loader warps cache
Gather4 coordinates, use a compact RoPE stage, and release K before reloading V
from L2. Split-KV and V-column partitions expose additional work on small grids;
the existing parallel GMEM reducer includes those partials in the final result.
Sustained work retains K/V tiles with eight loaders and CLC scheduling instead.
H128 uses the shared 2CTA schedule beyond the tiny split-grid region. Its packed
NVFP4 path retains the existing conversion ownership and DSMEM exchange for V.
Plans with `assume_valid_prefix=True` use the general path's prefix optimization;
the native reload loader retains its arbitrary-hole masks.
These are separate schedules within the shared kernel framework. D512, packed
formats, narrower heads, extra pools and sinks retain their qualified paths.

Sparse public LSE is natural-log; dense optional final LSE is base 2. Split
scratch stays base 2 until the final store. An empty sparse row writes zero output and negative-infinity LSE. Native
cache pools accept flat `[rows,D]` storage as well as paged views; indices always
include each pool's physical row stride. Scale arguments remain host-runtime
values with the same graph-capture semantics on both schedules.
An omitted `softmax_scale` uses `1/sqrt(D)` for query width D on every cache
format. Supply the model's effective attention/Q/K scale explicitly when its
normalization dimension differs from the absorbed query width.

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

For serving, validate metadata invariants upstream and use `run(validate=False)`
in the hot path. The default `validate=True` checks GPU values on the host and
synchronizes. Q quantization and query RoPE are caller-owned preparation.

Fixed plans retain their B/Sq geometry. A sparse wrapper can serve a range of
independent row counts with one profile: plan `batch_size=1`,
`packed_query=True`, `max_q_len=row_capacity`, then pass `[rows,H,D]` Q and
CUDA INT32 `qo_indptr=[0,rows]` with `rows <= row_capacity`. Choose capacity for
the desired workload range; each graph retains its captured views and addresses.

The 1CTA persistent/CLC scheduler requires `split_kv=1`: its work queue does
not enumerate KV splits. Whole-tile K/V reuse can still run with split-KV using
a nonpersistent schedule. Low-level profile overrides must obey this constraint;
2CTA scheduling has its own split-aware work queue.

Quantized H16/H32 plans retain whole converted K/V tiles for small grids too.
Below half an SM wave, a nonpersistent power-of-two KV split fills one wave
while keeping at least two KV tiles per split. Larger grids use the persistent
reuse schedule. The resource layout therefore does not fall back to duplicated
conversion just because the planned row count is small.

Quantized persistent 2CTA launches balance their waves using live tensor
extents. The compilation key retains the physical SM limit, so different
row buckets that choose the same profile reuse one compiled kernel.

Routine task-builder HEAD/LOOP overlap and unused-routing advisories are quiet
by default. Set `FLASHINFER_PRIMS_TS_DEBUG_CHECKS=1` before a cold build to display
them. Runtime `validate` controls input checks separately; other compiler and
schedule warnings/errors remain visible.

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

Dense plans select cached policies and compiled kernels from explicit problem
bounds; runs bind live metadata and reuse scratch. See the shared
[planning contract](../../README.md#reusable-plans),
[runtime split-KV policy](../../README.md#runtime-split-kv-policy), and
[final statistics contract](../../README.md#final-statistics-and-sink-semantics).

## Optional merge statistics for dense pages

Dense MLA plans accept `store_softmax_stats=True`. Every run then requires a
caller-owned contiguous FP32 `softmax_stats` buffer on the query device with
shape `[*query.shape[:-1], 2]` and at least 4-byte alignment. Its final axis is
`(m, s)`: the natural-log maximum of scaled QK scores and
`sum(exp(score - m))`. The denominator excludes FP8's internal probability
scale and output/V scaling. These statistics are independent of the optional
base-2 LSE output; either or both may be enabled without changing return types.

The statistics buffer must not alias inputs, output, LSE, or workspace. Retain
it along with the wrapper and other buffers for CUDA graph replay. Presence is
part of the compiled contract even when runtime validation is disabled.
Direct, serial/parallel split-KV, and 1CTA cluster reduction retain raw maxima
and sums; no reducer reconstructs the denominator from rounded absolute LSE.
Stats-enabled split scratch includes two additional FP32 sidecars, and cluster
scratch includes the matching raw-state exchange.

Statistics require an exact maximum (`softmax_rescale_threshold_log2=0`), which
is the dense default. Nonzero deferred-max tuning with statistics enabled is
rejected before allocation; stats-disabled tuning is unchanged. Sparse/NVFP4
entry points do not expose this option. Without per-query bounds, zero-KV
requests are inactive and preserve caller buffers. Fully masked active rows
export `(-inf, 0)`; an all-empty packed-Q launch remains a no-op.

## Per-query local-KV prefixes

Dense MLA `plan(use_kv_bounds=True)` specializes a graph-live `kv_bounds` input.
It is a contiguous, 4-byte-aligned CUDA int32 vector with one count per logical
query token: `[B*SQ]` for fixed Q and `[total_q]` for packed Q. All heads of a
token share its bound. Bounds must be nondecreasing within each request and
lie in `[0, seq_lens_kv[b]]`; they replace both dense and bottom-right causal
visibility, rather than intersecting the selected mask. The raw local-KV
length still determines producer/reducer split partitioning, preserving FP8
probability quantization across the same split/instance streams.

Supply bounds exactly when enabled, including with `validate=False`. Validation
checks structure and values outside capture; trusted runs perform no host
readback. Retain the vector and mutate its contents between completed launches
or graph replays. A zero prefix writes zero O, optional LSE `-inf`, and optional
merge statistics `(-inf, 0)`. A bounds-only neutral initialization launch also
covers requests with zero local KV when the ordinary task schedule skips them.
Unbounded zero-KV behavior is unchanged. The static K/V capacity remains
positive; the one-shot API defaults it to one for an all-empty bounded rank.

Per-query prefixes add no scratch and do not alter the public return type.
Sparse/NVFP4 entry points remain outside this option's scope. The bounds
specialization is independent of LSE/statistics; existing constraints on
statistics and nonzero rescale thresholds still apply.
