# Experimental Kimi-K3 FP8_PB_WO projection GEMMs (Cake backend)

Both the API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection. Tracking: flashinfer-ai/flashinfer#4568 (Kimi-K3 kernels),
tracker #4254.

`flashinfer.gemm.kimi_k3_fp8_projection(x, prepared, out=None)` runs one
TP-local KDA / MLA projection of `nvidia/Kimi-K3-NVFP4` whose weight is
serialized as `FP8_PB_WO` (E4M3 `[N_pad128, K]` + ModelOpt 128x128 FP32
`weight_scale`) on SM100 (B200) and SM103 (B300/GB300):

```
a_q[m, k] = E4M3_rn(x[m, k] / s_a[m, k // 128]),   s_a = 2^ceil(log2(max(amax_128(x[m]), 1e-4) / 448))
w2, s2    = requant_ue8m0(weight, weight_scale)     (once: s2 = 2^ceil(log2 scale), w2 = E4M3_rn(weight * scale / s2))
out       = bf16(sum_k a_q s_a * w2 s2)              [M, n_valid], FP32 accumulation, one BF16 rounding
```

The activation recipe is DeepGEMM's `per_token_cast_to_fp8(use_ue8m0=True)`
(bit-exact), the weight recipe vLLM's `requant_weight_ue8m0`; the GEMM is
tcgen05 `kind::mxf8f6f4` block-scaled MMA with hardware scale application.
Every launch of the call is a generated Cake program:

| Route (host dispatch) | Programs | When |
| --- | --- | --- |
| quantization launch + persistent 2-CTA GEMM | `quant:u<units>`, `gemm_tstore` (16-byte aligned output base and a row stride that is a multiple of 8 elements: TMA-store epilogue) or `gemm` (other strides: register epilogue); the `_n<bn>` instances of the same epilogues on the tabulated `gemm_bn` rows (`_n192`: the N = 576 `kv_a` family at M > 256, three 192-column tiles instead of 768 padded columns; `_n384`: the 128 x 384 pair tile; `_n224` / `_n352`, TMA-store epilogue only: the B200 `o_proj` and `q_proj` rows at M = 512 and the B300 `q_proj` rows at M = 497..512 (an in-bucket range), 32 tiles of 224 columns for N = 7168 and 35 tiles of 352 for N = 12288; the staged-register view of those B300 rows keeps the 384-wide program (table key `gemm_rstaged_bn`); `bn` sets the launch's N tiles, B scale tiles and grid); the `_k128` instances of the TMA-store and staged-register epilogues on the tabulated `gemm_bk` rows (every 128-K half of an operand stage is loaded and multiplied under its own barrier pair, so the MMA of one half overlaps the TMA of the other; same SMEM layout, same MMA sequence, bit-identical output; the token follows `_n192` in the program name, before the stream-K suffixes); the `_s2_ts4` instances of the TMA-store epilogue on the tabulated `gemm_stages` / `gemm_ts_slots` rows (round 9: two operand stages and four staging slots per epilogue group on the short-K large-M `f_b` / `kv_b` rows, so a 32-column store chunk waits for the store issued one tile earlier instead of the previous chunk's read; same MMA sequence, bit-identical output; the tokens follow `_k128`, before the stream-K suffixes; the register epilogues keep the untokened programs); the `_pdlb` instances of the TMA-store and staged-register epilogues on the tabulated `gemm_pdlb` rows (round 11: the loader issues every stage's weight boxes before its activation boxes and waits for the quantization launch once per CTA, right before its first activation box, so the first stage's weights, the weight L2 prefetches and the first CLC request land under the quantization kernel's drain; same MMA sequence, bit-identical output; the token closes the program stem, after the stream-K suffixes and before `_pf<D>`; the register epilogue and the 384-wide program keep the untokened programs); the `_st` instances of the TMA-store and staged-register epilogues on the tabulated `gemm_static` rows (round 11: the static one-tile form of the plain `_n192_k128` programs -- one output tile per CTA pair over 2 x tiles CTAs, no cluster-launch-control work ring, admissible when the launch's CTA-pair tiles fit the resident pairs; same MMA sequence, bit-identical output; the token takes the `_cs<S>` slot after `_k128`; the register epilogue keeps the untokened program); the `_gs` instances of the TMA-store epilogue on the tabulated `gemm_gs` rows (round 11: the static grid-stride form of the plain `_k128` / `_n192_k128` programs -- a grid of 2 x min(tiles, resident pairs) CTAs whose pair p owns the tiles p, p + P', p + 2 P', ..., no cluster-launch-control work ring; same MMA sequence per tile, bit-identical output; the token takes the `_cs<S>` / `_st` slot after `_k128`; the staged-register and register epilogues keep the untokened programs); the `_gs_ars` instance of the 256-wide `_k128_gs` program on the tabulated `gemm_ars` rows (round 12, B200: the A-resident form -- each CTA pair keeps its activation K range, at most 512, resident in shared memory across the tiles it owns, so only the weight boxes stream per tile; same launch, same MMA sequence, bit-identical output; the token follows `_gs`); on the tabulated `gemm_rstaged_pf` / `gemm_rstaged_skf` rows the staged-register epilogue takes the cell's weight prefetch (`gemm_rstaged_n192_k128_pf2`) or fix-up stream-K form (`gemm_rstaged_k128_skf`) that the TMA-store epilogue takes on every `gemm_pf` / `gemm_skf` row (round 12, B200: the `in_proj_qkvgfab` family, whose 6284-column edge refuses the TMA store); `DECODE_TABLE_M_RANGES` gives a tabulated cell in-bucket row ranges `(lo, hi, entry)` whose rows take `entry` instead of the cell's entry (round 12, B200: the fastest route changes inside the 2048 / 4096 buckets of the `in_proj_qkvgfab` and `kv_b` families); decode programs carry `_px<D>` when the row's table key `pfx` prefetches the BF16 token tile into L2 D stages ahead of its TMA load (round-6 next loop, lever PX on the small fused rows: the tile's DRAM access precedes the weight burst instead of queueing behind it) | `M > 256` unless the family is tabulated for the decode kernel at that row count, and the tabulated `(N, K)` families whose measured best route is the GEMM |
| quantization launch + decode | `quant:u<units>` (`u1` unless the table cell pins a width with `quant_units`), `decode:t<tok>_p<stages>[_a<n>][_e<n>]...` | measured table entry with `fused = false` |
| fused decode | `decode:t<tok>_p<stages>_fused[_w<n>][_a<n>][_e<n>][_res]` | measured table entry with `fused = true` (the token tile is quantized in-CTA; `_res` keeps the quantized token tiles resident for `K <= 256`); above 256 rows only the single-N-tile families (`f_a`, `b_proj`) are tabulated |

Decode programs carry `_a<n>` when the row's table key `acc_bufs` gives the
instance `n` TMEM accumulator sets and `_e<n>` when `epi_groups` gives it `n`
epilogue warp groups (64/128-token tiles only; the host forces one group below
64 tokens); both sit right after the `_fused[_w<n>]` block and are absent at
the default of 1. A two-launch row whose cell carries `quant_units` launches
the quantization program at that width (`quant:u<units>`, decode and GEMM
routes alike; it must divide `K / 128`) instead of the `quant_units` rule.

Decode programs carry `_dpf` when the row's table key `dpf` prefetches the
tensor-map descriptors in the kernel prelude, before the first TMA issue
(round-6 lever Q1, adopted in round 8 on the SM103 `tp1 f_b` cell of the
65..256-row bucket); the token follows the TMA-store token `_tso` and
precedes `_pi<n>` / `_xp`, as in the Cake kernel symbol, so a `tstore` row
that pins it ships the pair `..._tso_dpf` (16-byte-aligned output views) and
`..._dpf` (register epilogue). The prefetch changes no data path and no
launch argument.

`decode_table.py` is the measured dispatch table
(`"<n_tiles128>,<num_k_iters>,<m_bucket>"`, buckets `M <= 1 / 8 / 64 / 256 / 512 /
1024 / 2048` for every family and `M <= 4096 / 16384` for 20 and 21 of the 22
families)
over the 22 representative families (TP8 and TP1 `q_proj`, `fused_qkvg`,
`in_proj_qkvgfab`, `f_a`, `f_b`, `b_proj`, `kv_a`, `fused_qkv_a`, `q_b`,
`kv_b`, `o_proj`): one table shared by SM100 and SM103 (`DECODE_TABLE`) plus
the few cells whose measured best route differs per architecture
(`DECODE_TABLE_OVERRIDES[arch]`; `decode_table(arch)` merges them). It is
generated from the Cake sweeps and mirrors the production dispatcher row for
row. Families outside the table take the quantization + GEMM route (the Cake
dispatcher falls back to a calibrated cost model there; every representative
Kimi-K3 family is covered on both architectures). `ROUTE_TWINS[arch]` names,
per cell, the twin a launch takes instead of the key the table selects
(`route_twins(arch)`): the same program with L2 eviction-priority hints --
`_wef` (decode: weight and weight-scale loads evict-first), `_xef` (fused
decode: BF16 activation loads evict-first), `_bef` (GEMM: B / SFB loads
evict-first), `_xef_qel` (quantization: activation loads evict-first, E4M3 /
scale stores evict-last) -- or with a schedule change -- `_k128` (unfused
decode: each 256-K TMA stage loaded as two 128-K halves), `_sko<N>` (16-token
cluster split-K decode: the partial-sum exchange bounded to the first N rows,
taken only by cells of at most N rows), `_csrt` (4-way cluster split-K decode:
the closing cluster barrier arrives relaxed after a tcgen05 fence), `_cv100`
(quantization: a preferred shared-memory carveout of 100 percent, set as a
launch attribute, for the decode launch that starts beside it) -- built for
the architectures it was measured faster on; the output bits are the plain
program's. A table row's `tstore_twins` names twins that only its launches into
a TMA-store output view take (the B300 `q_proj` rows at M = 497..512: the
quantizer's `_xef_qel` twin); its other views keep the selected programs.

| Tensor | Shape | dtype |
| --- | --- | --- |
| `weight` | `[N_pad128, K]` serialized checkpoint tensor (128-row block padding), contiguous; `K % 128 == 0` | float8_e4m3fn |
| `weight_scale` | `[N_pad128 / 128, 1, K / 128, 1]` ModelOpt block scale (or its 2-D view) | float32 |
| `x` | `[M, K]` activations, contiguous (the quantization program reads rows at stride `K`) | bfloat16 |
| `out` | `[M, n_valid]` view with unit column stride and an even row stride (any view into a wider buffer; only the `n_valid` columns are written) | bfloat16 |
| `workspace.q` | `[M, K]` quantized activation (caller-owned) | float8_e4m3fn |
| `workspace.sf` | `[bytes]` activation scale tiles + per-tile counters (zero initialised once by the allocation) + split-K partials (written before read; caller-owned) | uint8 |

```python
import torch
from flashinfer.gemm import (
    allocate_kimi_k3_fp8_projection_workspace,
    prepare_kimi_k3_fp8_projection,
    prepare_kimi_k3_fp8_projection_weights,
)

prepared = prepare_kimi_k3_fp8_projection_weights(weight, weight_scale, n_valid=6144, splits=(1536, 1536, 1536, 1536))
workspace = allocate_kimi_k3_fp8_projection_workspace(prepared, M)      # once per M
runner = prepare_kimi_k3_fp8_projection(x, prepared, out, workspace)     # binds the launch sequence
runner()                                                                 # no allocation; CUDA-graph capturable
q, k, v, g = prepared.output_views(out)                                  # fused projections: split views
```

`prepare_kimi_k3_fp8_projection_weights` requantizes the weight to UE8M0
scales, pads it to the 256-row CTA-pair tile and stores it as 128x128 E4M3
tiles in the order the TMA streams (one contiguous 32 KB box per pipeline
stage) together with the swizzled weight scale tiles; do it once per weight.
`allocate_kimi_k3_fp8_projection_workspace` sizes the byte workspace for the
route the device's table selects for `M` and zero-fills only its scale-tile and
counter bytes. `prepare_kimi_k3_fp8_projection` validates the binding, resolves
the route once (architecture and SM count are read once per device and cached)
and binds the generated argument plans; the runner reads `x` on device at
every launch, so a CUDA graph capturing it stays valid when new activations
are written into `x`.

Correctness: `|out - ref| <= 1e-2 + 1e-2 |ref| + 2 bf16 ulp` element-wise
against the exact quantized-operand emulation (zero budget), quantized
activation bit-exact vs the DeepGEMM recipe; see
`tests/experimental/test_cake_kimi_k3_fp8_projection.py`.

Benchmark: `python benchmarks/bench_cake_kimi_k3_fp8_projection.py [--cupti]`
times the 132 representative rows (22 families x `M in {1, 8, 64, 256, 4096,
16384}`) as CUDA-graph replays with a cold L2 against FlashInfer's existing
`per_token_group_quant_8bit` + `gemm_fp8_nt_groupwise` chains (`cutlass` sm1 /
sm2, `trtllm`, `cutile`) on the same weight.

Generated sources live under `csrc/cake_kimi_k3_fp8_projection/`: one kernel
source per program or lever family and one launcher source per argument plan
(5 launcher sources, one per argument plan, for the 107 programs) for both
architectures (the architecture-specific lowering sits behind `__CUDA_ARCH__`
guards; the JIT compiles each source with the flags of the device it runs
on), the shared device-helper header, the host-helper header
(`cake_kimi_k3_fp8_projection_host_common.cuh`: the dynamic-shared-memory and
non-portable-cluster opt-ins and the receipt / chain text every launcher
includes verbatim inside its namespace), and nothing else. A program that
differs from a sibling only by a prefetch lever or by the raster group width
shares the sibling's kernel source and selects its lever value
(`-DCAKE_LEVER_*`, `-DACC_BUFS` or `-DGROUP_M`) on the compile line; the programs of one
argument plan share one launcher source, and a program's TMA box dimensions,
dynamic shared memory, block and cluster dimensions and its kernel symbol and
launcher namespace are `-D` constants on its compile line (`-DCAKE_BOX_*`,
`-DCAKE_DYN_SMEM`, `-DCAKE_BLOCK_X`, `-DCAKE_CLUSTER_*`,
`-DCAKE_PROGRAM_KERNEL`, `-DCAKE_PROGRAM_SHIM`); a source's defaults select
the program it was exported for.
`cake_jit.py` registers them (`MODULES`: one record per program, expanded
from one row per program and the five named argument plans of `ARG_PLANS`;
`KERNELS`: logical kernel key -> program and compile-line defines, e.g. the
three quantization widths are one program with `-DQUANT_UNITS=1|2|4`). Both
are written by the Cake generated-program export; the shared kernel and
launcher sources, the hoisted host header, the `-D` entries of the records
that use them, the row form of `MODULES`, the `shim` / `fast_math` row fields
and the `_KERNEL_PROGRAMS` table are a hand fold of the exported files,
pending regeneration by the export.
