# Lossless FP4 scale blocks

The checkpoint encoding `csf_32x2_v1` stores exact scale bytes for NVFP4 and
MXFP4 weights. NVFP4 uses four-bit offsets into a row palette; MXFP4 uses
one-bit offsets. Exceptions replace palette results with exact eight-bit scale
values. Weight nibbles and global calibration tensors retain their native
checkpoint representation.

## Checkpoint representation

Each logical scale matrix has `R` rows and `C` columns. `R` is a positive
multiple of 32; `C` is positive. A packed weight matrix of shape `[R, K/2]`
has `C = K/16` for NVFP4 and `C = K/32` for MXFP4. The quantization recipe
determines offset width and scale-byte interpretation.

For a scale tensor named `weight_scale`, its standard safetensors index maps
these five component names to shards:

| Suffix | Dtype | Contents |
| --- | --- | --- |
| `.csf32_bases` | uint8 | One palette base per logical row. |
| `.csf32_codes` | uint8 | Packed offset fields in block order. |
| `.csf32_page_bases` | uint32 | Byte origins in the exception arena. |
| `.csf32_offsets` | uint16 | Page-relative exception-record prefixes. |
| `.csf32_exceptions` | uint16 | Block-local positions and replacement bytes. |

Components may reside in different shards. Integer arrays use little-endian
storage. Recipes identify the encoding with
`weight_scale_encoding: "csf_32x2_v1"`.

Blocks cover 32 rows and two columns. Blocks are grouped into logical
128-row by four-column tiles. Tiles are visited in row-major order. Within
each tile, the first column pair's row blocks precede the second column
pair's row blocks. A boundary tile contains only the blocks inside the
matrix. A final single-column block contains only that column.

Each block packs row-major offset fields, least-significant field first.
The code stream contains exactly `R * C * offset_bits / 8` bytes. Row bases
are stored separately in logical row order. NVFP4 bases are in `0..240`;
MXFP4 bases are in `0..254`. There is no stored row or column padding.

An exception's low byte is `2 * local_row + local_column`; its high byte is
the replacement scale byte. Positions within a block are sorted, unique,
and inside the block's logical extent. Records follow block order. Decoding
first adds each offset to its row base, then applies the replacements.

An index page contains 32 whole tile groups, except for the final page.
Each page's prefix array starts at zero and includes a terminal entry after
its last block. Prefixes count records; page origins count bytes. A matrix
with `B` blocks and `P` pages has `B + P` prefix entries and `P` page origins.
The exception arena is consecutive and smaller than 4 GiB. A group's records
never cross a page, including after TP slicing and projection fusion.

## Loading and native destinations

The checkpoint contains one representation independent of TP size or selected
kernel. The loader selects logical rectangles, concatenates projection rows,
and rebuilds rank-local groups and indexes. TP boundaries require complete
32-row blocks and complete quantization groups. Partitions are derived from
geometry rather than stored TP directories.

Direct loaders batch verbatim checkpoint components into CUDA staging buffers.
GPU kernels validate palettes and exception indexes, gather selected scale
blocks into native order, and rebuild their indexes with a device prefix sum.
Only validation flags and allocation sizes return to the CPU; scale matrices
are neither decoded nor regrouped there. Staging is bounded to one expert layer.

The slab checkpoint encoding `weight_scale_encoding: "csf"` is also accepted.
Its fixed slabs and absolute exception records are uploaded verbatim and
converted on the GPU to the same resident payload. Both checkpoint encodings
use the same scale-expansion tuner and ordinary native compute kernels.

Preparation applies the selected native kernel's scale transformations,
including NVFP4 W4A16 normalization, MXFP4 exponent clamping, and projection
row order. It permutes compressed fields, row bases, and exception positions
for the destination. These operations occur during loading; they do not
introduce another checkpoint encoding.

The resident payload supports direct expansion into:

| Destination | Native consumers |
| --- | --- |
| F8_128x4 | NVFP4 quantized-activation kernels. |
| Packed MMA scales | NVFP4 and MXFP4 W4A16 kernels. |
| N256 scale tiles | MXFP4 A8 kernels with N256 weight packing. |
| N64 scale tiles | MXFP4 A8 kernels with compact N64 weight packing. |

Gate/up sections follow each destination's native packing rules. Single
scale-column packed MMA matrices use the native single-group row permutation.
The expansion kernel writes the destination directly, without an intermediate
row-major expansion or a second swizzle pass. Native compute-tile padding is
confined to reusable output scratch. A TP-padded zero-weight partition can
extend its logical scales during loading without padding the checkpoint.

Resident exception metadata uses group prefixes, two-byte destination
positions, and one-byte values. Offset-code storage remains four bits per
NVFP4 scale or one bit per MXFP4 scale. There is no per-layer persistent copy
of the expanded scales; serialized layers can reuse caller-owned output
scratch. Overlapping consumers require disjoint scratch or explicit stream
dependencies.

## Preparation and execution

`fused_moe.plan_scale_expansion` declares an expansion tuning request for a
route capacity. Its tuning contract sweeps 1, 2, 4, 8, 16, 32, or 64 groups
per CTA, exception staging, and expert-presence construction. CTAs use one
warp per group up to eight warps. Declare the expansion request as a
dependency of the corresponding MoE request and pass its plan through
`fused_moe.plan_execution(scale_expansion=...)`. Capacity families pass a
mapping from token count to expansion plan. The prepared MoE binding retains
the selected expansion programs and shares its fixed expert-presence scratch.

Short route lists use duplicate suppression. Larger lists mark the active
expert set once. Long prefills expand all experts. The resulting native scale
buffers feed ordinary expert kernels. CUDA graph replay uses retained
programs and fixed storage, with no compilation or allocation.

vLLM times CSF expansion and its consuming MoE candidates through CUDA graph
replay. Activation and routing producers run before every timed replay, so
the captured operation sees the complete tuning workload. Trial graphs and
their buffers are released after selection. Graph-timed requests use distinct
selection keys from direct-launch requests.

`scripts/convert_csf_blocks.py SOURCE OUTPUT` converts slab-format checkpoint
scales into this representation. It preserves packed weights and calibration,
compares every converted logical scale byte, and writes a standard safetensors
index. The output directory must not exist. Source checkpoints remain
read-only, and the completed output is installed by directory rename.

`benchmarks/experimental/b12x/benchmark_nvfp4_csf_checkpoint.py SOURCE
--block-checkpoint OUTPUT --tuning-execution graph` compares native NVFP4, slab
expansion, and block expansion using a checkpoint's routed experts. The block
arm reads `OUTPUT` through vLLM's checkpoint reader and weight preparation.
Weights, calibration, and logical expanded scale bytes must match `SOURCE`
exactly before graph comparisons. Activations and routes are seeded synthetic
inputs; measurements cover one MoE component rather than the complete model.
The arithmetic oracle matches the selected kernel's FP4 rounding. Its cosine
bound is 0.9999, or 0.99989 for GLM fast-dynamic kernels whose BF16 atomic
reductions vary across identical graph replays. Both retain maximum absolute
error 0.05, RMSE 0.0075, and mean absolute error 0.005. Scale-byte comparisons
remain exact.

Both checkpoint and destination benchmarks accept
`--tuning-execution {direct,graph}` to compare candidate timing methods. Their
selection caches are separate. Final measurements use cold-L2 graph replay in
both modes.
