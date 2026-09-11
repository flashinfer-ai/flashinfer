# PatchShift 3x3x3 Conv3d compute core

This directory contains only the framework-independent BF16 NDHWC compute
implementation. It deliberately excludes model integration, layout conversion,
bias, cache/concatenation logic, reference code, timing code, and fallback
operators.

## Public core files

- `problem.cuh`: fixed operator contract and runtime problem extents.
- `common.cuh`: CUDA, TMA, mbarrier, TMEM, and tcgen05 primitives.
- `kernels.cuh`: the stable single-translation-unit assembly point.

`kernels.cuh` is the only header a launcher should include. Its detail-header
order is intentional and preserves the original template visibility, inlining,
PTX, and kernel resource footprint.

## Detail kernel families

- `mainloop.cuh`: general M128 mainloop, C16/C32 paths, and compact spatial tails.
- `c64_and_hybrid.cuh`: C64/K64 pipeline and exact C96 hybrid paths.
- `cluster_a.cuh`: weight multicast across adjacent spatial CTAs.
- `cluster_a_hybrid_c96.cuh`: exact C96 cluster-A path.
- `cluster_b_c32.cuh`: activation multicast for logical M256 with C32 stages.
- `cluster_b_c64.cuh`: activation multicast for logical M256 with C64 stages.
- `output_tail.cuh`: M64 output-channel tails.
- `small_grid.cuh`: native M32/M64 P16 small-grid kernels.
- `m32_c64_small_grid.cuh`: M32 small-grid kernels with C64 activation stages.
- `m32_d1_shallow_c64.cuh`: shallow D1/C128/K128 M32 path.
- `m64_c64_small_grid.cuh`: M64 small-grid kernels with C64 activation stages.
- `m64_cluster_b.cuh`: logical M128 from two M64 cluster-B CTAs.
- `micro_d1.cuh`: D1/C32/K64 M32N128 micro path.
- `m64n128_micro_d1.cuh`: D1/C32/K64 exact M64N128 micro path.

Host-only TensorMap creation and shape routing live under
`csrc/patchshift_conv3d/`. They must not be moved into a device detail header.
