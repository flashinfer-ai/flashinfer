# W8A16 FP8 GEMM: CUTLASS C++ -> CuTe-DSL port

Branch `w8a16-dsl` (off `w8a16`, PR 7495). Goal set by Igor on Wed 29 Oct 2026: the DSL
kernel replaces the C++ one only once both run the benchmark
(`benchmarks/bench_w8a16_fp8_gemm.py`) at +-0.5% TFLOP/s on every row, on H100 and B200.
Until then the C++ stays as-is and the DSL kernel lives beside it.

## What exists to build on

- `flashinfer/gemm/kernels/cute_dsl/dense_gemm_bf16_fp4_sm100.py`: dense W4A16 kernel on
  sm100. Same architecture as our C++ Blackwell kernel: TMA warps (weight, activation),
  scheduler warp, MMA warp, 4 epilogue warps, two transform warpgroups (8 warps) that
  convert the weight tile in registers and store it to TMEM (or smem), 1SM and 2SM
  tcgen05, static persistent scheduler with a tile_info pipeline, TMA store epilogue with
  alpha, SIMT store for the N tail. The Blackwell port is a fork of this file.
- `cutlass.utils.mixed_input_helpers` (DSL library): transform partitions, smem layouts,
  epilogue partitions, TMA atom kinds.
- `cutlass.utils.PersistentTileSchedulerParams(..., swizzle_size, raster_along_m)`: the
  L2 tile grouping our C++ sets through `max_swizzle_size`.
- Hopper: `examples/python/CuTeDSL/hopper/dense_gemm_persistent.py` (SS wgmma, TMA,
  persistent) plus the RS wgmma path (`warpgroup.OperandSource.RMEM`) used by
  `hopper/fmha.py` and the MoE megakernel `kernel_mxfp4_fp8_glu_fc12_swapab.py` (register
  conversion in front of wgmma). No dense mixed-input Hopper example exists.
- Nothing in the DSL library does split-K or stream-K for GEMM.

## Contract to reproduce (from the C++)

- out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N]); act bf16 or fp16,
  weight e4m3 K-major, per-tensor fp32 scale read on device, bias per N.
- Conversion exact for every finite e4m3 code (`w8a16_convert.cuh`): Blackwell uses
  `cvt.rn.bf16x2.e4m3x2` (CUDA >= 13.2) or the 4-instruction ALU path; fp16 uses
  `cvt.rn.f16x2.e4m3x2`. NaN codes are the caller's problem.
- Blackwell variants (algo codes 1..7): 2SM 256x128/160/192 (A in TMEM, k-tile 128),
  1SM 128x128, 2SM 256x256 (A in smem, k-tile 64), 256x128 split-K, 256x128 stream-K.
  Dispatch: `W8A16Sm100PickAlgo` (dispatch.hpp), swizzle `W8A16Sm100PickSwizzle`.
- Hopper variants (codes 2..11): cooperative 128x{256,192,160,128}, ping-pong 64x{256,128},
  128x256 / 128x192 stream-K, 128x128 / 128x256 split-K, grouped raster (scheduler.hpp).
- Split-K = batch of K-slices (L mode) writing fp32 partials + reduce kernel (scale, bias,
  cast). Stream-K = CUTLASS scheduler; no DSL equivalent.
- Python API (`flashinfer/gemm/w8a16_fp8.py`): algo names, workspace query, checks.

## Plan

1. Blackwell plain variants in DSL (`w8a16_fp8_gemm_sm100.py` + `_utils.py`):
   fork the W4A16 kernel, drop the block scales, add the e4m3 convert (bf16 and f16),
   per-tensor alpha from a device pointer, per-row bias, a flag to put transformed A in
   smem for 256x256, k-tile 64/128, swizzle_size from PickSwizzle, raster from the
   CUTLASS heuristic. Verify exactness and correctness with the existing test matrix
   against the C++ kernel; bench row by row.
2. Blackwell split-K: L-mode batch + DSL reduce kernel. Stream-K: measure how much it
   buys on the bench rows where auto picks it; implement a DSL equivalent only if needed
   for the +-0.5%.
3. Hopper: build from the persistent dense GEMM example with RS wgmma and register
   conversion; cooperative + ping-pong, grouped raster, split-K; stream-K as in 2.
4. Wire a `dsl` column into the bench, run the full table on both GPUs, compare.
5. Replace: Python op dispatches to the DSL kernels, C++ removed, tests/docs updated.

## Status log

- Thu 29 Oct 2026: survey done, worktree created, plan written. Starting step 1.
- Fri 30 Oct 2026: Blackwell draft committed (0c7e5e166), reviewed, not yet run (no
  Blackwell node all night). Open: the L2 tile group (PickSwizzle) must be done by the
  scheduler warp, the DSL runtime scheduler cannot take a swizzled layout; algo 3
  (A in smem) needs its own compile check; split-K / stream-K not started.
- Fri 30 Oct 2026 (afternoon): first light on GB200 (gb-nvl-059-compute01, aarch64 venv
  `w8a16-aarch64-venv` + nvidia-cutlass-dsl 4.8.1, DSL CUDA 12.9 so the ALU e4m3 path).
  Hang on every CTA's second tile traced with the debug build to the tile_info consume
  helper: a PipelineState passed into a @cute.jit helper is rebuilt on entry, so its
  in-place advance() was lost (213d75789 returns it). Single-tile and multi-tile shapes
  pass for the 1SM tile; full matrix over the five tiles running.
- Fri 30 Oct 2026 (evening): all five Blackwell plain tiles correct, bias on and off,
  partial tiles included; max error identical to the C++ op, bitwise equal on the
  largest shape.  Scheduler warp now walks the tile grid itself with CUTLASS's sm100
  swizzle remap (L2 group from PickSwizzle, raster heuristic), replacing the library
  runtime scheduler.  First bench comparison on GB200 next; then split-K, stream-K
  decision, Hopper.
- Fri 30 Oct 2026, first bench (GB200, TP4, M 1024/4096/8192, CUDA-graph timing): DSL
  2-12% behind the C++ on plain tiles, 23-50% on the two rows where the C++ picks split-K
  (not ported yet), 24% on mlp.down 8192.  With fp16 activations (native convert on both
  sides) the gap is gone: mlp.down 372.0 vs 371.4 us, und.q 128.8 vs 132.6 (DSL ahead),
  gate_up 400 vs 384.  So the bf16 gap is the ALU e4m3->bf16 path: the aarch64 DSL wheel
  only bundles the CUDA 12 toolchain (CUDA_VERSION 12.9); the x86 wheel has cu13 and the
  one-instruction cvt.  Parity must be measured on x86 B200, or with the cu13 libs here.
- Fri 30 Oct 2026, night: split-K ported (algo 5: 2sm256x128 tile over K-slices writing
  fp32 partials [splits, m, n], then `w8a16_fp8_splitk_reduce.py` sums, scales, adds bias,
  casts).  Two dead ends on the way, both `cudaErrorIllegalInstruction` at launch, found
  with compute-sanitizer + cuobjdump: a 3-D operand view with the K-slice as the batch
  mode traps in the activation TMA load (UTMALDG), and a 3-D C view [n, m, splits] traps
  in the TMA store (UTMASTG).  Every tensor is now 2-D: the operands span the whole K and
  a tile's slice is a k-tile offset (`tile_l * k_tiles_per_slice`), the partials are
  C[n, splits*m] with the slice an activation-row offset.  `splits` is a compile-time
  constant.  Full matrix passes (algos 1,2,3,4,7,5; bias on/off; swizzled shapes), bitwise
  equal to the C++ split-K.  GB200 bench after split-K (C++ / DSL us): und.kv 15.3/17.4,
  21.8/23.1, 28.1/28.2 (was 25.1/25.9 for the first two).  Remaining gaps are the
  2sm256x160/192 tiles: attn.out 1024/4096 +16%, und.q 4096 +7%, gen.qkv +5-8%, mlp.down
  4096 +6%; attn.out 8192 the DSL is 4% ahead.  Note: `CUTE_DSL_DISABLE_FILE_CACHING=1`
  in the run scripts (the DSL keeps a compile cache under $TMPDIR keyed on the entry
  function; kept off while the kernel source changes often).
