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
- Fri 30 Oct 2026, late: the 160/192-tile gap was one accumulator stage.  The DSL
  sized an accumulator stage with the TMEM *allocation* helper, which rounds columns
  up to a power of two (160 -> 256), so those tiles got a single stage and the epilogue
  ran exposed after every tile (ncu: smem pipe 69% busy vs 82% in the C++).  Counting
  the exact N columns gives two stages (320 or 384 columns plus 3 or 2 TMEM A stages =
  512, as the C++).  GB200 table after the fix (C++ / DSL us, gap): und.q 25.4/25.8
  +1.6%, 72.7/74.2 +2.1%, 127.8/128.3 +0.4%; und.kv 15.5/17.5 +12.9%, 21.9/23.1 +5.5%,
  28.3/28.3; gen.qkv 30.2/31.5 +4.3%, 86.7/87.4 +0.8%, 164.1/165.4 +0.8%; attn.out
  25.8/27.5 +6.6%, 72.4/72.4, 130.4/125.4 -3.8%; gate_up 61.0/61.8 +1.3%, 196.4/197.0,
  383.7/384.6; mlp.down 62.2/63.9 +2.7%, 202.7/202.8, 371.3/371.8.  Left: the M=1024
  rows (+1.3..6.6%) and the two split-K rows (+5.5%, +12.9%).
- Sat 31 Oct 2026, early: small-M residue hunted with a debug timeline (%globaltimer
  stamps per warp role, slots 16-34 of the per-CTA debug record).  First tile on GB200:
  prologue 1.4 us (barrier init 0.5, cluster sync 0.3, TMEM alloc + wait 0.4), first
  TMA -> first MMA 1.7 us, mainloop ~0.55 us per k-tile (L2-bound, same for the C++),
  epilogue 1.6-2.3 us per tile (0.3-0.4 us per 8 KB subtile: smem write, two named
  barriers, TMA store issue; TMEM load latency is not it).  Done: every warp derives its
  first work item itself (saves the 0.6 us scheduler hop); reduce kernel with a
  compile-time slice count and 16-byte-aligned slice loads.  Tried and dropped: TMEM
  load one subtile ahead (no change).  ncu, one tile per CTA (und.kv split-K): GEMM 20.6
  vs 19.4 us, reduce 7.2 vs 6.6 (M=4096); the C++ is non-persistent there (one CTA per
  tile, 128 CTAs) and cannot be stamped without plumbing a debug pointer through CUTLASS.
  Current GB200 table (C++ / DSL us): within +-1% on 11 of 18 rows; behind on the M=1024
  rows (gen.qkv 30.2/31.8, attn.out 25.8/27.2, gate_up 60.9/62.0, mlp.down 64.9/66.5),
  und.q 4096 72.7/74.2 and the split-K rows (und.kv 15.4/16.4, 21.7/22.9); ahead on
  attn.out 8192 (130.4/125.2).  Next: Hopper port; the small-M residue after that.
- Sat 31 Oct 2026: Hopper port first light on H100 (s4124-0017, x86 venv
  `w8a16-x86-venv`, public CUDA 13.3.1 for the C++ JIT).  Kernel: persistent warp-
  specialized, producer warpgroup + 1 or 2 consumer warpgroups, A (fp8 weight) smem ->
  registers -> ALU convert -> register-sourced wgmma, B from smem, TMA-store epilogue,
  C++ grouped tile order, split-K with tile-padded partials (no predicated store path).
  Every variant (128x256/192/160/128, 64x256, 64x128, split-K 10/11; bias on/off) matches
  the C++ bitwise where the C++ auto picks the same tile.  Lessons: the per-element row
  of a wgmma accumulator = identity partition (warpgroup-uniform) + warp*16 + lane/4; the
  fp32 partials go through the plain 32-bit copy whose fragment order is (row, col pair,
  n8) while the accumulator chunk runs (col pair, row, n8).  First H100 bench: DSL 7-50%
  behind; the cost model picks the ping-pong tiles on 8 of 18 rows, which run here as
  single-warpgroup tiles (half the wgmma issue) -> implementing true ping-pong next;
  stream-K rows (gate_up 4096/8192, mlp.down 1024) pay the wave quantization (+17%).
- Sat 31 Oct 2026, later: Hopper tuning.  Ping-pong schedule added (ordered mbarriers
  between the two consumer warpgroups), the e4m3 pair conversion runs straight from the
  16-bit load register, the weight stage is addressed as its own swizzled tensor (static
  element offsets).  H100 table (C++ / DSL us): und.q 30.4/33.1, 107.1/107.3, 210.5/211.5;
  und.kv 14.6/15.1, 26.9/29.4, 44.0/45.7; gen.qkv 40.5/43.2, 138.7/147.4, 261.0/262.9;
  attn.out 36.0/38.1, 114.2/120.7, 224.9/236.5; gate_up 107.6/108.2, 342.1/375.3,
  677.5/754.5; mlp.down 118.6/142.4, 332.5/365.6, 673.2/728.8.  Cooperative plain rows at
  parity; ping-pong rows 3-9% behind (ncu: still +30% ALU / FMA instructions and 2x
  branch ops against the C++ mainloop); the three stream-K rows 10-20% behind (wave
  quantization: 800 tiles / 132 SMs); mlp.down 4096/8192 (128x256 plain, K 6400) 8-10%.
  DSL lessons: a tensor returned from or passed through a @cute.jit helper leaves layout
  values in the helper's region (inline instead); a kernel must use its layout arguments,
  not self's host-side values; a tensor rebound inside a staged loop becomes loop-carried.
- Sat 31 Oct 2026, night: Hopper stream-K replacement.  Work items = whole-wave body of
  plain tiles + the last partial wave cut into K-slices (up to 16, one round, workspace
  <= 64 MB); the slices of a tile reduce it among themselves inside the kernel (publish
  the fp32 block with cp.async.bulk.wait_group 0 + fence.proxy.async.global +
  fence.acq_rel.gpu, atom.acq_rel.gpu counter, spin with nanosleep, each slice sums its
  own column slab, second count resets the counter; cached zero counters per device).
  Split-K codes = body 0.  Lessons: the per-group thread index (tidx % 128) is wrong for
  anything per tile under cooperative (two warpgroups) -> use the epilogue-group index;
  a jit helper cannot return a tensor nor take a pointer-built smem tensor, so the output
  epilogue is a helper taking partitions while the tail epilogue is inline; variables
  used in both branches of a staged `if` must exist before it.  H100 table (C++ / DSL
  us): und.q 30.5/30.4, 107.4/107.1, 210.9/210.9; und.kv 14.6/15.8 (+8%), 26.8/27.3,
  43.9/44.0; gen.qkv 40.5/39.8, 138.5/140.0, 261.6/261.9; attn.out 35.6/35.5,
  112.1/112.7, 221.2/224.8; gate_up 108.7/108.5, 340.3/334.1, 684.0/679.2; mlp.down
  109.2/107.6, 331.3/331.2, 674.0/676.0.  mlp.down 8192 swings +0.3..+9% run to run: at
  locked clocks (ncu) and with 5 iterations the two kernels are identical; the 200-
  iteration run shows the DSL 10% slower -> a sustained-clock / power effect, not
  kernel work.
