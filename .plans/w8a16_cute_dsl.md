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
- Fri 30 Oct 2026, evening: the split-K reduces.  Hopper und.kv M=1024 (128x128 split-K,
  8 slices) was +8%: cutting the fused fix-up in stages showed GEMM + partial store at
  parity with the C++ GEMM (11.7 vs 11.55 us) and the fix-up at 4.0 us against the C++
  reduce kernel's 3.1 + 0.3 us gap.  Fixes: monotonic 64-bit tail counters (each slice
  adds lcm(1..16)/splits and waits for the next stride boundary; the atomic's return
  tells the last slice it is last, no reset, no second count) and 16-byte output stores
  (the output view's column stride spelled 8 * (rows / 8); with the dynamic row count
  the compiler emitted eight 2-byte stores).  15.8 -> 14.8 us (C++ 14.7).  Blackwell:
  the separate reduce kernel loaded the second and later slices as eight 4-byte loads
  (a host-computed stride arrives as an opaque integer); views built inside the kernel
  fix it, GB200 und.kv 1024 reduce 4.8 -> 3.55 us, rows 15.0/15.1 and 21.8/21.9 (DSL
  ahead).  B200 (x86, lease w8a16-b12) baseline differs from the GB200: M=8192 rows of
  gate_up / gen.qkv / mlp.down 4-13% behind on the same tile, at M=4096 the DSL wins
  most tiles -> suspected power-cap clock effect (B200 1000 W); measuring clocks and
  profiling at locked clocks next.
- Fri 30 Oct 2026, night: Blackwell mainloop.  Debug-mode cycle accounting per warp
  role (slots 40-63; the timeline script prints the shares) showed the transform warps
  ~92% busy and the leader MMA warp waiting 16-25% for transformed A: a k-tile's chain
  (4 LDS.128, 32 F2FP, 4 STTM, tcgen05.wait::st, arrive) is ~1000 cycles of latency
  with all eight warps on the same k-tile.  The two transform warpgroups now take
  alternate k-tiles (each converts a stage in two halves; raw stage released after the
  converts); arrival counts halved.  Not a power effect: both kernels run ~1865 MHz on
  the B200.  B200 at locked clocks (ncu, C++ / DSL us): mlp.down 8192 323.1/326.9,
  gate_up 8192 317.9/322.8, gate_up 4096 164.7/164.8, mlp.down 4096 166.7/168.2,
  attn.out 8192 108.2/107.9, gen.qkv 1024 26.1/27.0, attn.out 1024 22.8/23.6.  GB200
  bench: all M=4096/8192 rows within +-2% (attn.out 8192 -3.9%, gate_up 8192 -2.0%,
  mlp.down 8192 -2.1%); M=1024 rows +2.5..+6.6% (one tile per CTA: prologue 0.8 us,
  first TMA -> first MMA 1.4 us, epilogue 1.5 us per tile).  The B200 bench itself is
  +-4% noisy run to run (C++ mlp.down 8192 measured 331..357); use ncu for B200.
- Fri 30 Oct 2026, late night: Blackwell M=1024 rows.  Per-variant sweeps (GB200 bench,
  B200 locked clocks) show the DSL at parity or ahead on every tile except 2sm256x160
  and 2sm256x192, which the cost model picks for gen.qkv / attn.out at M=1024 (one or
  two tiles per CTA): +3..6%.  The mainloop rate per column matches the 128-wide tile,
  so the loss is per-tile fixed cost (C++ CTAs 35.4k cycles vs ours 38.2k on the 160
  tile).  Tried and reverted: a single accumulator stage for one-tile launches (five
  transformed-A stages instead of three: no change) and double-buffered TMEM loads in
  the epilogue (1.82 -> 1.73 us per 160-wide tile: the subtile cost is the store path
  and barriers, not the load).  Kept: 64-column epilogue subtiles where the tile allows
  (1.5 -> 1.0 us per 128-wide tile).  Open: the 160 tile's five 32-column subtiles
  (160 has no larger power-of-two divisor that fits a 128-byte TMA swizzle) and the
  first-tile fill (~2.2 us from the first TMA to the first MMA).
- Sat 31 Oct 2026, ~00:00: fixed-cost intercepts (B200 ncu, K = 1..16 k-tiles, M=1024 N=2560,
  C++ / DSL us at one k-tile): 2sm256x128 9.06 / 8.67, 1sm128x128 8.74 / 8.38 (DSL ahead),
  2sm256x160 8.77 / 9.60 (+0.83; slope 442 vs 461 ns per k-tile), 2sm256x192 8.80 / 10.02
  (+1.2; slopes equal at 514 ns).  Cycle counters: epilogue busy per tile 2866 (128, two
  64-column subtiles), 3358 (192, three), 3918 (160, five 32-column subtiles), end-of-kernel
  store drain included.  The CUTLASS SM100 epilogue issues the same per-subtile sequence
  (fence, barrier, TMA store by one warp, commit, acquire, barrier) and EpilogueTileAuto.
  Global-timer phases below ~1 us are quantization noise (the timer ticks coarsely); use
  the %clock accumulators.  H100 five-round table: all rows within +-1.3% except one
  throttled-GPU artifact (gate_up 8192 +17% while all times ran 30% slow; an isolated
  rerun gave 713.7 / 699.2).  Remaining Blackwell gap: gen.qkv / attn.out at M=1024
  (+4..6%, 160-wide tile, one tile per CTA) and mlp.down 1024 (+2.4%).
- Sat 31 Oct 2026, ~01:00: per-warp epilogue stores tried (each epilogue warp TMA-stores
  its own 32-row slab after a warp sync, no CTA barrier per subtile).  Wrong results
  (slab smem addressing, both the partitioned-pointer and the byte-address forms) and
  only -6% epilogue cycles on the 160 tile in the broken state (3918 -> 3676), so
  reverted rather than debugged: the whole epilogue is ~2 us of the 27 us row.  The
  160 / 192 fixed-cost gap (+0.8 / +1.2 us vs C++ at one k-tile; 128 and 1-SM tiles are
  0.4 us ahead) remains open; the epilogue is not the main part of it.
- Sat 31 Oct 2026, 02:00-04:30: the Blackwell small-M gap, found.  Three parts, all
  measured with %clock stamps (the global timer is only good for cross-CTA skew):
  (1) start-up: the tile derivation divided by two dynamic cluster counts (~0.4k cycles
  on every warp) and the TMA / scheduler warps waited at the TMEM-allocation barrier
  (~0.5k cycles) -> magic-number divides computed per launch, barrier only for TMEM
  users; (2) the TMA warps' suspending barrier wait and the activation pipeline depth
  were not it (tight poll and a sixth B stage changed nothing; the MMA's "waiting for
  B" is tensor-pipe backpressure); (3) the real one: a partial last activation tile
  (M not a multiple of the tile: 1024 vs 160 / 192) went through the predicated SIMT
  store, ~1.7 us slower than the TMA store, and those CTAs ended the kernel (per-CTA
  finish spread 2.3 us).  TMA stores clip out-of-bounds rows, so partial tiles use
  them whenever the output is final (split-K partials keep SIMT: slices abut).  GB200
  M=1024 rows now: gen.qkv -4.0%, attn.out -3.9%, mlp.down -1.8%, und.q -4.7%.  The
  C++ kernel has an opt-in -DW8A16_SM100_TRACE clock trace (block 0 roles, per-CTA
  end times) that made the per-phase comparison possible.
- Sat 31 Oct 2026, 05:00-06:30: Hopper small rows.  (1) A 64x128 split-K variant (algo
  12, DSL only) takes und.kv 1024 from 14.9 to 14.2 us (C++ 14.6): the cost model picks
  it.  (2) und.kv 4096 (M=4096, N=256: 128 tiles of 64x128 on 132 SMs, one tile per CTA,
  the ping-pong kernel's second consumer warpgroup idle) got a shared-item mode: a CTA
  with a single work item runs it on both consumer warpgroups, k-tiles by parity, group
  1 handing its accumulators to group 0 through the idle activation stages.  Correct on
  every shape (max error equals the C++'s; not bitwise, the halves sum in another
  order), but slower everywhere it engages: und.kv 4096 +4.1% (off: +1.5%), und.kv 8192
  +8%, und.q 1024 +8% (h20).  ncu at that shape: L2 52% of peak, DRAM 52%, L2 bytes
  100 MB (not the 4x activation re-read I expected), so it is not a bandwidth limit on
  paper; the stall breakdown is being collected.  Kept behind W8A16_DSL_KSPLIT (default
  on until the profile is read; to be removed or defaulted off).  ipp2-0146 lost its GPU
  a second time (h19, then h22); replacement h23 is s4124-0017.
- Sat 31 Oct 2026, 06:30-08:30: the Hopper ping-pong rows, closed.  ncu with and without
  the cache flush, boost clocks, h20 and h23: the C++ and DSL 64x128 / 64x256 kernels
  execute the same instruction count (31,782,452 vs 31,782,480 at gen.qkv 4096), the same
  smem loads, stalls, tensor-pipe activity and crossbar bytes, and the same cycles within
  0.1-0.3%; the time difference is the SM clock of the moment (sm__cycles_elapsed per
  second 1.659 vs 1.647 GHz).  und.kv 4096 is not L2-bandwidth bound either (crossbar
  8 TB/s vs 13.5 at gen.qkv 64x256; tensor pipe 42%, one warpgroup per SM) - the k-split
  helps only at ncu's locked clocks.  Measurement-order swap (--impls dsl kernel) left
  the +1% in place, so it is the bench conditions, not the kernel.
  Then the real finding: the DSL op's Python path cost 60 us per call (C++ op: 24 us),
  enough to starve the GPU on small rows (und.kv 4096 back to back: 60 us per call for a
  26 us kernel; the --no-l2-flush bench showed +6.6% for that reason).  cProfile: the
  Hopper cost model re-run per call (half), torch.cuda.current_stream() through the
  dynamo wrapper (a third), CuTe pointer objects built per call for an already compiled
  kernel.  Plan cached per shape, pointers built only at compile, raw stream handle:
  12 us per call, half the C++ op's.  Checks pass on H100 (algos 2, 7, 9, 12) and GB200.
- Sat 31 Oct 2026, 08:30-10:00: the Hopper ping-pong gap, found and closed.  A K sweep
  (CUPTI, M=4096, hot and cold L2) separated fixed cost from slope: the DSL 64x128 kernel
  had a lower intercept but a 4-10% steeper slope (hot: 4.06 vs 3.69 ns per K).  SASS of
  the two steady-state loops side by side (same LDS.U16 x4 / PRMT / SHL / LOP3 / HMUL2 /
  HGMMA.64x128x16 per k-block, DEPBAR.LE 3): the one structural difference is that
  CUTLASS issues SYNCS.PHASECHK.TRYWAIT on the next stage's full barrier right after the
  first HGMMA of the current k-tile and blocks at the loop top only if that probe failed;
  the DSL issued the blocking wait cold at the top.  Same probe now (consumer_try_wait
  token into consumer_wait).  First attempt probed unconditionally and cost 3.5 us per
  tile: try_wait on a stage nobody fills suspends the thread until the hardware time
  limit, so the probe is skipped on an item's last k-tile.  Result (h23): 64x128 slope
  3.41 ns per K (C++ 3.59); und.kv 4096 26.91 -> 26.56 cold (C++ 26.91), 22.98 -> 21.41
  hot; 64x256 gen.qkv 4096 139.10 -> 135.95 (C++ 139.10).  Five-round H100 table: 17 of
  18 rows DSL ahead (-0.1 to -4.8%), mlp.down 8192 +1.1% (a long row that swings with the
  node's thermals: -3.4 / +0.8 / +1.3 / +1.1 across runs; being re-measured).
- Sat 31 Oct 2026, 10:00-11:30: Hopper epilogue fixed cost.  K sweeps of the cooperative
  128x256 tile (M=8192 N=5120): slope equal or better, intercept +3.5 us (+6.7% at K=320),
  i.e. ~0.36 us per tile.  Cause: two epilogue staging buffers made the storing warp wait
  for every outstanding TMA store before each subtile's barrier, so the previous store's
  smem read sat on the critical path eight times per tile.  Three buffers (wait for all
  but the newest) where they do not cost a mainloop stage: K=320 +6.7% -> +0.2%, K=5120
  +3.1% -> +0.8% cold / -0.7% hot.  The 64x256 tile keeps two (a third dropped it from 6
  to 5 mainloop stages and lost more than it gained).  K sweeps of every tile the bench
  uses are now at or under parity at K=5120: 64x128 -2.3%, 64x256 -0.0%, 128x192 -0.1%,
  128x256 -1.0% (M=1024 N=6400) / +0.8% (M=8192 N=5120).
  Final five-round H100 table (h23): 15 of 18 rows under parity, -0.2 to -9.4%; und.q
  8192 +0.6, gen.qkv 8192 +1.6, gate_up 1024 +3.5 in that run, but the node's 8192 rows
  swing 30% between runs (gen.qkv 8192 C++ 282 then 331 us) and gate_up 1024 came back
  -0.5% on a seven-round rerun.  Under ncu locked clocks (both kernels identical
  conditions): und.q 8192 -0.3%, gen.qkv 8192 -0.4%, gate_up 8192 -0.2%, mlp.down 8192
  +0.3%, gate_up 1024 -0.1%.  GB200 three-round table: 16 of 18 under parity; the two
  8192 rows throttle in the bench (+1.4 / +2.4) and were -0.2 / -0.4% at locked clocks
  earlier.  Plot: scratchpad w8a16_parity.png.  B200 lease w8a16-b13 still queued (its
  last table predates today's fixes).  Open: remove the W8A16_DSL_KSPLIT, W8A16_DSL_FRAG,
  W8A16_DSL_AIN_SMEM and W8A16_DSL_FIXUP_CUT knobs before the PR; swap the public op to
  the DSL; tests and docs.
