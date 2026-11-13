# cake_rmsnorm_det_train

Deterministic BF16 RMSNorm training kernels for SM100 (`sm_100a`), SM103 (`sm_103a`) and
SM107 (`sm_107a`). The forward writes `y`, `rstd` and, with a residual, `h_new`; the
backward writes `dx` and `dw`. Each route (architecture, hidden size, direction) selects
one program per call from the token count `T` (`cake_windows.PROGRAM_WINDOWS`).

## What repeats bit for bit

Two calls with identical inputs and the same hidden size, token count `T`, weight dtype
and residual setting, on devices of one architecture with the same SM count, return
identical bits in every output. No output depends on the order in which CTAs run: the
backward CTAs claim chunks of rows through counters, but each chunk writes its partial
`dw` row to its own slot, and the reduce sums the slots in a fixed order. Recomputing the
forward (activation checkpointing) gives the same `y`, `rstd` and `h_new` bits, so the
backward after a recomputation matches the backward after the saved forward.

| Output | Repeats bit for bit | What can change its bits between two calls |
| --- | --- | --- |
| `y` | yes | `T` across a forward window edge where the per-row thread group changes |
| `rstd` | yes | `T` across such a forward window edge |
| `h_new` | yes | nothing: `x + residual` rounded once to BF16 |
| `dx` | yes | `T` across a window edge whose class in `cake_windows.EDGES` is not `same` |
| `dw` | yes | `T` (it sums every row); the SM count, outside the column-task windows |

`cake_windows.EDGES` gives the measured class of the outputs at every window edge and
`cake_windows.NOTES` the bounds of the differences.

## Accuracy

Statistics are computed in FP32. `dx` is one BF16 rounding of FP32 math. With a residual,
`h_new` is rounded to BF16 before the normalization reads it. The package tests compare
every output with an FP64 reference of the same math:

- `y` and `dx`: `atol = rtol = 1e-2` after scaling both sides to unit RMS;
- `rstd`: relative L2 error at most `1e-5`;
- `dw`: relative L2 error at most `1e-4`, its median error over the test matrix no worse
  than the eager FP32 sum of the same products, and every element within
  `atol = rtol = 1e-4` of the FP64 sum at the largest row counts of the long-chunk
  windows and one row above each accuracy edge.

## dw summation order per window

The backward splits the rows of a call over `ctas_per_sm` x SM count slots and cuts each
slot's share into chunks by the window's chunk table: one chunk per slot, or a halving
tail (`tail_div`, `max_halving`). Each chunk sums its rows into one FP32 partial row of
`dw`, plainly or compensated over blocks of rows; the reduce program then folds the
partial rows of each column slab in a fixed order (`chain` or `tree`) with Kahan or
TwoSum compensation and one FP32 rounding. A row-strided window gives slot `s` the rows
`s`, `s + n`, `s + 2n`, ... with `n` = min(`T`, slots), and each slot sums its rows
plainly in row order into its partial row; where its kernel runs beside a side stream,
the window leaves eight SMs out of the slot count (`ctas_per_sm` x (SM count - 8)). A
column-task window sums each column over every row in one CTA in a fixed order (exact
products and TwoSum per row lane, the lanes folded in FP64 or FP32 triple words, one FP32
rounding), so there `dw` does not depend on the SM count. The FP32-weight routes use the
same windows and orders, except the `_w32` routes the tables list: they keep the
programs of windows where the BF16 route runs a program without an FP32-weight build.
Program and reduce names are the first eight hex digits of the program name.

### sm_100a

| Route | Rows `T` | Program | Partial sums | Chunk table | Reduce |
| --- | --- | --- | --- | --- | --- |
| `bwd_h1536` | (0, 1] | 273b578b | column task | - | - |
| `bwd_h1536` | (1, 768] | 8d4ec255 | column task | - | - |
| `bwd_h1536` | (768, 1024] | 7dc29629 | plain | tail_div 12, max_halving 5 | 9b727c2c (tree, Kahan) |
| `bwd_h1536` | (1024, 4096] | c1a9efdf | plain | tail_div 6, max_halving 5 | 1863320f (tree, Kahan) |
| `bwd_h1536` | (4096, 32768] | 21c8bb55 | plain | tail_div 12, max_halving 2 | 1863320f (tree, Kahan) |
| `bwd_h1536` | (32768, 36352] | b457c4d1 | plain | tail_div 12, max_halving 5 | 9b727c2c (tree, Kahan) |
| `bwd_h1536` | (36352, inf) | b5bdabba | compensated, 8-row blocks | tail_div 12, max_halving 5 | 9b727c2c (tree, Kahan) |
| `bwd_h2048` | (0, 1] | 61a2d9fc | column task | - | - |
| `bwd_h2048` | (1, 768] | 39d33412 | column task | - | - |
| `bwd_h2048` | (768, 1024] | cc26e316 | plain | row-strided slots | 880768a0 (tree, Kahan) |
| `bwd_h2048` | (1024, 2047] | c73074cf | plain | tail_div 12, max_halving 5 | d0960be9 (tree, Kahan) |
| `bwd_h2048` | (2047, 22528] | cc26e316 | plain | row-strided slots | 880768a0 (tree, Kahan) |
| `bwd_h2048` | (22528, 58880] | ceeef043 | plain | tail_div 12, max_halving 5 | d0960be9 (tree, Kahan) |
| `bwd_h2048` | (58880, inf) | 2fe43a95 | compensated, 8-row blocks | tail_div 12, max_halving 5 | 631af6e5 (tree, TwoSum) |
| `bwd_w32_h2048` | (0, 1] | 5f20b731 | column task | - | - |
| `bwd_w32_h2048` | (1, 768] | 12875cc8 | column task | - | - |
| `bwd_w32_h2048` | (768, 1024] | 07f9f2fb | plain | tail_div 12, max_halving 5 | 880768a0 (tree, Kahan) |
| `bwd_w32_h2048` | (1024, 16384] | b187e598 | plain | tail_div 12, max_halving 5 | d0960be9 (tree, Kahan) |
| `bwd_w32_h2048` | (16384, 58880] | 7e6a0b63 | plain | tail_div 12, max_halving 5 | d0960be9 (tree, Kahan) |
| `bwd_w32_h2048` | (58880, inf) | e6347c71 | compensated, 8-row blocks | tail_div 12, max_halving 5 | 631af6e5 (tree, TwoSum) |
| `bwd_h256` | (0, 87] | c0ca220c | column task | - | - |
| `bwd_h256` | (87, 2048] | a385ae16 | plain | tail_div 4, max_halving 5 | 72138dce (chain, Kahan) |
| `bwd_h256` | (2048, 5120] | af005462 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, Kahan, column-split) |
| `bwd_h256` | (5120, 12288] | a385ae16 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, Kahan, column-split) |
| `bwd_h256` | (12288, 40960] | af005462 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, Kahan, column-split) |
| `bwd_h256` | (40960, 80896] | a385ae16 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, Kahan, column-split) |
| `bwd_h256` | (80896, 1572864] | 083ea823 | compensated dot | tail_div 4, max_halving 5 | b4b63d5b (chain, TwoSum, column-split) |
| `bwd_h256` | (1572864, inf) | c18297d7 | compensated dot | tail_div 4, max_halving 5 | e73321ce (chain, TwoSum, high / low rows) |
| `bwd_h4096` | (0, 203] | bbc80180 | column task | - | - |
| `bwd_h4096` | (203, 256] | 3e644ea1 | plain | tail_div 12, max_halving 5 | 5d75ea5f (tree, Kahan) |
| `bwd_h4096` | (256, 1024] | 59270fb2 | plain | tail_div 4, max_halving 5 | 5d75ea5f (tree, Kahan) |
| `bwd_h4096` | (1024, 22528] | 5572b267 | plain | row-strided slots, 8 SMs left out | 5d75ea5f (tree, Kahan) |
| `bwd_h4096` | (22528, 32768] | 5abcdc29 | plain | tail_div 12, max_halving 5 | 5d75ea5f (tree, Kahan) |
| `bwd_h4096` | (32768, 44032] | 35a0b0ca | plain | tail_div 12, max_halving 5 | fcbfa14d (chain, Kahan) |
| `bwd_h4096` | (44032, inf) | 7122dbcd | compensated, 8-row blocks | tail_div 12, max_halving 5 | fcbfa14d (chain, Kahan) |
| `bwd_h512` | (0, 1] | 43d6b60b | column task | - | - |
| `bwd_h512` | (1, 227] | 5f3e3fcc | column task | - | - |
| `bwd_h512` | (227, 1024] | 09601d42 | plain | tail_div 12, max_halving 5 | 7097069b (tree, Kahan) |
| `bwd_h512` | (1024, 4096] | 123db0b2 | plain | tail_div 12, max_halving 5 | fef03ce5 (tree, Kahan) |
| `bwd_h512` | (4096, 32768] | ad1a5a3c | plain | tail_div 6, max_halving 5 | 7097069b (tree, Kahan) |
| `bwd_h512` | (32768, inf) | 8877322a | plain | tail_div 12, max_halving 5 | 122721bd (tree, Kahan) |
| `bwd_h5120` | (0, 1] | 6a1cfc92 | column task | - | - |
| `bwd_h5120` | (1, 256] | baa071a2 | column task | - | - |
| `bwd_h5120` | (256, 6144] | b482c84d | plain | tail_div 4, max_halving 5 | c5831b29 (tree, Kahan) |
| `bwd_h5120` | (6144, 50688] | c145ef19 | compensated, 8-row blocks | tail_div 4, max_halving 5 | c5831b29 (tree, Kahan) |
| `bwd_h5120` | (50688, inf) | c145ef19 | compensated, 8-row blocks | tail_div 4, max_halving 5 | cdcd68d9 (tree, TwoSum) |
| `bwd_w32_h5120` | (0, 1] | 9560032b | column task | - | - |
| `bwd_w32_h5120` | (1, 256] | 9560032b | column task | - | - |
| `bwd_w32_h5120` | (256, 6144] | 40227b2a | plain | tail_div 4, max_halving 5 | c5831b29 (tree, Kahan) |
| `bwd_w32_h5120` | (6144, 50688] | 9479b129 | compensated, 8-row blocks | tail_div 4, max_halving 5 | c5831b29 (tree, Kahan) |
| `bwd_w32_h5120` | (50688, inf) | 9479b129 | compensated, 8-row blocks | tail_div 4, max_halving 5 | cdcd68d9 (tree, TwoSum) |
| `bwd_h6144` | (0, 384] | 7100a5f1 | column task | - | - |
| `bwd_h6144` | (384, 767] | 5c804aed | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree, Kahan) |
| `bwd_h6144` | (767, 4096] | 61421bc7 | plain | row-strided slots, 8 SMs left out | 5f5cdc51 (tree, Kahan) |
| `bwd_h6144` | (4096, 42496] | b93bf060 | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree, Kahan) |
| `bwd_h6144` | (42496, inf) | c440e57b | compensated, 8-row blocks | tail_div 12, max_halving 5 | 89047c18 (tree, TwoSum) |
| `bwd_w32_h6144` | (0, 384] | 76a09fc0 | column task | - | - |
| `bwd_w32_h6144` | (384, 4096] | 41b80b6a | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree, Kahan) |
| `bwd_w32_h6144` | (4096, 42496] | 62beb52f | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree, Kahan) |
| `bwd_w32_h6144` | (42496, inf) | 0ae6eb51 | compensated, 8-row blocks | tail_div 12, max_halving 5 | 89047c18 (tree, TwoSum) |
| `bwd_residual_h6144` | (0, 1152] | d2749326 | column task | - | - |
| `bwd_residual_h6144` | (1152, 42496] | c1a3be94 | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree, Kahan) |
| `bwd_residual_h6144` | (42496, inf) | 84385468 | compensated, 8-row blocks | tail_div 12, max_halving 5 | 89047c18 (tree, TwoSum) |

### sm_103a

| Route | Rows `T` | Program | Partial sums | Chunk table | Reduce |
| --- | --- | --- | --- | --- | --- |
| `bwd_h1536` | (0, 1] | 0eaab647 | column task | - | - |
| `bwd_h1536` | (1, 31] | ad547a3c | column task | - | - |
| `bwd_h1536` | (31, 214] | 53285b95 | column task | - | - |
| `bwd_h1536` | (214, 1024] | 7dc29629 | plain | tail_div 12, max_halving 5 | 9b727c2c (tree, Kahan) |
| `bwd_h1536` | (1024, 4096] | c1a9efdf | plain | tail_div 6, max_halving 5 | 1863320f (tree, Kahan) |
| `bwd_h1536` | (4096, 14928] | 1df7db17 | plain | tail_div 12, max_halving 5 | 9b727c2c (tree, Kahan) |
| `bwd_h1536` | (14928, 15566] | 4df08ce5 | plain | tail_div 12, max_halving 2 | 1863320f (tree, Kahan) |
| `bwd_h1536` | (15566, 40448] | f522f5b8 | plain | tail_div 12, max_halving 5 | 1863320f (tree, Kahan) |
| `bwd_h1536` | (40448, inf) | cbeae168 | compensated, 16-row blocks | tail_div 12, max_halving 5 | 882ec880 (tree, TwoSum) |
| `bwd_h2048` | (0, 1] | 56635d2c | column task | - | - |
| `bwd_h2048` | (1, 2] | ce658895 | column task | - | - |
| `bwd_h2048` | (2, 171] | 2b4c0f5b | column task | - | - |
| `bwd_h2048` | (171, 1024] | caf6978e | plain | tail_div 12, max_halving 5 | 880768a0 (tree, Kahan) |
| `bwd_h2048` | (1024, 16384] | cc26e316 | plain | row-strided slots, 8 SMs left out | 880768a0 (tree, Kahan) |
| `bwd_h2048` | (16384, 32768] | caf6978e | plain | tail_div 12, max_halving 5 | 596b457a (tree, Kahan) |
| `bwd_h2048` | (32768, 55296] | 302f64db | plain | tail_div 12, max_halving 5 | d0960be9 (tree, Kahan) |
| `bwd_h2048` | (55296, inf) | 018f0cd7 | compensated, 8-row blocks | tail_div 12, max_halving 5 | d0960be9 (tree, Kahan) |
| `bwd_h256` | (0, 1680] | 083ea823 | compensated dot | tail_div 4, max_halving 5 | b4b63d5b (chain, TwoSum, column-split) |
| `bwd_h256` | (1680, 2048] | a385ae16 | plain | tail_div 4, max_halving 5 | 72138dce (chain, Kahan) |
| `bwd_h256` | (2048, 5120] | af005462 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, Kahan, column-split) |
| `bwd_h256` | (5120, 12288] | a385ae16 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, Kahan, column-split) |
| `bwd_h256` | (12288, 40960] | af005462 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, Kahan, column-split) |
| `bwd_h256` | (40960, 99840] | a385ae16 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, Kahan, column-split) |
| `bwd_h256` | (99840, 720896] | 083ea823 | compensated dot | tail_div 4, max_halving 5 | b4b63d5b (chain, TwoSum, column-split) |
| `bwd_h256` | (720896, 1572864] | 0291949b | compensated dot | tail_div 4, max_halving 5 | b4b63d5b (chain, TwoSum, column-split) |
| `bwd_h256` | (1572864, inf) | c18297d7 | compensated dot | tail_div 4, max_halving 5 | e73321ce (chain, TwoSum, high / low rows) |
| `bwd_h4096` | (0, 512] | 10f7bdf2 | column task | - | - |
| `bwd_h4096` | (512, 1024] | ad2bb26e | plain | tail_div 6, max_halving 5 | 5d75ea5f (tree, Kahan) |
| `bwd_h4096` | (1024, 24576] | 5572b267 | plain | row-strided slots, 8 SMs left out | 5d75ea5f (tree, Kahan) |
| `bwd_h4096` | (24576, 32768] | 5abcdc29 | plain | tail_div 12, max_halving 5 | 5d75ea5f (tree, Kahan) |
| `bwd_h4096` | (32768, 48640] | 35a0b0ca | plain | tail_div 12, max_halving 5 | fcbfa14d (chain, Kahan) |
| `bwd_h4096` | (48640, 51200] | 850eec3a | compensated, 8-row blocks | tail_div 4, max_halving 2 | 5d75ea5f (tree, Kahan) |
| `bwd_h4096` | (51200, inf) | e813cc30 | compensated, 8-row blocks | tail_div 4, max_halving 2 | 5d75ea5f (tree, Kahan) |
| `bwd_w32_h4096` | (0, 1] | 11820fe4 | column task | - | - |
| `bwd_w32_h4096` | (1, 512] | f03fb1c6 | column task | - | - |
| `bwd_w32_h4096` | (512, 1024] | 16545056 | plain | tail_div 6, max_halving 5 | 5d75ea5f (tree, Kahan) |
| `bwd_w32_h4096` | (1024, 24576] | 47a01388 | plain | row-strided slots, 8 SMs left out | 5d75ea5f (tree, Kahan) |
| `bwd_w32_h4096` | (24576, 32768] | a8eb7547 | plain | tail_div 12, max_halving 5 | 5d75ea5f (tree, Kahan) |
| `bwd_w32_h4096` | (32768, 48640] | 626aed74 | plain | tail_div 12, max_halving 5 | fcbfa14d (chain, Kahan) |
| `bwd_w32_h4096` | (48640, 51200] | cb164a27 | compensated, 8-row blocks | tail_div 4, max_halving 2 | 5d75ea5f (tree, Kahan) |
| `bwd_w32_h4096` | (51200, inf) | cc019a16 | compensated, 8-row blocks | tail_div 4, max_halving 2 | 5d75ea5f (tree, Kahan) |
| `bwd_h512` | (0, 1] | 9dc76b22 | column task | - | - |
| `bwd_h512` | (1, 4] | 117a66e4 | column task | - | - |
| `bwd_h512` | (4, 227] | 877751ba | column task | - | - |
| `bwd_h512` | (227, 1024] | 09601d42 | plain | tail_div 12, max_halving 5 | 7097069b (tree, Kahan) |
| `bwd_h512` | (1024, 4096] | 8e7f4d81 | plain | tail_div 12, max_halving 5 | 7097069b (tree, Kahan) |
| `bwd_h512` | (4096, 32768] | 274b0055 | plain | tail_div 6, max_halving 5 | 122721bd (tree, Kahan) |
| `bwd_h512` | (32768, inf) | cfaae3ce | plain | tail_div 12, max_halving 5 | 122721bd (tree, Kahan) |
| `bwd_h5120` | (0, 64] | e73ed933 | column task | - | - |
| `bwd_h5120` | (64, 5120] | 2fc4932f | plain | tail_div 4, max_halving 2 | c5831b29 (tree, Kahan) |
| `bwd_h5120` | (5120, inf) | 7ffb443a | compensated, 8-row blocks | tail_div 4, max_halving 2 | c5831b29 (tree, Kahan) |
| `bwd_h6144` | (0, 256] | 5d6088e6 | column task | - | - |
| `bwd_h6144` | (256, 767] | 5c804aed | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree, Kahan) |
| `bwd_h6144` | (767, 4096] | 61421bc7 | plain | row-strided slots, 8 SMs left out | 5f5cdc51 (tree, Kahan) |
| `bwd_h6144` | (4096, 48640] | 322574af | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree, Kahan) |
| `bwd_h6144` | (48640, inf) | 289dc9ab | compensated, 8-row blocks | tail_div 12, max_halving 5 | 5f5cdc51 (tree, Kahan) |
| `bwd_w32_h6144` | (0, 1] | 017091c7 | column task | - | - |
| `bwd_w32_h6144` | (1, 256] | f9c641ac | column task | - | - |
| `bwd_w32_h6144` | (256, 4096] | 41b80b6a | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree, Kahan) |
| `bwd_w32_h6144` | (4096, 48640] | 9a8c0396 | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree, Kahan) |
| `bwd_w32_h6144` | (48640, inf) | aa46aa07 | compensated, 8-row blocks | tail_div 12, max_halving 5 | 5f5cdc51 (tree, Kahan) |
| `bwd_residual_h6144` | (0, 1216] | 07135550 | column task | - | - |
| `bwd_residual_h6144` | (1216, 48640] | 04ebec87 | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree, Kahan) |
| `bwd_residual_h6144` | (48640, inf) | 21b5e791 | compensated, 8-row blocks | tail_div 12, max_halving 5 | 5f5cdc51 (tree, Kahan) |

### sm_107a

| Route | Rows `T` | Program | Partial sums | Chunk table | Reduce |
| --- | --- | --- | --- | --- | --- |
| `bwd_h1536` | (0, 1] | 33551729 | column task | - | - |
| `bwd_h1536` | (1, 576] | 9e71ed0d | column task | - | - |
| `bwd_h1536` | (576, 1024] | 7dc29629 | plain | tail_div 12, max_halving 5 | 628248fe (tree, Kahan) |
| `bwd_h1536` | (1024, 16384] | 6a428da2 | plain | tail_div 12, max_halving 2 | db5c0b67 (tree, Kahan) |
| `bwd_h1536` | (16384, 45056] | 56d26996 | plain | tail_div 12, max_halving 5 | db5c0b67 (tree, Kahan) |
| `bwd_h1536` | (45056, inf) | e694b0a2 | compensated, 4-row blocks | tail_div 12, max_halving 5 | db5c0b67 (tree, Kahan) |
| `bwd_h2048` | (0, 1] | bc9d86b3 | column task | - | - |
| `bwd_h2048` | (1, 256] | 32eb12e2 | column task | - | - |
| `bwd_h2048` | (256, 1024] | 845cdcd8 | plain | tail_div 12, max_halving 5 | 271082c5 (tree, Kahan) |
| `bwd_h2048` | (1024, 16384] | 81c3d8ab | plain | tail_div 12, max_halving 2 | c4d8be67 (tree, Kahan) |
| `bwd_h2048` | (16384, 50688] | 845cdcd8 | plain | tail_div 12, max_halving 5 | c4d8be67 (tree, Kahan) |
| `bwd_h2048` | (50688, inf) | 694decfb | compensated, 8-row blocks | tail_div 12, max_halving 5 | c4d8be67 (tree, Kahan) |
| `bwd_h256` | (0, 1572864] | 083ea823 | compensated dot | tail_div 4, max_halving 5 | b4b63d5b (chain, TwoSum, column-split) |
| `bwd_h256` | (1572864, inf) | c18297d7 | compensated dot | tail_div 4, max_halving 5 | 158c8876 (tree, TwoSum, high / low rows) |
| `bwd_h4096` | (0, 256] | 44511fa0 | column task | - | - |
| `bwd_h4096` | (256, 1024] | a647fccc | plain | tail_div 12, max_halving 5 | 69bf6e10 (tree, Kahan) |
| `bwd_h4096` | (1024, 4096] | d41a514f | plain | tail_div 4, max_halving 5 | 2030d808 (tree, Kahan) |
| `bwd_h4096` | (4096, 24064] | ee4a9053 | plain | tail_div 12, max_halving 5 | 10e24516 (tree, Kahan) |
| `bwd_h4096` | (24064, 50176] | 99a01f2a | compensated, 8-row blocks | tail_div 12, max_halving 5 | 10e24516 (tree, Kahan) |
| `bwd_h4096` | (50176, inf) | 9daa9bc7 | compensated, 8-row blocks | tail_div 12, max_halving 5 | 10e24516 (tree, Kahan) |
| `bwd_h512` | (0, 1] | 9dc76b22 | column task | - | - |
| `bwd_h512` | (1, 768] | 3c06f3e2 | column task | - | - |
| `bwd_h512` | (768, 1024] | 3feed502 | plain | tail_div 12, max_halving 5 | f94861c3 (chain, Kahan) |
| `bwd_h512` | (1024, 4096] | 3feed502 | plain | tail_div 12, max_halving 5 | d75459d9 (tree, Kahan) |
| `bwd_h512` | (4096, 16384] | dfaf378a | plain | tail_div 12, max_halving 5 | 8f032213 (tree, Kahan) |
| `bwd_h512` | (16384, inf) | eb141cc9 | plain | tail_div 4, max_halving 5 | 8f032213 (tree, Kahan) |
| `bwd_h5120` | (0, 1] | 505283a5 | column task | - | - |
| `bwd_h5120` | (1, 256] | e73ed933 | column task | - | - |
| `bwd_h5120` | (256, 1280] | 505283a5 | column task | - | - |
| `bwd_h5120` | (1280, 4096] | 838c077d | plain | tail_div 12, max_halving 5 | 2af4cf2f (tree, Kahan) |
| `bwd_h5120` | (4096, 19968] | 6816fb80 | plain | tail_div 4, max_halving 5 | 2af4cf2f (tree, Kahan) |
| `bwd_h5120` | (19968, inf) | 47da668e | compensated, 8-row blocks | tail_div 4, max_halving 5 | cdcd68d9 (tree, TwoSum) |
| `bwd_h6144` | (0, 1] | 57d45549 | column task | - | - |
| `bwd_h6144` | (1, 256] | dab76189 | column task | - | - |
| `bwd_h6144` | (256, 1664] | 57d45549 | column task | - | - |
| `bwd_h6144` | (1664, 19456] | 5d292172 | plain | tail_div 12, max_halving 5 | fb3f2a96 (tree, Kahan) |
| `bwd_h6144` | (19456, inf) | 19e18371 | compensated, 8-row blocks | tail_div 12, max_halving 5 | fb3f2a96 (tree, Kahan) |
| `bwd_residual_h6144` | (0, 368] | c15868b8 | column task | - | - |
| `bwd_residual_h6144` | (368, 19456] | 36cad444 | plain | tail_div 12, max_halving 5 | fb3f2a96 (tree, Kahan) |
| `bwd_residual_h6144` | (19456, inf) | b73a6086 | compensated, 8-row blocks | tail_div 12, max_halving 5 | fb3f2a96 (tree, Kahan) |
