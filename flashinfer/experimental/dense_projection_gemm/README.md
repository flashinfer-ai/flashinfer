# Cake dense projection GEMM backend with an explicit numerical mode (SM100 / SM103 / SM107)

Experimental backend for the GLM-5.2 training GEMMs -- the BF16 dense projection GEMM (K1: BF16 operands, FP32
accumulation, BF16 or FP32 output, strided views, ragged token counts, batched head projections) and the FP32
router-gate GEMM through split-BF16x3 tensor-core emulation (K2) -- with a selectable numerical mode.  Family issue:
flashinfer-ai/flashinfer#5677.

The deterministic implementation is the package `flashinfer.experimental.dense_projection_det_gemm`
(flashinfer-ai/flashinfer#5677, #7043; see its README for the view contract, the host paths and the per-architecture
programs).  This package keeps its operations, dtypes, accuracy budgets and host paths, and adds programs whose
split-K reductions add the partials in the order they finish.  Only run-to-run bitwise reproducibility is relaxed,
and only where a row adds more than two partials.

Both the API and the backend are experimental: no compatibility guarantee.

## Numerical mode

Every entry point of `cake_backend` takes the keyword-only `deterministic: bool = True`.

| call | runs |
|---|---|
| `deterministic=True` (default) | the deterministic package's canonical entry point with the same arguments: the same programs, the same bits, the same errors |
| `deterministic=False` | this package's programs on the rows that carry a measured verdict for the device (below); the deterministic programs for every other row and operation |
| `deterministic=False` under `torch.use_deterministic_algorithms(True)` | `RuntimeError`; with `warn_only=True` a `UserWarning` and the deterministic path |

*Table: what a call runs, by its `deterministic` argument.  The default does not read PyTorch's setting: it is
deterministic already.*

| operation | `deterministic=False` |
|---|---|
| K1 forward (`projection_forward`, `kk` layouts) | deterministic program (whole output tiles, or the deterministic plan's two-way split of K with the partials added in a fixed order; a recomputed forward reproduces its result) |
| K1 input gradient (`projection_dgrad`, `kn` layouts) | deterministic program (whole output tiles, or the deterministic plan's two-way split of K with the partials added in a fixed order) |
| K1 weight gradient (`projection_wgrad`, `nn` layouts) | the rows of `K1_SPLIT_RULES`: split-K over the ragged token count -- every split writes its FP32 partial tile to a workspace slot; the last split of a tile to arrive (a per-tile ticket counter) adds the other slots in arrival order and rounds once into `out`, then returns the counter to zero.  Every other row: deterministic program |
| K2 forward (`router_forward`, `kk`) | deterministic program (its four splits) on every row -- without a split count, while no router table gives the forward a verdict, without the per-call check of the row; an explicit `splits` of 1 or 2 runs this package's stream-K program (at most two partials per element) |
| K2 input gradient (`router_dgrad`, `kn`) | deterministic program (one split) on every row: the exact-FP32 SIMT windows measured on 152-SM `sm_100a` and 212-SM `sm_107a` parts missed the FP64 error rule (the router FP64 error table below), so no router table gives the input gradient a window and this checkout registers no SIMT program |
| K2 weight gradient (`router_wgrad`, `nn_t`) | the rows of `ROUTER_RULES` with a partition: stream-K over the K blocks, in spans alike or in long head spans and short tail spans (the head/tail program) -- whole tiles store the one-split result, tiles split between units TMA reduce-add into `out` (zeroed before the program, on the same stream).  Every other row: deterministic program (its six splits) |
| BF16x3 split (`split_fp32_to_bf16x3_`) | the deterministic stage (exact) |

*Table: the routing of `deterministic=False` (`cake_backend.ROUTES`).*

## Numerics, architectures and parallelism

| property | contract |
|---|---|
| operands and accumulation | BF16 operands with FP32 accumulation (K1); FP32 operands split exactly into three BF16 terms with FP32 accumulation (K2), except in a K2 SIMT window (input and weight gradient): the FP32 operands as they are, one FP32 FFMA per product on the CUDA cores, no BF16x3 split.  No silent TF32 or BF16 for the router |
| accuracy | the deterministic package's budgets against FP64 of the exact operands, worst over repeated runs: BF16 out `atol = rtol = 1e-2`; FP32 out relative Frobenius error `<= 5e-5` and max error `<= 1e-3 * max abs(ref)`; router `<= 2e-6` relative Frobenius error and no row whose top-8 experts differ from FP64 (outside near ties of the 8th and 9th logits).  At the largest token counts some thin K1 rows miss the BF16 (FP32) budget with the deterministic program and with this package's program alike; there the row keeps its route.  The deterministic program: BF16 output of `sm_100a` `4096 -> 32`, `4096 -> 48`, `4096 -> 64`, `5120 -> 32`, `5120 -> 48` and of `sm_103a` `4096 -> 32`, `4096 -> 48`, `5120 -> 32`, `5120 -> 48`, `5120 -> 64`, `5120 -> 128` (from 45058, 49153, 65536 or 65537 tokens up by row: the K1 token tables below), and FP32 output of the rows of the odd-block K1 table at an odd number of K blocks from 98305 tokens up.  This package's program: BF16 output of `sm_100a` `5120 -> 32` at 53249 to 65536 tokens on parts of more than 148 SMs, and of `sm_103a` `5120 -> 48` from 49152 tokens and `5120 -> 64` from 49153 tokens on 148-SM parts |
| rounding | each output element is rounded once, from its complete FP32 sum |
| subnormals | kept: K1 adds the partials with IEEE round-to-nearest adds, K2 reduces with the TMA reduce-add; a K2 SIMT window (input and weight gradient) accumulates with round-to-nearest FFMA, adds the sums of a CTA's slices in a fixed order through shared memory and stores every element of `out` once, with no reduce-add; nothing flushes to zero |
| run-to-run bits, K1 | two splits (every `sm_100a` and `sm_103a` row, `sm_107a` `8192 -> 128`): repeat bit for bit (one add of two values); one split (`sm_107a` rows marked `p = 1` below): repeat bit for bit (whole tiles); three or four splits (the other `sm_107a` rows): vary in the last bits |
| run-to-run bits, K2 | forward: the deterministic program's bits; weight gradient on a `ROUTER_RULES` partition: varies in the last bits (more than two partials per element); a SIMT window (weight or input gradient): repeats bit for bit (every element stored once, its slices added in a fixed order); with an explicit forward `splits` of 1 or 2: repeats bit for bit (`0 + a + b == 0 + b + a`), so recomputed router logits and top-k match |
| architectures | `sm_100a`, `sm_103a`, `sm_107a`: one `-gencode=arch=compute_<arch>,code=sm_<arch>` build per program, minimum CUDA 12.8 / 12.9 / 13.4 (`build.json`) |
| streams and threads | planning takes a module lock; the convenience path keeps one workspace arena per (device, stream); prepared launches own their workspace |
| CUDA graphs | prepared launches allocate nothing on the device and capture into a CUDA graph (K2: the split, the memset of `out`, the program); no host synchronization |
| other kernels on the device | the programs walk their work persistently over at most `sm_count // 2` CTA pairs and wait on no CTA that is not resident; every routed row was measured next to a kernel holding 8, 16 or 32 SMs on another stream (no more than 3 % slower than the deterministic plan there) |
| context parallelism (KCP) | not applicable: a local operator on the caller's CP-local shards; no collectives; the reduction of parameter gradients across ranks stays with the caller |
| autograd, checkpointing, gradient accumulation | none in either package: callers compose the forward and both gradients; `out` is overwritten, not accumulated into |

*Table: the numerical, architecture and parallelism matrix of `deterministic=False`.*

## Planners and rule tables

A planner decides, per row and device, whether `deterministic=False` runs this package's program and with which
parameters; a row it leaves runs the deterministic program (`plan_dense_projection_gemm` / `plan_router_fp32_gemm`
say which).  Only measured rows carry a program: a row runs this package's program where that program was faster
than the deterministic package's shipped plan of the row in every measurement run: a one-sided paired sign test
(p < 0.05) with a median time ratio below 0.99, both in eager launches and in CUDA graph replays, and no more than
3 % slower than that plan next to a kernel holding 8, 16 or 32 SMs on another stream.  A `None` entry is a measured
row that missed; a row without an entry was not measured.  Both run the deterministic program.

The K1 verdicts were measured at 16172 to 16384 tokens.  Those of `K1_SPLIT_RULES` hold at every token count their split
count fits (`num_splits * 64 <= T + 63`), except the token counts of `K1_SPLIT_RULE_EXCLUDED_TOKENS` (the K1 token
tables below: `sm_100a` / `sm_103a` parts of 152 SMs or more, `sm_107a` parts of 208, 212 and 216 SMs), on parts
of exactly 148 SMs those of `K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` (the 148-SM K1 token table below), and, where the
token count spans an odd number of 64-token K blocks, those of `K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` (the odd-block K1
table below); the tables of
`K1_SPLIT_RULES_BY_SM` and the overrides of `K1_SPLIT_OVERRIDES_BY_SM` hold only at 16172 to 16384 tokens
(`K1_SPLIT_RULES_BY_SM_TOKENS`) and, for the rows of `K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`, at the token ranges of that
table (the 212-SM K1 token-range table below), except those token counts; the rows of
`K1_SPLIT_RANGE_OVERRIDES_BY_SM` run that table's program at its token ranges (the same table), and those of
`K1_TWO_SPLIT_RANGES_BY_SM` at its token ranges, at either number of 64-token K blocks (the same table on 212 SMs,
`4096 -> 128` FP32 included; on 152-SM `sm_100a` / `sm_103a` parts the 152-SM two-split table below).  Before all of
these, the rows of `K1_CHAIN_RANGES_BY_SM` run the one-split program of several accumulation chains at that table's
token ranges on parts of exactly its SM count, at both block parities (the K1 accumulation-chain table below).  A K1
program runs only at token counts where its output met the deterministic package's budgets against FP64 (the accuracy
row above) on every measured seed; the ratio of its FP64 relative Frobenius error to the deterministic program's on
the same operands (q) is reported in the tables below as data, not as a rule.  Where both programs miss the budget,
the row keeps its route.
The router verdicts hold only inside token windows (`ROUTER_RULES`; the last table below) and pass a stricter rule:
measured in one long-lived process and in a fresh process per token count, and no more than 3 % slower than the
deterministic plan next to such a kernel both in the same launch order and with the router partition launched before
that kernel against the deterministic plan launched after it.  A router window also keeps the FP64 error of the plan
the row takes without it: for every seed, a relative Frobenius error against FP64 of the exact operands (worst of 20
runs) at most 1.02 times that plan's, 10 seeds per measured token count (the router FP64 error table below).  Every
other token count of a router row runs the deterministic program.  The K1 verdicts hold from the floor of `K1_MIN_SM_COUNT` /
`K1_RULE_MIN_SM_COUNT` up (a part with more SMs than the measured ones takes them without a measurement, a part with
fewer runs the deterministic program); the tables of `K1_SPLIT_RULES_BY_SM` hold for exactly 208 and 212 SMs, the
overrides of `K1_SPLIT_OVERRIDES_BY_SM` for exactly 216 SMs.  The
router windows hold only on the SM count they were measured on (`ROUTER_SM_COUNT`: 152 on `sm_100a` / `sm_103a`, 216 on
`sm_107a`), and those of `ROUTER_RULES_BY_SM` only on exactly the SM count of their table (208 and 212 on `sm_107a`);
every other SM count runs the deterministic plan, except in the `"waves"` windows, which keep the earlier SM condition
of `"waves"` (the deterministic plan's last wave below 80 % full).  A SIMT window comes first in its row and may
overlap the windows after it: on an SM count where it does not hold, the token count takes the next window that holds
it.  The measured parts have 148 and 152 SMs (`sm_100a` /
`sm_103a` K1), 152 SMs (`sm_100a` / `sm_103a` router) and 208, 212 and 216 SMs (`sm_107a`).

| table | key | value |
|---|---|---|
| `K1_SPLIT_RULES` | `(arch, L, M, N, out_dtype)`: the transposed-store launch `out[l, n, m]` (`M` = the stored inner extent, `d_in` of a weight gradient; `N` = its rows, `d_out`), `out_dtype` `"bf16"` / `"f32"` | `(block_n, num_splits)` (this package's program; `block_n` 192 is the row-major-store program with the operands exchanged) or `None` (the deterministic program) |
| `K1_MIN_SM_COUNT` | `arch` | the fewest SMs for the K1 verdicts of the architecture (`sm_100a` / `sm_103a`: 148, `sm_107a`: 216) |
| `K1_RULE_MIN_SM_COUNT` | a key of `K1_SPLIT_RULES` | the fewest SMs for that row's verdict where it is above `K1_MIN_SM_COUNT` (`sm_103a` `4096 -> 48`: 152) |
| `K1_SPLIT_RULES_BY_SM` | `(arch, sm_count)` | the K1 verdicts of the parts with exactly `sm_count` SMs below a row's floor (keys and values as in `K1_SPLIT_RULES`), at the token counts of `K1_SPLIT_RULES_BY_SM_TOKENS` and of the row's ranges in `K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` only; a part without a table, and every other token count, runs every K1 row below the floor on the deterministic program (`sm_107a`: 208 and 212 SMs) |
| `K1_SPLIT_RULE_EXCLUDED_TOKENS` | a key of `K1_SPLIT_RULES` or of a table of `K1_SPLIT_RULES_BY_SM` | `((sm_count, min_tokens, max_tokens), ...)`: the row runs the deterministic program at a token count from `min_tokens` to `max_tokens` (`None`: no upper bound) on a part of at least `sm_count` SMs (`sm_count` at or above the row's SM-count floor) or of exactly `sm_count` SMs (below it); every other token count and part keeps the verdict |
| `K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` | `(arch, sm_count)`, then a key of `K1_SPLIT_RULES` | `((min_tokens, max_tokens), ...)`: the row runs the deterministic program at a token count from `min_tokens` to `max_tokens` (`None`: no upper bound) on a part of exactly `sm_count` SMs; every other token count and part keeps its routing (`sm_100a` / `sm_103a`: 148 SMs, the 148-SM K1 token table below) |
| `K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` | a key of `K1_SPLIT_RULES` or of a table of `K1_SPLIT_RULES_BY_SM` | `((sm_count, min_tokens, max_tokens), ...)` (parts as in `K1_SPLIT_RULE_EXCLUDED_TOKENS`): the row runs the deterministic program at a token count from `min_tokens` to `max_tokens` (`None`: no upper bound) that spans an odd number of 64-token K blocks (`ceil(T / 64)` odd); an even number, and every other token count and part, keeps its routing (FP32 two-split rows of `sm_100a` and `sm_103a`: the odd-block K1 table below) |
| `K1_SPLIT_RULES_BY_SM_TOKENS` | -- | `(min_tokens, max_tokens)`: the token counts (`K` of the weight gradient) the tables of `K1_SPLIT_RULES_BY_SM` and `K1_SPLIT_OVERRIDES_BY_SM` hold for, `(16172, 16384)` (where they were measured) |
| `K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` | `(arch, sm_count)`, then a key of that part's table in `K1_SPLIT_RULES_BY_SM` | `((min_tokens, max_tokens), ...)`: further token counts at which the row takes its verdict from the part's table, except the token counts of `K1_SPLIT_RULE_EXCLUDED_TOKENS`; every other row, part and token count keeps `K1_SPLIT_RULES_BY_SM_TOKENS` alone (`sm_107a`: 212 SMs, the 212-SM K1 token-range table below) |
| `K1_SPLIT_RANGE_OVERRIDES_BY_SM` | `(arch, sm_count)`, then a key of that part's table in `K1_SPLIT_RULES_BY_SM` | `((min_tokens, max_tokens, (block_n, num_splits)), ...)`: on a part of exactly `sm_count` SMs below the row's floor, a token count from `min_tokens` to `max_tokens` runs that program, except the token counts of `K1_SPLIT_RULE_EXCLUDED_TOKENS`; a range lies outside `K1_SPLIT_RULES_BY_SM_TOKENS` and the row's `K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` (`sm_107a`: 212 SMs, the 212-SM K1 token-range table below) |
| `K1_TWO_SPLIT_RANGES_BY_SM` | `(arch, sm_count)`, then a key as in `K1_SPLIT_RULES` (below the row's floor: a key of that part's table in `K1_SPLIT_RULES_BY_SM`) | `((min_tokens, max_tokens, (block_n, num_splits)), ...)`: on a part of exactly `sm_count` SMs, below the row's floor or at or above it, a token count from `min_tokens` to `max_tokens` runs that program at either number of 64-token K blocks, except the token counts of `K1_SPLIT_RULE_EXCLUDED_TOKENS`; a range lies outside `K1_SPLIT_RULES_BY_SM_TOKENS` and the row's ranges in `K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` and `K1_SPLIT_RANGE_OVERRIDES_BY_SM` (`sm_107a`: 212 SMs, the two-split rows of the 212-SM K1 token-range table below, `4096 -> 128` FP32 without a verdict in the part's table of `K1_SPLIT_RULES_BY_SM`; `sm_100a` / `sm_103a`: 152 SMs, the 152-SM two-split table below) |
| `K1_SPLIT_OVERRIDES_BY_SM` | `(arch, sm_count)` | verdicts (keys and values as in `K1_SPLIT_RULES`) that replace a row's `K1_SPLIT_RULES` verdict on parts of exactly `sm_count` SMs at or above the row's floor, at the token counts of `K1_SPLIT_RULES_BY_SM_TOKENS` only, except the token counts of `K1_SPLIT_RULE_EXCLUDED_TOKENS`; every other token count and SM count keeps the `K1_SPLIT_RULES` verdict (`sm_107a`: 216 SMs, `4096 -> 1024` FP32 (192, 1), the row-major-store program) |
| `K1_CHAIN_RANGES_BY_SM` | `(arch, sm_count)`, then a key of `K1_SPLIT_RULES` | `((min_tokens, max_tokens, (block_n, chains)), ...)`: on a part of exactly `sm_count` SMs, a token count from `min_tokens` to `max_tokens` runs this package's one-split program of that tile width with that many accumulation chains (`k1_chain_plan`; the chain counts this package ships are `K1_CHAINS_CHOICES`), before every table above, `K1_TWO_SPLIT_RANGES_BY_SM` included (`sm_107a`: 212 and 216 SMs, the K1 accumulation-chain table below) |
| `ROUTER_RULES` | `(arch, layout, hidden, experts)`: `layout` `"kk"` (forward, `hidden = K`), `"nn_t"` (weight gradient, `hidden` = the launch's `M`) or `"kn"` (input gradient, `hidden` = the launch's `N`), `experts` = the launch's `N` (`K` of the input gradient) | `None` (the deterministic program), token windows `((min_tokens, max_tokens, spec), ...)` (the spec of the window that holds the token count -- `K` of the weight gradient, `M` of the forward and of the input gradient; every other token count: the deterministic program) or a spec alone (every token count: a plan a test or a measurement forces).  A spec is `"waves"`, `"H<f>x<k>"`, `"H<f>x<k>p<h>"`, `"forward"`, `"pairs"`, `"tiles"`, `"S<n>"`, a unit count (this package's stream-K program) or `"simt_n<tile_n>m<tile_m>s<slices>"` (the exact-FP32 SIMT program of the row's layout, `router_simt`; the only spec of an input-gradient row) |
| `ROUTER_SM_COUNT` | `arch` | the SM count of the router verdicts of the architecture (`sm_100a` / `sm_103a`: 152, `sm_107a`: 216); a device with any other SM count runs the deterministic program, except in a `"waves"` window (the condition of `router_waves`; below a SIMT window too) or with a table of its own in `ROUTER_RULES_BY_SM` |
| `ROUTER_RULES_BY_SM` | `(arch, sm_count)` | the router verdicts of the parts with exactly `sm_count` SMs (keys and values as in `ROUTER_RULES`; `router_rules` picks the table of a device); such a part takes its own table only: a row without an entry, and every token count outside its windows, runs the deterministic program (`sm_107a`: 208 and 212 SMs) |

*Table: the rule tables.*

| architecture | K1 weight gradient `d_in -> d_out` (`block_n`, `num_splits`), BF16 and FP32 out unless marked | K2 forward | K2 weight gradient |
|---|---|---|---|
| `sm_100a` (>= 148 SMs) | `4096 -> 32, 48, 64`; `5120 -> 32, 48, 64`; `6144 -> 32`: (64, 2).  `4096 -> 128`: deterministic (measured, missed); FP32 on 152 SMs: (64, 2) from 1024 to 4096 tokens (the 152-SM two-split table) | deterministic (4 splits) | 256 experts, hidden 5120 from 1 to 128 tokens: `"waves"` (on every SM count where it applies).  Every other token count, SM count and hidden size: deterministic (6 splits); the SIMT and head/tail windows of 152-SM parts missed the FP64 error rule (the router FP64 error table below) |
| `sm_103a` (>= 148 SMs) | `4096 -> 32`; `5120 -> 32, 48, 64`; `6144 -> 32`: (64, 2); `4096 -> 48` (>= 152 SMs): (64, 2); `5120 -> 128`, `6144 -> 128`: (128, 2).  `4096 -> 64`, `4096 -> 128`: deterministic (measured, missed); `4096 -> 128` FP32 on 152 SMs: (64, 2) from 1024 to 6144 tokens (the 152-SM two-split table) | deterministic (4 splits) | 256 experts, hidden 5120 from 1 to 160 tokens: `"waves"` (on every SM count where it applies).  Every other token count, SM count and hidden size: deterministic (6 splits); the head/tail windows of 152-SM parts, and `"waves"` from 161 tokens, missed the FP64 error rule (the router FP64 error table below) |
| `sm_107a` (>= 216 SMs) | `4096 -> 32` BF16 (64, 4), FP32 (64, 3); `4096 -> 48` FP32 (64, 3); `4096 -> 64` FP32 (64, 3); `5120 -> 32` (64, 3); `5120 -> 48`, `5120 -> 64` FP32 (64, 3); `6144 -> 32` (64, 3); `8192 -> 128` (128, 2); `4096 -> 512` FP32 (128, 1) `p = 1`; `5120 -> 512` FP32 (128, 1) `p = 1`; `4096 -> 1024` FP32 (256, 1) `p = 1` (on exactly 216 SMs at 16172 to 16384 tokens: (192, 1) `p = 1`, the row-major-store program, 176 CTAs); `5120 -> 1024` (256, 1) `p = 1`, except the token counts of the `sm_107a` K1 token table (the four FP32 rows with one split: from 40961 tokens up).  `4096 -> 64` BF16, `4096 -> 128`, `4096 -> 512` BF16, `4096 -> 4096` FP32: deterministic (measured, missed) | deterministic (4 splits) | 256 experts, in the token window of the last table: on 216 SMs hidden 5120 `"H0.5x4"` (the head/tail program).  Every other token count and SM count, and hidden 4096 / 6144: deterministic (6 splits): at 4096 `"H0.5x4"` loses next to a kernel holding 16 SMs that starts during it, at 6144 it leaves the backward pass no faster |
| `sm_107a` (208 SMs, 16172 to 16384 tokens) | `4096 -> 32, 48, 64`, `5120 -> 32, 48, 64` FP32: (64, 3); `4096 -> 512`, `5120 -> 512` FP32 (128, 1) `p = 1`; `4096 -> 1024`, `5120 -> 1024` FP32: (256, 1) `p = 1`; except the token counts of the `sm_107a` K1 token table.  `4096 -> 32, 64` BF16, `5120 -> 32` BF16, `6144 -> 32`: deterministic (no verdict: its stream-K program did not complete next to a resident kernel); `4096 -> 512` BF16, `5120 -> 1024` BF16, `4096 -> 4096` FP32, `8192 -> 128`: deterministic (measured, missed) | deterministic (4 splits) | 256 experts, hidden 5120 at 16172 to 16384 tokens: `"H0.5x4"` (the head/tail program).  Every other token count, and hidden 4096 / 6144: deterministic (6 splits): next to a resident kernel the head/tail program missed at both hidden sizes on the parts at full SM clock |
| `sm_107a` (212 SMs, 16172 to 16384 tokens and the ranges of the 212-SM K1 token-range table) | `4096 -> 32, 48, 64` FP32 (64, 3); `5120 -> 32, 48, 64` FP32 (64, 3); `8192 -> 128` (128, 2); `4096 -> 512`, `5120 -> 512` FP32 (128, 1) `p = 1`; `4096 -> 1024` FP32: (192, 1) `p = 1` (row-major-store program, 176 CTAs); `5120 -> 1024` FP32: (256, 1) `p = 1`; except the token counts of the `sm_107a` K1 token table.  At the two-split ranges of the 212-SM K1 token-range table, at either number of 64-token K blocks (`K1_TWO_SPLIT_RANGES_BY_SM`): `4096 -> 512`, `5120 -> 512` FP32 (128, 2); `4096 -> 1024` FP32 (192, 2) (row-major-store program); `5120 -> 1024` FP32 (256, 2); `4096 -> 128` FP32 (64, 2) from 1024 to 7680 tokens (no verdict in the 212-SM table: deterministic at every other token count).  `4096 -> 32, 64` BF16, `5120 -> 32` BF16, `6144 -> 32`: deterministic (no verdict, as on 208 SMs); `4096 -> 512` BF16, `4096 -> 4096` FP32: deterministic (measured, missed; `4096 -> 4096` FP32 against the deterministic package's 160-wide tiles of the row); `5120 -> 1024` BF16: (256, 1) `p = 1` at its range of the 212-SM K1 token-range table (`K1_SPLIT_RANGE_OVERRIDES_BY_SM`), deterministic at 16172 to 16384 tokens (1.3 to 2.5 % faster there, missed on 208 SMs) | deterministic (4 splits) | 256 experts, hidden 5120 and 6144 at 16172 to 16384 tokens: `"H0.5x4"` (the head/tail program with the BF16 parts of `A` in tensor memory, `ROUTER_HEAD_TAIL_TEMPLATES_BY_SM`).  Every other token count, and hidden 4096: deterministic (6 splits): the SIMT windows (hidden 4096 and 5120 from 1 to 256 tokens, hidden 6144 from 1 to 192) missed the FP64 error rule (the router FP64 error table below); next to a resident kernel `"H0.5x4"` missed at hidden 4096 with both head/tail programs, and at hidden 6144 with the three-chunk-buffer program |

*Table: the rows that run this package's programs.  `4096 -> 128` misses on `sm_100a` / `sm_103a` next to a kernel
holding 32 SMs (its 128 CTAs no longer fit the free SMs) and on `sm_107a` at 16384 tokens; on `sm_107a`, `4096 -> 512`
BF16 (192 CTAs) and `4096 -> 4096` FP32 (216 CTAs) miss next to a kernel holding 32 SMs the same way.  `4096 -> 64`
(`sm_103a`; `sm_107a` BF16) and `4096 -> 48` on 148-SM `sm_103a` parts are not faster than the deterministic plan in
every run.  The `sm_107a` one-split rows store whole tiles: they win where the deterministic package's `sm_107a` program
of the row is slower (the FP32 ones on 216 SMs and more up to 40960 tokens: from 45056 tokens up they miss the FP32
budget, the `sm_107a` K1 token table; on 212-SM and 216-SM parts these FP32 rows run three accumulation chains at one
split at the ranges of the K1 accumulation-chain table).  Every K1 choice also needs `num_splits * 64 <= T + 63`
(shorter token counts run the deterministic program); the token counts of the K1 token tables below run the
deterministic program (`sm_100a` / `sm_103a`: parts of 152 SMs or more, and parts of exactly 148 SMs; `sm_107a`: 208,
212 and 216 SMs); the 208-SM and 212-SM choices and the 216-SM `4096 -> 1024` FP32 choice hold only at 16172 to 16384
tokens (`K1_SPLIT_RULES_BY_SM_TOKENS`), and the 212-SM choices of the rows of the 212-SM K1 token-range table also at
its ranges.*

| K1 row `d_in -> d_out` | `sm_100a` BF16 | `sm_100a` FP32 | `sm_103a` BF16 | `sm_103a` FP32 |
|---|---|---|---|---|
| `4096 -> 32` | 512; 65536 and up† | -- | 512; 65536 and up§ | 512; 32768 and up§ |
| `4096 -> 48` | 512; 49153 and up† | 512 | 512; 32768 and up§ | 512; 32768 and up§ |
| `4096 -> 64` | 512; 65536 and up† | 512 | no verdict | no verdict |
| `5120 -> 32` | 512; 65537 and up† | 512 | 512; 45058 and up‡ | 512 |
| `5120 -> 48` | 512 to 2048; 49153 and up† | 512 to 2048 | 512 to 2048; 40961 and up‡ | 512 to 2048 |
| `5120 -> 64` | 512 to 2048; 8192 | 512 to 2048; 8192 | 512 to 2048; 45058 and up‡ | 512 to 2048 |
| `5120 -> 128` | no verdict | no verdict | 512 to 8192; 45058 and up‡ | 512 to 4096 |
| `6144 -> 32` | 1024 | -- | 512 | 512 |
| `6144 -> 128` | no verdict | no verdict | 512 to 8192 | 512 to 8192 |

*Table: the K1 token counts that run the deterministic program on `sm_100a` / `sm_103a` parts of 152 SMs or more
(`K1_SPLIT_RULE_EXCLUDED_TOKENS`; `--`: none; `no verdict`: the row runs the deterministic program at every token
count).  Every other token count keeps the row's verdict.  Measured at 512, 1024, 2048, 4096, 8192, 16231, 32768 and
65536 tokens on two 152-SM parts per row and token count (GB200: two parts; GB300: four parts, because the strict runs
of 8 `sm_103a` points, 3 of them in this table (`4096 -> 32` BF16, `5120 -> 32` FP32, `6144 -> 32` FP32 at 512 tokens),
ran on two other parts than their screen): a screen of 9 paired groups per row and token count in one long-lived process
per GPU, eager launches and CUDA graph replays, next to a kernel holding 8, 16 or 32 SMs in both launch orders (a launch
order without paired groups: the slowest split-program time against the fastest deterministic time), with the backward
pass and the checkpointed step timed; where a screen run failed, 15 paired groups in a long-lived process and in a fresh
process per row, on two parts per token count (one part for `sm_100a` `6144 -> 32` FP32 at 1024 tokens); and the FP64
error of 5 seeds on each of two parts per architecture (prepared launches, CUDA graph replays, the convenience call)
next to the deterministic program's on the same operands.  A measured token count runs the deterministic program where
the split program failed in at least one run (the K1 rule above; the backward pass or the checkpointed step more than
1.3 % slower than with the deterministic program; or the deterministic package's FP64 error budget missed for some seed)
and the deterministic program leaves the backward pass and the checkpointed step no more than 1.3 % slower (medians and
paired medians, eager launches and graph replays), or where the split program's output missed the deterministic
package's FP64 error budget for some seed (marked ‡, †).  The ratio of the split program's error to the deterministic
program's (q) is data, not part of the rule.  A token count between two consecutive such measured token counts runs the
deterministic program too (not measured).  Marked ‡: the BF16 output of `sm_103a` `5120 -> 32`, `5120 -> 64` and
`5120 -> 128` met the BF16 budget on every seed at 32768 tokens (30 seeds on two parts) and at every measured token
count up to 45057 tokens on a 152-SM GB300 part (10 seeds, 20 runs each); at the next measured token count, 47105, both
programs missed it for some seed, so the row keeps the deterministic program from the token count after the last passing
one (the split program misses it on every seed at 65536 tokens, 30 seeds on two parts).  `5120 -> 48` met it on that
part on every seed up to 49152 tokens (the deterministic program missed it for one seed at 45057 and at 49152 tokens),
but on a 148-SM B300 part, which runs the same program, it missed it for one of 10 seeds at 40961 and at 40962 tokens
while the deterministic program met it (the 148-SM table below), so its entry starts at 40961 tokens; from 49153 tokens
both programs miss it on the 152-SM part.  Marked §: the entries of `sm_103a` `4096 -> 32` and `4096 -> 48` reach above
65536 tokens (up to 65536 tokens they come from the rule above).  There neither program meets the BF16 budget on any
seed (10 seeds per token count from 65537 tokens up on a 148-SM part; the deterministic plan of these rows is the same
on 148 and 152 SMs), and the FP32 split program gives the deterministic program's bits at an even number of 64-token K
blocks; at an odd number it met the FP32 budget on every seed at 65537 tokens (q 1.021 to 1.024) and both programs miss
it from 98305 tokens up.  It was not measured faster there, so these rows keep the deterministic program from their
entry up.  Marked †: `sm_100a` BF16 entries from 148 SMs up (also in the 148-SM table below; `4096 -> 64` from 152 SMs
up, its 148-SM entry is below) where the split program misses the BF16 budget for some seed (10 seeds per token count on
a 148-SM B200 part; the deterministic plan of these rows is the same on 148 and 152 SMs): at 49153 tokens 8 of 10 seeds
met it for `4096 -> 48` and 9 for `5120 -> 48`; at 65536 tokens 3 for `4096 -> 32`; from 65537 tokens up none.  The
deterministic program misses it there too (`4096 -> 48` 9 of 10 seeds and `5120 -> 48` 8 of 10 at 49153 tokens,
`4096 -> 32` 3 of 10 at 65536 tokens and none from 65537 tokens up), so these rows keep the deterministic program from
their entry up.  `5120 -> 32` (10 seeds, 20 runs each, at 32769 to 65025 tokens): on a 152-SM GB200 part the split
program met the budget on every seed up to 49153 tokens, and from 53249 tokens both programs missed it for some seed
there and on a 148-SM B200 part, so the row keeps its entry from 65537 tokens (on 148 SMs it starts at 49153 tokens: the
148-SM table below).  Token counts below 512, above 65536 (other rows), between two measured token counts with different
outcomes, and on parts with fewer than 152 SMs were not measured and keep the verdict.  Measured token counts that
failed the rule but keep the verdict, because the deterministic program makes the backward pass or the checkpointed step
more than 1.3 % slower there: BF16 output at 1024 tokens on `sm_103a` `4096 -> 48`, `6144 -> 32`; BF16 output at 2048
tokens on `sm_100a` `5120 -> 32`, `6144 -> 32`; `sm_103a` `4096 -> 48`; FP32 output at 512 tokens on `sm_100a`
`4096 -> 32`, `6144 -> 32`; FP32 output at 1024 tokens on `sm_100a` `6144 -> 32`; FP32 output at 8192 tokens on
`sm_103a` `4096 -> 48`, `5120 -> 128`.  At 65536 tokens neither program meets the BF16 error budget.*

| K1 row `d_in -> d_out` | `sm_100a` BF16 | `sm_100a` FP32 | `sm_103a` BF16 | `sm_103a` FP32 |
|---|---|---|---|---|
| `4096 -> 32` | 65536 and up† | -- | 65536 and up‡ | 65536 and up‡ |
| `4096 -> 48` | 512; 49153 and up† | 512 | no verdict | no verdict |
| `4096 -> 64` | 512; 65536 and up‡ | 512; 65536 and up‡ | no verdict | no verdict |
| `5120 -> 32` | 49153 and up | -- | -- | -- |
| `5120 -> 48` | 512 to 2048; 49153 and up† | 512 | 512 to 2048; 40961 to 49151 | 512 to 2048 |
| `5120 -> 64` | 512 to 2048; 8192 | 512 to 1024; 8192 | 512 to 1024; 47105 to 49152 | 512 to 1024 |
| `5120 -> 128` | no verdict | no verdict | 512 to 4096; 32768 to 65536 | 512 to 4096; 32768 to 65536 |
| `6144 -> 32` | -- | -- | -- | -- |
| `6144 -> 128` | no verdict | no verdict | 512 to 8192 | 512 to 8192 |

*Table: the 148-SM K1 token table: the K1 token counts that run the deterministic program on `sm_100a` / `sm_103a` parts
of exactly 148 SMs (`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`; †: the entries of the 152-SM table above that hold from 148
SMs up; `--`: none; `no verdict`: the row runs the deterministic program at every token count, `sm_103a` `4096 -> 48`
below its 152-SM floor).  Every other token count keeps the row's verdict, and parts of any other SM count keep their
routing.  Measured with the protocol and the rule of the 152-SM table above at the same token counts, on B300
(`sm_103a`) and B200 (`sm_100a`) parts of 148 SMs: a screen of every split-routed row and token count on one part (9
paired groups, one long-lived process per GPU), the points that failed there again on a second part (15 paired groups in
a long-lived process and in a fresh process per point; B300: a part of a second node, which also screened every point;
B200: another part of the same node), and the FP64 error of 5 seeds on two parts at 32768 and 65536 tokens and at the
first and the last token count of every entry and one token either side of it (98304 and 131072 tokens too for the
entries that reach 65536 tokens).  ‡: entries from the FP64 error and the time.  `sm_100a` `5120 -> 32` BF16 from 49153
tokens: from the FP64 error alone (the † text of the 152-SM table above); its 152-SM entry from 65537 tokens holds on
148 SMs too.  `sm_103a` `5120 -> 48` BF16 at 40961 to 49151 tokens and `5120 -> 64` BF16 at 47105 to 49152 tokens: from
the FP64 error alone, measured with 10 seeds (20 runs each) at 36865 to 61441 tokens: the split program missed the BF16
budget for one of 10 seeds at 40961 and at 40962 tokens (`5120 -> 64`: 47105) while the deterministic program met it on
every seed (`5120 -> 48` met it at 41985 to 47105 tokens inside its range); from 49152 tokens (`5120 -> 64`: 49153) both
programs miss it for some seed, so the split program stays there.  Above 65536 tokens the split program of `4096 -> 32`
(`sm_103a`) and `4096 -> 64` (`sm_100a`) gives other bits than the deterministic program where the token count spans an
odd number of 64-token blocks (measured at 65537, 65538, 65600 and 65665 tokens; `4096 -> 64` BF16 on `sm_100a` also at
11 more odd counts from 65728 to 98240 tokens) and the same bits where it spans an even number (65601 to 131072 tokens,
12 counts measured).  There the BF16 output of both programs misses the BF16 budget for some seed, and the FP32 output
of both misses the FP32 budget at an odd count from 98305 tokens up (q 1.021 to 1.023 at the measured odd counts).  At
the even counts the split program does not make the backward pass or the checkpointed step 1 % faster (`sm_100a`: the
backward pass up to 3.5 % slower).  So these rows run the deterministic program from 65536 tokens up.  The BF16 output
of `5120 -> 32` (23 token counts from 57336 to 131072 tokens) and `5120 -> 128` (65536 to 131072 tokens) on `sm_103a`
keeps the split program on 148 SMs (q median 1.000 and 1.015, largest 1.26 and 1.15); its absolute error against the
BF16 budget there is not measured with 10 seeds yet.  Unmeasured token counts keep the verdict.  Measured token counts
that failed the rule but keep the verdict, because the deterministic program makes the backward pass or the checkpointed
step more than 1.3 % slower there on a part: `sm_100a`: BF16 output at 512 tokens on `4096 -> 32`, `5120 -> 32`,
`6144 -> 32`, at 1024 on `5120 -> 32`, at 2048 on `6144 -> 32`, at 32768 on `5120 -> 48`, at 65536 on `4096 -> 32`,
`4096 -> 48`, `5120 -> 48`, `5120 -> 64`, `6144 -> 32`; FP32 output at 512 tokens on `4096 -> 32`, at 1024 and 2048 on
`5120 -> 48`, at 2048 on `5120 -> 64`, at 65536 on `5120 -> 48`; `sm_103a`: BF16 output at 512 tokens on `4096 -> 32`,
`5120 -> 32`, `6144 -> 32`, at 2048 on `5120 -> 64`, at 8192 on `5120 -> 128`, at 65536 on `5120 -> 48`, `5120 -> 64`,
`6144 -> 32`, `6144 -> 128`; FP32 output at 512 tokens on `6144 -> 32`, at 2048 on `5120 -> 64`, at 8192 on
`5120 -> 128`, at 32768 on `4096 -> 32`.*

| K1 row `d_in -> d_out` | parts | token counts with an odd number of 64-token K blocks that run the deterministic program |
|---|---|---|
| `sm_100a` `4096 -> 32` FP32 | 148 SMs and more | 65538 and up |
| `sm_100a` `4096 -> 48` FP32 | 148 SMs and more | 65538 and up |
| `sm_100a` `4096 -> 64` FP32 | 148 SMs and more | 65538 and up |
| `sm_100a` `5120 -> 32` FP32 | 148 SMs and more | 65538 and up |
| `sm_100a` `5120 -> 48` FP32 | 148 SMs and more | 65538 and up |
| `sm_103a` `4096 -> 32` FP32 | 148 SMs and more | 65538 and up |
| `sm_103a` `4096 -> 48` FP32 | 152 SMs and more | 65538 and up |

*Table: the odd-block K1 table (`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS`): these rows run the deterministic program at a
token count of the third column that spans an odd number of 64-token K blocks (`ceil(T / 64)` odd); an even number, and
every other row, keeps the routing of the tables above.  In these ranges the deterministic program of the row cuts the
token reduction of its tiles, taken in order, into units of half the tile's K blocks, rounded up (at an odd block count
a unit runs on into the next tile; its automatic two-way split), and adds the partials in a fixed order.  At an even
block count the two-split program's output is the deterministic program's bit for bit.  At an odd count its FP32 output
met the deterministic package's FP32 budget on every seed at every measured odd token count from 1025 to 65537 tokens
(10 seeds each: the `4096` rows at 1025, 1088, 2049, 3000, 4097, 8193, 12289, 16173, 24577, 32769, 49153 and 65537
tokens, `sm_100a` on a 148-SM B200 part and `sm_103a` on a 148-SM B300 part, the deterministic plan of these rows being
the same on 148 and 152 SMs; `sm_100a` `5120 -> 32` and `5120 -> 48` at 12337 to 65537 tokens), so those token counts
run the two-split program; q was 1.03 to 1.15, 1.021 to 1.024 at 65537 tokens and 1.007 to 1.075 for the `5120` rows
(data).  From 98305 tokens up both programs miss the FP32 budget at an odd count; no odd token count from 65538 to 98304
was measured, and the rows run the deterministic program at an odd count from 65538 tokens up.  `sm_107a` `8192 -> 128`
FP32 has no entry: at odd block counts from 1025 to 8191 tokens its two-split program met the FP32 budget on every seed
(10 seeds on 212-SM R200 parts; q 1.04 to 1.14 at 1025, 1088, 2049 and 4097 tokens).  The BF16 outputs of these rows had
the deterministic program's relative Frobenius error and met the error budget on every seed where the deterministic
program met it.*

| K1 row `d_in -> d_out` (FP32 out) | 152 SMs: token ranges, every K-block count (`K1_TWO_SPLIT_RANGES_BY_SM`) | measured token counts outside them |
|---|---|---|
| `sm_100a` `4096 -> 128`: (64, 2) | 1024 to 4096 | 512; 6144; 8192 |
| `sm_103a` `4096 -> 128`: (64, 2) | 1024 to 6144 | 512; 8192 |

*Table: the 152-SM two-split table (`K1_TWO_SPLIT_RANGES_BY_SM` on `sm_100a` / `sm_103a` parts of exactly 152 SMs, above
the SM-count floor): the FP32 row `4096 -> 128`, which has no verdict in `K1_SPLIT_RULES`, runs two splits of 64-wide
tiles, each adding half of an output's K blocks, at every token count of its range, at either number of 64-token K
blocks.  Outside the range, with BF16 output and on parts of any other SM count (148 SMs included) it runs the
deterministic program.  At an even block count its output was the deterministic program's bit for bit (5 seeds at 1024,
2048, 4096, 6144 and 8192 tokens); at 1088 and 2112 tokens (17 and 33 blocks) it was not, with q 1.107 and 1.142 (5
seeds on three parts; data: the deterministic program's FP64 relative Frobenius error there is about 1 % of the FP32
budget).  Measured at 512, 1024, 2048, 4096 and 8192 tokens on two 152-SM parts of each GPU (two GB200 nodes, two GB300
nodes) and at 6144 tokens on one GB200 part and two GB300 parts: 15 paired groups of 20 launches in four runs per part
(two launch orders, a long-lived process and a fresh process per row), eager launches and CUDA graph replays, next to a
kernel holding 8, 16 or 32 SMs in both launch orders, with the backward pass and the checkpointed step timed.  A range
runs between measured token counts at which the two-split program passed the K1 rule in every run on every part, left
the backward pass and the checkpointed step no more than 1.3 % slower than the deterministic program, and met the
deterministic package's FP64 error budget for every seed; the token counts between two consecutive measured ones inside
it, odd block counts included, take the verdict too (not measured).  From 1024 to 4096 tokens its time was 0.70 to 0.79
times the deterministic program's (graph replays, both GPUs; 8.65 against 12.14 us on a GB200 part at 1024 tokens).  At
512 tokens it is slower than the deterministic program, and at 8192 tokens slower next to a kernel holding 32 SMs; at
6144 tokens on the GB200 part its time next to a kernel holding SMs on another stream was 1.099 times the deterministic
program's.*

| K1 row `d_in -> d_out` (FP32 out unless marked BF16) | 208 SMs | 212 SMs | 216 SMs |
|---|---|---|---|
| `4096 -> 32` | -- | -- | 16231 |
| `4096 -> 32` BF16 | -- | -- | 512¶ |
| `4096 -> 48` | 16231; 16384 | -- | 512¶; 16231; 16384 |
| `4096 -> 64` | 16231; 16384 | -- | 512¶; 16231; 16384 |
| `4096 -> 512` | 16231; 16384 | 16231; 16384 | 16231; 16384; 40961 and up§ |
| `4096 -> 1024` | 16231 | 16231 | 16231; 40961 and up§ |
| `5120 -> 32` | -- | -- | 512¶ |
| `5120 -> 32` BF16 | -- | -- | 512¶ |
| `5120 -> 48` | 16231; 16384 | 16231 | 512¶; 16231; 16384 |
| `5120 -> 64` | 16231 | 16231 | 512¶; 16231; 16384 |
| `5120 -> 512` | -- | 16231; 16384 | 40961 and up§ |
| `5120 -> 1024` | -- | -- | 40961 and up§ |
| `5120 -> 1024` BF16 | -- | -- | 16231 |
| `6144 -> 32` BF16 | -- | -- | 512¶ |
| `8192 -> 128`, BF16 and FP32 | -- | -- | 512¶; 65536¶ |

*Table: the K1 token counts that run the deterministic program on `sm_107a` parts (`K1_SPLIT_RULE_EXCLUDED_TOKENS`;
`--`: none).  Unmarked ones are token counts of the deterministic package's `sm_107a` rules for single token counts
(16231 and 16384 tokens; `4096 -> 1024` at 16231 only), which give the row a faster deterministic plan there; every
other token count keeps the row's verdict.  On 212 SMs `4096 -> 4096` FP32 runs the deterministic program at every token
count (`K1_SPLIT_RULES_BY_SM`): the deterministic package's rule for that row holds at every token count.  On 208, 212
and 216 SMs the rows were measured at 16172, 16231 and 16384 tokens against those plans: 15 paired groups of 20 steps
per row and token count with the L2 cache flushed before every launch, in a long-lived process per GPU and in a fresh
process per row, in both arm orders, eager launches and CUDA graph replays, next to a kernel holding 8, 16 or 32 SMs in
both launch orders, with the backward pass and the checkpointed step timed (9 paired groups); 212 SMs: eight parts in
two nodes (`4096 -> 1024`: sixteen parts in five nodes), 208 SMs: two parts (`5120 -> 1024`: three), 216 SMs: two parts.
A measured token count runs the deterministic program where the row's program failed the K1 rule above in some run, or
left the backward pass or the checkpointed step more than 1.3 % slower than the deterministic program (ratio of the
medians over the runs, or median of the paired ratios, in eager launches or in graph replays).  `4096 -> 1024` on 212
SMs and `5120 -> 1024` on 208 SMs were measured twice; the rule takes the runs of both measurements together.  On 216
SMs the 192-wide row-major program of `4096 -> 1024` FP32 (the 216-SM override) was measured in the same runs against
the 256-wide tiles and the deterministic plan: faster than both at 16172 and 16384 tokens in every run, not faster than
the deterministic package's 16231-token plan, which that token count keeps.  The 216-SM entry of `5120 -> 1024` BF16
(16231 tokens) is not a token count of those rules: the deterministic plan there is the row's plan at every token count,
and the row's program left the backward pass 1.5 % slower in eager launches (ratio of the medians over the runs).
Marked §: entries of their own, not from the rules above.  On parts of at least 216 SMs these FP32 rows run the
deterministic program from 40961 tokens up because their one-split program's FP64 relative Frobenius error exceeds the
deterministic package's FP32 error budget (5e-5) on every seed there.  That program adds the whole token reduction of an
output in one chain, and its error grows about linearly with the token count.  Measured with the 216-SM plan on 212-SM
R200 parts (the same program and accumulation order as on 216 SMs; the error does not depend on the SM count) against an
FP64 reference, each row on two parts, 5 seeds, at 32768, 36864, 40960, 45056, 49152, 57344, 65536 and 131072 tokens:
within the budget on every seed up to 40960 tokens (1.86e-5 at 16384 tokens, 3.77e-5 at 32768), above it on every seed
from 45056 tokens up (7.6e-5 at 65536 tokens and 1.5e-4 at 131072; the deterministic program: 3.8e-5 and 7.6e-5).  From
961 tokens (16 K blocks) up the deterministic program of these rows cuts the token reduction into units of half an
output's K blocks, rounded up (its automatic two-way split), so below 40961 tokens q is 2.0 to 2.4 (data; up to 960
tokens both programs add one chain and give the same bits).  The token counts between 40960 and 45056 run the
deterministic program too (not measured), and the range has no upper bound: the error keeps growing with the token
count.  The deterministic program is slower there: at 65536 tokens on a 212-SM part it takes 1.81 (`4096 -> 512`), 1.05
(`4096 -> 1024`), 1.72 (`5120 -> 512`) and 1.02 (`5120 -> 1024`) times the one-split program's time (CUDA graph replays)
and leaves the backward pass or the checkpointed step up to 51.7, 4.8, 41.5 and 3.9 % slower.  Marked ¶: single token
counts from a re-measurement of chosen points at 512, 8192, 32768 and 65536 tokens on two 216-SM R200 parts per point
(the protocol above: 15 paired groups, eager launches and graph replays, next to a kernel holding 8, 16 or 32 SMs in
both launch orders, the backward pass and the checkpointed step timed).  A point is an entry where the row's program
failed the K1 rule in some run and the deterministic program leaves the backward pass and the checkpointed step no more
than 1.3 % slower on both parts (the worst of the ratio of the medians and the median of the paired ratios, eager
launches and graph replays: the rule of the `sm_100a` / `sm_103a` entries); the neighbouring token counts were not
measured and keep the verdict.  Measured points that failed but keep the verdict, because the deterministic program
makes the backward pass or the checkpointed step more than 1.3 % slower there on a part: BF16 output at 8192 tokens on
`4096 -> 32`, at 32768 tokens on `4096 -> 32` and `5120 -> 1024`, at 65536 tokens on `5120 -> 1024`; FP32 output at 512
tokens on `4096 -> 32` and `6144 -> 32`.*

| K1 row `d_in -> d_out` (FP32 out unless marked BF16) | 212 SMs: token ranges besides 16172 to 16384 | measured token counts outside them |
|---|---|---|
| `4096 -> 32`, `4096 -> 48`, `4096 -> 64` | 1024 to 16172; 16384 to 65536 | 512 |
| `5120 -> 32`, `5120 -> 48`, `5120 -> 64` | 1024 to 16172; 16384 to 65536 | 512 |
| `8192 -> 128`, BF16 and FP32 | 1024 to 16172; 16384 to 32768 | 512; 65536 |
| `4096 -> 512`, `4096 -> 1024`, `5120 -> 512`, `5120 -> 1024` | 512 | 1024 to 8192; 32768; 65536 |
| `5120 -> 1024` BF16: (256, 1) `p = 1` (`K1_SPLIT_RANGE_OVERRIDES_BY_SM`) | 512 to 12288 | 32768 |
| `4096 -> 512`, `5120 -> 512`: (128, 2), every K-block count (`K1_TWO_SPLIT_RANGES_BY_SM`) | 1024 to 8192; 32768 to 65536 | 512 |
| `4096 -> 1024`: (192, 2), every K-block count (`K1_TWO_SPLIT_RANGES_BY_SM`) | 2048 to 8192; 32768 to 65536 | 512; 1024 |
| `5120 -> 1024`: (256, 2), every K-block count (`K1_TWO_SPLIT_RANGES_BY_SM`) | 1024 to 8192 | 512; 32768; 65536 |
| `4096 -> 128`: (64, 2), every K-block count (`K1_TWO_SPLIT_RANGES_BY_SM`; no verdict in the 212-SM table) | 1024 to 7680 | 512; 640; 768; 896; 8000; 8192; 12288; 16384 |

*Table: the 212-SM K1 token-range table (`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`): further token counts at which these rows
of the 212-SM table of `K1_SPLIT_RULES_BY_SM` run their program (the third column runs the deterministic program),
except the token counts of the `sm_107a` K1 token table above.  Measured at 512, 1024, 2048, 4096, 8192, 32768 and 65536
tokens on two 212-SM R200 nodes of four parts each, with the protocol of the window above (one part per launch layout
and row order), and the FP64 error of 5 seeds on two parts.  A measured token count joins a range where the row's
program passed the K1 rule in every run, left the backward pass and the checkpointed step no more than 1.3 % slower than
the deterministic program, and met the deterministic package's FP64 error budget for every seed on both parts.  A range
ends at a measured token count that passed (or at the window) and holds no measured token count that failed; the token
counts between two consecutive measured ones inside it take the verdict too (not measured).  Token counts below 512,
above 65536 and between a measured token count that passed and one that failed run the deterministic program.  The four
one-split FP32 rows take this table at 512 tokens only (their 212-SM program in the window as well); from 960 tokens the
K1 accumulation-chain table below runs three chains first, and the two-split ranges hold more token counts (q of the
one-split program about 2 from 1024 tokens up, data: it adds an output's whole token reduction in one chain).  The other
rows are not faster than the deterministic program in every run at 512 tokens, nor `8192 -> 128` at 65536.  The
two-split rows (`K1_TWO_SPLIT_RANGES_BY_SM`) run two splits of the row's 212-SM tile width, each adding half of an
output's K blocks, at every token count of their ranges, at either number of 64-token K blocks.  So the token counts in
the third column of the one-split FP32 rows run the two-split program where a two-split range holds them.  At an even
block count from 1024 tokens up their output was the deterministic program's bit for bit (5 seeds on two parts at 1024,
2048, 4096, 8192, 16231, 16384, 32768 and 65536 tokens, on one part at 6000, 12000 and 48000); at an odd count from 17
blocks up it was not, with q 1.01 to 1.17 (data) and within the FP32 budget on every seed (5 seeds, 1025 to 65537
tokens; relative Frobenius error at most 3.8e-5, at 65537 tokens).  They were measured with the protocol above at 512,
1024, 2048, 4096, 8192, 32768 and 65536 tokens (`4096 -> 512` also at 40961, an odd block count) on two 212-SM R200
nodes of four parts each; a range runs between measured token counts at which the two-split program passed the K1 rule
in every run and left the backward pass and the checkpointed step no more than 1.3 % slower than the deterministic
program, and holds none that failed; the token counts between, odd block counts included, take the verdict too (not
measured).  At 512 tokens the two-split program is slower than the deterministic plan; `4096 -> 1024` is not faster in
every run at 1024 tokens; `5120 -> 1024` is not faster in every run at 32768 tokens and slower at 65536.  `4096 -> 128`
FP32 has no program in the 212-SM table either; its two-split range was measured with the same protocol at 512, 640,
768, 896, 1024, 2048, 4096, 6144, 6656, 7168, 7680, 8000, 8192, 12288 and 16384 tokens on two nodes of each of two R200
systems (6144, 8000, 8192, 12288 and 16384 tokens and 640 on one system).  From 1024 to 7680 tokens it passed in every
run (0.70 to 0.76 times the deterministic program's time) and its output was the deterministic program's bit for bit at
an even block count (5 seeds); at an odd count (1088 to 5000 tokens) q was 1.11 to 1.16 (data), so these token counts
run it too.  It failed the rule at 512, 640, 768 and 896 tokens; from 8000 tokens the deterministic plan changes and the
two-split error was 1.54 times its error, and at 12288 and 16384 tokens it was 5 to 7 % slower.  `5120 -> 1024` BF16 has
no program in the 212-SM table (at 16172 to 16384 tokens the deterministic plan is faster); its range runs the 256-wide
one-split program of `K1_SPLIT_RULES` (its 216-SM verdict), measured at 512, 1024, 2048, 4096, 8192, 12288 and 32768
tokens with the same protocol on a node of four parts, twice with the parts' launch layouts and orders rotated: faster
than the deterministic plan in every run up to 12288 tokens, with the FP64 error of the deterministic program on every
seed; at 32768 tokens it failed the K1 rule in every run (0.1 to 0.6 % faster than the deterministic plan alone), and
its checkpointed step measured 2.5 to 4.3 % slower, as did a second deterministic launch against the first there (1.9 to
4.4 %).  On 212 SMs `4096 -> 1024` FP32 keeps the 192-wide program at 512 tokens: the 256-wide program is 15 % faster
alone and leaves the backward pass 4 to 6 % faster, but the checkpointed step 0.2 to 1.2 % slower in graph replays
(median of the paired ratios, ratio of the medians).*

| K1 row `d_in -> d_out`, FP32 out | 212 SMs: `(block_n, chains)`, token ranges | 216 SMs: `(block_n, chains)`, token ranges |
|---|---|---|
| `4096 -> 512` | (128, 3): 960 to 16172; 32768 to 40961 | (128, 3): 961 to 16172; 32768 to 40961 |
| `4096 -> 1024` | (192, 3): 960 to 16172; 32768 to 40961 | (192, 3): 961 to 16172; 32768 to 40961 |
| `5120 -> 512` | (128, 3): 960 to 16172; 32768 to 40961 | (128, 3): 961 to 40961 |
| `5120 -> 1024` | (256, 3): 960 to 40961 | (256, 3): 961 to 16384; 40961 |

*Table: the K1 accumulation-chain table (`K1_CHAIN_RANGES_BY_SM`): token ranges at which these rows run the one-split
program of three accumulation chains (templates `dense_proj_gemm_atomic_nn_n128_f32_t_c3`,
`dense_proj_gemm_atomic_nn_n256_f32_t_c3`, `dense_proj_gemm_atomic_nn_n192_f32_c3`) on parts of exactly that SM count,
in place of the tables above, the 212-SM two-split ranges included: at the even block counts of those ranges both
programs were timed at (1024 to 8192 and 32768 to 40960 tokens) three chains ran the weight gradient 7 to 52 % and the
checkpointed step 2 to 21 % faster than two splits (medians, eager launches and graph replays).  The program cuts each
tile's K blocks into three contiguous chains, each summed from an `init` MMA in an accumulator of its own, and its
epilogue adds them once, in order: no workspace and the same bits on every run.  Its FP64 relative Frobenius error was
0.31 to 0.73 times the deterministic program's on every seed (q, data) (5 seeds on a 212-SM R200 part at 960, 961, 1024,
2048, 4096, 8192, 16172, 16231, 16384, 32768 and 40961 tokens, even and odd block counts), where the one-chain program's
is 2.0 to 2.3 times.  The ranges follow the rule and the construction of the 212-SM K1 token-range table (the K1 rule in
every run, the backward pass and the checkpointed step no more than 1.3 % slower than the deterministic program by the
ratio of the medians over the runs and the median of the paired ratios, eager launches and graph replays).  212 SMs:
measured in two jobs on four parts of one node each (one launch layout and row order per part), at 960, 4096, 16172,
16384, 32768 and 40961 tokens and at 960, 1024, 2048, 4096, 4097, 8192, 16172, 16384, 32768, 32769, 40960 and 40961
tokens; `4096 -> 1024` left the checkpointed step 1.7 % slower at 16384 tokens in one of them.  216 SMs: measured at
960, 961, 1024, 2048, 4096, 8192, 16172, 16231, 16384, 32768 and 40961 tokens on one part (two row orders); up to 960
tokens these rows keep their one-chain program, which gives the deterministic program's bits there.  Token counts
outside the ranges keep the routing of the tables above (216 SMs: the one-chain program up to 40960 tokens, within the
FP32 budget with q 2.0 to 2.4, and the 216-SM override of `4096 -> 1024` at 16172 to 16384 tokens).*

`"waves"` (`router_waves`) launches whole waves of the device's CTA pairs (`pairs = sm_count // 2`): `c * pairs`
units with `c = max(2, round(tiles * k_iters / (68 * pairs))) + 1`, and only where the deterministic plan's six
splits fill their last wave below 80 % (`w / ceil(w)`, `w = 6 * tiles / pairs`); otherwise the row runs the
deterministic program.  At hidden 5120 on 152 SMs that is 80 units at 64 tokens (at most one unit per (tile,
K block) step) and 380 units at 16k tokens; on 148 SMs the deterministic plan's last wave is 81 % full and the row
runs the deterministic program.

`"simt_n<tile_n>m<tile_m>s<slices>"` (`router_simt`, the weight gradient only) runs the exact-FP32 program
`router_fp32_gemm_simt_n<tile_n>m<tile_m>s<slices>` on the CUDA cores: one CTA per `tile_n x tile_m` tile of the
transposed `out` over all tokens, `slices` slices of the CTA each taking every `slices`-th 16-token block of the FP32
operands (read in place through their row strides), one FFMA per product, the slices added in slice order, every
element of `out` stored once (no BF16 parts, no workspace, no zero fill; the bits are fixed by the token count and the
tile, not by the SM count).  It launches after its predecessor, never as a programmatic dependent launch: launched
early, its CTAs took only the SMs the input gradient left and the backward pass was slower.  Its result differs from
the deterministic program's in the last bits (FFMAs of the FP32 operands instead of products of their BF16 parts, in
another summation order), and each slice sums its products in one FFMA chain along the tokens: from about 100 tokens
its FP64 error is about twice the deterministic program's or more (the router FP64 error table below), so the windows
of 152-SM `sm_100a` and 212-SM `sm_107a` parts, though faster, run the plan without them (the deterministic program; at
hidden 5120 up to 128 tokens on `sm_100a`, `"waves"`).  This checkout registers no SIMT program (weight or input
gradient): a SIMT spec set in a router table plans its template, which is not registered, so the row runs the
deterministic program.

On an input-gradient row (`kn`) `"simt_n<tile_n>m<tile_m>s<slices>"` runs the exact-FP32 program
`router_fp32_dgrad_simt_m<tile_m>n<tile_n>s<slices>`: one CTA per `tile_m x tile_n` tile of `out = G @ W` (`tile_m`
tokens by `tile_n` hidden columns, the last row tile partial) over all experts, `slices` slices of the CTA each taking
every `slices`-th 16-expert block of `G` and `W` (read in place through their row strides), one FFMA per product, the
slices added in slice order, every element of `out` stored once.  Like the weight gradient's program it launches after
its predecessor.  On 212-SM `sm_107a` parts the 32 x 64 tile in eight slices (`"simt_n64m32s8"`, 256 threads) from 1
to 64 tokens and the 64 x 64 tile in four slices (`"simt_n64m64s4"`, 256 threads) from 65 to 128 at hidden 4096 and
5120 (`"simt_n64m32s8"` up to 64 tokens at hidden 6144), and on 152-SM `sm_100a` parts the 32 x 64 tile in four slices
(`"simt_n64m32s4"`, 128 threads) up to 128 tokens at hidden 4096 and 5120 and up to 64 at hidden 6144, were faster than
the deterministic program, but their FP64 error was 1.30 to 1.86 (`sm_107a`) and 1.66 to 2.07 (`sm_100a`) times the
deterministic program's (the router FP64 error table below), so every part runs the deterministic program.

`"H<f>x<k>"` (`router_partition_of`, the weight gradient only) runs the head/tail program
`router_fp32_gemm_atomic_nn_t_ht_c3_nreg` (on `sm_103a` and on 212-SM `sm_107a` parts:
`router_fp32_gemm_ts_r6p6_atomic_nn_t_ht_c2_nreg`, the same products, accumulators and part segments with the BF16 parts
of `A` in tensor memory, `ROUTER_HEAD_TAIL_TEMPLATES` / `ROUTER_HEAD_TAIL_TEMPLATES_BY_SM`):
one head unit per CTA pair, whose spans share about the fraction `f` of
the (row-tile pair, K block) steps, then `k` tail units per CTA pair over the rest (each count a multiple of the
column tile count).  Every pair starts a long head span at once; the short tail spans go to the pairs that finish
first, so a kernel holding SMs on another stream delays a short span rather than a long one.  `"H0.5x3"` is 76 head
and 228 tail units at 152 SMs; `"H0.5x4"` is 104 head and 416 tail units at 208 SMs, 106 head and 424 tail units at
212 SMs and 108 head and 432 tail units at 216 SMs.  A token count too short for one step per span runs the
deterministic program.
`"H<f>x<k>p<h>"` starts only `h` head units (at most one per CTA pair), each about `f` of a pair's even share of the
steps; the other pairs start on the tail spans, so a kernel holding the SMs of up to `pairs - h` CTA pairs delays no
head span.  At 152 SMs `"H1x4p44"` is 44 head and 304 tail units and `"H1x5p44"` 44 head and 380 tail units.

| row of `ROUTER_RULES` (weight gradient, 256 experts) | token window | measured at (tokens) | part | time / deterministic plan, alone | worst next to a resident kernel: same launch order / router launched first |
|---|---|---|---|---|---|
| `sm_100a` hidden 5120 `"waves"` | 1 to 128 | 64, 128 | GB200, 152 SMs | x0.691-0.884 | x0.864 / x0.846 |
| `sm_103a` hidden 5120 `"waves"` | 1 to 160 | 64, 128, 192 | GB300, 152 SMs | x0.706-0.919 | x1.005 / x0.917 |
| `sm_107a` hidden 5120 `"H0.5x4"` | 16172 to 16384 | 16172, 16231, 16384 | R200, 216 SMs (two parts) | x0.793-0.819 | x0.960 / x0.963 |
| `sm_107a` hidden 5120 `"H0.5x4"` (`ROUTER_RULES_BY_SM`, 212 SMs, the head/tail program with the `A` parts in tensor memory) | 16172 to 16384 | 16172, 16231, 16384 | R200, 212 SMs (four parts in two nodes) | x0.735-0.770 | x0.922 / x0.925 |
| `sm_107a` hidden 6144 `"H0.5x4"` (`ROUTER_RULES_BY_SM`, 212 SMs, the head/tail program with the `A` parts in tensor memory) | 16172 to 16384 | 16172, 16231, 16384 | R200, 212 SMs (four parts in two nodes) | x0.831-0.870 | x0.981 / x0.961 |
| `sm_107a` hidden 5120 `"H0.5x4"` (`ROUTER_RULES_BY_SM`, 208 SMs) | 16172 to 16384 | 16172, 16231, 16384 | R200, 208 SMs (two parts, and a third at a lower SM clock) | x0.816-0.830 | x1.002 / x1.015 |

*Table: the token windows of the router weight gradient (eager launches and CUDA graph replays, one long-lived process
and a fresh process per token count; the resident kernel holds 8, 16 or 32 SMs).  The `"waves"` rows are the FI-main
`"waves"` plan (same kernel text, measured on this package's host path); the `sm_103a` `"waves"` window ends at 160
tokens, the last token count where its FP64 error keeps the rule (the router FP64 error table below).  On the R200
parts `nvidia-smi` read `clocks_throttle_reasons` 0x4 after some rows (sampled every 200 ms only during 6 of the 12
runs at hidden 5120 and 6 of the 10 at hidden 6144, those on two of the four 212-SM parts: 0x4 in 17-47 % of the
non-idle samples, median SM clock 2388-2400 MHz, no hardware slowdown or thermal reason).  The arms of each row were
measured interleaved, in paired groups.  Inside the head/tail windows cuBLAS FP32 in
`CUBLAS_FP32_EMULATED_BF16X9_MATH` (the benchmark's `bf16x9` arm) took x1.630-1.715 (R200, 216 SMs), x1.809-1.915
(R200, 212 SMs, hidden 5120 and 6144) and x1.639-1.768 (R200, 208 SMs) of the partition's time: at least x1.630.
On 212 SMs the head/tail program with
the BF16 parts of `A` in tensor memory took x0.916-0.954 of the three-chunk-buffer program's time at hidden 5120 and
x0.906-0.950 at hidden 6144 (same partition, alone); with the three-chunk-buffer program hidden 6144 was up to 9.2 %
slower than the deterministic plan next to a resident kernel, so its window holds only with the tensor-memory program.
On the 208-SM and 212-SM parts at full SM clock the three-chunk-buffer program with `"H0.5x3"` was 0.9-2.1 % faster
than with `"H0.5x4"`, but on a 208-SM part at a lower SM clock (about 2000 MHz; there `"H0.5x4"` took x0.931-0.959
alone) it was up to 4.9 % slower than the deterministic plan when launched before a resident kernel, so both SM counts
take `"H0.5x4"`.  The head/tail windows hold
only on the SM count of their part (`ROUTER_SM_COUNT`, or the part's own table in `ROUTER_RULES_BY_SM`), the
`"waves"` windows on every SM count where `"waves"` applies (`router_waves`); every other token count and SM count runs
the deterministic plan.
Hidden 5120 on `sm_100a` / `sm_103a` ran `"waves"` at every token count before these windows: below 64 tokens the
`"waves"` windows keep it without a measurement; above them the row runs the deterministic plan -- there `"waves"` was
more than 3 % slower than that plan next to a resident kernel at every measured token count from 224 to 10240
(x1.046-1.156 on GB200 at 224 to 12288 tokens, x1.059-1.167 on GB300 at 224 to 10240 tokens), on `sm_103a` from 161
to 192 tokens its FP64 error missed the rule, and above 32768 tokens nothing was measured.  The head/tail windows of
152-SM `sm_100a` / `sm_103a` parts (`"H1x5p44"`, `"H0.5x3"`, `"H1x4p44"`; 0.75 to 0.96 times the deterministic
plan's time alone) missed the FP64 error rule and run the deterministic plan.*

The router windows' FP64 error: per seed the relative Frobenius error against FP64 of the exact operands (worst of 20
runs into a NaN-filled `out`) of the window's program, divided by that of the plan the row takes without the window
(the deterministic program; at hidden 5120 up to 128 tokens on 152-SM `sm_100a` parts, `"waves"`), on the same
operands (the benchmark's router inputs: FP32 activations, router weight and output gradient), 10 seeds per token
count.  A window keeps the rule where this ratio is at most 1.02 for every seed.  The error of each program is within
the router budget (`<= 2e-6`, at most 3.2e-7 here); the ratio measures what the window adds to it.

| row (256 experts) | token window | measured at (tokens, 10 seeds each) | part | error / error without the window | verdict |
|---|---|---|---|---|---|
| `sm_100a` weight gradient hidden 4096 `"simt_n128m64s2"` | 1 to 224 | 1, 113, 224 | GB200, 152 SMs | x2.226-2.423 (x0.932-0.959 at 1 token) | missed |
| `sm_100a` weight gradient hidden 5120 `"simt_n64m64s2"` | 1 to 256 | 1, 128, 129, 256 | GB200, 152 SMs | x2.300-2.536 (x0.928-0.951 at 1 token) | missed |
| `sm_100a` weight gradient hidden 6144 `"simt_n64m64s2"` | 1 to 256 | 1, 129, 256 | GB200, 152 SMs | x2.298-2.531 (x0.927-0.951 at 1 token) | missed |
| `sm_100a` input gradient hidden 4096 `"simt_n64m32s4"` | 1 to 128 | 1, 64, 128 | GB200, 152 SMs | x1.697-2.067 | missed |
| `sm_100a` input gradient hidden 5120 `"simt_n64m32s4"` | 1 to 128 | 1, 64, 128 | GB200, 152 SMs | x1.679-1.967 | missed |
| `sm_100a` input gradient hidden 6144 `"simt_n64m32s4"` | 1 to 64 | 1, 32, 64 | GB200, 152 SMs | x1.661-1.933 | missed |
| `sm_100a` weight gradient hidden 4096 `"H1x5p44"` | 10240 to 16384 | 10240, 12288, 16384 | GB200, 152 SMs | x1.177-1.201 | missed |
| `sm_100a` weight gradient hidden 5120 `"H0.5x3"` | 10240 to 32768 | 10240, 16384, 32768 | GB200, 152 SMs | x1.014-1.027 | missed |
| `sm_100a` weight gradient hidden 5120 `"waves"` | 1 to 128 | 1, 64, 128 | GB200, 152 SMs | x1.000-1.002 | kept |
| `sm_103a` weight gradient hidden 4096 `"H1x4p44"` | 20480 to 49152 | 20480, 32768, 49152 | GB300, 152 SMs | x1.198-1.294 | missed |
| `sm_103a` weight gradient hidden 5120 `"H0.5x3"` | 16172 to 32768 | 16172, 24576, 32768 | GB300, 152 SMs | x1.013-1.021 | missed |
| `sm_103a` weight gradient hidden 6144 `"H1x5p44"` | 16172 to 20480 | 16172, 16384, 20480 | GB300, 152 SMs | x1.354-1.399 | missed |
| `sm_103a` weight gradient hidden 5120 `"waves"` | 1 to 160 | 1, 65, 97, 113, 128, 144, 160 | GB300, 152 SMs | x0.979-1.016 | kept |
| `sm_103a` weight gradient hidden 5120 `"waves"` | 161 to 192 | 161, 176, 192 | GB300, 152 SMs | x1.026-1.040 | missed |
| `sm_107a` weight gradient hidden 4096 `"simt_n128m64s2"` | 1 to 256 | 1, 129, 256 | R200, 212 SMs | x2.300-2.535 (x0.929-0.950 at 1 token) | missed |
| `sm_107a` weight gradient hidden 5120 `"simt_n128m64s2"` | 1 to 256 | 1, 129, 256 | R200, 212 SMs | x2.301-2.536 (x0.928-0.951 at 1 token) | missed |
| `sm_107a` weight gradient hidden 6144 `"simt_n128m64s2"` | 1 to 192 | 1, 97, 192 | R200, 212 SMs | x1.995-2.659 (x0.927-0.951 at 1 token) | missed |
| `sm_107a` input gradient hidden 4096 `"simt_n64m32s8"` | 1 to 64 | 1, 32, 64 | R200, 212 SMs | x1.299-1.530 | missed |
| `sm_107a` input gradient hidden 4096 `"simt_n64m64s4"` | 65 to 128 | 65, 96, 128 | R200, 212 SMs | x1.823-1.860 | missed |
| `sm_107a` input gradient hidden 5120 `"simt_n64m32s8"` | 1 to 64 | 1, 32, 64 | R200, 212 SMs | x1.348-1.535 | missed |
| `sm_107a` input gradient hidden 5120 `"simt_n64m64s4"` | 65 to 128 | 65, 96, 128 | R200, 212 SMs | x1.819-1.859 | missed |
| `sm_107a` input gradient hidden 6144 `"simt_n64m32s8"` | 1 to 64 | 1, 32, 64 | R200, 212 SMs | x1.307-1.585 | missed |
| `sm_107a` weight gradient hidden 5120 `"H0.5x4"` | 16172 to 16384 | 16172, 16231, 16384 | R200, 212 SMs | x0.943-0.952 | kept |
| `sm_107a` weight gradient hidden 6144 `"H0.5x4"` | 16172 to 16384 | 16172, 16231, 16384 | R200, 212 SMs | x0.974-0.983 | kept |

*Table: the router FP64 error.  The SIMT programs repeat bit for bit; an exact model of their FFMA order reproduces
their output bit for bit, so the ratio follows from the order (each slice's products summed in one FFMA chain), not
from the SM count.  The head/tail partitions split a tile's K range over more spans than the deterministic plan's six
and add them with the TMA reduce-add.  At one token the SIMT weight gradient rounds each product once (below the
deterministic program's error), but a window of one token is not kept.*

`prepare_*` and `plan_*` take `splits` (and, for K1, `block_n`) to force a plan, for tests and measurements; the K2
forward takes at most two splits.  `benchmarks/bench_cake_dense_projection_gemm.py` prints, per row, the
`rule_key` / `rule_value` of the plan it measured, in the format of these tables.

## Workspace

A K1 launch with more than one split uses an FP32 slot workspace (`tiles * num_splits * 256 * block_n` elements,
zero at creation: the last split of a partial tile reads the padding columns no split writes, and discards them) and
the ticket counters (`tiles * 512` words, zero at creation, zero again after every complete launch).  A prepared
launch owns both; the convenience path keeps them in an arena per (device, stream), so concurrent launches on
different streams never share them.  After an aborted launch, call `PreparedAtomicGemm.reset_counters()` (or
`clear_convenience_caches()`).  A K2 launch with part tiles zeroes `out` on the launch stream (one memset) between
the split and the program; prepared launches of both programs allocate nothing and can be captured in a CUDA graph.
A prepared K2 launch binds everything but the stream when prepared (its FFI arguments are converted once), so a call
issues its launches back to back.  A cached K1 launch of the convenience path orders its arguments once per cache key
(the PDL flag is bound, so it is part of the key) and converts its workspace once per arena growth; a call rebuilds and converts only its operand views.  A K1 call that
runs the deterministic program -- the forward, the input gradient, a weight-gradient row without a verdict -- launches
it through a cached binding of this package: the deterministic package's plan and program (its bits, with its `sk` /
`cta_rows` knobs and its weight-gradient orientation), with its stream-K slabs and slice counters in this package's
arena, keyed by the entry point's own tensors.  A batched forward that the deterministic package runs on its swapped
small-token program makes that package's call (with `enable_pdl` and `weights_static`).
A cached K2 launch of the convenience path orders its arguments once per cache key
and converts the arena's BF16x3 stack once per arena growth, so a call passes only the caller's views (each
converted in its FFI call) and enters the arena's stream context, made once.  A K2 call
that runs the deterministic program -- the input gradient, the forward without a split count, a call without a split
count whose row has no verdict in the router table of the device's SM count (`ROUTER_RULES`, or the part's own table
in `ROUTER_RULES_BY_SM`; not planned here) -- launches it through a cached binding of this package: the
deterministic package's plan, programs and `torch.sum` of its split-K partials (its bits), with the stack and the
partials in this package's arena, bound once per arena growth, and the entry point's own tensors as the launch
views.  It issues the same launches as the deterministic package's convenience path in fewer host operations.  Whether a row has a verdict is kept per device and view geometry, and whether any
router table gives the forward one is kept too, until a router table changes: the router tables count their changes,
so a verdict set or removed at run time is seen at the next call (a table rebound to a plain `dict` is read on every
call).  The router entry points (`router_forward`, `router_dgrad`, `router_wgrad`) cache each call once more by their
own tensors, the operation, the split count and the PDL flag: a call goes from that key to the cached launch its
route takes, without building the transposed operand, routing the row or building the launch's own key; the key is
routed again when a router table or `ROUTES` changes (with a table rebound to a plain `dict`, every call takes the
full path).

## Benchmark

`benchmarks/bench_cake_dense_projection_gemm.py` times every row of the deterministic family's benchmark with the
arms `det` (the deterministic package's prepared launch), `atomic` (`deterministic=False`), `atomic_p1` (the same
program at one split), `cublas` (`torch.matmul`), `cublaslt` (a cuBLASLt heuristic sweep) and `bf16x9` (cuBLAS FP32
in `CUBLAS_FP32_EMULATED_BF16X9_MATH`, the router rows), in paired groups, eager and in CUDA graphs; `--resident`
re-times `det` and `atomic` next to a kernel holding `--resident-sms` SMs on another stream.

## Names

This import path held the dense projection GEMM package until #8046 renamed it to `dense_projection_det_gemm`; #8046
left it as an alias of the renamed modules.  The names keep their spelling and, with the default, their behavior:

| name before #8046 (`dense_projection_gemm.cake_backend`) | deterministic package (`dense_projection_det_gemm.cake_backend`) | this path |
|---|---|---|
| `dense_projection_gemm` | `dense_projection_det_gemm` | `dense_projection_gemm(..., deterministic=True)` |
| `projection_forward`, `projection_dgrad`, `projection_wgrad` | `projection_det_forward`, `projection_det_dgrad`, `projection_det_wgrad` | the same names with `deterministic=` |
| `prepare_dense_projection_gemm`, `prepare_projection_wgrad` | `prepare_dense_projection_det_gemm`, `prepare_projection_det_wgrad` | the same names with `deterministic=` |
| `plan_dense_projection_gemm` | `plan_dense_projection_det_gemm` | the same name with `deterministic=` |
| `router_fp32_gemm`, `router_forward`, `router_dgrad`, `router_wgrad` | `router_fp32_det_gemm`, `router_det_forward`, `router_det_dgrad`, `router_det_wgrad` | the same names with `deterministic=` |
| `prepare_router_fp32_gemm`, `plan_router_fp32_gemm` | `prepare_router_fp32_det_gemm`, `plan_router_fp32_det_gemm` | the same names with `deterministic=` |
| `split_fp32_to_bf16x3_` | `split_fp32_to_bf16x3_det_` | the same name with `deterministic=` (the deterministic stage in both modes) |
| `instance_symbol`, `router_symbol` | `instance_symbol`, `router_symbol` | the same names with `deterministic=` |
| `GemmPlan`, `PreparedGemm` | `DetGemmPlan`, `PreparedDetGemm` | read through; `deterministic=False` plans `AtomicGemmPlan`, `PreparedAtomicGemm` |
| `RouterPlan`, `PreparedRouterGemm` | `RouterDetPlan`, `PreparedRouterDetGemm` | read through; `deterministic=False` plans `AtomicRouterPlan`, `PreparedAtomicRouterGemm` |
| every other module name (`ROW_RULES`, `stream_k_plan`, `wgrad_views`, ...) | the same name | read through to the deterministic package |
| program templates `dense_proj_gemm_*`, `router_fp32_gemm_<layout>`, `split_fp32_bf16x3` | `dense_proj_det_gemm_*`, `router_fp32_det_gemm_<layout>`, `split_fp32_det_bf16x3` | this package's templates are new: `dense_proj_gemm_atomic_nn_n<64\|128\|256>[_f32]_t`, `dense_proj_gemm_atomic_nn_n192_f32` (row-major store, `sm_107a`), `router_fp32_gemm_atomic_{kk,nn_t,nn_t_ht_c3_nreg}`, `router_fp32_gemm_ts_r6p6_atomic_nn_t_ht_c2_nreg` (`sm_103a`) |

*Table: the names before #8046, their deterministic names, and what this path offers
(`cake_backend.DETERMINISTIC_NAMES`).*

| name (`flashinfer.experimental.dense_projection_gemm`) | default | `deterministic=False` |
|---|---|---|
| the entry points of the table above | the deterministic package's `*_det_*` object | this package's schedule (routing table above) |
| `cake_backend.clear_convenience_caches` | clears the caches and arenas of both packages | -- |
| `cake_jit.{MODULES, KERNELS, ARG_PLANS}` | this package's programs only | -- |
| `cake_jit.select_module`, `cake_jit.load_cake_dense_projection_gemm_module` | a deterministic template or program resolves through the deterministic package | -- |
| `cake_build` | this package's loader (byte-identical to every Cake package's) | -- |

*Table: the names of this path and what they reach.*

A module-level assignment through this path (for example `CONVENIENCE_CACHE_SIZE`) does not reach the deterministic
package; set it there.

## Layout of this package

| file | content |
|---|---|
| `__init__.py` | imports nothing; `cake_backend`, `cake_jit`, `cake_build` load on first attribute access |
| `cake_backend.py` | entry points, mode resolution, routing, the planners, rule tables and the convenience state of this package's programs |
| `cake_jit.py` | the registry of this package's programs (`ARG_PLANS`, `MODULES`, `KERNELS`) and the loader |
| `cake_build.py`, `build.json`, `csrc/cake_dense_projection_gemm/` | the `cake-build/1` loader, manifest and generated sources of this package's programs (written by the Cake exporter) |

The deterministic package is resolved as this package's sibling on first use; the package never imports `flashinfer`
by absolute name.  A consumer that vendors this package vendors the deterministic package too and binds it with
`cake_backend.use_deterministic_backend(<its cake_backend module>)`.
