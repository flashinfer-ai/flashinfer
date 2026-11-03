# Legacy NVFP4 GIN implementation status

Updated 2026-11-03. Migration is complete for the fixed N2R2 scope, including final
sparse-poison/example and retained-finalizer qualification.
Default-off behavior is retained. Performance corrections and their limits are
reported in [performance-fix-blackwell.md](performance-fix-blackwell.md).

Receiver: branch `port/legacy-nvfp4-nccl-gin`, worktree
`port/flashinfer-nvfp4-gin`, based on
`e74a1677ab95a1b97aa1c6b6a80b3cf5c0ebac34`. Original FI and donor checkouts
remain unchanged. Implementation commits: `fb033d522`, `79a519e20`. No merge, push or
publication has been performed.

| Phase | Result | Evidence |
| --- | --- | --- |
| P0 | Fixed sources/vendor recovery, source/ABI/import/hunk audit complete | migration-audit.md and evidence/ |
| P1 | Real single-root NCCL session and peer-read/lifetime qualification passed | 3576298; host contracts 16 passed |
| P2 | Targeted device port; FI sparse/activation/scale/shared behavior retained | EP1/EP2/EP4 and shared compatibility results below |
| P3 | Formal public runtime, collective errors, session isolation/recreation and capture rejection passed | 3576839: 4 tests/rank |
| P4 | Fixed-term, independent math, transport/L1 byte and relaunch checks passed | 3576795; augmented final run 3576874 passed |
| P5 | Provenance, supported-case and review material prepared | Final review/evidence complete; subsequent Blackwell measurements reported separately |

## Current measured evidence

| Job | Scope | Result |
| --- | --- | --- |
| 3576158 / 3576298 | Fixed donor balanced N2R2 | Stages and final max_diff=0 all ranks; second runtime probe also passes |
| 3576310 | F_BASE/off and receiver/off EP4 | Two cases/rank/source, including per-token scale, no skips |
| 3576340 | Fixed donor local-only, empty experts, zero remote messages | All stages and final output exact |
| 3576704 | Receiver/off EP1 and EP2/EP4 graphs | EP1 33 passed; graphs 15 passed/rank at each world size; layer first 21 passed/rank, then missing development-source link blocked native compilation |
| 3576822 | Completed remaining off checks after standard data links/submodules were restored | 263 host-layer/workspace passed, 1 unrelated optional DeepGEMM skip; BF16/MXFP8 two cases/rank passed; remaining nine existing precision cases/rank passed |
| 3576795 | Matrix-first and balanced strong GIN oracles | 24 calls/rank, max_diff=0 and mismatches=0; transport/L1 bytes and unchanged Torch tolerance pass. Job later cancelled after a distinct rank-local dtype-validation escape |
| 3576839 | Corrected public control/lifetime checks | Four cases/rank passed: single-rank errors/recovery, two live sessions/recreation, capture rejection, packed input parity |
| 3576874 | Final strong run with sparse holes/NaN poison, full public tests and executable example | COMPLETED, exit 0; 14 tests/rank, no skips; 24 exact oracle calls/rank, 96 byte checks total; example 12 calls/rank |
| 3576878 | Retained off persistent/external finalizers and workspace cache/graphs | COMPLETED, exit 0; four finalizer tests and one cache/graph test per rank |

The protocol-generation race and the separate rank-local dtype escape are
fixed and explained in [investigation](investigation.md). Failed intermediate
jobs remain part of the record; their whole-job statuses are not relabeled.
The final independent Torch maximum relative L2 is 0.002649 (limit 0.02);
protocol mismatch counts remain zero. Final Python source hashes match the
launch snapshot. Current source checks: 16 standalone host contracts and changed-file pre-commit
hooks, including mypy/Ruff, pass.

Raw logs/launchers: `port/migration-evidence/`. Per-rank node IDs and outcomes:
[evidence/qualification-results.json](evidence/qualification-results.json).
GPU environment: independent GB300 NVLink domains, Torch
2.13.0a0+9186a08b2c.nv26.07, DSL 4.6.2, CUDA 13.3, NCCL 2.30.7,
nccl4py 0.3.1, driver 580.173.10.

## User scope clarification

The user stated: “我们的这个feature不涉及rubin环境，可以不用在本地验证”.
This waived performance measurements at the original correctness delivery.
The user subsequently requested Blackwell measurements; they are now complete.
The initial results remain in [performance-blackwell.md](performance-blackwell.md).
Subsequent corrections pass all24 off performance gates. Final source74c036287
passes17 strong/public tests per rank plus4 off finalizer tests and1 cache/graph
test per rank in3578176 (COMPLETED, exit0). Internode public host overhead is
reduced; native raw comparisons remain inconclusive because baseline variation
exceeds5%. See [the correction report](performance-fix-blackwell.md).
