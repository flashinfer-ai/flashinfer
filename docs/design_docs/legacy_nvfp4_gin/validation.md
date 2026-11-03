# Legacy NVFP4 GIN validation record

Updated 2026-11-03. Host checks, real device results and incomplete gates are
reported separately. The fixed-scope migration is accepted: numerical/public-runtime, final
sparse-poison/example and retained-finalizer checks pass. Job outcomes
are tracked in [implementation-status.md](implementation-status.md); failure
mechanisms and hypotheses are tracked in [investigation.md](investigation.md).

## Environment and commands

The login Python is `/usr/bin/python3`, version 3.12. The independent host
environment is `/tmp/fi-legacy-nvfp4-gin-host` (same Python), with pytest 8.4.2
and Ruff 0.12.8. It is not a FlashInfer GPU development environment. Torch,
DSL, NCCL and NVSHMEM dependencies remain unavailable in it.

Run from the receiving worktree:

```bash
python3 scripts/audit_legacy_nvfp4_gin.py
/tmp/fi-legacy-nvfp4-gin-host/bin/python -m unittest discover \
  -s tests/cpu/moe_ep -p test_legacy_nvfp4_gin_contracts.py -v
/tmp/fi-legacy-nvfp4-gin-host/bin/ruff check \
  scripts/audit_legacy_nvfp4_gin.py \
  flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/shim/gin.py \
  tests/cpu/moe_ep/test_legacy_nvfp4_gin_contracts.py
git diff --check
```

The host tests deliberately load the two dependency-free leaf modules and
run under unittest. They do not change normal pytest collection, stub GPU
imports or bypass a GPU test. The normal repository pytest collection still
requires Torch/FlashInfer; a missing dependency is BLOCKED, not a skip/pass.

| Check | Result | Scope |
| --- | --- | --- |
| Fixed-source audit | PASS | 29 changed candidates, 27 modules in static kernel closure |
| Arena/wire layout and LSA host contracts | PASS | Explicit offsets, padding, capacity and same-domain address failures |
| Control-plane contracts | PASS | In-memory four-participant all-gather; errors, subgroup and config mismatch reach all participants |
| Lifecycle contracts | PASS | Instrumented host resources; drain/reverse close order, idempotence, failed drain/close and partial initialization |
| Ruff and diff whitespace checks | See [host-checks log](evidence/host-checks.log) | Targeted changed adapter/script/tests |
| Normal pytest collection | PASS in GPU container | Login-only attempt was blocked by missing Torch and remains recorded; actual CPU/GPU selections collected and ran in the qualified container |
| CUDA arena typed views, real NCCL init/windows/LSA mapping | PASS, 3576298 | Actual GPU reads of every LSA peer, three create/close cycles/rank |
| F_BASE/off versus receiver/off EP4 smoke | PASS, 3576310 | Two cases/rank/source, including per-token scale |
| Donor and FI on-mode stage/numeric/relaunch checks | PASS for fixed case | 3576795: 24 strong calls/rank, exact protocol/message/L1 bytes and unchanged Torch tolerance; 3576839: four public control/lifetime cases/rank |
| GPU performance | Final off24/24 PASS; internode raw inconclusive | [Correction results and limits](performance-fix-blackwell.md); initial negative results retained |

Raw command output and exact counts are in [host-checks.log](evidence/host-checks.log).
The all-gather and resource harnesses exercise host control flow; they are not
NCCL resource, RDMA, CUDA or multi-node correctness evidence.

## GPU qualification contract

Before any GPU qualification, record container/image, Python, Torch, DSL,
NCCL/nccl4py, CUDA/driver, NVSHMEM, GPU/SM, NIC, node and PG/domain mapping.
Verify the fixed donor on the same quantized tensors as FI. Use a job-wide
timeout, full-rank logs and whole-job failure exit. A damaged communicator
must never run the next case.

Off validation must cover EP1/EP2/EP4, singleton/empty experts, sparse holes,
capacity edges, activation/per-token-scale/finalize modes and cache invalidation.
Generate separate EP2 and EP4 graph node-ID manifests by collecting in the real
launcher environment; verify nonzero NVFP4 case counts and actual world size.
Do not infer world size from a marker or accept all-skipped collections.

On validation starts with N2R2. Check Stage 1 metadata/message bytes, DATA/SF,
weights, counters, node-deduplicated routes and final output on every rank.
For fixed BF16 FC2 terms, the communication/ordered-reduce oracle requires
bitwise equality and per-rank max difference/mismatch counts. For independent
GEMM/activation, preserve the exact existing reference tolerance; it cannot
excuse a protocol-oracle mismatch. Include ten synchronized relaunches with
dense/sparse, partial/empty ranks, changed routes and complete credit/tail
cycles, without zeroing away the protocol state. Test resource recreation,
failed initialization, two-session isolation or explicit mutual exclusion,
and every supported/rejected mode through the public backend.

## Original performance proposal (superseded by the subsequent Blackwell experiment)

Use alternating baseline/receiver runs on identical hardware, clocks, routing,
inputs, tuning and cache protocol. For off, use F_BASE as the control; for on,
use the fixed donor. Report fused-kernel and full-call times separately, the
slowest rank, medians and dispersion. Use at least five alternating rounds
with 20 measured calls after five warmups per round. Report cold-L2 and
steady-state warm-L2 separately; do not combine them. Respect donor platform
restrictions and do not change any fixed DeepSeek CI guard.

Proposed off regression gate: receiver median no more than 5% above F_BASE
for each qualified case in both fused-kernel and full-call intervals. If
baseline dispersion exceeds 5%, label the comparison inconclusive and fix the
measurement environment before evaluating the gate. The subsequent [Blackwell protocol and results](performance-blackwell.md) explicitly
separate the CUDA-event pipeline gate from independently profiled fused-kernel
durations. The original waiver applied at correctness delivery; the user
later explicitly requested the measurements. The final corrections pass all24 off gates; internode raw comparisons remain
inconclusive. See [the correction report](performance-fix-blackwell.md).

## User scope clarification

On 2026-11-02 the user stated: “我们的这个feature不涉及rubin环境，可以不用在本地验证”.
This waived performance measurements at the original correctness delivery.
The user later explicitly requested Blackwell performance measurements,
superseding this waiver for that experiment. The subsequent results are
reported below; the fixed DeepSeek CI guard remains unchanged.

## Real execution summary

- GB300 container: `nvcr.io#nvidia/pytorch:26.07-py3`, Torch
  `2.13.0a0+9186a08b2c.nv26.07`, CUDA 13.3, DSL 4.6.2, nccl4py 0.3.1,
  NCCL 2.30.7, driver 580.173.10. Real Slurm node/domain details are in each
  job's `.out` log; actual LSA/rail support is checked by the runtime.
- Off 3576704: EP1 33 tests passed; EP2 and EP4 graphs 15 tests/rank passed;
  NVFP4 layer first 21 tests/rank passed. A missing editable-install data
  link blocked the next native compile. No skipped GPU case is counted.
- Off 3576822: after standard data links and recorded submodules were
  restored, nine remaining precision tests/rank and BF16/MXFP8 two tests/rank
  pass. Layer/workspace host checks: 263 passed, one optional DeepGEMM
  preprocess-output check skipped (not a GIN/NVFP4 acceptance case).
- On 3576795: each rank reports 24 exact protocol results, zero mismatches,
  exact message/split/L1 bytes and successful independent math. A later
  dtype error escaped local validation, and the owned job was cancelled
  after read-only stack capture. Its aggregate status is not PASS.
- On 3576839: four public tests/rank pass, covering the corrected rank-local
  errors/recovery, two live sessions and recreation, graph rejection and
  pre-staged packed input parity. Whole job COMPLETED, exit 0.

Final jobs 3576874 (sparse-poison strong oracles/example) and 3576878
(retained finalization/cache) both COMPLETED with exit 0. Launchers and full logs reside in
`/lustre/fsw/coreai_devtech_all/qiangf/ncclGIN/port/migration-evidence/`.

Single-node CI must not interpret four local ranks as N2R2. The GIN helper
requires torchrun `WORLD_SIZE=4`, `LOCAL_WORLD_SIZE=2`, `GROUP_WORLD_SIZE=2`
before starting the control group. Other launch geometries skip only these
topology-specific tests; runtime LSA/RDMA checks still verify actual domains.
The recorded N2R2 passes execute every selected case and are not skip results.

## Final acceptance

- 3576874: four ranks each collected and passed 14 tests, no skips. Each
  rank reports 24 exact fixed-term protocol calls (`max_diff=0`,
  `mismatches=0`); all 96 transport/L1 byte checks pass. Sparse route holes
  carry NaN weights to check that invalid payloads/scores are ignored.
  Independent Torch math passes its unchanged tolerance on 93 nonempty
  batches; maximum relative L2 0.0026485608 is below 0.02. Three empty output
  batches still execute the full communication protocol. The executable
  example then passes 12 finite/shape calls/rank in fresh worker processes.
- 3576878: four finalizer cases/rank pass (persistent partial/full capacity,
  per-token scale, external finalization with early/late route weighting
  and graphs); one explicit-workspace/cache/graph case/rank passes.
- Runtime code is committed as `fb033d522` plus `79a519e20`. The final
  launcher captured base/source hashes and a bounded patch; all Python
  hashes match the final implementation. Log SHA-256 and per-rank execution
  manifests are retained in [qualification-results.json](evidence/qualification-results.json).
- Some inherited outer launcher banners say “donor”; the actual receiving
  cwd, pytest rootdir and captured source hashes identify FI execution.
  Job classifications above use the executed code, not that old banner.

The default-off migration is complete within the fixed supported case.
Broader topologies, precision/mode additions and performance promotion are
separate work. No push, merge or publication was performed.

## Subsequent Blackwell performance request

The user subsequently requested Blackwell performance verification. That
request supersedes the earlier local-performance waiver for this experiment.
The predeclared matrix/protocol and live measurement record are in
[performance-blackwell.md](performance-blackwell.md). Correctness results
above are unchanged; they are not timing measurements.

Blackwell measurements are now complete: EP1/EP2/EP4 off against F_BASE and
N2R2 on against the fixed donor, five alternating rounds and 100 primary
samples per group. All accepted output-bit/payload checks passed. The initial off
performance gate did not pass overall; the initial on measurements show regression.
See the linked report for every case, dispersion, scope differences and
retained failed/partial runs. No kernel tuning was included.


## Performance-correction qualification, 2026-11-03

Final3578176 COMPLETED, exit0, on theia0048/theia0115 using74c036287:
17 strong/public tests per rank, no skips;4 selected off finalizer cases per
rank and1 explicit-workspace/cache/graph case per rank pass. The final two
providers complete1600 timing samples with exact bit/payload/profile checks.
The old38f650e63 rank-local stream rejection reproduces before the corrected
source passes stream-handle observation and current-device recovery tests.
All111 runtime/test source hashes match the launch snapshot; all17 device
hunk ledger hashes are unchanged by the performance fixes.

Off acceptance uses4800 samples and24 passing gates from3578010/3578087/3578088.
On standard raw350.37→349.31 us warm and363.98→356.43 us cold remains
inconclusive (baseline round CV~6.4%). Native/public full-call scopes differ.
The complete measured correction and retained failed jobs are documented in
[performance-fix-blackwell.md](performance-fix-blackwell.md).
