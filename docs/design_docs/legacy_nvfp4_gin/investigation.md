# GIN bring-up investigation

## Question and boundary

Complete the user's fixed-source legacy NVFP4 GIN migration, retaining FI off
behavior and qualifying the formal FI N2R2 path. Diagnose the first public
multi-node failure without reclassifying host checks as device correctness.
The accepted migration plan remains the completion contract.

## Observations

- **O1, measured:** Fixed donor f863d754, balanced T32/H7168/J3072/E48/K4,
  N2R2, passed Stage 1/2/3 and final all-rank max_diff=0 in jobs 3576158 and
  3576298. Full logs: `port/migration-evidence/donor-{job}.out` and `.err`.
- **O2, measured:** The initial adapter rejected every rank's self window VA
  as different from root allocation VA (3576158). The installed nccl4py
  `RegisteredWindowHandle.get_peer_device_pointer` promises a peer-window
  pointer, not original-VA identity. After preserving the returned deltas,
  3576298 read every local peer's actual bytes on GPU, including self, and
  passed three create/close cycles on every rank. Subgroup rejection passed.
- **O3, measured:** Job 3576310 on GB300 theia0248 ran F_BASE and port/off in
  separate torchrun subprocesses. Each of four ranks collected/passed two
  tests: layer/reference and per-token-scale/reference at T64. No skips.
- **O4, measured:** In 3576298, the formal FI layer's first launch (local-only
  routing, 4 populated and 8 empty experts/rank) compiled and then reported
  unspecified CUDA launch failure at synchronize. Cleanup errors followed
  the original CUDA error and are not its source. No relaunch completed.
  This case differs from O1's balanced routing and from O2's plain pointer
  probe. The first cause is not established.
- **O5, source fact:** FI `_construct_kernel()` omitted the donor runner's
  `publish_relay_counts_to_gemm=True` in GIN mode. That flag affects expert
  scheduled counts and adaptive intra/relay pool merging. This is an adapter
  contract defect whether or not it explains O4's local-only failure.

## Knowledge lookup

Glean search on 2026-11-02 used the concrete NCCL GIN/MegaMoE/GB300 empty-expert
launch-failure context, then `"MegaMoE" "GIN" "empty"`. Results included
different NCCL-EP bugs and SM120 project updates; no directly comparable
fixed-source N2R2 case was established. These snippets are not adopted as
mechanistic evidence. Local fixed-source code and actual job results drive
the checks below. No external report has been treated as a reproduced result.

## Hypotheses and coverage

| ID | Mechanism and prediction | O1–O5 assessment | State |
| --- | --- | --- | --- |
| H1 | Missing relay publication flag changes scheduled rows/readiness. Restore the donor flag; balanced GIN output must include remote terms. | Compatible with O1/O2/O3; O5 proves mismatch; explanation of local-only O4 remains uncertain. | Contract fix required; causal scope unproven |
| H2 | Zero remote messages / empty-expert routing exposes a donor or port protocol defect. Run the same local-only routes through fixed donor. | O1 uses a different routing case; O2 and O3 do not exercise GIN MoE scheduling. | Candidate |
| H3 | FI arena/view/mapper binding differs from donor. Compare shapes/offsets and directly probe the DSL mapper. | O2 validates raw peer addresses, not the full mapper/argument ABI or every view. | Candidate |

## Next checks

1. Fix O5 and retain it as a distinct source-contract correction; do not claim
   that changing the flag alone explains O4.
2. Reproduce donor local-only routing with identical capacity/topology and no
   force-enable or support-check bypass. Use a fresh process/job on failure.
3. Compare formal FI balanced and local-only cases, preserving full-rank logs
   and checking protocol terms and independent math separately.
4. Keep the O1/O2/O3 passes as constraints on every revised explanation.

Current conclusion: runtime ownership and the limited off smoke pass, while
formal FI GIN numerical execution remains failed and migration is incomplete.
No performance conclusion is justified.


## Revision 2 and checkpoint R1

- **O6, measured:** Fixed donor local-only routing passed exact metadata,
  payload/L1 and final all-rank max_diff=0 in 3576340. Empty experts and zero
  remote messages therefore do not inherently fail in that donor case.
- **O7, measured:** After restoring the relay-publication flag, 3576341 passed
  both balanced and matrix relaunch tests (12 launches/case/rank, shape/finite
  checks only). The following config test failed because per-token scale was
  not rejected at layer construction. The backend init preflight now checks
  unsupported activation/scale/topology flags consistently.
- H2's donor-defect form is contradicted by O6 for this local-only case. H1's
  contract correction is supported by O5/O7; the exact first failing device
  instruction in O4 was not localized. H3 is not established as a cause; O7
  is evidence against an unconditional mapper/arena defect.
- R1: [checkpoint-01.md](checkpoint-01.md), Markdown snapshot, O1–O7. The next
  discriminating validation is fixed-actual-BF16-term protocol equality plus
  independent Torch math through the same public FI layer.

## Revision 3: initial-route dependence

- **O8, measured:** 3576514 executes the route-matrix oracle first, before
  balanced routing. Its first local-only forward fails at synchronize after
  about 70 seconds, before snapshotting terms or comparing weights/references.
  The restored publication flag is already present. O7 therefore does not
  establish that the flag alone fixes first-local-route startup.
- H1 remains a proven adapter-contract correction, but its sufficiency is
  contradicted by O8. O7 started with balanced routing before the matrix case;
  O8 starts with the matrix case and retains source BF16 weights for math.
  Initialization state, routing order, and retained-allocation differences
  must be distinguished. O6 still excludes an unconditional donor-local-route
  failure for its reproduced geometry.
- 3576773 adds opt-in device-guard source-site diagnostics and repeats O8 in
  a fresh job. No protocol tolerance has been loosened.

## Revision 4: localized token-back generation race

- **O9, measured:** 3576773 passes first launch and protocol/Torch comparisons
  for matrix iterations 0–8 on all ranks. Iteration 9 fails on rank 0 at
  diagnostic source-site 6357: the token-back deferral wait. Peer CPU gathers
  fail only after that rank exits. No payload/math mismatch preceded failure.
- **O10, source fact:** The imported donor deferral reads the *current* issue
  counter then waits for `current+1`. The relay producer can already have
  issued and incremented this counter before a late token-back warp reads it.
  Such a consumer waits for the next launch, while the current launch's tail
  waits for that consumer. Empty/short relay paths make this ordering feasible.
  The previous claim in the donor comment that this cannot occur is not an
  enforced synchronization edge. Passing donor cases O1/O6 do not exclude a
  scheduling race.
- Fix: derive the issue target from `internode_barrier_counter+1`, whose
  generation advances once at kernel tail after token-back retirement. The
  target stays stable whether the producer is early or late. Off mode never
  uses this branch. This is a documented local correctness diff against the
  donor, pending upstream submission. Diagnostic-off 3576795 will exercise
  matrix-first numeric checks and the new public failure/session tests.
- This mechanism predicts both startup and relaunch sensitivity. O4/O8 were
  not instrumented, so they remain consistent with, rather than direct
  measurements of, this exact guard failure.

## Revision 5: protocol pass and local validation escape

- **O11, measured:** 3576795, diagnostics disabled, passes all 12 matrix-first
  iterations and all 12 balanced iterations on every rank: Stage 1 complete
  message bytes and split counts, Stage 2/3 DATA/SF/weights, ordered fixed-term
  BF16 output (`max_diff=0`, `mismatches=0`) and unchanged Torch tolerance.
  Finite relaunch and all four unsupported-config tests also pass. This
  supports the generation correction; it does not certify performance.
- **O12, measured:** The subsequent single-rank-error test hangs. Read-only
  py-spy stacks show rank 1 already in pytest PG teardown and other ranks
  waiting in `stage_inputs` preflight, not in a GIN device kernel. Saved stacks:
  `port/migration-evidence/control-hang-3576795.txt`. The owned job was cancelled
  after collecting evidence; strong-oracle passes remain valid observations.
- **O13, source fact:** Shared shape validation does not check BF16 input
  dtype. GIN collective validation inherited that omission, and core
  `_resolve_quantize_input` then rejected one rank locally. Explicit input
  dtype/device/contiguity and route dtype checks now occur in the GIN local
  validator inside collective preflight, before that core check.
- 3576839 targets public errors, session mismatch/recreation, capture rejection
  and pre-staged input parity. Its explicit failure-evidence plugin prints the
  first local failure and exits through the launcher rather than entering a
  damaged collective test-session teardown. Normal collection is unchanged.

## Final checkpoint R2 and conclusion

- **O14, measured:** 3576839 completed, four public control/lifetime tests
  passed on each rank. The rank-local dtype escape is corrected; negative
  validation no longer strands peers.
- **O15, measured:** 3576874 completed on theia0018/theia0020, 14 tests/rank
  passed without skips. Matrix-first and balanced routes each run 12 times;
  sparse holes carry NaN weights. All 96 protocol comparisons have zero bit
  mismatches, all 96 dispatch-byte checks pass, and 93 nonempty Torch
  comparisons retain the original tolerance (max relative L2 0.0026485608).
  The standalone public example passes another 12 calls/rank in fresh workers.
- **O16, measured:** 3576878 completed, off persistent/external finalization
  and explicit-workspace/cache/graph tests pass on all four ranks. Together
  with 3576704/3576822, the recorded EP1/EP2/EP4 and BF16/MXFP8 matrix passes.
- **O17, checked:** The captured final Python hashes match implementation
  commits fb033d522 + 79a519e20. The 17-file device hunk ledger hashes also
  match. Changed-file pre-commit/type/style checks and 16 host contracts pass.
- H1 is a retained adapter-contract correction; H2's unconditional donor-local
  failure was contradicted. The localized generation race explains a real
  measured guard failure and is corrected by a stable current-launch target.
  O15/O16 support the correction without erasing O1–O13 or weakening numeric
  checks. No unconditional mapper defect remains supported by the evidence.
- The user waived Rubin/local performance validation for this feature. No
  latency/speedup conclusion follows from correctness observations.
- R2: [checkpoint-02.md](checkpoint-02.md), final Markdown evidence checkpoint.
  Fixed-scope migration is complete; upstreaming the local race correction
  and qualifying additional modes/topologies are separate follow-up work.

## Performance regression follow-up (2026-11-02)

Task: fix the measured Blackwell regression before further migration work.
User constraint: use the fixed legacy cutedsl_megamoe baseline, never next.
Keep the existing ordered/numeric/control/lifetime qualification; do not buy
latency by removing rank-local error agreement or weakening support checks.
Completion requires paired Blackwell measurements against legacy, preserved
correctness, and explicit disposition of the noisy off cases. The earlier
measurement report and raw samples remain immutable historical evidence.

- **O18, measured previously:** GB300 jobs 3577461/3577696 off and 3577625 on
  produced the complete 56-group matrix in performance-blackwell.md. Off has
  10 passes, 2 pipeline regressions and 12 inconclusive baseline-noise cases.
  On warm raw pipeline is 315.76→375.26 us; independent fused profile is
  244.610→310.243 us. These are separate scopes, not a causal attribution.
- **O19, measured previously:** On full call is 380.63→6876.83 us. Native donor
  adapter and FI public API include different host checks. This factor is not
  an isolated device-kernel slowdown.
- **O20, source audited:** Benchmark donor imports the fixed commit f863d754
  pre_unification/moe_nvfp4_swapab runner. FI config defaults to the explicit
  legacy registry key sm100_nvfp4_nvfp4_bf16_cutedsl, mapped to
  nvfp4_nvfp4_bf16_cutedsl.backend.Nvfp4CutedslMegaKernelBackend. No next
  provider is selected. Subsequent runs will assert and record these paths.
- **O21, source fact:** Public forward invokes collective validate_forward
  before _prepare_inputs, which invokes it again. Staging revalidates and
  agrees; internode _prepared_thunk_state unconditionally rebuilds its thunk;
  make_launch_thunk agrees after preparation and before every launch.
  collective_gin_preflight uses all_gather_object on the default PG. Counts
  and contribution to O19 have not yet been measured.

| Hypothesis | O18 pipeline/fused | O19 full-call | O20 baseline | O21 source |
| --- | --- | --- | --- | --- |
| PH1 redundant collective/preparation work dominates full-call overhead | Undetermined; cannot explain raw compiled call directly | Candidate, predicts high measured preflight/prepare share | Consistent | Consistent, execution contribution unmeasured |
| PH2 retained FI device features or layout/codegen differences slow the fused kernel | Candidate, requires matched-device intervention | Could explain only the device fraction | Consistent | Outside this mechanism |
| PH3 cross-rank timing/launch ordering or node noise affects pipeline ratios | Candidate; EP4 baseline noise is measured | Undetermined | Consistent | Undetermined |

**PT1, planned before execution:** on one exclusive N2R2 GB300 allocation,
reuse the validated fixture, assert legacy module/config paths, measure full
call/preflight/prepare counts, and separately capture raw kernels. Compare
unchanged object agreement with CPU-control transport and compact fixed-size
agreement as diagnostic interventions. Checks/errors remain enabled; these
are mechanism probes, not production claims. A drop in full-call time with
unchanged raw timing supports PH1; it cannot establish PH2. No configuration
tuning is selected at this stage.

Knowledge lookup: Glean searched the concrete PyTorch all_gather_object/Gloo
and CuTe DSL host-latency context. Original NVIDIA CUTLASS host-latency guide
was retrieved through nvidiadocs:
https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/003_host_latency_best_practices.html.
Its compile-once and argument-wrapper guidance motivates checking preparation
reuse. It describes CUTLASS Operator API, not this DSL 4.6.2 legacy runner;
its example speedups are not applicable measurements for this investigation.

- **O22, measured/PT1 completed:** job 3577851, theia0095/theia0201, 20 warm
  diagnostic samples, bitwise pass. Original public full 6978.907 us; CPU
  object agreement 5565.586 us; compact CPU agreement 3317.156 us. There are
  five preflights/call (backend-forward twice), rank-0 medians 1104–1251 us,
  and 214 us kwargs reconstruction. PH1 is supported for the full-call scope
  by the controlled transport intervention. PH1 does not explain raw pipeline.
  Raw donor/FI medians remain 304.480/334.672 us (+9.9%), preserving PH2/PH3.
- R3: [checkpoint-03.md](checkpoint-03.md), Markdown intermediate report.
- **PT2, launched 3577867:** pristine receiver c597d35ba isolated in
  flashinfer-nvfp4-gin-before-perf, fixed native legacy donor, record actual
  compile settings and PTX/cubins, and compare compact CUDA agreement with
  the already measured compact CPU intervention. No next baseline/provider.
  Its outcome determines the control transport; device code is still unchanged.

- PT2 job 3577867 failed in the diagnostic compiler-output extractor before
  timing: this DSL returns PTX text, whereas the extractor assumed a filename.
  No timing is accepted. The correction saves text/bytes directly and isolates
  dump directories per rank. Replacement 3577883 also runs real negative
  preflight/session/capture tests against the host optimization.
- **PH4, candidate:** additional optional GIN parameters on the non-GIN JIT
  entry point add host marshalling/launch-envelope overhead. Donor already
  uses launch_without_gin for its off entry, but FI compiles __call__ directly.
  This could affect the small-T off raw span without changing fused duration;
  it cannot explain the on fused difference. **PT3, planned:** original versus
  existing lean non-GIN entry on pristine FI c597d35ba, EP1/EP2, T64/T1024,
  two alternating diagnostic rounds, unchanged internal kernel and existing
  exact output fixture. No production device edit or next provider involved.

- **O23, measured/PT2 replacement completed:** 3577883 on theia0090/theia0201
  reproduces a counterexample with unchanged device code: raw medians donor
  359.728 us versus pristine FI 316.848 us; separate fused medians
  291.775 versus 287.107 us. Both compile with 152 active clusters. Compiler
  outputs have near-identical instruction counts; this does not establish
  identical machine code or a GPU root cause. The compact-CUDA host probe
  reaches 1872.808 us but adds CUDA binding requirements, so the production
  candidate uses CPU headers when the existing default PG supplies Gloo.
  The CUDA-versus-CPU numbers come from different allocations and are not a
  paired transport ranking. PH2 is not established as an unconditional device
  slowdown; PH3 remains candidate. O18 is retained, not discarded.
- **O24, measured:** the same job runs the production host candidate's five
  control tests on all four ranks: per-rank errors, compact-header phase/config/
  subgroup errors and recovery, two live sessions/recreation, capture rejection,
  and prequantized parity. All pass. Host contracts: 16 passed; changed-file
  pre-commit including mypy/Ruff passes. Full oracle requalification is pending.
- Production candidate caches the existing session-bound thunk (its per-launch
  session/capture agreement is retained), removes the duplicate same-call
  forward validation only for the identical backend object, and exchanges
  a fixed CPU error/configuration header before falling back to full reports
  on any mismatch. All boundaries use that header protocol, so phase mismatch
  cannot mix compact and object wire formats. NCCL-only default PGs retain
  the original object protocol. No kernel/tuning change is included.
- **PT4, planned:** keep legacy donor, pristine FI and optimized FI resident
  in independent processes/contexts on the same GPUs, with strictly serialized
  source turns and five alternating rounds. Primary protocol retains five
  warmups/20 calls and separate cold/warm measurements. Secondary 200-raw-call
  conditioning is reported separately to test state sensitivity. Profiles
  run only after ALL primary samples. Record every rank; require exact output
  checks, stable baseline dispersion for any 5% no-regression claim, and the
  complete public/strong-oracle correctness suite. No next provider is allowed.

Source S2: the public PyTorch distributed source documents get_backend_config
and warns about object-collective overhead:
https://github.com/pytorch/pytorch/blob/main/docs/source/distributed.md.
This supports API selection only; installed-container execution O24 is the
compatibility evidence. No documentation example timing is used as a baseline.

- **O25, measured/PT3:** job 3577900, theia0122, EP1/EP2, 40 samples/group
  from two alternating rounds. Original versus existing non-GIN entry raw
  medians: EP1 T64 warm 95.74→84.53 us, cold 105.44→96.19 us;
  T1024 warm 142.32→130.00 us, cold 146.50→135.84 us. EP2 T64 warm
  115.89→105.26 us, cold 125.22→109.23 us; T1024 warm 163.14→150.94 us,
  cold 170.03→157.26 us. Exact output checks pass. Most five-call fused
  profiles remain close; EP2 T64 also changes by about 5 us. This supports
  the entry-point intervention; it does not attribute every microsecond to
  host marshalling. PH4 is supported within this diagnostic scope.
- The off candidate selects the donor's existing launch_without_gin only
  when persistent finalize and per-token FC1 scaling are disabled. Those FI
  extensions retain their original entry; no vendor device source is edited.
  The disabled persistent num_valid_tokens keyword is omitted from the lean
  entry's host ABI. On keeps its original entry unchanged.
- **PT5, planned:** final F_BASE versus optimized off, EP1/EP2/EP4, T64/T1024,
  five alternating rounds, 5 warmups/20 calls, same cold/warm protocols and
  predeclared 5%/baseline-CV gate. All primary calls run with --skip-profiles;
  separate --profile-only workers run after the complete primary matrix on
  each allocation. This avoids interleaving profiling with later primary
  rounds without asserting profiling caused the old noise. API entry/module
  assertions forbid next. Retained finalize/cache/graph tests follow.

- **O26/PT4 completed:** 3577960 on theia0151/theia0248, all three legacy
  providers resident and turns serialized. 2400 primary critical-rank samples
  (24 groups ×100) across separate standard/conditioned protocols, all bitwise
  checks pass. Standard warm public full: before 6155.76 us → host candidate
  2208.60 us; cold 6147.29→2202.81 us. Per-call agreement counts fall 5→3.
  Before and after PTX AND cubin SHA-256 match exactly on rank 0 (all-rank
  equivalence still to audit), establishing no device-binary change there.
  Standard donor warm raw 361.95 us versus after 349.68 us, but baseline
  round CV 6.89% makes that no-regression gate inconclusive. Cold raw
  377.73→363.44 us with baseline CV4.92% passes the 5% gate. Conditioned
  results are separate: donor raw warm359.63/cold346.56, after314.78/326.21 us;
  donor dispersion remains >5%. A device-state/launch/runtime contribution is
  supported by different timings for identical before/after binaries, but its
  exact mechanism remains unresolved. This does not erase O18's old regression.
  The complete strong protocol/math plus public correctness suite passes
  15 tests on each of four ranks, with original tolerances retained.

- R4: [checkpoint-04.md](checkpoint-04.md), the device-direction counterexample
  and the reason not to attribute the old ratio to an intrinsic device defect.
- All four before/host-candidate PTX and cubin pairs in 3577960 are byte-identical.
- **PT6, planned:** move the existing GIN route-value check into the already
  mandatory per-call collective validator and remove staging's duplicate
  metadata/agreement pass. Public staging still enters that validator through
  _prepare_inputs. Retain the independent per-launch session guard, so prepared
  compute and mixed live sessions remain protected. Extend the real negative
  test to call public stage_inputs as well as forward before recovery. Compare
  fixed legacy donor and the two-agreement candidate with two resident providers
  (same standard/conditioned protocols, all profiles after primary), and rerun
  the complete 15-test strong/public qualification. No device/tuning changes.

- **O27/PT6 completed:** 3578050, theia0083/theia0121, two resident providers,
  final candidate38f650e63. All1600 primary samples pass bit checks/payload
  identity;15 strong/public tests pass on each rank, including invalid public
  staging and recovery. Standard warm full1513.24 us versus native387.31 us;
  cold1594.61 versus393.88 us. Wrapper scopes differ. Standard raw warm
  donor329.55→candidate299.54 us, cold342.62→321.68 us. Donor round CV6.58%
  and5.76% still prevent a formal5% no-regression claim. The 200-call conditioned
  results stay separate. Full sampled data: resident-v2/job-3578050/summary.json.
- **O28, final unbound off matrix:** 3578010 (theia0248 EP1/EP2) and3578011
  (theia0154 EP4) complete30 paired source points,4800 critical-rank samples,
  all output/payload/profile checks.9 PASS,0 REGRESSION,15 inconclusive baseline
  dispersion. All medians are within the5% limit, but that does not make noisy
  cases pass.8 EP1 gates and EP2 T64 warm-raw gate pass. Deferred profiling
  alone did NOT fix EP4 noise, so it is not established as that noise's cause.
  Four finalizer tests, one cache/graph test and three layer/scale tests pass
  per rank at EP4, retaining the extended-entry fallback's correctness.
- **O29, profiled local arrival skew:** old3577696 F_BASE EP4 traces contain
  15 same-host paired calls with GPU start skew>100 us; the largest is6328.8 us
  while GPU end skew is0.1 us. Other samples show~3.3–3.6 ms start skew with
  near-simultaneous completion. Thus those long profiled spans include waiting
  for late peers, not only arithmetic. This does not establish the cause of
  every unprofiled sample. Derived CSV: blackwell-fix-20261102/profile-launch-skew.csv.
- **O30, topology/placement observations:** read-only topology probe on
  theia0154 maps GPU0/1 to CPU NUMA0 and GPU2/3 to NUMA1. A live read-only probe
  during3578050 finds EVERY worker restricted to CPUs0-31 on both nodes. The
  same32-CPU Slurm task configuration is used by the unbound off jobs; their
  workers exited before a direct affinity read completed. Do not relabel that
  inference as a direct EP4 process observation.
- **PH5, candidate:** shared CPU placement can delay some ranks' submissions
  and make collective device spans include peer-arrival waits. This predicts
  lower variance with GPU-local disjoint CPU sets. It is compatible with O29
  and O30 but not yet proven by those observations. **PT7, launched3578088:**
  on ONE exclusive GB300 node, interleave fixed F_BASE/shared32,
  F_BASE/GPU-local8-per-rank, and candidate/GPU-local8-per-rank over five rounds.
  Both placements use32 physical CPUs total, all original timing/cache rules
  and output checks retained. NVML queries and bindings precede Torch/CUDA
  context creation; sidecars record actual sets. The final pinned pair also
  receives deferred profiles.3578087 repeats unresolved EP2 gates with the same
  binding; its eight primary gates now pass. EP4 results remain pending.
- **PT8, launched3578114:** repeat the final two-provider resident N2R2
  comparison under the same pre-context NVML local binding. Old unbound and
  conditioned measurements remain separate; no baseline cherry-picking.

Sources S3/S4 (guidance, not current-root-cause proof): Glean retrieved the
NVIDIA Grace tuning guide and NVBug5914874 after the affinity/launch-latency
query. The latter concerns Strata/Rubin and cannot establish a GB300 diagnosis:
https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/index.html
https://nvbugswb.nvidia.com/NVBugs5/redir.aspx?url=/5914874.
NVML's documented node-scope CPU mask API is used only for the benchmark
placement experiment; installed driver calls succeed and actual masks are
recorded. No GPU clocks or memory policy are modified:
https://docs.nvidia.com/deploy/archive/R535/nvml-api/group__nvmlAffinity.html.

- **O31/PT7 completed:**3578087 (theia0154) and3578088 (theia0095) complete.
  Together with retained EP1 from3578010, all24 off gates PASS (48 groups,
  4800 critical-rank samples), all bits/per-rank payloads/profile counts valid.
  On the SAME EP4 allocation, unchanged F_BASE shared32 raw round-CVs are
  6.31/16.85/88.38/40.03% for T64warm/cold,T1024warm/cold; GPU-local binding
  gives2.42/3.09/0.97/2.01%. This is a controlled placement intervention using
  32 distinct worker CPUs either way. PH5 is supported within this launcher;
  it does not distinguish NUMA locality, per-rank partitioning or detailed
  driver mechanisms. CPU masks/UUIDs are observed, not inferred, in sidecars.
- R5: [checkpoint-05.md](checkpoint-05.md), the off qualification and bounded
  placement conclusion. PT8 is the remaining final on confirmation.

Current coverage revision: PH1 explains the measured public-call envelope,
not raw device cost. PH4 explains the ordinary-off entry intervention and the
confirmed raw regressions are now absent under qualified measurement. PH5
accounts for reproducible EP4 placement noise; it does not explain every old
EP1/on case. PH2 remains unestablished as an unconditional device slowdown.
PH3 covers observed state/arrival sensitivity but the exact mechanism behind
all old N2R2 ratios remains unresolved. No prior observation has been dropped.

- **O32/PT8:**3578114 completed all1600 timing samples, bit checks, payload
  checks and profiles, but the job FAILED afterward: the affinity wrapper
  called main() on a worker module that executes at import and has no main.
  Consequently its later correctness phase did not run. Retain whole-job
  failure and do not label it a qualification pass. Standard raw medians are
  donor364.75→port322.38 us warm and369.39→338.30 us cold; baselineCV~11%
  remains inconclusive. Thus local binding alone is NOT established as a fix
  for all N2R2 noise. The corrected wrapper will not double-invoke that module.
- **O33, source review/prediction:** the cached GIN thunk key still included
  the local CUDA stream. A stream change on only one rank would make that
  rank enter launch-prepare while cached peers enter launch; compact phase
  checking would reject the call. The cache optimization must not add this
  restriction. **PT9, planned:** reproduce the previous38f650e63 rejection,
  bind the current stream at execution for managed eager-GIN thunks (keep
  off/graph thunks stream-specific), retain launch/session/capture agreement,
  and test actual compiled-call stream handles plus bitwise outputs while only
  one rank alternates streams. Explicit bound-device validation stays inside
  the existing collective boundary, before staging. Final17-test qualification
  also checks wrong-current-device recovery. No device binary change is intended.


- **O34/PT9 completed:** final3578176 COMPLETED exit0 on theia0048/0115.
  Old38f650e63 rejects the predicted one-rank stream change on all ranks;
  corrected74c036287 passes17 strong/public tests per rank, including actual
  compiled-call stream handles, bitwise outputs and bound-device recovery.
  Its4 off finalizer tests and1 cache/graph test per rank also pass.
- **O35:** final resident comparison validates16 groups/1600 samples, all
  bit/payload/profile checks. Standard raw donor350.368→receiver349.312 us
  warm and363.984→356.432 us cold; baseline round CV6.38/6.39% makes both
  inconclusive. Conditioned200 protocol remains separate:359.440→319.792 us
  warm,373.904→321.760 us cold, CV8.51/9.16%, also inconclusive. Public
  standard medians1630.15/1656.86 us; native prepared402.74/404.10 us have
  different wrapper scopes. No paired cross-allocation speedup is inferred.
- **O36:** fixture audit corrects the initial performance plan's balanced
  label: actual seeded random top-k has512 slots, all48 experts,4–23/expert.
  Providers share identical payloads; balanced/sparse correctness fixtures
  are distinct. Routing histogram is saved with the final evidence.
- **R6:** [performance-fix-blackwell.md](performance-fix-blackwell.md) is the
  coherent final correction report. The original report and failures remain.

Final coverage: PH1 has paired host-intervention support (O18–O27) and the
final guard count/latency is measured; it does not explain raw device spans.
PH4 ordinary-off entry correction is qualified by all24 off gates (O31).
PH5 placement intervention is causal within the EP4 launcher (O31), but the
claim that placement eliminates ALL N2R2 noise is contradicted by O32/O35.
PH3 arrival/state sensitivity remains supported as an intermediate observation;
its deeper cause for all old N2R2 ratios is unknown. PH2 unconditional intrinsic
kernel regression remains unestablished: direction reversals and identical
paired binaries preclude attributing every raw ratio to changed device code,
while noisy final baselines also preclude a universal no-regression claim.
No old observation is removed. Practical off correction and public-host cost
reduction are delivered; internode promotion remains an explicit evidence gap.
A useful future check would require a demonstrably stable N2R2 launch baseline;
repeating unchanged allocations until a passing sample appears is not warranted.

Archive handoff for R6: optional per-invocation authorization requested;
status pending. No session trace was collected, packaged or uploaded.
The benchmark artifact index is ordinary experiment evidence, not a session
trace archive. No destination is inferred from project remotes.
