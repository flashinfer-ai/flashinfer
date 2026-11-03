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
