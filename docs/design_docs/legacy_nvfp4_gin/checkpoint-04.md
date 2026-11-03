# Blackwell checkpoint: the GPU difference changes direction

The device regression is not yet attributable to an intrinsic kernel slowdown.
A new fixed-source legacy comparison changes the direction of the raw timing
difference. The previously reported regression remains valid for its recorded
conditions; this counterexample prevents a general device-cause claim.

| Warm diagnostic, GB300 theia0090/theia0201 | Legacy donor | Pristine FI |
| --- | ---: | ---: |
| Unprofiled critical-rank raw GPU median, 20 calls | 359.728 us | 316.848 us |
| Separately profiled fused median, 5 calls | 291.775 us | 287.107 us |
| Compiler active-cluster argument | 152 | 152 |

Job 3577883 saves actual PTX/cubins and constructor fields for every rank.
Selected legacy source/config paths and exact output bits pass. The FI source
is still c597d35ba; its compact-CUDA host diagnostic changes no kernel source.
Profiling follows that host diagnostic, so these short profile values are
not a replacement for unprofiled primary results. Constructor differences
include FI's four additional local counter words and retained optional-feature
flags. Their causal relevance remains unknown.

The production host candidate separately passes five control tests per rank,
including compact-header mismatch/recovery and session/capture lifetime checks.
The 16 host contracts and changed-file mypy/Ruff/pre-commit checks also pass.
Full oracle requalification and five-round performance confirmation remain pending.

The next check keeps legacy donor, pristine FI and optimized FI resident on the
same GPUs with strictly serialized turns, captures all primary timings before
profiling, and reports extra conditioning as a separate diagnostic protocol.
This addresses process-start/state confounders without selecting a next baseline
or changing the device implementation on speculation.

Evidence: `migration-evidence/blackwell-fix-20261102/device-probe-v3/`,
`job-3577883/{donor,port}/rank-*.json`, compiler JSON and PTX/cubins alongside.
