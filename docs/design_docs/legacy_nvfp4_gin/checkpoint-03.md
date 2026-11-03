# Blackwell performance checkpoint: repeated host agreement

The large full-call gap is dominated by repeated host agreement. This finding
does not yet explain the compiled GPU pipeline regression.

On exclusive GB300 nodes theia0095/theia0201, job 3577851 reused the same
ordered-BF16-validated fixture for fixed legacy donor f863d754 and FI c597d35ba.
The selected FI backend and donor source paths were asserted to be legacy.
Every output comparison passed bitwise. This was a 20-sample warm diagnostic,
not a replacement for the earlier five-round qualification matrix.

| Diagnostic path | Critical-rank full-call median |
| --- | ---: |
| Native legacy donor adapter | 372.289 us |
| Unchanged FI public call, instrumented | 6978.907 us |
| FI with object agreement on the explicit CPU control group | 5565.586 us |
| FI with compact CPU tensor agreement, same boundary checks | 3317.156 us |

The instrumented original executes five preflights per call: backend-forward
twice, before-input-staging, launch-prepare and launch. Rank 0 median boundary
time is 1104–1251 us; rebuilding the CuTe runtime arguments costs another
214 us. Compact agreement keeps error/configuration checks but uses a
fixed-size success message and the detailed object slow path on mismatch.
It reduces the full-call envelope while the compiled target is unchanged.
These are diagnostic interventions, not production changes or accepted APIs.

Unprofiled raw GPU medians in the same run remain 304.480 us donor and
334.672 us FI (+9.9%). Host agreement alone cannot explain this difference.
No host-boundary time is added to the independently measured GPU span.

Next: cache session-bound preparation, eliminate duplicate validation with
control-error tests retained, compare compact CUDA agreement, and inspect
actual legacy kernel compiler outputs before changing device code.

Raw records and drivers:
`/lustre/fsw/coreai_devtech_all/qiangf/ncclGIN/port/migration-evidence/blackwell-fix-20261102/`.
Per-rank files: `job-3577851/{donor,port}/rank-*.json`; traces are adjacent.
