# Unit-test timing data

These files are versioned inputs to `scripts/task_run_unit_tests.sh`:

- `unit_test_duration_estimates.csv.gz` contains one conservative duration per timing profile and canonical pytest node ID.
- `unit_test_duration_estimates_summary.csv` is the reviewable function-level summary of the compressed file.
- `unit_test_overhead_estimates.csv` contains per-source pytest startup and warm-up estimates.

The initial `sm100-cuda13` duration profile was seeded from the repository's most recent legacy JUnit-derived `test_function_timings.csv`. This matches the runner's architecture fallback when a B100 driver reports the generic device name `NVIDIA Graphics Device`. Recognized device models use model-specific labels; otherwise automatic profiles use `sm<major><minor>-cuda<major>`. Future updates should use the manifest-aware workflow below so exact node identity, synthetic status, and warm-up telemetry are available.

Ordinary test runs never edit these files. To rebuild observations from a chosen window of completed or partial sharded runs:

```bash
python scripts/rebuild_test_duration_estimates.py scan \
  /path/to/junit-run-1 /path/to/junit-run-2 \
  --output-dir /tmp/test-duration-review
```

After reviewing the observation CSV and diagnostics, refresh the tracked estimates with the same explicit input window:

```bash
python scripts/rebuild_test_duration_estimates.py refresh \
  /path/to/junit-run-1 /path/to/junit-run-2 \
  --output-dir /tmp/test-duration-review
```

The refresh uses nearest-rank p90 observations, grows estimates immediately, decreases them by at most 10% per refresh, ignores skipped and synthetic timeout cases, and writes deterministic gzip output. Add newly produced estimate and summary changes to the same review as the test-suite change that motivated them.

During planning, exact node estimates remain preferred. Function- and
source-level p90 fallbacks are used only when the timing database covers at
least 80% of the corresponding nodes in the current collection. Sparser
aggregate data falls back to the selected profile's arithmetic mean (about 1.5
seconds for the current `sm103` profiles), with `--unknown-case-seconds` as its
floor. This prevents a biased subset of slow parametrizations from assigning
its p90 to thousands of unseen cases.

The default checkpoint and logical-unit targets are both 1,000,000 seconds.
This preserves the original one-pytest-process-per-source-file behavior for
ordinary files while retaining deterministic manifests and per-file resume
boundaries. Configure lower targets when finer-grained load balancing or resume
checkpoints are more important than avoiding repeated process warm-up.

For a source with high measured process startup and warm-up cost, the planner
deterministically expands a smaller configured checkpoint up to the
logical-unit target so estimated work can reach fifteen times the per-process
overhead.

The planner also caps each source at four pytest batches per estimated
logical-unit target. This bounds process churn when overhead telemetry is
missing or underestimates warm-up cost. A capped batch may exceed the soft
checkpoint target, but it remains an independently finalized JUnit/resume
boundary.

GitLab's artifact cleaner truncates XML attributes, so long parametrized node IDs
cannot be used directly. Reconstruct a manifest with the same source revision,
collection environment, timing inputs, and packing options as the jobs, then
scan the downloaded ZIPs and sibling job logs:

```bash
python scripts/rebuild_test_duration_estimates.py scan \
  --cleaned-artifact-dir /path/to/downloaded-job-artifacts \
  --reconstruction-manifest /path/to/reconstructed/junit/manifest.json \
  --output-dir /tmp/test-duration-review
```

The scanner restores a node ID only when the deterministic batch ID, testcase
order, and truncated prefix agree. For batches that differ from the
reconstruction manifest, it combines the independently truncated
`pytest_nodeid` and testcase `name` prefixes and accepts only a unique match.
Exact failed/skipped results printed in sibling logs fill gaps from batches
that did not finalize XML. Ambiguous or absent node IDs are reported and
omitted rather than guessed. Use `refresh` instead of `scan` after reviewing
those diagnostics. JUnit from failed or infrastructure-error jobs remains
useful: every finalized batch is scanned independently of the job's final
status.

Pruning is intentionally separate. It requires `--prune` and a manifest from a complete, unsampled collection rooted at `tests/`:

```bash
python scripts/rebuild_test_duration_estimates.py refresh /path/to/runs/* \
  --prune \
  --complete-collection-manifest /path/to/complete/junit/manifest.json
```
