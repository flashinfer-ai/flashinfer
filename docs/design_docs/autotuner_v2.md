# Autotuner v2 — Managed Persistence, Deployment-Matched Measurement, and the Runner Contract

**Scope**: `flashinfer/autotune_cache.py` (the whole module) and the `autotune_v2` state and hook
sites in `flashinfer/autotuner/autotuner.py` — the managed store, the measurement policy, and the
runner contract that tactic identities must satisfy.

**Not** in scope: the v1 path — `autotune()`, `save_configs()`, `load_configs()`, and the
profiling machinery they share. That code predates this document and has none of its own; changing
it does not oblige you to update this doc. §4 is where the two are planned to converge, and the
day v1 is folded into v2 this scope line should widen to match.

**Status**: proposed. RFC [#3920](https://github.com/flashinfer-ai/flashinfer/issues/3920); this
document ships with the v2 MVP in
[#3861](https://github.com/flashinfer-ai/flashinfer/pull/3861) (opt-in; `autotune()` and the v1
cache path are untouched). Sections 1–3 describe the design as implemented in this branch;
section 4 (graduation) is the part that is **not** yet agreed and is the reason this document
exists.

## 1. Motivation

FlashInfer's autotuner picks a tactic per operation by profiling candidates at warmup, then
replays that choice at serving time. Persisting those choices across processes is what makes
warmup a one-time cost instead of a per-restart one.

In v1 the *caller* owns persistence: `autotune(enable, cache="<path>.json")` plus
`save_configs(path)` / `load_configs(path)`. The filename is the cache identity, so every
consumer had to invent the same policy independently:

- **vLLM**: rank 0 tunes into a vLLM-computed `sha256(compile-factors)` directory, broadcasts
  the raw JSON bytes, and every rank rewrites the file and calls `load_configs(path)`.
- **SGLang**: every rank tunes simultaneously (the warmup forward contains collectives, so
  rank-0-only tuning deadlocks DeepEP), with one JSON file per rank, its own model/quant/
  parallelism hash, and its own disable knob.

Both reimplemented cache hashing, atomic writes, staleness handling, and an enable knob — all of
which are FlashInfer policy. The cost of getting that boundary wrong is on record:
[vllm-project/vllm#43119](https://github.com/vllm-project/vllm/issues/43119) — the v1 key omitted
`use_8x4_sf_layout`, invalid tactics were replayed, and vLLM disabled the persistent cache for
all multi-rank deployments. One key-completeness bug cost the feature its default.

v2's thesis: **FlashInfer owns keys, schema, placement, and invalidation; frameworks decide only
*when* to warm up and how to coordinate ranks.** Autotune data is a disposable performance
optimization — cross-version migration is an explicit non-goal, following the Triton /
`torch.compile` compiler-cache pattern (key by program + environment, publish atomically, treat
unusable entries as work to rebuild).

## 2. Design

### 2.1 API surface

v2 is a standalone context manager, deliberately disjoint from `autotune()`:

```python
graph_policy = MeasurementPolicy(execution_mode="cuda_graph")
with flashinfer.autotune_v2(measurement_policy=graph_policy):
    model(dummy_inputs)      # warmup: tune misses, publish each winner atomically
model(inputs)                # serving: reuses entries, no context needed

# fresh process, same environment and measurement policy
graph_policy = MeasurementPolicy(execution_mode="cuda_graph")
with flashinfer.autotune_v2(mode="replay", measurement_policy=graph_policy):
    pass                     # hydrate only
model(inputs)
```

The measurement policy is part of the managed-cache identity. Reusing the same
cache root without passing the original policy selects a different store; it
does not hydrate the graph-tuned winners. Bare serving after the context exits
uses the attached store under the default process scope.

| `mode` | `persistent_cache` | meaning |
|---|---|---|
| `"tune"` | True *(default)* | tune misses, publish to disk |
| `"tune"` | False | tune in-memory only (disk forbidden) |
| `"replay"` | True | serve from the on-disk store, no profiling |
| `"replay"` | False | memory-only replay (already-hydrated winners) |

`mode` names a positive action rather than a negated flag: `mode="replay"` is the serving path,
which does not read like "throw the tuning results away" the way `enable_tuning=False` did.
Bucketing (`tuning_buckets`, `round_up`, `skip_ops`) is delegated to `autotune()` unchanged;
only persistence and measurement differ.

**Attach semantics.** The default `cache_scope="process"` keeps the managed store
attached after warmup, so bare inference continues using it. Sequential process
attachments are last-wins. `cache_scope="context"` instead selects the store only
inside the context, without changing the ambient pointer. Both use the same
store registry, measurement identities and atomic persistence.

An identified context scope (a managed store or a nonempty explicit measurement
policy) excludes legacy-loaded and bundled winners. The legacy serving state
remains available outside the scope. With `persistent_cache=False` and no policy
identity, legacy memory/file behavior is unchanged. Shortlists and delegated
protocol records follow the same identity as ordinary winners; only the first
shortlisted tactic is persisted, as in process scope.

Tune before capture, then select the same identity for capture-time warmups and
capture. A completed CUDA graph replays no Python cache lookup. Uncaptured eager
tails must use their appropriate policy or the ambient path. For example:

```python
graph_policy = MeasurementPolicy(execution_mode="cuda_graph", cold_l2=False)
with autotune_v2(cache_scope="context", measurement_policy=graph_policy):
    graph_region(inputs)  # tune before capture
graph = torch.cuda.CUDAGraph()
with autotune_v2(mode="replay", cache_scope="context", measurement_policy=graph_policy):
    graph_region(inputs)  # capture-time warmup, using cached winners
    with torch.cuda.graph(graph):
        output = graph_region(inputs)
graph.replay()
eager_tail(output)  # existing ambient/legacy behavior
```

Nested v2 contexts remain unsupported; nesting ordinary `autotune()` is
unchanged. `autotune_v2_reload()` must run outside context scopes: it clears all
in-process partitions, immediately hydrates the ambient store, and other stores
hydrate when next selected. Overlapping read-only replay scopes are supported;
concurrent tuning/serving and distributed scoped-store use require separate
qualification. Scoped ownership does not make the process-global tuning mode
thread-local.

**Replay coverage.** An optional `ReplayCoverage` report observes Python cache
lookups without profiling. Pass `replay_report=report` to `mode="replay"`; the
context yields the same report. Counters cover all observed decisions, while
`records` keeps only the first `max_records` (64 by default). Reusing a report
accumulates counts. Each record includes the op, input shapes, cache identity,
source, selected runner/tactic and known profiling-policy provenance. An unknown
stored policy is `None`. Tactic representations are truncated to 256 characters.

A cached default tactic `-1` is a hit. An absent or rejected winner is a miss;
explicit `skip_ops` exclusions are counted separately. Ordinary observation
preserves source precedence and compatibility behavior, including legacy-file
hits and replay of entries whose profiling policy is unknown. A report describes
tuner choices, not backend-internal fallback or the kernels actually executed.
Delegated protocols report their own `TuningResult.cache_hit`; protocols remain
responsible for honoring disabled tuning and validating their records.

For single-rank preflight, `strict_replay=True` additionally requires a compatible
cached winner for every non-skipped lookup, including known matching profiling
policy for ordinary winners. It requires an identified `cache_scope="context"`
and raises on a miss without profiling or publishing a replacement. It creates
and yields a report when none is supplied:

```python
with autotune_v2(
    mode="replay", cache_scope="context", measurement_policy=graph_policy,
    strict_replay=True,
) as coverage:
    for inputs in planned_capture_inputs:
        graph_region(inputs)  # preflight every intended capture profile
```

Run preflight before capture, then use the same identity and profiling controls
for capture-time lookup. Strict lookup also works during capture, but a miss then
aborts that capture; it cannot prove coverage for shapes never exercised. CUDA
graph replay executes no Python and adds no records. Shortlist calls in replay
return and report the single cached winner, as before. Diagnostics reject nested
tuning, forced protocol tuning and direct library profiling. The default path
allocates no report and retains fallback behavior.

Strict replay rejects configured tuning groups or a distributed world larger
than one: a unilateral exception during a model forward could strand peers in a
collective. Distributed callers can collect non-strict reports, then coordinate
coverage checks at a framework-owned boundary before capture. This API adds no
serving collectives and does not provide concurrent tuning/serving support.

### 2.2 Store layout on disk

```text
<root>/v2/<environment_hash>/
├── manifest.json                 # canonical environment, human-readable
└── entries/
    ├── <operation_hash>.json     # one atomic file per tuned operation
    └── ...
```

- `<root>` defaults to `FLASHINFER_CACHE_DIR/autotune`, overridable with
  `FLASHINFER_AUTOTUNE_CACHE_DIR`. It is **placement only** — the schema and environment
  namespaces live below it, so no choice of root can mix incompatible entries (unlike the v1
  filename, which was identity-bearing).
- `<environment_hash>` = first 16 hex of `sha256(canonical manifest)`.
- `<operation_hash>` = first 24 hex of `sha256(file_key)`, where `file_key` is the autotuner's
  canonical lookup key (op identity, bucketed dynamic dims, dtypes/layouts, runner extras).
  Repeated layers and different models share an entry whenever their keys are identical.
- Each entry is `{"key": <file_key>, "runner": <class name>, "tactic": <json>}`. The embedded
  key guards against hash collisions and foreign files.

### 2.3 Environment identity and invalidation

The manifest reuses v1's `_collect_metadata()` — `flashinfer_version`, `cuda_version`,
`cublas_version`, `cudnn_version`, `cudnn_frontend_version`, `cutlass_dsl_version` (the CuTe-DSL
compiler-stack fingerprint — DSL runners' tactic spaces are tile configs compiled by it), `gpu` —
plus `cache_schema` and any
non-default `MeasurementPolicy` fields. v1 compares those fields at load time and rejects a
mismatched file wholesale; v2 *hashes* them into the directory name instead.

That difference matters more than it looks. Under v1, a changed environment means the file is
unusable and `save_configs` refuses to write back to it — the user must pick a new path.
Under v2 a changed environment is simply a different directory: nothing to invalidate manually,
old generations stay on disk, and downgrading finds its own entries intact. Invalidation is
therefore structural, not procedural.

**Autotune caches do not persist across FlashInfer versions**, and never have —
`flashinfer_version` is in v1's metadata and in v2's manifest, so a patch bump alone retires
every entry. This is intentional: paying the tuning cost once after an upgrade is cheaper than
maintaining a migration path for data that is, by construction, an optimization.

### 2.4 Concurrency and crash safety

- **Publish on tune, not at exit.** Each winner is written as soon as it is measured, via
  tempfile + `os.replace` in the same directory (same-filesystem rename, atomic). There is no
  read/merge/write cycle and no exit-time save, so a crashed or killed tuning run keeps every
  winner it had already measured.
- **No locks.** Concurrent writers do redundant work and the last valid write wins. Atomic
  publication protects individual files, not agreement on cache hits. It is a deliberate
  divergence from FlashInfer's JIT kernel cache, which takes a cross-process
  `FileLock` for single-flight: that is correct there and wrong here. Compiling the same kernel
  twice wastes minutes of CPU, whereas ranks tune *inside collectives*, so a cross-rank lock
  would either serialize warmup or deadlock it. Redundant measurement is the cheaper failure.
- **Invalid is a miss, never an error.** A missing file, malformed JSON, or an embedded-key
  mismatch logs a warning and returns "not found". A corrupt entry costs one retune, not a dead
  server, and cannot take the other entries with it.
- **One filesystem probe per key per process.** Positive and negative lookups are memoized
  inside the store object — never in v1's `_file_configs` — so the serving hot path touches
  neither the filesystem nor the JSON decoder, and v1/v2 state mixing is impossible by
  construction.

### Custom tuning protocols

`AutoTuner.choose_with_protocol()` supports operations that own candidate
enumeration, measurement and result installation, on one device or across
processes. It supplies the tuner's controls, lock, statistics and record storage.
The protocol defines cache compatibility and handles any distributed coordination.
The tuner always invokes it, even when tuning is disabled or the operation is
skipped, so a protocol can coordinate its fallback or cache decision.

`TuningContext.records()` reads the active store's protocol records.
`context.publish()` keeps a record in memory and writes it to that store when
ambient tuning is enabled and the operation is not in `skip_ops`.
`force_tune` can request a fresh search but does not override these write
controls. Returning a `TuningResult` does not publish it. Replay imports remain
in memory. No JSON filename is required.

Protocol results use the same `entries/` files, writer and preload as runner
tactics. Their structural key is `("__protocol__", namespace, record_key)`;
`runner` holds the stable protocol namespace and `tactic` holds the JSON result,
including any metadata needed for selection. Neither field requires a runner
instance. The protocol owns its record keys and payload schema. Changing
aggregation or selection requires a distinct key or compatibility check when
previous results are no longer reusable.

Preload indexes these results by namespace for `context.records()`, preserving
JSON lists and dictionaries. `context.lookup(key)` reads one result without
copying the namespace. Both return copies. Ordinary runner tactics keep their
existing decoding. Atomic writes preserve unrelated entries; writes to the same
key can overwrite each other. `clear_cache()` and `autotune_v2_reload()` clear
the namespace index along with other lookup memos. The protocol retains its
own compatibility and capacity selection rules.

Root and environment switches isolate protocol records. With persistence disabled,
records use the in-memory measurement-policy partition, and the protocol must
also disable any compatibility-file reads and writes. Measurement policies,
inherited graph replay counts, `tuning_buckets` and `round_up` are passed to the
protocol, which must implement or reject them. Protocol records are separate
from v1 JSON import/export.
Existing `autotune()` controls still compose with v2 as described above.
`context.cache_identity` includes the store or policy identity, reload generation
and skip state so protocols can invalidate prepared selections. Replay counts
and bucket overrides are exposed separately; protocols must account for them
when reusing prepared state or tuning results.
Serving outside a context uses the attached store's measurement policy.

### 2.5 MeasurementPolicy — tune the way you deploy

```python
MeasurementPolicy(
    execution_mode="auto" | "cuda_graph" | "eager",
    cold_l2=None,
    graph_timing=None | "amortized" | "individual",
)
```

The primary axis is **whether per-call host cost counts**, not the timer implementation. Under
CUDA-graph serving the host cost is paid once at capture, so excluding it is correct; under eager
serving it is paid every call, and excluding it mis-ranks host-heavy candidates. Measured on
SM100 (`bmm_fp8`, M=8, kernel ≈ 8 µs): the same cuDNN candidate reads **8 µs host-excluded** and
**≈ 330 µs host-included**, and the eager ranking flips to cublas < cutlass ≪ cuDNN. Even the
fastest backends are host-bound at decode sizes, so this is a regime property, not a one-backend
anecdote.

`execution_mode` is the deployment statement; capture behavior and timing implementation are
derived from it (`eager` → event timing with no delay kernel; otherwise standard event timing).
`cold_l2` is orthogonal. The policy is part of the store's environment identity, so entries tuned
under different policies land in different directories and never overwrite each other — which is
what closes the eager-warmup / graph-serving aliasing.

`graph_timing` optionally selects the **hot-L2 graph** measurement objective.
`"amortized"` divides a repeated replay window by its invocation count;
`"individual"` takes the median of per-call intervals recorded inside the graph.
The latter uses external CUDA events, or timestamp kernels when the autotuner
selects `%globaltimer`. These nodes can change scheduling, so individual timing
needs validation against the deployment's uninstrumented graph. It is not model
decode latency or a guarantee of better tactic selection.

The default `None` inherits `TuningConfig.use_cuda_graph_individual_timing`, which
defaults to `False`. Existing policies retain their cache identity. An explicit
objective creates a distinct managed-cache namespace and can override a runner's
default in either direction. It does not enable graph capture by itself; specify
`execution_mode="cuda_graph"` for that. Cold-L2 profiling keeps its existing
measurement path. Explicit eager or CUPTI policies reject `graph_timing`;
CUPTI also overrides an inherited per-op individual-timing flag.

`prepare_tactic_profile` applies the same measurement and replay-count overrides
before preparing the input schedule used by `profile_tactic`. Delegated tuning
contexts also preserve the attached store's graph-timing objective after warmup.

Event timing initializes its delay-kernel module before candidate warmup. Lazy
module loading between warmup and timing can bias the first candidate in a fresh
process. Timing paths that do not use the delay kernel do not initialize it.

Each newly tuned entry also records an optional `profiling_policy` payload: the effective
CUDA-graph replay count (or `null` without graphs), L2 policy (`hot`, `cold`, or
`cold-chain:<n>`, see `TuningConfig.cold_chain_launches`), and active individual-call
graph timing after context and measurement overrides. These inherited settings
are not fully represented by the manifest.
When `choose_one` can profile, it reuses a managed winner only when this provenance matches,
including after preload, reload, and promotion to the root/environment-partitioned memory
cache. A missing or different policy is a profiling miss, not deletion; replay, bare inference,
and direct `search_cache` lookups still accept the winner, even inside a tuning context,
unless strict replay explicitly requires matching provenance.
Entries without provenance are retuned by `choose_one` even under the default policy, including
after a same-version build upgrade that leaves the environment directory unchanged.
Successful retuning retires the previous candidates before publishing the new winner, including
when the winning runner changes. A single-process replacement keeps the entry being republished
on disk until its atomic overwrite, avoiding a missing-file window for concurrent readers.
Single-process retuning preserves the previous winner if no candidate succeeds.
Coordinated tuning must retire rejected selections before profiling, so a
failed retune cannot restore a selection that the group already rejected (§2.7).

`rank_tactics` uses the same measurement overrides, store/policy partition, and winner publication
as `choose_one`. Full shortlists remain memory-only and are rebuilt after reload; their first
tactic is persisted as the serving winner. Loaded v1 or bundled entries remain compatibility
fallbacks unless that key has been retired in the active identity. This exclusion is local to
the process and identity, survives `autotune_v2_reload`, and resets with `clear_cache`; it does
not delete global v1 entries used by another identity. This is an additive v2 field, not a new
schema, manifest identity, or operation key. Legacy v1 file-policy gating is unchanged.

`"auto"` currently preserves legacy behavior. Flipping the default to `"cuda_graph"` (the
dominant serving mode) is gated on validating capture-safety across the op suite.

### 2.6 Runner contract

Persistence is only as good as the tactic identities it stores. A backend audit found one disease
in three forms: a tactic interpreted *relative to runner-internal shape-keyed state* — cuDNN plan
indices into per-bucket plan lists, cuBLASLt algo-list indices re-enumerated at raw runtime
shapes, a device-dependent compile-cache key gap in CuteDSL. The contract:

1. **Tactics are self-describing** (explicit parameter tuples) or index a shape-independent
   static table — never an index into something that varies with shape, library version, or
   enumeration order. ([#3707](https://github.com/flashinfer-ai/flashinfer/pull/3707)'s
   structured `(engine, knobs)` tactics bring cuDNN into compliance.)
2. **Runner-internal caches are keyed by the tactic**, never by a runner-derived shape bucket.
3. **Compile/graph/algo cache keys include device identity** and every trace-time-baked
   parameter.
4. **Revalidate at runtime where cheap** (the CUTLASS `isValidConfig` pattern): a stale tactic is
   a loud fallback, never a silent different kernel. The autotuner-side hook
   `validate_tactic(inputs, tactic)` covers in-memory and on-disk hits; per-runner adoption is
   separate work, so this class is currently *guardable*, not guarded.
5. **`get_cache_key_extras` is synthesis-invariant** (no raw dynamic dims). Violating this is
   exactly the vllm#43119 class. It extends to any consumer-visible execution dimension a runner
   exposes — e.g. CuteDSL compile caches treat dynamic-vs-static batch as separate entries, so a
   warmup exercising only one silently misses the other unless that dimension is in the key.
6. **A serving-time miss must not trigger unbounded host-side compilation.** Reported on sm_12x
   as a 12–17 s cuDNN graph build inside the engine loop; measured to be ~325 ms/shape
   accumulation across dozens of untuned serving shapes. Largely closed in-bucket by cuDNN
   override-shape support; the residual is out-of-bucket shapes.

CuteDSL is otherwise the model architecture here: its caches are keyed by the tactic itself and
kernels take shapes as dynamic arguments, so the tactic *selects* a cache entry rather than
indexing into one.

### 2.7 Distributed

Persistence and orchestration stay separate responsibilities.

- **Shared filesystem**: per-entry publication prevents torn entries. It does not provide a
  snapshot across ranks: concurrent publishers or an interrupted run can produce mixed hits.
- **Profiling participation**: `set_autotune_process_group` adds startup gates
  for operation/runner order, skip decisions, profile count, effective measurement
  policy and ordered candidates. A mismatch raises on all participating ranks
  before they enter incompatible tactic loops. Rank-local shapes and cache roots
  need not match; runners still own the validity of their distributed inputs and
  kernel collectives. Every rank must enable tuning and enter the same calls.
  A missing rank remains a process-group timeout, not a recoverable plan mismatch.
- **Preparation failures**: input synthesis, candidate discovery and runner
  preparation coordinate OOM fallback and propagate caught non-OOM Python errors
  across the group. A fixed-size phase/selection exchange also detects mismatched
  nested shortlist preparation. It does not recover sticky CUDA errors or
  failures inside arbitrary runner collectives. Delegated protocols still own
  their distributed coordination.
- **Cache agreement**: each profile reduces its hit flag and a digest of the
  cached runner/tactic. Only an all-hit group with matching selections skips
  profiling; any miss or disagreement makes all ranks discard that profile's
  winner/shortlist and re-profile. Malformed persisted selections remain cache
  misses. Warm tuning still performs startup gates; serving with tuning disabled
  adds no collective. Distributed tuning must run outside CUDA Graph capture.
- **Timer fallback**: CUPTI availability and initialization are checked across the tuning
  group before candidate warmup or graph capture. If tracing fails on any rank, all ranks
  fall back to CUDA events. Candidate exceptions still disqualify the tactic rather than
  retrying it with another timer.
- **In-session rank consistency**: coordinated tuning re-profiles divergent cached winners and
  tactic shortlists. Independent tuning without a process group can still diverge until
  `autotune_v2_reload()` runs — a finalize step (tune → barrier → reload) that drops in-process
  winners and ephemeral tactic shortlists, bulk re-hydrates the attached store's final
  published state into memory, and thereby makes shared keys use the store's final tactics
  (no lazy per-key disk reads afterwards). Later tuning rebuilds shortlists by profiling.
  Rank-specific keys remain distinct, and independent deployments must not keep publishing
  into the store while another group finalizes if identical snapshots are required.
  Composes with [#3187](https://github.com/flashinfer-ai/flashinfer/pull/3187), which fixes the
  in-session window by all-reducing measured times before the argmin.
- **No shared filesystem**: a small opaque boundary — `export()` produces bytes the framework
  broadcasts without parsing, `install()` verifies the environment fingerprint and publishes
  locally. Not yet implemented.

## 3. Why v2 is a separate entry point

The reason commonly given — "the on-disk format may be different" — is **not** load-bearing.
Autotune caches are already per-version disposable (§2.3): `_collect_metadata()` stamps
`flashinfer_version` and a mismatch is hard-rejected, so a v1 file is dead on the first patch bump
regardless of what v2 does. Neither format has to read the other's data, ever.

Two things do argue for a separate name, and both are migration-window problems:

1. **Call-site signature.** v1's `cache=<file path>` is identity-bearing; v2's `cache_root` is a
   placement-only *directory*. An in-place swap silently redefines an argument that downstream
   code already passes — `cache="my_configs.json"` would become "use that filename as the root
   directory". vLLM and SGLang both call this today.
2. **Lifetime.** v1 scopes tuning to the `with` block; v2 attaches the store for the process.
   Changing that under existing `with autotune(...)` call sites changes what happens *after* the
   block exits — the subtle-behavior-change class that a distinct symbol avoids.

Neither justifies permanent coexistence, which is what section 5 exists to prevent.

## 4. Graduation plan

**The problem this section addresses**: `autotune_v2` is a version number in a public symbol
name. If graduation is left implicit, the number becomes permanent API surface and the next
iteration is structurally forced to be `autotune_v3`. That outcome arrives by inertia unless the
end state is written down *before* downstream code adopts the name — and the vLLM migration
draft posted on #3861 asks vLLM to adopt it by name now.

### 4.1 End state

**`autotune_v2` is a transitional name.** At graduation:

1. `autotune()` becomes the v2 implementation.
2. `autotune_v2` becomes a deprecated alias of `autotune()`, emitting a `DeprecationWarning`.
3. `autotune(cache=<path>)`, `save_configs(path)`, and `load_configs(path)` are retained as thin
   shims forwarding to the managed store, with `cache=<path>` honored as *placement only*.
4. The alias and the v1 shims are removed no earlier than the next major version (§4.3).

The version number never survives into the stable API. Step 3 is what makes this cheap: it
collapses the two-autotuner surface immediately, without waiting on a removal window, and no
framework call site breaks at the moment of graduation.

### 4.2 Gates

"Deprecate v1 afterwards" currently hides at least four preconditions. Graduation requires all
of:

| # | gate | why it blocks |
|---|---|---|
| 1 | vLLM and SGLang migrated **and released** | the migration is the point; an unreleased patch proves nothing |
| 2 | `validate_tactic` adopted by ≥1 runner (cuDNN, via #3707) | otherwise graduation ships a contract that is "guardable, not yet guarded" (§2.6 rule 4) |
| 3 | `execution_mode` default resolved | if `"auto"` → `"cuda_graph"` flips *after* graduation, that is a second silent tactic-selection change under the final name |
| 4 | accuracy harness: regret ≤ v1 on ≥2 architectures | production B200 data exists; SM120 has none yet |

Gates 2 and 3 are the substantive ones: both would otherwise land as behavior changes *after*
users have migrated, which is exactly the cost the separate entry point was meant to avoid.

### 4.3 Version policy

Under the right-shifted scheme in `CLAUDE.md`, **removing** `autotune(cache=<path>)` /
`save_configs` / `load_configs` is an incompatible API change and requires a **major** bump — as
does eventually dropping the `autotune_v2` alias. So:

- Graduation itself (steps 1–3 of §4.1) is backwards-compatible and can land in a **minor**.
- Removal (step 4) is deferred to the next major, whether or not that is stated. Given that the
  shims are a few lines, keeping them indefinitely is a legitimate end state.

Nothing about graduation deletes user cache files: old environment directories are left on disk
and simply stop being consulted.

### 4.4 Documentation debt

`docs/autotuning.rst` (419 lines) currently documents `autotune(cache=path)` / `save_configs` /
`load_configs` as *the* public API and does not mention v2. #3861 does not touch it. Until it
does, the shipped documentation describes v1 only and nothing in-tree forces the reckoning.
Graduation must update that file in the same change that swaps the implementation.

## 5. Alternatives considered

**New parameters on `autotune()` instead of a new symbol.** Rejected for the two reasons in §3:
the `cache=` argument would change meaning under existing callers, and attach-vs-scope would
change behavior after the `with` block exits. A distinct symbol makes the eventual convergence a
rename rather than a silent semantic drift — provided §4 is committed to.

**Context scope as the default.** Rejected: existing integrations serve bare
Python calls after warmup. Process attachment remains the default; explicit
context scope is available for integrations that select the correct identity
during every uncaptured lookup, including capture-time forwards.

**Tune on first call (Triton-style).** Rejected for serving: latency jitter in the engine loop,
collective deadlocks when ranks tune at different times, must-precede-graph-capture ordering, and
profiling OOM risk.

**Cross-file locks / single-flight for tuning.** Rejected: ranks tune inside collectives, so a
cross-rank lock either serializes warmup or deadlocks it. Redundant measurement is the cheaper
failure mode.

**Mixed execution policies.** Eager and graph execution can share a bucket.
Frameworks should select policy by actual capture coverage and fallback routing,
not by prefill/decode labels or an M threshold alone. Explicit context scopes
allow managed graph winners alongside existing eager serving. Frameworks still
own region routing and must track warmup by identity/region, not one global
"warmed up" flag.

## 6. Limitations and future work

- **Opaque `export()` / `install()`** for multi-node distribution without a shared filesystem —
  designed, not implemented.
- **cuDNN plan sidecar** (after #3707): serialize the winning plan next to the entry to skip
  plan-list enumeration on load.
- **Per-runner `validate_tactic` adoption** — the hook exists; runners opt in separately.
- **Policy-level aggregation knob** (rounds + reducer, interleaved min-of-medians): sub-0.05 ms
  kernels go bimodal under co-tenancy and single-shot medians flip winners run to run, so stable
  ordering on noisy hosts should be a production-tuning property, not just an offline-harness one.
- **No GC, size limits, or pruning.** Old environment directories accumulate; `rm -rf` is the
  supported cleanup.
- **Representative probe construction** for MoE (skewed routing / EP-deflated shapes) is a
  separate op-level track; balanced probes cannot exercise the regression class in #3622.
- **Built-in collective orchestration** is not provided. A custom tuning protocol must
  coordinate candidates and timing aggregation, agree on results, and include any required
  topology information in its cache keys.
