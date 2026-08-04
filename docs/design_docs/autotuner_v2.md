# Autotuner v2 — Managed Persistence, Deployment-Matched Measurement, and the Runner Contract

**Scope**: `flashinfer/autotune_cache.py` (the whole module) and the `autotune_v2` state and hook
sites in `flashinfer/autotuner/autotuner.py` — the managed store, the measurement policy, and the
runner contract that tactic identities must satisfy.

**Not** in scope: the v1 path — `autotune()`, `save_configs()`, `load_configs()`, and the
profiling machinery they share. That code predates this document and has none of its own; changing
it does not oblige you to update this doc. §5 is where the two are planned to converge, and the
day v1 is folded into v2 this scope line should widen to match.

**Status**: proposed. RFC [#3920](https://github.com/flashinfer-ai/flashinfer/issues/3920); this
document ships with the v2 MVP in
[#3861](https://github.com/flashinfer-ai/flashinfer/pull/3861) (opt-in; `autotune()` and the v1
cache path are untouched). Sections 1–4 describe the design as implemented in this branch;
section 5 (graduation) is the part that is **not** yet agreed and is the reason this document
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
with flashinfer.autotune_v2(measure=MeasurementPolicy(execution_mode="cuda_graph")):
    model(dummy_inputs)      # warmup: tune misses, publish each winner atomically
model(inputs)                # serving: reuses entries, no context needed

# fresh process, same environment
with flashinfer.autotune_v2(mode="replay"):
    pass                     # hydrate only
model(inputs)
```

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

**Attach semantics.** `persistent_cache=True` attaches the store for the remainder of the
*process*; the context scopes only *when profiling cost may be paid*. This is forced by how the
consumers actually run: both vLLM and SGLang serve **outside** any context (serving replays a
captured CUDA graph — no Python runs at replay, and the autotuned ops are buried in layer
forwards), so a context-scoped store would silently regress serving to heuristics the moment
warmup exits. Consequently a process serves under one ambient policy (last attach wins), and
`autotune_v2` does **not** nest — a v2-in-v2 context has ambiguous store targeting and fails
fast. Nesting a plain `autotune()` inside or around it stays supported.

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
`cublas_version`, `cudnn_version`, `cudnn_frontend_version`, `gpu` — plus `cache_schema` and any
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
- **No locks.** Concurrent writers do redundant work and the last valid write wins. This is what
  makes SGLang's all-ranks-tune-simultaneously pattern safe with zero framework code — and it is
  a deliberate divergence from the JIT kernel cache (§4), where single-flight is correct.
- **Invalid is a miss, never an error.** A missing file, malformed JSON, or an embedded-key
  mismatch logs a warning and returns "not found". A corrupt entry costs one retune, not a dead
  server, and cannot take the other entries with it.
- **One filesystem probe per key per process.** Positive and negative lookups are memoized
  inside the store object — never in v1's `_file_configs` — so the serving hot path touches
  neither the filesystem nor the JSON decoder, and v1/v2 state mixing is impossible by
  construction.

### 2.5 MeasurementPolicy — tune the way you deploy

```python
MeasurementPolicy(execution_mode="auto" | "cuda_graph" | "eager", cold_l2=None)
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

- **Shared filesystem**: solved with zero framework code. Per-entry atomic publish makes
  all-ranks-tune-simultaneously safe; per-rank cache files and framework hash directories become
  unnecessary.
- **In-session rank consistency**: ranks may still hold divergent locally-measured winners until
  `autotune_v2_reload()` runs — a finalize step (tune → barrier → reload) that drops in-process
  winners so every rank re-reads the store's canonical entries and serves byte-identical tactics.
  Composes with [#3187](https://github.com/flashinfer-ai/flashinfer/pull/3187), which fixes the
  in-session window by all-reducing measured times before the argmin.
- **No shared filesystem**: a small opaque boundary — `export()` produces bytes the framework
  broadcasts without parsing, `install()` verifies the environment fingerprint and publishes
  locally. Not yet implemented.

## 3. Relationship to the CuTe-DSL kernel cache

[`cute_dsl_kernel_cache.md`](cute_dsl_kernel_cache.md) describes a disk cache with a strikingly
similar shape — environment record file, one artifact per specialization, atomic publish,
invalid-is-a-miss. The convergence is evidence the primitives are right. The two caches are
nonetheless **not** unifiable at the payload level, because they store different classes of
artifact:

| | CuTe-DSL kernel cache | autotune v2 store |
|---|---|---|
| payload | compiled `.o` (opaque binary) | chosen tactic (small JSON) |
| key origin | author-written specialization name, known statically | derived per call at runtime from live tensors |
| reproducible from key? | yes — recompile, seconds | no — requires GPU profiling; depends on silicon, clocks, library versions, measurement policy |
| validity axes | *compile* env: arch, `nvidia-cutlass-dsl` stack, source SHA | *performance* env: GPU, cuBLAS/cuDNN versions, measurement policy |
| concurrency | `FileLock` single-flight — double-compiling wastes minutes | no lock, last-valid-write-wins — a cross-rank lock would serialize or deadlock collective warmup |
| portability | must never leave its arch | the thing you specifically want to broadcast to peers |

The locking contracts are outright opposite, so a shared implementation is not available. What
*should* be shared is the mechanics layer, and today it is duplicated:

- **One word for the environment record.** `meta.json` (kernel cache) vs `manifest.json`
  (autotune store) name the same concept.
- **One helper for atomic-write + invalid-is-a-miss.** Both implement roughly the same 40 lines;
  those crash-safety semantics deserve to be tested once, in `flashinfer/jit/`.
- **One cache-clearing story.** Both live under `FLASHINFER_CACHE_DIR`, so `rm -rf
  ~/.cache/flashinfer/` covers both, but `clear_cache_dir()` semantics and the env-var surface
  (`FLASHINFER_AUTOTUNE_CACHE_DIR`, and the unrelated MLA-specific `FLASHINFER_AUTOTUNE_DIR`)
  should be documented as one story.

One lesson runs the other way: the kernel cache selects artifacts by a hand-written name string
(`_nvfp4_kernel_name`), with `meta.json` guarding arch/DSL-version/source-SHA but not per-kernel
codegen parameters. That is structurally the vllm#43119 failure class. It is currently mitigated
by a test; contract rule 5 above (synthesis-invariant keys backed by a debug-mode completeness
check) is the stronger form.

## 4. Why v2 is a separate entry point

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

## 5. Graduation plan

**The problem this section addresses**: `autotune_v2` is a version number in a public symbol
name. If graduation is left implicit, the number becomes permanent API surface and the next
iteration is structurally forced to be `autotune_v3`. That outcome arrives by inertia unless the
end state is written down *before* downstream code adopts the name — and
`framework_patches/vllm_autotune_v2.patch` in #3861 asks vLLM to adopt it by name now.

### 5.1 End state

**`autotune_v2` is a transitional name.** At graduation:

1. `autotune()` becomes the v2 implementation.
2. `autotune_v2` becomes a deprecated alias of `autotune()`, emitting a `DeprecationWarning`.
3. `autotune(cache=<path>)`, `save_configs(path)`, and `load_configs(path)` are retained as thin
   shims forwarding to the managed store, with `cache=<path>` honored as *placement only*.
4. The alias and the v1 shims are removed no earlier than the next major version (§5.3).

The version number never survives into the stable API. Step 3 is what makes this cheap: it
collapses the two-autotuner surface immediately, without waiting on a removal window, and no
framework call site breaks at the moment of graduation.

### 5.2 Gates

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

### 5.3 Version policy

Under the right-shifted scheme in `CLAUDE.md`, **removing** `autotune(cache=<path>)` /
`save_configs` / `load_configs` is an incompatible API change and requires a **major** bump — as
does eventually dropping the `autotune_v2` alias. So:

- Graduation itself (steps 1–3 of §5.1) is backwards-compatible and can land in a **minor**.
- Removal (step 4) is deferred to the next major, whether or not that is stated. Given that the
  shims are a few lines, keeping them indefinitely is a legitimate end state.

Nothing about graduation deletes user cache files: old environment directories are left on disk
and simply stop being consulted.

### 5.4 Documentation debt

`docs/autotuning.rst` (419 lines) currently documents `autotune(cache=path)` / `save_configs` /
`load_configs` as *the* public API and does not mention v2. #3861 does not touch it. Until it
does, the shipped documentation describes v1 only and nothing in-tree forces the reckoning.
Graduation must update that file in the same change that swaps the implementation.

## 6. Alternatives considered

**New parameters on `autotune()` instead of a new symbol.** Rejected for the two reasons in §4:
the `cache=` argument would change meaning under existing callers, and attach-vs-scope would
change behavior after the `with` block exits. A distinct symbol makes the eventual convergence a
rename rather than a silent semantic drift — provided §5 is committed to.

**Context-scoped store instead of process attach.** Rejected: both consumers serve outside any
context, so the store would detach at the end of warmup and serving would silently fall back to
heuristics. The `with` belongs to tuning; serving is unavoidably context-free.

**Tune on first call (Triton-style).** Rejected for serving: latency jitter in the engine loop,
collective deadlocks when ranks tune at different times, must-precede-graph-capture ordering, and
profiling OOM risk.

**Cross-file locks / single-flight for tuning.** Rejected: ranks tune inside collectives, so a
cross-rank lock either serializes warmup or deadlocks it. Redundant measurement is the cheaper
failure mode.

**Per-bucket or per-region measurement policy.** Explicit non-goal. Prefill (large M) and decode
(small M) land in disjoint shape buckets, so each bucket is served in one consistent mode; a
single ambient policy matching the mode-sensitive path is sufficient.

## 7. Limitations and future work

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
- **Comm/collective op tuning** is out of scope: it needs lockstep candidate enumeration, a
  MAX-reduction, one group-level decision, and topology fields in the manifest.
