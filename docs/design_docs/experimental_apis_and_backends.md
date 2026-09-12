# Experimental APIs and Backends in FlashInfer

This document records the motivation and design decision; the normative policy is [`flashinfer/experimental/README.md`](../../flashinfer/experimental/README.md).*
How an experimental PR is tested and lands is covered in [`experimental_track_ci.md`](experimental_track_ci.md); the general CI triggering design it builds on is [`ci_bot_and_targeted_testing.md`](ci_bot_and_targeted_testing.md).

## Overview

FlashInfer currently applies the same review and support expectations to most user-facing functionality.

This works well for functionality intended for long-term support, especially on datacenter architectures where stability is critical. It works less well for fast-moving client-GPU support, where users often value functional availability over API or implementation stability.

The initial proposal was motivated by expedited SM12x kernel delivery, but the problem is broader. FlashInfer needs to distinguish and accommodate:

* an **experimental API**, whose interface may change or disappear (e.g. new op from a new model); and

* an **experimental backend**, whose implementation is not yet ready for stable support (e.g. Kernel Factory-generated kernel).

These are separate concerns. A stable API may expose an experimental backend through explicit opt-in, while an experimental API must be clearly identified and gated.

This document records the selected placement and identification design for experimental APIs and backends, and the rules that govern them.

Throughout, experimental behavior must be explicitly enabled, tested for its intended use case, and governed by ownership and lifecycle rules. In this document, **core** refers to the existing non-experimental codebase.

## Design: Tagged experimental APIs in core; experimental backends in `flashinfer.experimental`

API stability and backend stability are separate decisions.

|  | Stable Backend | Experimental Backend |
| ----- | :---: | :---: |
| Stable API | Normal Path | Allowed with Opt in |
| Experimental API | Not a target use case | Allowed with Opt in |

Experimental APIs live in core and are marked with `@flashinfer_experimental_api`.

Experimental backends and backend-specific logic live under the `flashinfer.experimental` directory.

Core contains only a thin entry point:

* public API signature;
* general validation;
* feature-gate check;
* minimal handoff.

Support checks, heuristics within the experimental backend, routing, compilation, caching, and kernels remain under `flashinfer.experimental`.

This supports both:

* a tagged experimental API backed by an experimental backend; and
* a stable API exposing an experimental backend once the opt-in is set.

This design preserves a unified API surface while strongly isolating fast-moving implementation code.

### **Feature gating**

Opting in to experimental functionality is explicit and needs no environment variable:

* calling an API marked `@flashinfer_experimental_api`; or
* naming an experimental backend in a stable API (`backend="<name>"`).

Both emit an `ExperimentalWarning` once. What is gated is **automatic selection**: `backend="auto"` — the dispatch heuristics and autotuning behind it — may pick an experimental backend only with

`FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`

**Without** this opt-in, automatic selection considers only stable backends; a stable API never routes to an experimental backend on its own.

**With** it, experimental backends join the automatic candidate list alongside stable ones, for dispatch and autotuning. Trace-apply is outside this gate: experimental backends are not required to support trace, and a trace-apply solution is the deployer's explicit choice.

Mechanically, an experimental backend's `@backend_requirement` checker is marked `@experimental_backend`. The decorator filters marked backends out of `backend="auto"` unless the variable is set, warns once per API/backend pair when one is selected, and rejects at import time a checker defined under `flashinfer.experimental` that lacks the marker. Stable APIs that route `"auto"` by hand call `require_experimental_auto_backends(...)` on the automatic branch. See `flashinfer/experimental/README.md` for a worked example.

## Common Rules for Experimental APIs and Backends

The following rules apply to all experimental APIs and backends.

### **1\. Ownership and lifecycle**

Every experimental feature must have:

* a named owner;
* a tracking issue that
  * documents the feature's use case;
  * states the reason for entering the experimental path (see the justification list in §2);
  * states a graduation plan and target release for graduation.
    * e.g. we expose this kernel first and the goal is to finalize the API and graduate by 0.6.xx release in four weeks.

Default intent for any experimental PR should be graduation within four weeks (see §7). Experimental features are periodically reviewed and either continue incubating, graduate through the normal stable process, or are removed.

Continued incubation should not be automatic. At each lifecycle review, the owner must document why the feature is still experimental, provide evidence of active use or progress, and identify the remaining graduation blockers. A maintainer must approve the extension. Repeating the same justification without meaningful progress is not sufficient.

Broken tests, loss of ownership, or lack of meaningful usage are sufficient reasons for removal without stable deprecation guarantees.

### **2\. Initial scope and admission**

The admitted use cases are the fast-moving work this policy exists for:

* client-GPU kernels and backends (e.g. SM12x);
* new operations from the latest models, where functional support can land ahead of a finalized stable API;
* highly specialized kernels for specific problem sizes, which enter as experimental backends behind an existing stable API rather than as new APIs.

A feature requires documented justification, such as:

* a new operator family or algorithm;
* a materially different API contract;
* a new compiler, backend, or architecture-specific implementation path;
* a use case FlashInfer has not committed to maintain.

A new experimental API should not be created for a parameter-space extension of an existing stable API. An experimental backend may implement an existing stable API when it introduces a materially new implementation path.

### **3\. Placement and integration**

Experimental backend logic lives under `flashinfer.experimental`.

Where core contains an experimental entry point, it may include only:

* API definition;
* shared validation;
* feature-gate check;
* backend selection;
* direct handoff.

Backend-specific support checks, routing, compilation, caching, and kernels remain outside core.

### **4\. API identification**

Public experimental APIs are marked with `@flashinfer_experimental_api`.

Stable APIs exposing experimental backends remain marked with `@flashinfer_api`.

The decorator sets `is_experimental = True` on the decorated function so that tooling can mechanically distinguish intentionally experimental public APIs from internal helpers and accidentally untagged APIs.

### **5\. Testing and review**

Every experimental feature must include:

* correctness tests against a reference;
* at least one representative supported configuration;
* validation on intended hardware;
* a runnable example.

Experimental tests run in a separate CI lane and must pass for PRs that modify the feature.

The PR declares which targets that lane runs, in a fenced `experimental-tests` block in the PR body (see `.github/pull_request_template.md`):

```
​```experimental-tests
tests/experimental/test_my_backend.py
​```
```

The declaration is required, and targets must live under `tests/experimental/`. Running the whole tree is permitted but gets slower and less relevant to any one change as the track grows, so authors should declare the narrowest scope that covers the change. `scripts/pr_checks/experimental_test_scope.py` parses and validates the block and emits a `TEST_PATH` (`--test-path`), the value both CI systems already accept — GitHub via the `run-ci` label or `@flashinfer-bot run`, GitLab via `/bot run TEST_PATH` — so neither lane reimplements the parsing, and a reviewer triggering a run by hand can paste the same value. Narrowing the scope reduces what runs within each GPU/toolkit matrix cell; it does not change the matrix itself.

Both CI systems take the scope as an argument: GitLab as `/bot run TEST_PATH`, GitHub as `@flashinfer-bot run <paths>`. A parameterised GitHub run retargets the GPU lanes at those paths; `@flashinfer-bot run` with no arguments is unchanged.

There is no arch selection, and none is needed: tests carry their own arch guards and skip themselves where unsupported, so a targeted run still spans the GPU matrix and the right subset executes on each runner. Narrowing the scope reduces what runs *within* each cell rather than which cells run — and no path-to-arch mapping has to be kept in sync with the tests.

**CI runs what it is told, and does not read the declaration.** Turning a declared block into paths happens once, in whoever issues the trigger — a reviewer, or the screening watcher posting under its own identity with the declared scope. There is deliberately no second parser in the workflow: a reviewer who wants to run something else just says so, and the declaration stays the default rather than becoming a constraint.

Because a `run-ci` label carries no payload, `ci-bot-commands.yml` — the one place that already parses this comment — publishes the requested paths as `ci/test-scope-N` commit statuses on the head SHA, and `pr-test.yml` reassembles them. A status description caps at 140 characters — only two or three real test paths (median 41, p90 60) — so the scope is chunked on target boundaries and rebuilt in index order. Indices start at 1 even for a single chunk, so there is one code path rather than a special case; when a scope shrinks, leftover chunks are updated to empty — there is no DELETE for statuses, but re-posting a context supersedes it — since they would otherwise leave stale trailing paths. Both sides read the *combined* status endpoint, which returns one entry per context already resolved to the latest, so nothing depends on the raw list's ordering. Nothing downstream re-reads comments, and keying on the SHA makes staleness structural: a new push is a new SHA with no scope status, so an old scope cannot be resurrected by re-labelling.

The convention is deliberately the same on both systems — `@flashinfer-bot run <paths>` and `/bot run TEST_PATH` — with the status being an implementation detail of the GitHub side. That symmetry also means automation gets no privileged channel: a watcher triggering a screened PR types exactly what a reviewer would, so there is one path to test and no way for the two to drift.

Requested targets are validated with the same rules as a declared scope — strict charset, must exist — but against a wider root (`tests/` rather than `tests/experimental/`), since a reviewer may legitimately want to run anything. Being authorised to trigger CI is not the same as a string being safe to hand to a shell, so the value is passed via `env`, never interpolated into script text, and read into an array with globbing disabled.

Changes under `flashinfer.experimental` receive narrower review focused on eligibility, correctness, containment, licensing, and obvious safety or maintainability risks.

Changes to core follow the normal core review process. Reviewers should verify that integration is explicit, stable behavior is unchanged by default, and backend-specific logic has not leaked into core.

Broad portability, comprehensive performance coverage, and long-term maintainability are not required for the experimental implementation at admission.

### **6\. User contract**

Experimental APIs and backends provide no compatibility or long-term support guarantees and may change or be removed without deprecation.

Documentation must identify:

* whether the API, backend, or both are experimental;
* supported use cases and limitations;
* required feature gates;
* whether the feature participates in automatic routing (dispatch, autotuning, trace-apply) once the gate is set.

Experimental features are intended primarily for main-branch users, community containers, and local framework integrations. They should not be enabled by default in supported framework releases.

A request for default or supported release-path integration is a signal that the feature should be considered for graduation.

### **7\. Miscellaneous**

* Experimental APIs and backends are JIT-only. AOT registration is prohibited.
* We set four weeks as the default lifecycle review cadence, but each experimental feature author can propose a planned lifecycle in their tracking issue with justification.


## Non-goals

This document does not:

- commit FlashInfer to supporting every new model, operator, or backend;

- guarantee that experimental features will graduate;

- provide compatibility or long-term maintenance guarantees;

- allow experimental behavior to activate by default;

- lower review or support expectations for stable core functionality.
