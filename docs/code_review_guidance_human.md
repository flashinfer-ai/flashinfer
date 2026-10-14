# Code Review Guidance (Humans)

This is the human-reviewer version of FlashInfer's code review guidance. The agent-reviewer
rule set lives in [`code_review_guidance.md`](code_review_guidance.md) — agents do **not**
read this file, so it can carry human-specific reality (limited attention, where to spend it)
without confusing them. The two docs share the same focus areas; they differ only on kernel
internals (see [Non-focus](#non-focus)).

## Why this exists

Code is generated much faster than before, which makes review a **speed-vs-accuracy** problem:
more diff per unit time, much of it plausible-looking but unverified. Spend your limited
attention where a mistake is most expensive and least likely to be caught elsewhere.

## Focus

### Crash-prone coding style
OOB indexing, unchecked pointer/tensor math, integer overflow in offset/stride/size math
(int32 vs int64), unvalidated shapes/dtypes/device assumptions, under-allocated
workspace/buffers on large problem sizes, missing synchronization.

### Interface (API, internal OOP design)
Interfaces get **replicated** — new code copies the shape of existing code, so review with
"how will this be copied?" in mind. Check argument order, plan/run split, wrapper patterns,
and especially **naming-convention adherence**. Framework separation: no Torch headers under
`include/`.

### Testing surface
Tests guard software quality and tighten the verification loop. Does the change add/extend
unit tests for the new behavior and edge cases? Are numerical references present
(`--refcheck`)? Are architecture guards correct? Is the code testable at all?

### Documentation / doc-strings
Agent-written comments are often **low signal-to-noise and too verbose**. We want concise,
to-the-point comments that explain *why*, not restate *what*. High-SNR exceptions worth
keeping: short rationale for a non-obvious algorithmic choice on a hot path.

### PR description hygiene
The PR description should keep the repository's default PR template
(`.github/pull_request_template.md`), filled in — not overwritten with a custom or
tool-generated format.

For a **performance optimization**, the PR description must report the observed performance
improvement: before/after numbers from a reproducible benchmark (e.g.
`benchmarks/flashinfer_benchmark.py`), with the GPU and problem sizes used. A perf claim
without numbers is not reviewable — ask for them.

Note that the PR title and description normally become the commit title and message on
(squash) merge. Humans and agents both rely on them when bisecting changes to identify
owners and possible bugs — one more reason to keep both accurate.

### PR defendability
A PR should be defendable by its human author — **to some extent**. Nowadays it is not
necessarily line-by-line defendable, and that is accepted. But watch for design that the
author cannot explain.

The principle: we may raise questions about the design especially if the code touches a relatively
durable area of the library. If the author, upon being asked, cannot walk through the rationale,
we may reject the PR submission. We support AI-assisted contributions, but we expect the authors
to understand the idea or rationale of their changes. This principle is also stated in
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

Possibly flag: **coding style that deviates from surrounding code** without reason —
flag it and discuss rather than silently accepting.

Where defendability matters more:
- **Perf / kernel-selection logic, high-level interfaces, widely-used operations.** Design
  rationale drift in these *non-disposable* areas is expensive because it propagates.
- **A lot of this risk is mitigated by good tests.**
- **Disposable code areas** — model-specific ops (DSv3, MSA) that matter today but likely lose
  relevance within months — warrant a lighter touch on long-term design purity.

## Non-focus

### Backwards compatibility (API breakage)
Delegated to a GitHub pre-merge check that QA puts in and maintains, rather than code
review, so reviewers do not need to audit for API breakage themselves.

### Kernel implementation details
Deprioritized **for human reviewers**, because of limited attention span and a genuinely
different programming model:

- Kernel programming uses many corner-cutting techniques.
- Compiler / template abstractions hide semantics.
- The computation is complex (MLA, etc.) before any optimization.

Instead, rely on **passing unit tests and benchmarks + fuzz testing** as the backstop for
kernel correctness.

> Note: this is the one place the agent guidance diverges. Agents are not attention-limited,
> so for them kernel internals are **in scope** — they read the logic and report bugs (with
> confidence labels). See [`code_review_guidance.md`](code_review_guidance.md).

## Experimental APIs

FlashInfer does not gate PRs by size. Instead, an **experimental** track is being introduced:
some PRs may be submitted on experimental terms — a separate lifecycle, workflow management,
and quality bar — declared via a tracked issue.

<!-- TODO(@bkryu): declaration workflow (tracked issue), review quality bar for experimental
code, and graduation / removal criteria. -->

## How to leverage agent review

**Design principle → design doc as markdown → code owner / agent enforce.** When a change
embodies a design decision for durable code, capture the principle as a markdown design doc
(see [design_docs/](design_docs/)) so reviewers — human or agent — can enforce against a written
rationale instead of re-deriving it each PR.

## Checklist

- [ ] Crash/OOB/overflow/allocation defects
- [ ] API shape, naming, convention consistency; `include/`/`import` stays Torch-free library-side
- [ ] Tests cover new behavior/edge cases; refcheck for numerics
- [ ] Comments concise, to the point; docs in sync
- [ ] Author can explain the design; style deviations from surrounding code flagged
- [ ] Default PR template kept (not overwritten); perf-optimization PRs report observed
      before/after numbers in the description
- [ ] Kernel internals **not** line-audited — instead, tests/benchmarks/fuzzing confirmed as
      the correctness backstop
