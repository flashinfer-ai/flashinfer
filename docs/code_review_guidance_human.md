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

### PR defendability
A PR should be defendable by its human author — **to some extent**. Nowadays it is not
necessarily line-by-line defendable, and that is accepted. But watch for design that the
author cannot explain.

A concrete flag: **coding style that deviates from surrounding code** without reason —
flag it and discuss rather than silently accepting.

**If the smell test fails:** ask the author to walk through the rationale — green CI alone
does not settle it. Our principle (the same one
[Dynamo](https://docs.nvidia.com/dynamo/getting-started/contribution-guide#contribution-workflow)
applies): we encourage AI-assisted contributions, and we ask that authors understand their
changes well enough to explain them when asked — a change its author cannot explain is not
ready to merge yet. We state this as a principle rather than a hard policy: the goal is
review quality, not discouraging contributions. This principle will also be added to the
contribution guide.

Where defendability matters more:
- **Perf / kernel-selection logic, high-level interfaces, widely-used operations.** Design
  rationale drift in these *non-disposable* areas is expensive because it propagates.
- **A lot of this risk is mitigated by good tests.**
- **Disposable code areas** — model-specific ops (DSv3, MSA) that matter today but likely lose
  relevance within months — warrant a lighter touch on long-term design purity.

## Non-focus

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

<!-- TODO: to be filled in by the owning team member — declaration workflow (tracked issue),
review quality bar for experimental code, and graduation / removal criteria. -->

## How to leverage agent review

**Design principle → design doc as markdown → code owner / agent enforce.** When a change
embodies a design decision for durable code, capture the principle as a markdown design doc
(see [design_docs/](design_docs/)) so reviewers — human or agent — can enforce against a written
rationale instead of re-deriving it each PR.

## Checklist

- [ ] Crash/OOB/overflow/allocation defects
- [ ] API shape, naming, convention consistency; `include/` stays Torch-free
- [ ] Tests cover new behavior/edge cases; refcheck for numerics
- [ ] Comments concise/high-SNR; docs in sync
- [ ] Author can explain the design; style deviations from surrounding code flagged
- [ ] Kernel internals **not** line-audited — instead, tests/benchmarks/fuzzing confirmed as
      the correctness backstop
