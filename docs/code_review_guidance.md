# Code Review Guidance (Agents)

Rule set for agent reviewers of FlashInfer changes. (Humans: see
[`code_review_guidance_human.md`](code_review_guidance_human.md) for the human-oriented
version with rationale.)

Review is a **speed-vs-accuracy** problem: lots of plausible-looking, unverified diff. Spend
attention where a mistake is most expensive and least likely to be caught elsewhere.

**Key divergence from human review: kernel implementation details are IN SCOPE.** Humans
deprioritize them due to attention span; agents do not have that constraint and catch real
bugs there. See [Kernel review](#kernel-review).

## Focus areas

1. **Crash-prone code.** OOB indexing, unchecked pointer/tensor math, int32/int64 overflow in
   offset/stride/size math, unvalidated shapes/dtypes/device assumptions, under-allocated
   workspace/buffers on large problems, missing synchronization.
2. **Interface & API design.** Interfaces get replicated — review with "how will this be
   copied?" in mind. Check convention adherence (argument order, plan/run split, wrapper
   patterns, `@flashinfer_api` / `@backend_requirement`), naming, and extensibility. Framework
   separation: no Torch headers under `include/`.
3. **Testing surface.** New behavior and edge cases covered by unit tests; numerical refcheck
   present; correct architecture guards; and the code is actually testable.
4. **Comments & docs.** Flag low-SNR / verbose comments — want concise, explain *why* not
   *what*. Short rationale for a non-obvious hot-path choice is high-SNR and wanted. Keep
   `CLAUDE.md` / `.claude/skills/` docs in sync when touched.
5. **Design deviations.** Flag (don't silently accept or rewrite) coding style that deviates
   from surrounding code. Scale scrutiny by durability: **hard** on perf/kernel-selection
   logic, high-level interfaces, and widely-used ops; **lighter** on disposable model-specific
   ops (DSv3, MSA). When unsure, treat as durable.
6. **PR description hygiene.** The description must keep the repo's default PR template
   (`.github/pull_request_template.md`), filled in — flag descriptions that overwrite it with
   a custom or tool-generated format. A **performance optimization** must report observed
   before/after numbers in the description (reproducible benchmark, GPU, problem sizes) —
   flag perf claims without numbers. The PR title/description normally become the commit
   title/message on (squash) merge and are relied on when bisecting — hold them to that
   standard.

**Out of scope:** backwards-compatibility / API-breakage auditing — delegated to a GitHub
pre-merge check that QA puts in and maintains, not code review.

## Kernel review

Read kernel logic and report genuine bugs — indexing/stride math, boundary/predication,
accumulation/dtype/scaling errors, sync/barrier placement, masking, tile/loop off-by-ones,
unsafe shape/alignment assumptions.

- **Calibrate confidence.** Distinguish "this is a bug, here's the breaking input" from "this
  looks unusual, confirm intent." Don't rewrite an intentional optimization for looking
  non-idiomatic — flag and ask.
- **Tests complement, not replace.** A green suite is not a reason to skip reading the kernel;
  unverifiable-by-static-reading code is where tests/benchmarks/fuzzing are the backstop.

## Effort calibration

- **Low / medium:** fewer, high-confidence findings — crashes, correctness, interface breaks.
- **High / max:** broader coverage, uncertain-but-worth-flagging findings, closer kernel reading.

## Experimental APIs

FlashInfer does not gate PRs by size. Some PRs may instead be submitted on **experimental**
terms — a separate lifecycle, workflow management, and quality bar — declared via a tracked
issue. Review such PRs against the experimental quality bar, not the durable-code bar.

<!-- TODO(@bkryu): declaration workflow (tracked issue), review quality bar for experimental
code, and graduation / removal criteria. -->

## Checklist

- [ ] Crash/OOB/overflow/allocation defects
- [ ] API shape, naming, convention consistency; `include/` stays Torch-free
- [ ] Tests cover new behavior/edge cases; refcheck for numerics
- [ ] Comments concise/high-SNR; docs in sync
- [ ] Style deviations flagged (esp. durable/high-leverage code)
- [ ] Default PR template kept (not overwritten); perf-optimization PRs report observed
      before/after numbers in the description
- [ ] Kernel logic read for real bugs, findings labeled by confidence
