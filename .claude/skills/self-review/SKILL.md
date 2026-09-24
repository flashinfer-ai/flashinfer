---
name: self-review
description: Check your own diff against the repository's review guidance and PR rules before opening a PR
---

# Self-Review Before Opening a PR

Run this when a change is being prepared for a PR, after it is complete and before writing
the PR description. It walks the diff through the repository's published review guidance and
PR rules, so that common issues are caught before a human looks and review time goes to the
design. It is informal and optional: it exists to make review easier for reviewers and
contributors, not to gate anything, and changes made only to test something locally do not
need it. This skill does not restate the rules; it points at where they live and says how to
apply them to a diff.

## Step 1: Get the diff

```bash
git fetch upstream main 2>/dev/null || git fetch origin main
BASE=$(git merge-base HEAD upstream/main 2>/dev/null || git merge-base HEAD origin/main)
git status --short          # untracked files that belong to the change must be added
git diff --stat "$BASE"     # merge-base vs working tree: committed and uncommitted changes
git diff "$BASE"
```

Review the whole diff, not only the files you remember editing. Unrelated changes (submodule
pointer drift, scratch files) must be left out.

## Step 2: Apply the review guidance

Read [`docs/code_review_guidance.md`](../../../docs/code_review_guidance.md) and walk its
focus areas and checklist over the diff. In particular:

- **Kernel logic is in scope.** Read indexing/stride math, boundaries and predication,
  accumulation dtype and scaling, barrier placement, alignment assumptions. Do not rely on a
  green test run as a substitute for reading the code.
- **Interfaces get copied.** Check argument order, plan/run split, decorator use
  (`@flashinfer_api`, `@backend_requirement`), naming, and that `include/` stays Torch-free.
- **Tests.** New behavior and edge cases are covered, numerics have a reference check, and
  architecture guards are correct. Run the touched test files, not the whole suite:

  ```bash
  pre-commit run --files $(git diff --name-only --diff-filter=d "$BASE")  # skip deleted files
  pytest tests/<touched files>
  ```

- **Comments.** Concise, explain *why*, not *what*. Remove narration of the obvious; keep a
  short rationale for any non-obvious hot-path choice.
- **Style.** Match the surrounding code. Where the diff deliberately deviates, say why in the
  PR description rather than leaving it for the reviewer to discover.

## Step 3: Apply the PR rules

From [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) (Pull Request Guidelines):

- The description uses the default template in
  [`.github/pull_request_template.md`](../../../.github/pull_request_template.md), filled in
  and not replaced by a custom or tool-generated format. Title and description become the
  squash-merge commit and are what `git bisect` surfaces later.
- A performance change reports observed before/after numbers from a reproducible benchmark
  (`benchmarks/flashinfer_benchmark.py` or a named script), with the GPU and problem sizes.
  A speedup ratio without absolute numbers, or numbers without a GPU, is not enough. Use the
  `benchmark-kernel` skill to produce them.
- Documentation referenced by the change (`CLAUDE.md`, `.claude/skills/`, `docs/`) is
  updated in the same PR.
- **Backwards compatibility.** If the change removes or renames a public API, or changes a
  signature, default, or semantics, say so explicitly in the PR description.
- **Defendability.** For each non-obvious choice, the author can explain the rationale when
  asked. If a choice cannot be explained, it is not ready; either understand it or remove it.

## Step 4: Report, then fix

Produce a findings list labeled by confidence, in two groups:

1. **Fix before opening** — defects and rule violations (crash risk, missing tests, wrong
   guards, template not used, perf numbers missing, unrelated files in the diff).
2. **Mention in Reviewer Notes** — intentional deviations, known limitations, anything a
   reviewer would otherwise have to rediscover.

Fix group 1, re-run Step 1 and the checks in Step 2, then draft the PR description in the
template with group 2 under "Reviewer Notes". A self-review that finds nothing on a
non-trivial diff is a signal to look again, not a pass.
