# API gate: same-head validation

On 2026-11-26, fork experiments did **not** demonstrate independent merge
authorization for two PRs sharing one head commit. The requested acceptance
criterion remains unmet: approving A must allow A to merge while B stays blocked.
The current gate retains its conservative shared-head rejection and its
[documented race limitation](pr_api_compatibility.md#trusted-execution-and-gate-states).

## Native Actions checks

[PR A](https://github.com/kangbintNV/flashinfer/pull/244) and
[PR B](https://github.com/kangbintNV/flashinfer/pull/246) shared head
`a2545591c210b4e078bbcd47ebacc7e707987dad` and base
`d627c895f58fe6512be16f670cd5151c3a696611`.
An isolated base branch required `API compatibility native probe` from GitHub
Actions (app ID 15368), with enforcement for administrators and no strict
up-to-date requirement.

| Probe | Check result | Actual merge request |
| --- | --- | --- |
| [A workflow](https://github.com/kangbintNV/flashinfer/actions/runs/45813490838) | Success | HTTP 405: required check failing |
| [B workflow](https://github.com/kangbintNV/flashinfer/actions/runs/45813501743) | Failure | HTTP 405: required check failing |

These were fixed A/B probe outcomes, not maintainer `expected`/`revoke`
interactions. The check-run API associated each run with its own PR, but both
check runs used the shared head SHA and appeared in both PRs' check rollups.
Actual merge attempts rejected both PRs; a successful A check did not establish
independent authorization. Neither PR was merged.

## Commit statuses on test-merge commits

[PR A](https://github.com/kangbintNV/flashinfer/pull/226) and
[PR B](https://github.com/kangbintNV/flashinfer/pull/227) shared head
`a387d8f141ce81ef8e42ae623ea51426e9fa6326`, with different test-merge commits.
The probes wrote success on A's test merge and failure on B's test merge.

| Required status placement | A merge request | B merge request |
| --- | --- | --- |
| Same context on head and test merges; head failed | HTTP 405: required check failing | HTTP 405: required check failing |
| Fresh context only on test merges | HTTP 405: required check expected | HTTP 405: required check expected |

These simulated outcomes also failed the acceptance criterion. Neither PR was
merged. Moving the status destination alone was insufficient in this setup.

## Required-workflow policy feasibility

An organization required-workflow policy is a candidate for further testing,
not a verified fix. Attempts to create a disabled repository ruleset targeting
only the isolated fork base failed with HTTP 422:

```text
Invalid rule 'workflows': Invalid parameter workflows:
Workflow error at index 0:
```

The workflow source was registered on the fork's default branch. Requests using
`refs/heads/main`, `main`, and an immutable source SHA all failed validation.
GitHub's [required-workflow documentation](https://docs.github.com/en/enterprise-cloud@latest/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets#require-workflows-to-pass-before-merging)
describes organization/enterprise rules. This personal-fork experiment could
not validate that policy; no upstream organization rules were changed.

Before replacing the current gate, validate the policy in a supported test
environment, integrate the trusted scanner and PR-scoped exceptions, and repeat
the real maintainer approval and merge-authorization test. The existing scan
workflow's success alone is insufficient: scan completion and the final
`API compatibility` decision are separate outcomes in the current implementation.

Temporary default-branch probe workflow files were removed after the experiment.
Probe PRs and run links remain available for inspection. This report was prepared
with AI assistance; it does not claim a completed same-head isolation fix.
