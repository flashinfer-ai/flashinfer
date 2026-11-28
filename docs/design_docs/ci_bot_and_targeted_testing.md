# CI Triggering and Targeted Testing

This document records the design of FlashInfer's CI triggering: how a run starts, how a run
can be narrowed to specific tests, and the constraints that follow. The user-facing command
reference lives in [`CONTRIBUTING.md`](../../CONTRIBUTING.md); this document records *why*
the mechanism is shaped the way it is.

It is infrastructure, not feature-specific. Other documents are expected to refer to it
rather than restate it.

## Two CI systems

FlashInfer's public CI is bifurcated:

| | trigger | tests |
|---|---|---|
| GitHub Actions | a `run` comment addressed to the CI bot | JIT tests |
| GitLab | `/bot run TEST_PATH` | the same JIT tests |

**The split is an infrastructure artifact, not a design choice.** The two systems run the
same tests on different GPUs. Automation may drive either or both, and a caller should not
have to care which.

Historically only GitLab accepted a test path. The GitHub command took no arguments, so a
GitHub run was always the full matrix.

Both now take a scope, and the policy is that **a scoped run is the default**:
`@flashinfer-bot run tests/<dir-or-file> [tests/...]` / `/bot run tests/<dir-or-file> [tests/...]`.
The full matrix is the exception — a final pre-merge check of a cross-cutting change — because
an unscoped run is the most expensive thing a PR comment can do.

## Runner workflow identity

`pr-test-dispatch.yml` owns PR, scheduled and manual triggers, authorization,
scope selection, and result reporting. Its authorized test job calls
`flashinfer-ai/flashinfer/.github/workflows/pr-test.yml@main`. The existing
runner-group entry remains unchanged:

```text
flashinfer-ai/flashinfer/.github/workflows/pr-test.yml@refs/heads/main
```

`pr-test.yml` exposes only `workflow_call` and directly defines the self-hosted
test matrix and provider-spec lint. The job definitions come from `main`; the
checkout SHA, optional merge snapshot, Docker tag, and matrices come from the
caller's setup job. The test matrix retains PR fail-fast and scheduled-run
coverage. The caller passes only the two sccache secrets; the reusable jobs
retain their existing token permissions. The reusable workflow independently
authorizes the GitHub-supplied triggering event on a GitHub-hosted runner before
either self-hosted job can start. It accepts only `run-ci`/`run-ci-focused` label events on PRs targeting this
repository's `main`, or scheduled/manual events on this repository's `main`.
Caller inputs cannot grant admission. For `run-ci-focused`, the trusted check
requires both snapshot fields, binds checkout to the event's PR head, reads the
published commit statuses independently, and verifies the base, tree, and scope
digest. Runner jobs consume its verified outputs. Ordinary `run-ci` and main
scheduled/manual runs retain their no-snapshot path. This check is necessary
because the PR-editable dispatcher cannot enforce the runner boundary.

Runner cleanup removes only Docker containers with a mount source exactly
matching the current checkout. It applies the same ownership check on every
host because callers can change both the lane kind and requested runner labels;
cleanup must not stop neighboring jobs on shared GPU hosts.

Keep the remote `@main` reference: a local `./.github/workflows/pr-test.yml`
reference resolves to the caller's commit and would lose the allowed workflow
identity on pull requests. Deployment requires the reusable `workflow_call`
interface to exist on `main` before the new dispatcher is enabled. The repository
requires `Test Results Summary`, so this PR cannot validate its pinned call
against the old `main`. Land a compatible interface first while preserving the
existing entry workflow, then deploy the dispatcher, or obtain an authorized
merge exception for the cutover. Do not weaken the runner allowlist or branch
protection to make the call validate. No runner allowlist change is needed.
Manual dispatches and workflow-specific API queries now use `pr-test-dispatch.yml`; the workflow name
and required `Test Results Summary` job name remain unchanged. API consumers of
individual lane names must account for the reusable call's `Self-hosted checks / `
prefix. GPU execution and runner admission must be verified after deployment.

PR result lookups consult both dispatcher and legacy workflow history for the
same head SHA, but reuse requires an artifact recording the same head, base,
and merge tree as the event merge, with the base still at the target branch tip.
The trusted authorization job records this source for each run attempt, for both
full and focused runs. Partial reruns inherit the latest successful authorization
record from the same run when that dependency is not rerun. Legacy runs without
that record, expired records (14 days), and stale event merges require a fresh
CI request. Active runs without a record
are followed until authorization publishes one or the run finishes. Only older
runs are followed, preventing summary jobs from waiting on each other. History
refreshes every five minutes, caches source records per attempt, and backs off
on API errors; an unreadable history remains unknown and cannot fail the check
on its own (the GitHub job timeout still applies). Offline scope replay recognizes
the split routing marker and requires dispatcher evidence only for revisions using the split layout.
Avoid rerunning legacy scheduled runs after cutover: their original issue-report
code cannot see the new workflow history and may reopen the shared tracking
issue. If a legacy scheduled run is rerun, verify the tracking issue afterward.

## The command vocabulary

The bot answers a comment in one of four ways, and this vocabulary is load-bearing —
automation depends on being able to tell the outcomes apart:

| outcome | response | effect |
|---|---|---|
| accepted run | react `+1` | CI label applied, run starts |
| advisory | react `+1` and post AI rationale | no status or label changes, no run |
| rejected | react `confused` | nothing; no label, no run |
| unrecognized | nothing | nothing |

A malformed *argument* is a malformed *command*, and gets the rejection response. It is not
downgraded into a different, still-successful command — see "Reject, do not degrade" below.

## What actually starts CI

It is the *application* of the CI label that starts a run, not the label being present. The
handler therefore removes the label before adding it, so that re-issuing a command works.
Consequently a label already sitting on a PR after a run does nothing.

A run is bound to the commit that was current when it started. Pushing, rebasing, or merging
does not start a new run.

## Transporting a test path

A narrowed scope travels from the requester to the test lanes as **commit statuses** on the
head SHA (`ci/test-scope-1..N`), not by re-scanning comments.

**Why not comment scanning.** A PR accumulates commands; scanning makes "the current scope"
ambiguous and lets an old request resurface. Keying to the head SHA makes staleness
structural instead: a new push is a new SHA carrying no scope, so a stale scope cannot be
revived by re-triggering.

**Why chunked.** A status description caps at 140 characters, which fits only two or three
real test paths. A scope is therefore split on path boundaries across numbered statuses and
reassembled by index. Because commit statuses cannot be deleted, only superseded, a scope
that shrinks re-publishes its leftover chunks as empty.

**Charset.** Requested paths are validated against a strict charset before use. The value
reaches a shell, so "starts with `tests/` and has no `..`" is not sufficient — a legal
filename can still carry shell metacharacters.

**Manual paths retain shape-only validation.** Explicit path requests validate charset,
root and count, without an extra contents lookup. A well-formed manual path that does not
exist is accepted and published. Automatic suggestions instead verify the pinned merged
tree and inspect test blobs in an isolated temporary repository. Automatic preparation
checks out source with hooks and external Git configuration disabled, without importing,
executing, or collecting PR code. Manual validation needs no source checkout.

The cost of that lands later and is not small. The lanes start, pull the image and install,
and only then does the runner's selection check reject the path, failing every lane. The
failure is loud rather than silent, so nothing merges on a false green — but it consumes
several minutes on each of the on-demand runners to say "that file is not there".

Whoever *issues* the command is the right place to catch this, because unlike the handler it
has the files.

## Reject, do not degrade

When a requested scope does not validate, the command is rejected. The tempting alternative
is to ignore the bad argument and run the full suite, on the reasoning that starting some CI
beats starting none.

That reasoning fails for two reasons.

**It is not what was asked for.** A narrowed run and a full run differ in duration by hours.
Silently substituting one for the other, and then reporting success, tells the requester
their request succeeded when it did not.

**The caller is frequently automation.** A machine can retry a loud rejection. It has no way
to detect a quiet substitution, because the observable outcome — a green check — is identical.

Rejection is also simpler: there is no heuristic separating "a mistyped path" from "an
English sentence", because neither is a valid scope.

## Constraints worth knowing

**The comment handler cannot be tested by CI on a pull request.** `issue_comment` workflows
are loaded from the repository's default branch, so no run on a PR ever executes that PR's
version of the handler; it takes effect only once merged. Changes to it must be carried by
review, and are best verified out-of-band against a scratch repository where the handler can
be merged and exercised directly.

**The handler uses the first unfenced command line.** Quoted replies, code fences,
inline code and mentions embedded in prose do not issue commands. This also keeps
the inline example command in an AI recommendation from starting another run.
Commands posted by `flashinfer-bot`, the account used by `FLASHINFER_BOT_TOKEN`,
are authorized by that token, so automation holding it can start CI. Other
services post as the same account and echo text an outsider controls: CI reports
quote test names, replies quote the original comment, and AI notes quote model
explanations. The permission step therefore admits the account only when the
whole comment is one command line, the form automation posts, and never falls
back to the account's repository role. The notes this handler posts end with the
hidden marker `<!-- flashinfer-bot:ci-bot-commands-note -->`, and the job guard
skips bot comments carrying it before checkout. Keep the guard, the marker and
the account name in sync. Generated replies retain copyable commands; model
explanations and validation errors remove bot mentions, including repeated `@`,
without regard to case.

**A narrowed run is not a full run.** The GPU lanes change name and the summary states the
scope, but the required check reports success either way. Green on a narrowed run means
"what was asked for passed", not "everything passed".

## AI scope recommendations

Bare `run` asks NVIDIA Inference Hub for a recommendation using the current target
branch merged with the pinned PR head. The trusted selector reads that isolated
tree, inspected test bodies, and current CI routing; it never imports, collects or
executes PR code. Source hooks and external Git configuration are disabled.

The two model calls request evidence files, then a structured decision:
`selected` (up to ten focused targets), `full_suite`, `no_pytest_change`, or
`needs_maintainer`. Non-target decisions require a rationale. A full-suite decision
must cite broad impact supported by source and CI evidence; uncertainty or missing
evidence is not a reason to run everything.

The 300,000-character JSON context cap applies to both requests after credential
redaction. Evidence is budgeted before the first request: retain changed-file
metadata first, prefer changed tests and nearby files in the inventory, and omit
whole patches or whole source bodies when they cannot fit. Omission counts are
included in `omitted_evidence`, and unread source paths in `unavailable_evidence`.
CI routing bodies stay complete. Eager file reads reserve space for the model's
second-stage file requests, including retries of omitted `configuration_files`
that exist in the pinned tree. Per-path omission records have a separate bounded
allowance; excess records increment a count without consuming source space.
A full-suite decision is downgraded to maintainer review if the PR description
or changed-file metadata was incomplete, or the files response count differs
from the captured PR count. Matching counts alone do not prove snapshot identity.

The Actions log records each request stage, serialized size, section sizes, and
omission counts without logging source contents or credentials. The internal
daemon uses the same budgeting module, deployed separately.

For an offline reproduction, save the PR and paginated files API responses as
JSON and ensure the base and head histories are already in a local checkout:

```bash
python scripts/infra/pr_checks/replay_ai_scope.py \
  --source-repo /path/to/flashinfer \
  --pr-json /tmp/pr.json --files-json /tmp/files.json \
  --base-sha <full-base-commit> \
  --read-path tests/path/test_file.py
```

The replay runs the real selector with a fake model, reports both request sizes,
and returns `needs_maintainer`; it does not assess coverage. No inference key or
network access is used, and no PR code is imported. To replay the internal
daemon, also pass `--internal-bot-root /path/to/ci-bot`,
`--ci-repo /path/to/flashinfer-ci`, and `--ci-ref <CI-definition-commit>`.

Focused or inconclusive results produce an advisory comment for developer review
and an explicit run command. They publish no statuses and apply no labels, leaving
any current CI run alone. Only `full_suite` starts an automatic run, with an empty
scope. Explicit paths and `run all` bypass inference; `run full` is no longer valid.

An automatic full-suite run clears previous scope chunks, then publishes
`ci/test-snapshot` on the head: base, diagnostic merge and tree SHAs, plus the first
16 hex digits of SHA-256 of the empty scope (139 characters total). It uses the
existing `run-ci-focused` label, whose event requires snapshot metadata. Explicit
paths and `run all` clear that marker and use `run-ci`. Status reads, writes and
clears must succeed before labeling. Head and current target tip are rechecked.

`pr-test-dispatch.yml` verifies the scope fingerprint (including an empty full-suite scope),
reconstructs the pinned base/head merge, and checks its tree before scheduling.
GPU jobs reconstruct the same source before initializing submodules. A stale
GitHub merge ref is not authoritative. Missing metadata, changed scope, conflicts
or mismatched trees fail preparation. Manual commands retain their existing source
preparation. An advancing base after a run still requires a new CI request.

The inference key is an Actions secret available only to the authorized command
step. Model and endpoint are repository variables; explicit commands need no
inference key. See CONTRIBUTING.md for names and defaults.

The public selector accepts at most 420 seconds. Its helper has a 450-second
outer limit, including the initial lookup and cleanup. GitHub CLI calls are
bounded to 15 seconds each, with a five-second forced-termination grace period.
A helper timeout follows the rejection path before publishing statuses or
applying a CI label. Feedback still depends on GitHub availability and the runner
remaining alive within the ten-minute job limit.
