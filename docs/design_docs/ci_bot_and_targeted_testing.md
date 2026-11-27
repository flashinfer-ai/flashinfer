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

`pr-test.yml` verifies the scope fingerprint (including an empty full-suite scope),
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

## Bot Run Request Check

`Bot Run Request Check` is a separate merge gate for internal `/bot run`
requests. It does **not** aggregate pipeline results or establish test coverage.
All prompts and diagnostics appear in Check Details and the Actions summary;
the check never posts conversation comments or reactions.

The gate first scans the PR's paginated comment history for a response from the
trusted `flashinfer-bot` account to an internal `/bot run` request. Pipeline
acknowledgements, AI advice saying no pipeline was started, authorization
rejections and trigger errors all qualify. A qualifying historical response
continues to satisfy this policy after new commits. The gate does not require
pipeline creation, a passing result or a matching tested commit. Unrelated
bot reports, public `@flashinfer-bot` suggestions and copied replies do not count.
Finding a qualifying response skips base SHA lookup, source inspection and
inference. The gate still verifies that the PR head and target branch are current.

Without a reply, the gate exempts documentation-only, CPU-only and
experimental-only changes. It checks for documentation first, including
Python comment and docstring edits verified against both versions of the file.
Changes confined to the experimental source/test trees and documentation can
also be classified without inference. An experimental
opt-in in core can qualify if the diff establishes that stable GPU behavior is
unchanged. Mixed changes that affect stable GPU behavior still require a bot
response. Experimental tests run in their own lane. To run a specific
experimental test via the internal bot, use
`/bot run tests/experimental/<test-file>.py`. The default `/bot run all`
collection skips this directory.

The optional GPU-impact analysis extends `ai_test_scope.py` with the same
inspected source/test/CI evidence and an independent impact assessment. File
extensions and `tests/cpu/` alone cannot establish CPU-only behavior. Kernels,
launchers, bindings, dispatch, shapes/dtypes/layouts, workspace, JIT/AOT, CUDA
graphs, synchronization and communication generally need GPU validation. New
PR tests are included even if they are not among the recommended targets.
Stable GPU changes still require a bot response when no test case was selected.
The gate skips missing or invalid supplemental file requests and drops invalid
test suggestions. Check Details lists warnings alongside the impact result.
The gate can read requested regular files from the pinned PR tree. Ordinary
bot commands use the model's bounded file inventory.
The GPU-impact response uses compact JSON with default reasoning. If analysis
fails validation or returns `unknown`, the gate retries once with
`reasoning_effort=minimal`. A validated evidence-file request is reused only
when its context is unchanged. If the second impact analysis is still
uncertain or invalid, the check fails and requires
`/bot run` and an internal bot response. A known stable GPU change requires a
reply immediately, without a model retry.

Use retained patches and supplied file bodies whenever they are available.
Only when neither content source is supplied and the context budget omitted it
may GPU classification fall back to the filename/path. The check records this
as filename-only evidence. The model still must not infer CPU-only merely from
a `.py` suffix or a `tests/cpu/` location. Under this gate's policy, filename-only
evidence does not by itself trigger a hard rejection of a model's CPU/docs
classification; the model should return `unknown` if it cannot support one. Missing patch text
is informational and does not automatically override a documentation/CPU-only
model conclusion. The model should return `unknown` when the available evidence
cannot support a conclusion. Missing changed-file records, invalid evidence
paths and invalid test targets still prevent an exemption. A verified bot
response remains sufficient even when analysis is unavailable.

| State without a qualifying bot response | Check result |
| --- | --- |
| Documentation/CPU-only change | Success: bot run not required, with reason |
| Experimental-only change | Success: bot run optional, with an explicit experimental test command |
| Stable GPU change, no command | Failure: comment `/bot run` |
| Stable GPU change, command found | Failure: waiting for `flashinfer-bot`, with command link |
| Unable to establish GPU impact | Failure: specific missing evidence/error and next action |

Commands with multiple spaces or tabs, such as `/bot  run tests/attention`, are
recognized. This check does not repeat the internal bot's path validation;
invalid trailing arguments still count as a request for the waiting message,
and the bot's rejection response satisfies the agreed response-only policy.
Quoted examples, fenced code and prose mentions do not issue commands.

Changed files are read from an immutable comparison of the captured base and
head commit SHAs, and that same list/base is used throughout source analysis.
A partial comparison (including GitHub's 300-file compare limit) cannot grant
an exemption. The gate rechecks the live PR before completing the Check.
For conclusions that depend on source, it also rechecks the base after
preparing the Check payload, immediately before writing it. A detected
snapshot change completes the captured Check as stale and re-evaluates once.
A second snapshot change finishes with a failure and rerun instruction.
Post-start evidence read failures complete the Check with a failure reason.
The ref read and Check write are separate GitHub API calls, so a base advance
between them can still occur.

The trusted default-branch workflow handles PR open/synchronize/reopen/ready and
base-edit events, plus PR comment creation/edit/deletion. It publishes a stable
Check on the PR head through the Checks API (not on `issue_comment`'s default
branch SHA). A per-head/base/analysis-version result can reuse its impact
assessment, but comments are read again on every event so deleted responses do
not persist as cached exemptions. Stale head/base completions are discarded.
A manual dispatch with `pr_number` can re-evaluate an existing PR.
PR source is evidence only: no imports, installs or test collection are allowed
in the credential-bearing job. A separate read-only PR workflow runs CPU tests
of this implementation before its default-branch handler is deployed.

### Enabling the merge blocker

After the implementation is merged to the default branch and has emitted its
first check, a repository administrator must add the exact **Bot Run Request
Check** context from GitHub Actions as a required status check in the `main`
branch protection/ruleset. Preserve other required checks and review/bypass
settings. Verify that a missing/failing result blocks merging and that a
qualifying response or CPU/docs exemption satisfies this requirement. The
workflow alone does not configure branch protection. If merge queue is enabled,
add merge-group evaluation before requiring this context for queued merges;
this version handles individual PR heads.

Verify comment-trigger refresh in the target repository after deployment;
GitHub suppresses follow-up workflows for events produced using `GITHUB_TOKEN`,
whereas the internal bot uses its separate account token. Existing bot replies
can be consumed without changing the internal service or accessing GitLab APIs.
The existing `PR Test / Test Results Summary` remains independent: passing this
response check does not waive any other required test result.
