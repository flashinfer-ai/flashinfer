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
The job guard excludes comments by `flashinfer-bot`, the account used by
`FLASHINFER_BOT_TOKEN`, before checkout and permission checks. Keep that guard in
sync if the posting account changes. Generated replies retain copyable commands;
model explanations and validation errors remove bot mentions without regard to case.

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
python scripts/pr_checks/replay_ai_scope.py \
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
