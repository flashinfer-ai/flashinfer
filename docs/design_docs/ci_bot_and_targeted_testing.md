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

## The command vocabulary

The bot answers a comment in one of three ways, and this vocabulary is load-bearing —
automation depends on being able to tell the outcomes apart:

| outcome | response | effect |
|---|---|---|
| accepted | react `+1` | CI label applied, run starts |
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

**Existence is deliberately not checked.** The handler validates shape — charset, root,
count — but not whether the paths exist. It cannot: the job holds a write token, so it does
not check out the pull request, and it therefore has no files to look at. A well-formed path
that does not exist is accepted and published.

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

**The trigger phrase is matched anywhere in a comment body.** Writing *about* a command
therefore issues it, including from inside code spans, fenced blocks and quoted replies.
Since the handler re-applies the CI label, and that cancels in-progress runs, an incidental
mention can cancel and restart a multi-hour run. Automation that comments on a PR it is also
driving must avoid reproducing the phrase.

**A narrowed run is not a full run.** The GPU lanes change name and the summary states the
scope, but the required check reports success either way. Green on a narrowed run means
"what was asked for passed", not "everything passed".
