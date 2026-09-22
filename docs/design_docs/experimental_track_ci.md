# CI for the Experimental Track

This document records how an experimental-track pull request is tested and lands. It builds
on [`ci_bot_and_targeted_testing.md`](ci_bot_and_targeted_testing.md), which records the
general CI triggering and targeted-testing design; the policy for what an experimental API
or backend *is* lives in [`experimental_apis_and_backends.md`](experimental_apis_and_backends.md).

## Why the track needs its own CI story

The experimental track exists to land fast-moving features **quickly**, under a declared
lifetime and a distinct workflow. Speed is the point, not a side effect. The intended path is

```
open -> screen -> review -> approve -> CI -> auto-merge -> nightly release, same day
```

Most of that is meant to be automated: a background watcher shepherds the pull request and
is authorized to trigger CI on the author's behalf.

**Targeted testing is the mechanism that makes same-day landing possible.** An experimental
PR runs the tests its author declares rather than the full matrix — minutes instead of hours.
This is why targeted testing is not a cost-saving exception for this track; it is the reason
the track can exist at all.

Because GitHub's command historically took no arguments while GitLab's accepted a test path,
the same declared scope could narrow a GitLab run but not a GitHub one. Closing that
asymmetry is what makes the track's CI story work on both systems.

## The declared scope

An experimental PR declares its test scope in an `experimental-tests` block in the pull
request body. The body is the source of truth:

- it is what a reviewer reads;
- it is what the watcher reads, to decide what to ask CI for;
- it survives independently of how many commands have been issued on the PR.

The comment is transport, not the record. `scripts/pr_checks/experimental_test_scope.py`
parses the block for the watcher and validates targets for the handler, so both entry points
share one definition of a well-formed scope.

## Why a rejected scope must not fall back to the full suite

The general design already rejects malformed commands rather than degrading them. For this
track the argument is stronger, because degrading is actively harmful in two ways at once:

**It defeats the mission.** Falling back to the full matrix drops the PR off the fast lane —
hours instead of minutes — which is precisely the outcome the track is built to avoid.

**It runs none of the declared tests.** `tests/experimental/` is excluded from the default
suite by `norecursedirs`, so the substituted "full suite" does not cover the experimental
tree at all. The author asked for specific tests, none of them ran, and every check reported
green.

A watcher driving the PR cannot detect this, because a degraded run and an honoured one both
end green.

## Watcher responsibilities

The watcher is the component that turns a declared scope into a CI request. Two of its
obligations are not enforceable anywhere else, and should be covered by its own tests.

**Verify that every declared path exists at the PR head before issuing a command.** The
handler cannot do this — it holds a write token and so deliberately does not check out the
pull request, leaving it with no files to inspect (see
[`ci_bot_and_targeted_testing.md`](ci_bot_and_targeted_testing.md)). A well-formed path that
does not exist is therefore accepted, published, and only rejected much later by the test
runner, after every lane has pulled an image and installed. The watcher holds the PR's file
list and can settle this instantly.

Suggested test: declare a scope naming a path that does not exist at the head commit, and
assert the watcher reports the missing path and issues **no** command.

**Do not reproduce the bot's trigger phrase in any comment it posts.** The trigger is matched
anywhere in a comment body, so a status update that quotes the command re-enters the handler,
re-applies the CI label, and cancels the run the watcher is shepherding.

Suggested test: have the watcher post each of its status messages on a pull request and
assert none of them causes the handler to fire.

## Scope of testing, and what it does not prove

A green targeted run means the declared tests passed on the declared hardware. It is not
evidence that the feature is ready for stable support, and graduation is governed by
[`experimental_apis_and_backends.md`](experimental_apis_and_backends.md), not by CI.

## Non-goals

This document does not:

- unify the two CI systems, or claim the split is desirable;

- make the experimental tree part of the default suite — that would slow every pull request
  in the repository, which is the opposite of the intent;

- describe the watcher's own implementation, which lives outside this repository.
