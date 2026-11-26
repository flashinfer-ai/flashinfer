# Per-PR API compatibility gate

The API checker compares a PR's merge base with its head using Python ASTs.
The target commit comes from the first parent of GitHub's PR test merge;
the second parent must match the expected PR head. A stale event base can
predate main changes already synchronized into the branch, as in #9244,
so applying merge-base to that stale SHA alone does not isolate the PR.
The scanner, gate, report publisher, inline anchor selection, REST client, and
regression tests all use Python and its standard library. No Node.js runtime or
third-party Python package is required by this API check.
The required status is **API compatibility**. Static compatibility does not
establish numerical or runtime equivalence, and the decorated surface does not
cover every public class contract.

## Policy

| Classification | Level | Gate | Heading |
| --- | --- | --- | --- |
| BREAKING | error | Block unless accepted for this head | 🚨 API BREAK 🚨 |
| LIKELY_BREAKING | warning | Pass with findings | ⚠️ API LIKELY BREAK |
| NON_BREAKING | notice | Pass | No PR comment |

Errors cover removed public functions/exports/modules without compatibility
entries, removed/renamed parameters, new required parameters, removed defaults,
changed required positional binding, removed variadic support, and sync/async
changes. Positional-only parameter names are not part of the keyword contract.

Warnings cover default changes, optional positional insertion/reordering or
conversion to keyword-only, unresolved wrappers/dynamic exports, decorator-only
removal, kwargs interception, and annotation changes that are not recognized
widenings. Removing a decorator cannot hide an independent breaking signature
change. Optional-binding warnings are a deliberate gating policy: old positional
calls can still fail or silently bind differently and must be reviewed.

Input and return annotations follow the same widening rule: retain every old
type alternative and the annotation change is NON_BREAKING. For example,
`torch.Tensor` to `Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`
does not alert. Other signature changes are classified independently. This
annotation rule requires no runtime-return investigation.

## Reports

One sticky API comment lists changed APIs, linked source locations and signature
diffs, grouped under the two severity headings. It does not repeat explanations,
rule identifiers or daily-report metadata. Inline comments explain each API's
impact and affected old calls. NON_BREAKING items and their counts are omitted
from PR comments. Complete scans with no errors or warnings post no summary;
when such a result resolves earlier findings, the previous API summary is removed.
Warning, expected-break, and incomplete-scan reports remain visible.
Inline anchors use source signature ranges and actual changed lines. Multi-line
signature changes can attach to a changed argument, removals use LEFT-side lines,
and renamed files retain their old/new coordinate mapping. If no precise line is
available, use an explicit file-level review comment rather than an unrelated
line. Findings with no changed file are logged as unplaced. Full evidence and
`comment_delivery` counts remain in the `trusted-public-api-report` artifact.
New line comments are submitted in one review; reruns update matching active
comments without duplicating threads. Summaries identify the analyzed head and
are bounded below GitHub's comment limit, retaining full counts and a link to
the complete artifact even when some changes are omitted. Removed-source links
use the base revision. Head/base identity is checked again after comment and
diff reads so stale findings are not attached to a newer diff.

All report text is English. A blocked report ends with:

> A non-author maintainer/admin can comment `@flashinfer-bot API BREAK expected, reason: ...` to unblock all API BREAKs for merging this head.

At most three matching code owners are shown, with a link to the target revision's
CODEOWNERS file for the rest. Each suggested owner must be a non-author human
whose current repository `role_name` is `maintain` or `admin`. Owners with only
`write` access, teams, bots, and owners whose permissions cannot be verified are
omitted. GitHub's coarse `permission` field is not used: `maintain` also reports
`permission: write`. The hint list does not grant approval permission; commands
recheck the actor's live role independently.
Signature diffs fold unchanged parameters and annotations. Inline explanations
combine argument shifts caused by one moved parameter; full signatures and
per-parameter diagnostics remain in the report artifact.

## Maintainer exception

```text
@flashinfer-bot API BREAK expected, reason: intentional API cleanup
```

The original standalone `@flashinfer-bot API BREAK expected` is also accepted.
When `, reason:` is present, its value must be nonempty and stay on the same line.
The reason is retained in the source comment; it does not change authorization,
revision binding, or edit/deletion invalidation. `revoke` remains standalone.

A non-author human maintainer with repository `maintain` or `admin` role can
accept **all** API breaks for the current PR head. Ordinary CI permission
(`write`, `triage`, or `ci-users`) does not grant exception approval.

The existing bot command workflow adds 👀 when processing starts, and 👍 only
after recording an authorized exception and successfully publishing the passing
gate for that head. It posts no acknowledgment or approval-summary comment.
The existing sticky report may update to `PASS (expected)`; findings remain
BREAKING/error in the evidence.

A trusted complete scan for the same head/base generation must predate the
command. The most recent scan must also be healthy; a newer healthy rescan does
not hide qualifying earlier evidence. A partially completed acceptance can be
retried after revalidating the live comment, permissions, revision and command
order. An evaluation API failure leaves the gate in error and preserves completed
scan evidence. A queued command cannot approve a new revision whose scan
completed after the command was posted. Every new commit
requires a new command. Approval and complete-scan records bind PR, head SHA,
base SHA, and a revision generation derived from trusted PR timeline events.
Returning to an earlier head by force-push cannot reuse that head's old approval;
rescanning without an intervening revision change preserves approval. Records
also bind command ID, actor and workflow provenance. Older unbound records are
not eligible; rollout requires a fresh scan and approval.

Editing or deleting the latest recorded approval queues a gate reevaluation through `issue_comment` edited/deleted events. Until GitHub runs that
queued job, the previous status may still be visible. Reevaluation invalidates
the approval without resurrecting an older one. A deleted comment does not
invalidate completed scan evidence.
A successful rescan from either trusted workflow refreshes that evidence; a
failed revoke rescan leaves the gate in error and refreshes the summary as
incomplete when comment delivery is available. A later explicit revocation takes
precedence over an older command even if workflow execution is delayed:

```text
@flashinfer-bot API BREAK revoke
```

Invalid or unauthorized commands do not receive a success 👍. Errors are visible
in workflow logs. Execution failures/incomplete scans cannot be waived.

## Trusted execution and gate states

`pr-api-compatibility.yml` uses `pull_request_target` with a default-branch
checkout. It fetches PR git objects and parses source; it never checks out,
installs, imports or executes PR-head code. PR-produced artifacts are not gate
inputs. Both trusted scans and bot command rescans fetch `refs/pull/<PR>/merge`
as git objects and resolve the aligned target/head pair without checking out
the merge. Missing merge snapshots or a mismatched head fail the scan; they
never fall back to the stale event base. The bot also executes default-branch
code. The advisory documentation producer resolves the same pair from its
checked-out PR test merge. This integrates the baseline fix from #9338;
the documentation producer/reporter still handles documentation only.

The CLI preserves `--strict` (fail for any alert) and adds `--fail-on error`:

```sh
python3 scripts/infra/check_pr_api_diff.py --base BASE_SHA --head HEAD_SHA \
  --fail-on error --report-json reports/public-api.json
```

Scanner exit codes are 0 (no blocking error), 1 (breaking findings), and 2
(incomplete/failed scan). The final trusted gate incorporates a valid exception.
A complete scan with no unaccepted errors passes; an incomplete scan never does.
Commit statuses store the scan and exception audit references separately from
the final required `API compatibility` status. The scanner and bot serialize
updates per PR using `queue: max` (up to 100 queued jobs; additional jobs are
canceled by GitHub) and recheck head/base identity before publishing gate results.
Retargeting a PR to `main` also schedules a scan; title/body edits do not.

GitHub commit statuses are shared by all PRs using the same head SHA. Before
publishing success, the gate checks for other open PRs at that SHA and fails
closed if any exist. Use a distinct head commit for each open PR (an empty commit
is sufficient). An exception in one PR cannot waive this restriction or unblock
another PR. The shared-head check runs before and after publishing success;
a newly detected peer repairs success to failure. GitHub does not offer an
atomic transaction covering PR enumeration, revision changes and commit-status
publication. Event delivery/queue delay and the final API read/write window
remain; this checker is not an atomic per-PR merge authorization service.
The required check name remains **API compatibility**.

The [same-head validation](pr_api_same_head_validation.md) records live fork
experiments with ordinary Actions checks and statuses on test-merge commits.
Neither demonstrated independent merge authorization for two PRs sharing a
head. An organization required-workflow policy remains an unverified candidate;
it is not implemented by this gate. The shared-head race above remains open.

## Validation

`scripts/infra/tests/test_check_pr_api_diff.py` is retained for local validation. PR workflows run
the API scan and merge gate without running the checker unit tests.

```sh
python3 -m unittest discover -s scripts/infra/tests -p 'test_check_pr_api_diff.py' -v
pre-commit run --files <changed-paths>
```

The retained tests cover signature comparison, type aliases, compatible
re-exports, table-driven exports, and experimental API exclusions. Git-history
regressions cover main synchronization with a stale event base, preservation
of PR-introduced API/documentation findings, invalid merge/head rejection,
the comparison CLI, and the trusted scanner's git-object-only path.

## Rollout and rollback

1. Merge the reviewed implementation. The privileged workflow and bot handlers
   must exist on the default branch before they can be exercised end to end.
2. Validate a passing PR, a blocking PR, a warnings-only PR, global expected with
   👀/👍 only, a new push invalidating approval, and revoke restoring the block.
3. Configure the repository ruleset to require **API compatibility**, selecting
   the GitHub Actions source where supported. This is a separate repository
   administration action; creating this PR does not change rules or merge it.
4. Roll back by removing the required status requirement before reverting the
   workflow, to avoid leaving PRs waiting forever on a removed status.

GitHub references: [commit statuses](https://docs.github.com/en/rest/commits/statuses),
[concurrency queues](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)
and [pull_request_target](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request_target).


## Documentation and API check separation

Blocked summaries list code owners matched to BREAKING API paths using the
target revision's `.github/CODEOWNERS` (last matching rule wins). Bot owners are
omitted. These are ownership hints; accepting a break still requires a human
with maintain/admin permission who is not the PR author. The PR head cannot
replace the ownership hints by editing its own CODEOWNERS file.

API scanning and merge gating run in `pr-api-compatibility.yml` from trusted
code. Its `trusted-public-api-report` artifact, API sticky marker, and inline
marker are independent of documentation reports. Evidence-upload failures warn
without replacing a completed gate decision.

`pr-api-doc-checks.yml` runs the read-only advisory documentation producer and
uploads only documentation evidence. `pr-api-doc-comment.yml` reads
only documentation artifacts and writes only documentation comments.

The privileged documentation reporter runs from the default branch. Until this
PR's reporter update is deployed there, the old reporter can still complain that
`public-api.json` is missing from the now-documentation-only artifact. That is a
rollout mismatch, not evidence that the trusted API scan failed. Do not fabricate
an empty API report to hide it.

The sticky API summary intentionally lists changes only; detailed reasons remain
in inline comments. When only a local type alias changes, the diff includes the
alias declarations so the summary does not show two identical signatures. Invalid anchors receive individual retry only for positively
identified anchor-validation errors; spam/rate-limit responses never fan out.
