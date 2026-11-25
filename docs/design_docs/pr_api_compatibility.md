# Per-PR API compatibility gate

The API checker compares a PR's merge base with its head using Python ASTs.
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

> Maintainers @xxxx can comment `@flashinfer-bot API BREAK expected` as the entire comment to unblock all API BREAK findings for the current head. Reason: … (provide separately in the PR description or discussion).

`@xxxx` is replaced by maintainer accounts where accessible, with a permission
based fallback. Reason is explanatory text outside the command, not a mandatory
argument. The accepted command itself is the standalone line below.

## Maintainer exception

```text
@flashinfer-bot API BREAK expected
```

A non-author human maintainer with repository `maintain` or `admin` role can
accept **all** API breaks for the current PR head. Ordinary CI permission
(`write`, `triage`, or `ci-users`) does not grant exception approval.

The existing bot command workflow adds 👀 when processing starts, and 👍 only
after recording an authorized exception and successfully publishing the passing
gate for that head. It posts no acknowledgment or approval-summary comment.
The existing sticky report may update to `PASS (expected)`; findings remain
BREAKING/error in the evidence.

A trusted complete scan must predate the command. A queued command cannot approve
a new head whose scan completed after the command was posted. Every new commit
requires a new command. Approval records bind PR, SHA, command ID, actor and
workflow provenance. Editing or deleting an approval invalidates it without
resurrecting an older approval. A deleted comment does not invalidate completed scan evidence.
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
inputs. The bot also executes default-branch code. The existing documentation
producer/reporter remains advisory and handles documentation only.

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
another PR. The required check name remains **API compatibility**.

## Validation

```sh
python3 -m unittest discover -s scripts/infra/tests -p 'test*api*.py' -v
node --test scripts/infra/tests/test_api_*.cjs
pre-commit run --files <changed-paths>
```

Fixtures cover widening returns, optional binding policy, mixed severity reports,
decorator-plus-signature changes, dynamic exports, authorization, source-comment
provenance, delayed approval, head races, revocation and scan failures.

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
