"""Authorization, trust and merge-gate contracts migrated from Node."""

import copy
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pr_checks.api_gate import (
    Context,
    Gate,
    approved,
    can_approve,
    gate_state,
    parse_command,
)
from pr_checks.github_api import GitHubError
from api_test_support import FakeGitHub

ROOT = "repos/org/repo"
REVISION = hashlib.sha256(("a" * 40 + ":" + "b" * 40 + ":0").encode()).hexdigest()[:24]
BREAK = {
    "execution_status": "COMPLETE",
    "findings": [{"classification": "BREAKING", "level": "error"}],
}


class Fixture:
    def __init__(
        self,
        role="maintain",
        author="bob",
        scan_state="success",
        scan_at="2026-11-25T00:00:00Z",
    ):
        self.api = FakeGitHub()
        self.pr = {
            "number": 3,
            "state": "open",
            "user": {"login": author},
            "head": {"sha": "a" * 40},
            "base": {"sha": "b" * 40},
        }
        self.comment = {
            "id": 10,
            "body": "@flashinfer-bot API BREAK expected",
            "user": {"login": "alice", "type": "User"},
            "html_url": "https://github.com/org/repo/pull/3#issuecomment-10",
            "created_at": "2026-11-25T01:00:00Z",
            "updated_at": "2026-11-25T01:00:00Z",
        }
        self.context = Context(
            "org/repo", "2", {"issue": {"number": 3}, "comment": self.comment}
        )
        self.records = [
            {
                "id": 1,
                "context": "api-scan/pr-3",
                "state": scan_state,
                "created_at": scan_at,
                "target_url": "https://github.com/org/repo/actions/runs/1",
                "creator": {"login": "github-actions[bot]"},
            }
        ]
        self.api.routes.update(
            {
                ("GET", ROOT + "/pulls/3"): lambda _: copy.deepcopy(self.pr),
                ("GET", ROOT + "/pulls"): [self.pr],
                ("GET", ROOT + "/collaborators/alice/permission"): {"role_name": role},
                (
                    "GET",
                    ROOT + "/commits/" + "a" * 40 + "/statuses",
                ): lambda _: copy.deepcopy(self.records),
                ("POST", ROOT + "/statuses/" + "a" * 40): self.add_status,
                ("GET", ROOT + "/issues/comments/10"): lambda _: copy.deepcopy(
                    self.comment
                ),
                ("POST", ROOT + "/issues/comments/10/reactions"): {},
                ("GET", ROOT + "/collaborators"): [],
                ("GET", ROOT + "/issues/3/comments"): [],
                ("POST", ROOT + "/issues/3/comments"): {},
                ("GET", ROOT + "/pulls/3/files"): [],
                ("GET", ROOT + "/pulls/3/comments"): [],
            }
        )
        for run, event, file in [
            (1, "pull_request_target", "pr-api-compatibility.yml"),
            (2, "issue_comment", "ci-bot-commands.yml"),
        ]:
            self.api.routes[("GET", ROOT + f"/actions/runs/{run}")] = {
                "event": event,
                "path": ".github/workflows/" + file,
                "repository": {"full_name": "org/repo"},
            }
        self.api.routes[("GET", ROOT + "/issues/3/timeline")] = []
        self.records[0]["description"] = "complete:" + REVISION
        self.gate = Gate(self.api, self.context, warning=lambda _: None)

    def add_status(self, data):
        item = {
            **data,
            "id": max(x["id"] for x in self.records) + 1,
            "creator": {"login": "github-actions[bot]"},
            "created_at": "2026-11-25T01:00:01Z",
        }
        self.records.insert(0, item)
        return item

    def statuses(self, name="API compatibility"):
        return [
            c[2]
            for c in self.api.calls
            if c[0] == "POST" and "/statuses/" in c[1] and c[2]["context"] == name
        ]

    def reactions(self):
        return [c[2]["content"] for c in self.api.calls if c[1].endswith("/reactions")]

    def run(self, scan=None):
        # Keep fixture publication out of the source checkout.
        with patch.object(self.gate, "write_report"):
            self.gate.handle_command(scan=scan or (lambda _: copy.deepcopy(BREAK)))


class GateTest(unittest.TestCase):
    def test_exact_standalone_commands_only(self):
        self.assertEqual(parse_command(" @flashinfer-bot api break revoke\n"), "revoke")
        self.assertEqual(
            parse_command("@flashinfer-bot API BREAK expected"), "expected"
        )
        for body in [
            "text @flashinfer-bot API BREAK expected",
            "> @flashinfer-bot API BREAK expected",
            "```\n@flashinfer-bot API BREAK expected\n```",
            "@flashinfer-bot API BREAK expected all",
            "@flashinfer_bot API BREAK expected",
            "@flashinfer-bot API BREAK expected\nmore",
        ]:
            self.assertIsNone(parse_command(body))

    def test_roles_and_self_approval(self):
        self.assertTrue(can_approve("maintain", "alice", "bob", "User"))
        for role, actor, author, kind in [
            ("admin", "alice", "alice", "User"),
            ("write", "alice", "bob", "User"),
            ("triage", "alice", "bob", "User"),
            ("none", "alice", "bob", "User"),
            ("admin", "bot", "bob", "Bot"),
        ]:
            self.assertFalse(can_approve(role, actor, author, kind))

    def test_only_complete_classified_errors_are_waived(self):
        self.assertEqual(gate_state(BREAK, False), "failure")
        self.assertEqual(gate_state(BREAK, True), "success")
        self.assertEqual(
            gate_state({**BREAK, "execution_status": "FAILED"}, True), "error"
        )
        self.assertEqual(
            gate_state(
                {
                    "execution_status": "COMPLETE",
                    "findings": [
                        {"classification": "LIKELY_BREAKING", "level": "warning"}
                    ],
                },
                False,
            ),
            "success",
        )
        with self.assertRaises(ValueError):
            gate_state({"findings": []}, True)
        with self.assertRaises(ValueError):
            gate_state(
                {
                    "execution_status": "COMPLETE",
                    "findings": [{"classification": "BREAKING", "level": "warning"}],
                },
                True,
            )

    def test_unauthorized_only_eyes(self):
        f = Fixture(role="write")
        f.run()
        self.assertEqual(f.reactions(), ["eyes"])
        self.assertEqual(f.statuses(), [])

    def test_command_cannot_approve_a_later_completed_scan(self):
        f = Fixture(scan_at="2026-11-25T02:00:00Z")
        f.run()
        self.assertEqual(f.reactions(), ["eyes"])
        self.assertEqual(f.statuses(), [])

    def test_failed_scan_cannot_be_approved(self):
        f = Fixture(scan_state="error")
        f.run()
        self.assertEqual(f.reactions(), ["eyes"])
        self.assertEqual(f.statuses(), [])

    def test_authorized_exact_head_then_gate_and_thumb(self):
        f = Fixture()
        f.run()
        self.assertEqual(f.statuses("api-break-exception/pr-3")[0]["state"], "success")
        self.assertEqual(f.statuses()[-1]["state"], "success")
        self.assertEqual(f.reactions(), ["eyes", "+1"])
        self.assertTrue(
            all(
                c[1].endswith(f.pr["head"]["sha"])
                for c in f.api.calls
                if c[0] == "POST" and "/statuses/" in c[1]
            )
        )

    def exception(self, number=10, **extra):
        return {
            "id": number,
            "context": "api-break-exception/pr-3",
            "description": f"expected:{number}:{REVISION}",
            "state": "success",
            "target_url": "https://github.com/org/repo/actions/runs/2",
            "creator": {"login": "github-actions[bot]"},
            **extra,
        }

    def test_forged_provenance_is_rejected(self):
        f = Fixture()
        self.assertFalse(
            f.gate.valid_exception(
                f.pr,
                self.exception(target_url="https://github.com/org/repo/actions/runs/1"),
            )
        )
        self.assertFalse(
            f.gate.valid_exception(f.pr, self.exception(creator={"login": "alice"}))
        )

    def test_author_cannot_approve(self):
        f = Fixture(author="alice")
        f.run()
        self.assertEqual(f.reactions(), ["eyes"])
        self.assertEqual(f.statuses(), [])

    def test_head_change_during_scan_never_records_approval(self):
        f = Fixture()

        def scan(_):
            f.pr["head"]["sha"] = "c" * 40
            return copy.deepcopy(BREAK)

        f.run(scan)
        self.assertEqual(f.statuses("api-break-exception/pr-3"), [])
        self.assertEqual(f.reactions(), ["eyes"])

    def test_incomplete_rescan_cannot_be_waived(self):
        f = Fixture()
        f.run(lambda _: {"execution_status": "FAILED", "findings": []})
        self.assertEqual(f.statuses("api-break-exception/pr-3"), [])
        self.assertEqual(f.statuses()[-1]["state"], "error")
        self.assertEqual(f.reactions(), ["eyes"])

    def test_newer_revoke_wins_over_delayed_expected(self):
        f = Fixture()
        f.records.append(self.exception(11, description="revoke:11", state="failure"))
        f.run()
        self.assertEqual(f.statuses(), [])
        self.assertEqual(f.reactions(), ["eyes"])

    def test_edited_latest_never_resurrects_valid_older_approval(self):
        f = Fixture()
        older = copy.deepcopy(f.comment)
        newer = {
            **older,
            "id": 11,
            "html_url": older["html_url"].replace("-10", "-11"),
            "updated_at": "2026-11-25T02:00:00Z",
        }
        f.api.routes[("GET", ROOT + "/issues/comments/10")] = older
        f.api.routes[("GET", ROOT + "/issues/comments/11")] = newer
        self.assertTrue(approved(f.gate, f.pr, [self.exception(10)]))
        self.assertFalse(
            approved(f.gate, f.pr, [self.exception(10), self.exception(11)])
        )

    def test_revoke_reblocks_without_thumb(self):
        f = Fixture()
        f.comment["body"] = "@flashinfer-bot API BREAK revoke"
        f.run()
        self.assertEqual(f.statuses()[-1]["state"], "failure")
        self.assertEqual(f.reactions(), ["eyes"])

    def test_trusted_scanner_does_not_execute_pr_code_or_inherit_git_location(self):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(
                    ["git", *args],
                    cwd=root,
                    env=clean,
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()

            git("init", "-q")
            git("config", "user.name", "API test")
            git("config", "user.email", "api@example.com")
            git("remote", "add", "origin", str(root))
            (root / "flashinfer").mkdir()
            (root / "flashinfer/example.py").write_text(
                "@flashinfer_api\ndef api(x): pass\n"
            )
            (root / "flashinfer/__init__.py").write_text(
                'raise RuntimeError("PR package must never execute")\n'
            )
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            (root / "flashinfer/example.py").write_text(
                "@flashinfer_api\ndef api(x,workspace): pass\n"
            )
            (root / "scripts/infra").mkdir(parents=True)
            (root / "scripts/infra/check_pr_api_diff.py").write_text(
                'raise RuntimeError("untrusted checker executed")\n'
            )
            git("add", ".")
            git("commit", "-qm", "head")
            head = git("rev-parse", "HEAD")
            f = Fixture()
            cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(
                    os.environ,
                    {
                        "GIT_DIR": str(root / "nonexistent-caller"),
                        "GIT_WORK_TREE": str(root / "wrong-worktree"),
                    },
                ):
                    report = f.gate.scan({"base": {"sha": base}, "head": {"sha": head}})
            finally:
                os.chdir(cwd)
            self.assertEqual(report["execution_status"], "COMPLETE")
            self.assertEqual(report["findings"][0]["classification"], "BREAKING")
            self.assertFalse((root / "nonexistent-caller").exists())

    def test_revoke_persists_before_failed_or_incomplete_rescan(self):
        for throwing in (True, False):
            f = Fixture(scan_state="error")
            f.comment["body"] = "@flashinfer-bot API BREAK revoke"

            def scan(_):
                if throwing:
                    raise RuntimeError("scan failed")
                return {"execution_status": "FAILED", "findings": []}

            f.run(scan)
            self.assertEqual(
                f.statuses("api-break-exception/pr-3")[0]["description"],
                "revoke:10:" + REVISION,
            )
            self.assertEqual(f.statuses()[-1]["state"], "error")
            self.assertEqual(f.reactions(), ["eyes"])

    def test_deleted_latest_invalidates_without_masking_other_errors(self):
        f = Fixture()
        f.api.routes[("GET", ROOT + "/issues/comments/11")] = GitHubError(
            404, "deleted"
        )
        self.assertFalse(f.gate.valid_exception(f.pr, self.exception(11)))
        f.api.routes[("GET", ROOT + "/issues/comments/11")] = GitHubError(
            403, "rate limited"
        )
        with self.assertRaises(GitHubError):
            f.gate.valid_exception(f.pr, self.exception(11))

    def test_successful_retry_restores_evidence_for_later_expected(self):
        f = Fixture(scan_state="error")
        f.comment["body"] = "@flashinfer-bot API BREAK revoke"
        f.run()
        self.assertEqual(f.statuses("api-scan/pr-3")[-1]["state"], "success")
        f.comment.update(
            id=11,
            body="@flashinfer-bot API BREAK expected",
            created_at="2026-11-25T03:00:00Z",
            updated_at="2026-11-25T03:00:00Z",
            html_url="https://github.com/org/repo/pull/3#issuecomment-11",
        )
        f.api.routes[("GET", ROOT + "/issues/comments/11")] = lambda _: copy.deepcopy(
            f.comment
        )
        f.api.routes[("POST", ROOT + "/issues/comments/11/reactions")] = {}
        f.run()
        self.assertEqual(f.reactions()[-1], "+1")

    def test_failed_revoke_refreshes_expected_summary(self):
        f = Fixture()
        f.comment["body"] = "@flashinfer-bot API BREAK revoke"
        f.api.routes[("GET", ROOT + "/issues/3/comments")] = [
            {
                "id": 5,
                "user": {"login": "github-actions[bot]"},
                "body": "<!-- flashinfer-api-compatibility -->PASS (expected)",
            }
        ]
        f.api.routes[("PATCH", ROOT + "/issues/comments/5")] = {}

        def fail(_):
            raise RuntimeError("syntax failure")

        f.run(fail)
        bodies = [c[2]["body"] for c in f.api.calls if c[0] == "PATCH"]
        self.assertIn("INCOMPLETE", bodies[-1])
        self.assertNotIn("PASS", bodies[-1])

    def test_shared_sha_cannot_be_unblocked_by_one_pr(self):
        f = Fixture()
        f.api.routes[("GET", ROOT + "/pulls")] = [f.pr, {**f.pr, "number": 4}]
        f.run()
        self.assertEqual(f.statuses()[-1]["state"], "failure")
        self.assertIn("shared", f.statuses()[-1]["description"])
        self.assertEqual(f.reactions(), ["eyes"])

    def test_force_push_back_does_not_restore_an_old_approval(self):
        f = Fixture()
        item = self.exception(10)
        self.assertTrue(f.gate.valid_exception(f.pr, item))
        f.api.routes[("GET", ROOT + "/issues/3/timeline")] = [
            {
                "id": 91,
                "event": "head_ref_force_pushed",
                "commit_id": f.pr["head"]["sha"],
            }
        ]
        self.assertFalse(f.gate.valid_exception(f.pr, item))

    def test_later_healthy_scan_does_not_hide_earlier_eligible_scan(self):
        f = Fixture()
        f.records.insert(
            0, {**f.records[0], "id": 2, "created_at": "2026-11-25T02:00:00Z"}
        )
        f.run()
        self.assertEqual(f.reactions(), ["eyes", "+1"])

    def test_partial_approval_can_resume_after_api_recovery(self):
        f = Fixture()
        f.api.routes[("GET", ROOT + "/pulls")] = GitHubError(503, "unavailable")
        f.run()
        self.assertEqual(f.statuses()[-1]["state"], "error")
        self.assertEqual(f.statuses("api-scan/pr-3")[-1]["state"], "success")
        f.api.routes[("GET", ROOT + "/pulls")] = [f.pr]
        f.run()
        self.assertEqual(f.statuses()[-1]["state"], "success")
        self.assertEqual(f.reactions()[-1], "+1")

    def test_peer_opened_during_status_write_reblocks_success(self):
        f = Fixture()
        calls = 0

        def peers(_):
            nonlocal calls
            calls += 1
            return [f.pr] if calls == 1 else [f.pr, {**f.pr, "number": 4}]

        f.api.routes[("GET", ROOT + "/pulls")] = peers
        f.run()
        self.assertEqual(f.statuses()[-1]["state"], "failure")
        self.assertEqual(f.reactions(), ["eyes"])

    def test_approval_edit_or_delete_triggers_recheck_without_reaction(self):
        for action in ("edited", "deleted"):
            with self.subTest(action=action):
                f = Fixture()
                f.run()
                self.assertEqual(f.statuses()[-1]["state"], "success")
                f.context.payload["action"] = action
                if action == "edited":
                    f.comment["body"] = "approval withdrawn"
                    f.comment["updated_at"] = "2026-11-25T04:00:00Z"
                else:
                    f.api.routes[("GET", ROOT + "/issues/comments/10")] = GitHubError(
                        404, "deleted"
                    )
                f.run()
                self.assertEqual(f.statuses()[-1]["state"], "failure")
                self.assertEqual(f.reactions(), ["eyes", "+1"])

    def test_generation_change_during_success_write_reblocks(self):
        f = Fixture()

        def write(data):
            result = f.add_status(data)
            if data["context"] == "API compatibility" and data["state"] == "success":
                f.api.routes[("GET", ROOT + "/issues/3/timeline")] = [
                    {"id": 92, "event": "head_ref_force_pushed"}
                ]
            return result

        f.api.routes[("POST", ROOT + "/statuses/" + "a" * 40)] = write
        f.run()
        self.assertNotEqual(f.statuses()[-1]["state"], "success")
        self.assertEqual(f.reactions(), ["eyes"])

    def test_unrelated_comment_edit_does_not_rescan(self):
        f = Fixture()
        f.context.payload["action"] = "edited"

        def fail(_):
            raise AssertionError("unrelated comment must not trigger a rescan")

        f.run(fail)
        self.assertEqual(f.statuses(), [])
        self.assertEqual(f.reactions(), [])

    def test_old_base_evidence_cannot_approve_same_head(self):
        f = Fixture()
        f.pr["base"]["sha"] = "c" * 40
        f.run()
        self.assertEqual(f.reactions(), ["eyes"])
        self.assertEqual(f.statuses(), [])

    def test_rescan_without_push_preserves_approval(self):
        f = Fixture()
        f.run()
        f.records.insert(
            0,
            {
                **f.records[0],
                "id": 99,
                "context": "api-scan/pr-3",
                "description": "complete:" + REVISION,
                "state": "success",
            },
        )
        with patch.object(f.gate, "write_report"):
            self.assertTrue(
                f.gate.evaluate(f.gate.bind(copy.deepcopy(f.pr)), copy.deepcopy(BREAK))
            )

    def test_healthy_earlier_evidence_does_not_override_current_scan_error(self):
        f = Fixture(scan_state="error")
        f.records.append({**f.records[0], "id": 0, "state": "success"})
        f.run()
        self.assertEqual(f.reactions(), ["eyes"])
        self.assertEqual(f.statuses(), [])

    def test_partial_acceptance_retry_cannot_override_newer_revoke(self):
        f = Fixture()
        f.api.routes[("GET", ROOT + "/pulls")] = GitHubError(503, "unavailable")
        f.run()
        f.api.routes[("GET", ROOT + "/pulls")] = [f.pr]
        f.records.append(
            self.exception(11, description="revoke:11:" + REVISION, state="failure")
        )
        previous = len(f.statuses())
        f.run()
        self.assertEqual(len(f.statuses()), previous)
        self.assertNotIn("+1", f.reactions())
