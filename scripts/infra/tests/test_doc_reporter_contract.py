"""Replay the trusted legacy reporter against both artifact contracts."""

import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[3] / ".github/workflows/pr-api-doc-comment.yml"
)
RUNNER = r"""
const {AsyncFunction} = {AsyncFunction:Object.getPrototypeOf(async function(){}).constructor};
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const calls=[];
const github={paginate:async()=>input.comments,rest:{issues:{
  createComment:async x=>calls.push({kind:'create',...x}),
  updateComment:async x=>calls.push({kind:'update',...x}),
  deleteComment:async x=>calls.push({kind:'delete',...x})
}}};
new AsyncFunction('require','github','core','context',input.script)(
  require,github,{info(){},warning(){}},{repo:{owner:'org',repo:'repo'}}
).then(()=>process.stdout.write(JSON.stringify(calls))).catch(e=>{process.stderr.write(e.stack);process.exitCode=1;});
"""


class ReporterContractTest(unittest.TestCase):
    def run_reporter(self, files, previous=False):
        step = WORKFLOW.read_text().split(
            "      - name: Update PR documentation summary\n", 1
        )[1]
        code = textwrap.dedent(
            step.split("          script: |\n", 1)[1].split("\n      - name:", 1)[0]
        )
        comments = (
            [
                {
                    "id": 314,
                    "user": {"login": "github-actions[bot]"},
                    "body": "<!-- flashinfer-pr-documentation-summary -->\nreports/public-api.json is missing",
                }
            ]
            if previous
            else []
        )
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory) / "reports"
            reports.mkdir()
            for name, data in files.items():
                (reports / name).write_text(
                    data if isinstance(data, str) else json.dumps(data)
                )
            result = subprocess.run(
                ["node", "-e", RUNNER],
                cwd=directory,
                input=json.dumps({"script": code, "comments": comments}),
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PR_NUMBER": "3",
                    "RUN_URL": "https://github.com/org/repo/actions/runs/1",
                },
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def doc_only(self, **extra):
        return {
            "execution-status.json": {"static_documentation": "success"},
            "static-documentation.json": {"findings": []},
            **extra,
        }

    def test_doc_only_clean_report_is_silent(self):
        self.assertEqual(self.run_reporter(self.doc_only()), [])

    def test_doc_only_clean_report_removes_stale_false_warning(self):
        calls = self.run_reporter(self.doc_only(), previous=True)
        self.assertEqual(
            [(c["kind"], c.get("comment_id")) for c in calls], [("delete", 314)]
        )

    def test_doc_findings_are_still_reported_without_api_requirement(self):
        files = self.doc_only(
            **{
                "static-documentation.json": {
                    "findings": [
                        {
                            "check": "args_consistency",
                            "path": "flashinfer/example.py",
                            "line": 4,
                            "message": "document y",
                        }
                    ]
                }
            }
        )
        body = self.run_reporter(files)[0]["body"]
        self.assertIn("document y", body)
        self.assertNotIn("public-api.json is missing", body)

    def test_legacy_api_findings_are_preserved(self):
        files = self.doc_only(
            **{
                "execution-status.json": {
                    "public_api": "success",
                    "static_documentation": "success",
                },
                "public-api.json": {
                    "findings": [
                        {
                            "path": "flashinfer/example.py",
                            "line": 4,
                            "message": "required parameter removed",
                        }
                    ]
                },
            }
        )
        body = self.run_reporter(files)[0]["body"]
        self.assertIn("POTENTIAL API BREAK", body)
        self.assertIn("required parameter removed", body)

    def test_legacy_missing_api_is_still_incomplete(self):
        files = self.doc_only(
            **{
                "execution-status.json": {
                    "public_api": "success",
                    "static_documentation": "success",
                }
            }
        )
        self.assertIn("public-api.json is missing", self.run_reporter(files)[0]["body"])

    def test_doc_only_failure_is_not_mistaken_for_a_clean_result(self):
        files = self.doc_only(
            **{"execution-status.json": {"static_documentation": "failure"}}
        )
        body = self.run_reporter(files)[0]["body"]
        self.assertIn("static_documentation did not complete", body)
        self.assertNotIn("public-api.json is missing", body)

    def test_missing_or_malformed_status_keeps_reports_required(self):
        for status in (None, "{", "null", "[]"):
            files = {"static-documentation.json": {"findings": []}}
            if status is not None:
                files["execution-status.json"] = status
            body = self.run_reporter(files)[0]["body"]
            self.assertIn("execution-status.json is", body)
            self.assertIn("public-api.json is missing", body)

    def test_doc_report_stays_required(self):
        body = self.run_reporter(
            {"execution-status.json": {"static_documentation": "success"}}
        )[0]["body"]
        self.assertIn("static-documentation.json is missing", body)
        self.assertNotIn("public-api.json is missing", body)
