"""Tests for scripts/guardword_hooks.py on throwaway repositories.

The scanner checkout is a stand-in: its flashinfer_hooks.py records how it was called
and reports findings when the staged changes, the pushed commits or the pushed
ref names contain CANARY. The real runner is tested in the scanner repository.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "guardword_hooks.py"

FAKE_RUNNER = textwrap.dedent(
    """\
    import os
    import subprocess
    import sys

    args = sys.argv[1:]
    log = os.environ["FAKE_LOG"]
    with open(log + ".runner", "a") as stream:
        stream.write(" ".join(args) + "\\n")
    command = args[0] if args else ""
    if os.environ.get("FAKE_RUNNER_EXIT"):
        sys.exit(int(os.environ["FAKE_RUNNER_EXIT"]))
    if command == "check":
        report = os.environ.get("FAKE_RUNNER_CHECK", "")
        print(report)
        sys.exit(1 if report.startswith("problem") else 0)

    def git(*command):
        return subprocess.run(["git", *command], capture_output=True, text=True).stdout

    if command == "pre-commit":
        text = git("diff", "--cached")
    elif command == "pre-push":
        if "github" not in args[2].lower():
            sys.exit(0)
        pushed = sys.stdin.read()
        with open(log + ".pushed", "a") as stream:
            stream.write(pushed)
        text = pushed + "".join(
            git("log", "-p", fields[1], "--not", "--remotes")
            for fields in (line.split() for line in pushed.splitlines())
            if fields[1].strip("0")
        )
    else:
        sys.exit(0)
    sys.exit(1 if "CANARY" in text else 0)
    """
)

SCANNER_PRE_PUSH = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    # FlashInfer Guardwords Pre-Push Hook (test double)
    exit 0
    """
)

RECORDING_PRE_PUSH = '#!/bin/sh\ncat > "$FAKE_LOG.$(basename "$0")"\n'

PRE_COMMIT_CONFIG = textwrap.dedent(
    """\
    repos:
      - repo: local
        hooks:
          - id: formatter
            name: formatter
            entry: sh -c 'touch "$FAKE_LOG.formatter"'
            language: system
            pass_filenames: false
            always_run: true
          - id: guardwords
            name: guardwords
            entry: python3 scripts/guardword_hooks.py check-commit
            language: system
            pass_filenames: false
            always_run: true
    """
)

needs_pre_commit = unittest.skipUnless(
    shutil.which("pre-commit") or importlib.util.find_spec("pre_commit"),
    "pre-commit is not installed",
)


class GuardwordHooksTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.home = self.tmp / "home"
        self.home.mkdir()
        (self.home / ".gitconfig").write_text(
            "[user]\n\tname = dev\n\temail = dev@example.com\n"
        )
        self.cache = self.home / "scanner-cache"
        self.scanner = self.tmp / "scanner"
        # Set as FLASHINFER_SCAN_REPO_URL; "" leaves the location unconfigured.
        self.scan_url = str(self.scanner)
        self.log = self.tmp / "log"
        # Detection must not reach GitHub from a test.
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self._write_hook(self.bin / "gh", "#!/bin/sh\nexit 1\n")

        self.scanner.mkdir()
        self._git(self.scanner, "init", "-q", "-b", "main")
        (self.scanner / "flashinfer_scan.py").write_text("print('stand-in')\n")
        (self.scanner / "flashinfer_hooks.py").write_text(FAKE_RUNNER)
        self._commit_all(self.scanner, "scanner")

        self.github = self._bare(self.tmp / "github" / "flashinfer.git")
        self.work = self.tmp / "work"
        (self.work / "scripts").mkdir(parents=True)
        shutil.copy(SCRIPT, self.work / "scripts" / SCRIPT.name)
        self._git(self.work, "init", "-q", "-b", "main")
        self._git(self.work, "config", "flashinfer.internal", "true")
        self._git(self.work, "remote", "add", "origin", str(self.github))
        (self.work / "README").write_text("hello\n")
        self._commit_all(self.work, "initial")
        self.hooks = self.work / ".git" / "hooks"

    def _env(self, **extra: str) -> dict[str, str]:
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("GIT_", "FLASHINFER_SCAN_", "PRE_COMMIT"))
            and k not in ("CI", "SKIP")
        }
        env.update(
            PATH=f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
            GIT_CONFIG_GLOBAL=str(self.home / ".gitconfig"),
            GIT_CONFIG_NOSYSTEM="1",
            FLASHINFER_SCAN_CACHE_DIR=str(self.cache),
            PRE_COMMIT_HOME=str(self.home / "pre-commit"),
            FAKE_LOG=str(self.log),
        )
        if self.scan_url:
            env["FLASHINFER_SCAN_REPO_URL"] = self.scan_url
        env.update(extra)
        return env

    def _git(
        self, cwd: Path, *args: str, check: bool = True, **env: str
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=self._env(**env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=check,
        )

    def _commit_all(self, cwd: Path, message: str) -> None:
        self._git(cwd, "add", "-A")
        self._git(cwd, "commit", "-q", "--no-verify", "-m", message)

    def _hooks(
        self, *args: str, cwd: Path | None = None, stdin: str = "", **env: str
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=cwd or self.work,
            env=self._env(**env),
            input=stdin,
            capture_output=True,
            text=True,
        )

    def _bare(self, path: Path) -> Path:
        path.mkdir(parents=True)
        self._git(path, "init", "-q", "--bare")
        return path

    def _rev(self, ref: str, cwd: Path | None = None) -> str:
        return self._git(cwd or self.work, "rev-parse", ref).stdout.strip()

    def _global_config(self, key: str) -> str:
        return self._git(
            self.work, "config", "--global", "--get", key, check=False
        ).stdout.strip()

    def _push(
        self, remote: str = "origin", ref: str = "HEAD:refs/heads/main", **env: str
    ) -> subprocess.CompletedProcess:
        return self._git(self.work, "push", "-q", remote, ref, check=False, **env)

    def _commit(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return self._git(self.work, "commit", "-q", *args, check=False, **env)

    def _add(self, name: str, text: str) -> None:
        (self.work / name).write_text(text)
        self._git(self.work, "add", name)

    def _runner_calls(self) -> list[list[str]]:
        log = Path(f"{self.log}.runner")
        return [line.split() for line in log.read_text().splitlines()]

    def _write_hook(self, path: Path, script: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script)
        path.chmod(0o755)

    def _pre_commit(self, *args: str) -> None:
        command = shutil.which("pre-commit")
        base = [command] if command else [sys.executable, "-m", "pre_commit"]
        subprocess.run(
            [*base, *args],
            cwd=self.work,
            env=self._env(),
            capture_output=True,
            check=True,
        )

    def _use_pre_commit_config(self) -> None:
        (self.work / ".pre-commit-config.yaml").write_text(PRE_COMMIT_CONFIG)
        self._commit_all(self.work, "config")

    def _set_up(self) -> None:
        result = self._hooks("setup", "--non-interactive")
        self.assertEqual(result.returncode, 0, result.stderr)

    def _is_shim(self, path: Path) -> bool:
        return "flashinfer-guardword-shim" in path.read_text()

    # -- detection and installation ------------------------------------------

    def test_external_contributor_gets_nothing(self) -> None:
        self._git(self.work, "config", "flashinfer.internal", "false")
        for command in ("ensure", "check-commit"):
            result = self._hooks(command)
            self.assertEqual((result.returncode, result.stderr), (0, ""))
        self.assertFalse((self.hooks / "pre-push").exists())
        self.assertFalse(self.cache.exists())

    def test_source_tree_without_git_gets_nothing(self) -> None:
        tree = self.tmp / "unpacked"
        tree.mkdir()
        for command in ("ensure", "check-commit"):
            result = self._hooks(command, cwd=tree)
            self.assertEqual((result.returncode, result.stderr), (0, ""), command)
        status = self._hooks("status", cwd=tree)
        self.assertEqual(status.returncode, 1)
        self.assertIn("not inside a git checkout", status.stderr)

    def test_an_opt_out_is_kept_and_reported(self) -> None:
        self._git(self.work, "config", "--unset", "flashinfer.internal")
        self._git(self.work, "config", "--global", "flashinfer.internal", "false")
        setup = self._hooks("setup", "--non-interactive")
        self.assertEqual(setup.returncode, 1)
        self.assertIn("git config --global --unset flashinfer.internal", setup.stderr)
        self.assertFalse((self.hooks / "pre-push").exists())
        status = self._hooks("status")
        self.assertIn("opted out", status.stderr)
        self.assertNotIn("Fix:", status.stderr)

        # Opting one clone back in leaves the other clones opted out.
        self._git(self.work, "config", "flashinfer.internal", "true")
        self.assertEqual(self._hooks("setup", "--non-interactive").returncode, 0)
        self.assertEqual(self._global_config("flashinfer.internal"), "false")

    def test_ci_gets_nothing(self) -> None:
        result = self._hooks("ensure", CI="true")
        self.assertEqual(result.stderr, "")
        self.assertFalse((self.hooks / "pre-push").exists())

    def test_github_org_membership_is_recorded_and_absence_rechecked_daily(
        self,
    ) -> None:
        self._git(self.work, "config", "--unset", "flashinfer.internal")
        self.scan_url = ""
        calls = self.tmp / "gh-calls"
        self._write_hook(self.bin / "gh", f'#!/bin/sh\necho >> "{calls}"\nexit 1\n')
        self.assertEqual(self._hooks("ensure").stderr, "")
        self.assertEqual(self._hooks("ensure").stderr, "")
        self.assertEqual(len(calls.read_text().splitlines()), 1)

        (self.work / ".git" / "flashinfer-guardwords-checked").unlink()
        self._write_hook(self.bin / "gh", "#!/bin/sh\necho active\n")
        result = self._hooks("ensure")
        self.assertIn("member of the NVIDIA GitHub org", result.stderr)
        # Kept where the opt-out goes, so a later gh logout changes nothing.
        self.assertEqual(self._global_config("flashinfer.internal"), "true")

    def test_scanner_checkout_or_location_means_nvidia(self) -> None:
        self._git(self.work, "config", "--unset", "flashinfer.internal")
        location = self._hooks("ensure")
        self.assertIn("FLASHINFER_SCAN_REPO_URL is set", location.stderr)

        self.scan_url = ""
        other = self.tmp / "other"
        self._git(self.tmp, "clone", "-q", str(self.work), str(other))
        # The devcontainers mount an empty directory there for everyone.
        self.cache.mkdir()
        self.assertEqual(self._hooks("ensure", cwd=other).stderr, "")
        self.cache.rmdir()
        self._git(self.tmp, "clone", "-q", str(self.scanner), str(self.cache))
        checkout = self._hooks("ensure", cwd=other)
        self.assertIn("guardword scanner checkout on this machine", checkout.stderr)

    def test_ensure_installs_the_push_hook_once_without_cloning(self) -> None:
        first = self._hooks("ensure")
        self.assertIn("installed guardword checks", first.stderr)
        self.assertIn("Guardword scan unavailable", first.stderr)
        self.assertTrue(self._is_shim(self.hooks / "pre-push"))
        self.assertFalse(self.cache.exists())
        self._set_up()
        self.assertEqual(self._hooks("ensure").stderr, "")

    def test_an_existing_push_hook_is_chained(self) -> None:
        self._set_up()
        self._write_hook(self.hooks / "pre-push.chained", RECORDING_PRE_PUSH)
        self.assertEqual(self._push().returncode, 0)
        self.assertIn(
            "refs/heads/main", Path(f"{self.log}.pre-push.chained").read_text()
        )

    def test_an_existing_push_hook_moves_aside(self) -> None:
        self._write_hook(self.hooks / "pre-push", RECORDING_PRE_PUSH)
        result = self._hooks("ensure")
        self.assertIn("pre-push.chained", result.stderr)
        self.assertTrue(self._is_shim(self.hooks / "pre-push"))
        self.assertEqual(
            (self.hooks / "pre-push.chained").read_text(), RECORDING_PRE_PUSH
        )

    def test_installed_scanner_hook_is_replaced(self) -> None:
        self._write_hook(self.hooks / "pre-push", SCANNER_PRE_PUSH)
        self._hooks("ensure")
        self.assertTrue(self._is_shim(self.hooks / "pre-push"))
        self.assertFalse((self.hooks / "pre-push.chained").exists())

    def test_hooks_path_is_left_alone_and_its_line_works(self) -> None:
        custom = self.tmp / "custom-hooks"
        custom.mkdir()
        self._git(self.work, "config", "core.hooksPath", str(custom))
        setup = self._hooks("setup", "--non-interactive")
        self.assertEqual(setup.returncode, 1)
        advice = setup.stderr
        self.assertIn("core.hooksPath", advice)
        self.assertEqual(list(custom.iterdir()), [])

        line = next(
            line.partition("] ")[2].strip()
            for line in advice.splitlines()
            if "git-common-dir" in line
        )
        self._write_hook(custom / "pre-push", f"#!/bin/sh\n{line}\n")
        self.assertEqual(self._hooks("ensure").stderr, "")
        self.assertEqual(self._hooks("status").returncode, 0)
        self._add("leak.txt", "CANARY\n")
        self._commit_all(self.work, "leak")
        self.assertNotEqual(self._push().returncode, 0)

        # A shared hooks directory also serves repositories without the copy.
        other = self.tmp / "other"
        other.mkdir()
        self._git(other, "init", "-q", "-b", "main")
        self._git(other, "config", "core.hooksPath", str(custom))
        (other / "f").write_text("CANARY\n")
        self._commit_all(other, "other")
        self._git(other, "remote", "add", "origin", str(self.github))
        push = self._git(
            other, "push", "-q", "origin", "HEAD:refs/heads/o", check=False
        )
        self.assertEqual(push.returncode, 0, push.stderr)

    # -- pushes ---------------------------------------------------------------

    def test_pushes_go_through_the_runner(self) -> None:
        self._set_up()
        self.assertEqual(self._push().returncode, 0)
        self._add("leak.txt", "CANARY\n")
        self._commit_all(self.work, "leak")
        push = self._push()
        self.assertNotEqual(push.returncode, 0)
        self.assertIn("Push blocked by the guardword scan", push.stderr)
        self.assertEqual(
            self._runner_calls()[-1], ["pre-push", "origin", str(self.github)]
        )
        self.assertIn(
            f"HEAD {self._rev('HEAD')} refs/heads/main",
            Path(f"{self.log}.pushed").read_text(),
        )

    def test_a_failing_runner_blocks_pushes(self) -> None:
        self._set_up()
        push = self._push(FAKE_RUNNER_EXIT="2")
        self.assertNotEqual(push.returncode, 0)
        self.assertIn("the scan could not run", push.stderr)

    def test_missing_scanner_blocks_github_pushes_but_not_commits(self) -> None:
        self.scan_url = ""
        result = self._hooks("ensure")
        self.assertIn("scanner location is not configured", result.stderr)
        # Shown wherever someone recognized by mistake keeps ending up.
        opt_out = "git config --global flashinfer.internal false"
        self.assertIn(opt_out, self._hooks("ensure").stderr)
        self.assertIn(opt_out, self._hooks("status").stderr)
        self._add("clean.txt", "nothing to see\n")
        commit = self._hooks("check-commit")
        self.assertEqual(commit.returncode, 0, commit.stderr)
        self.assertIn("pushes to GitHub stay blocked", commit.stderr)
        internal = self._bare(self.tmp / "internal" / "flashinfer.git")
        self._git(self.work, "remote", "add", "internal", str(internal))
        self.assertEqual(self._push("internal").returncode, 0)
        push = self._push()
        self.assertNotEqual(push.returncode, 0)
        self.assertIn("Push blocked until", push.stderr)
        self.assertIn(opt_out, push.stderr)

    def test_github_is_recognized_by_case_and_ssh_alias_before_setup(self) -> None:
        self.scan_url = ""
        self._hooks("ensure")
        upper = self._bare(self.tmp / "GitHub" / "flashinfer.git")
        self._git(self.work, "remote", "add", "upper", str(upper))
        self.assertNotEqual(self._push("upper").returncode, 0)
        self._write_hook(
            self.bin / "ssh",
            '#!/bin/sh\n[ "$1" = -G ] && [ "$2" = work ] && echo "hostname github.com"\n',
        )
        pushed = f"refs/heads/main {self._rev('HEAD')} refs/heads/main {'0' * 40}\n"
        alias = self._hooks("pre-push", "work", "work:flashinfer.git", stdin=pushed)
        self.assertNotEqual(alias.returncode, 0)
        other = self._hooks("pre-push", "lab", "lab:flashinfer.git", stdin=pushed)
        self.assertEqual(other.returncode, 0, other.stderr)

    def test_first_push_from_a_worktree_clones_the_scanner_cleanly(self) -> None:
        self._hooks("ensure")
        worktree = self.tmp / "worktree"
        self._git(self.work, "worktree", "add", "-q", "-b", "topic", str(worktree))
        (worktree / "topic.txt").write_text("topic\n")
        self._git(worktree, "add", "topic.txt")
        self._git(worktree, "commit", "-q", "--no-verify", "-m", "topic")
        push = self._git(
            worktree, "push", "-q", "origin", "HEAD:refs/heads/topic", check=False
        )
        self.assertEqual(push.returncode, 0, push.stderr)
        self.assertTrue((self.cache / "flashinfer_hooks.py").is_file())
        self.assertEqual(self._git(worktree, "status", "--porcelain").stdout, "")
        self.assertEqual(self._git(self.cache, "status", "--porcelain").stdout, "")

    @needs_pre_commit
    def test_push_hook_runs_behind_the_frameworks_push_hook(self) -> None:
        self._use_pre_commit_config()
        self._set_up()
        self._pre_commit("install", "--hook-type", "pre-push")
        # The framework moved the push hook to pre-push.legacy, which it runs
        # first; that is where it stays.
        self.assertTrue(self._is_shim(self.hooks / "pre-push.legacy"))
        self.assertEqual(self._hooks("ensure").stderr, "")
        self.assertEqual(self._hooks("status").returncode, 0)
        self.assertEqual(self._push().returncode, 0)
        self._add("leak.txt", "CANARY\n")
        self._commit_all(self.work, "leak")
        self.assertNotEqual(self._push().returncode, 0)

    # -- commits --------------------------------------------------------------

    def test_commit_check(self) -> None:
        self._set_up()
        self._add("clean.txt", "nothing to see\n")
        self.assertEqual(self._hooks("check-commit").returncode, 0)
        self.assertEqual(self._runner_calls()[-1], ["pre-commit"])

        self._add("leak.txt", "CANARY\n")
        blocked = self._hooks("check-commit")
        self.assertEqual(blocked.returncode, 1)
        self.assertIn("Commit blocked", blocked.stderr)
        self.assertEqual(self._hooks("check-commit", CI="true").returncode, 0)

        self._git(self.work, "config", "flashinfer.guardwords.scanCommits", "false")
        self.assertEqual(self._hooks("check-commit").returncode, 0)

    @needs_pre_commit
    def test_commits_are_scanned_by_the_framework_next_to_the_formatters(
        self,
    ) -> None:
        self._use_pre_commit_config()
        setup = self._hooks("setup", "--non-interactive")
        self.assertIn("the formatters and the guardword scan", setup.stderr)
        self.assertIn(
            "File generated by pre-commit", (self.hooks / "pre-commit").read_text()
        )
        # `commit -a` hands the hooks a temporary index, which has to reach
        # the runner.
        (self.work / "README").write_text("CANARY\n")
        leak = self._commit("-a", "-m", "leak")
        self.assertNotEqual(leak.returncode, 0)
        self.assertIn("Commit blocked", leak.stdout + leak.stderr)
        self.assertTrue(Path(f"{self.log}.formatter").exists())
        skipped = self._commit("-a", "-m", "leak", SKIP="guardwords")
        self.assertEqual(skipped.returncode, 0, skipped.stdout + skipped.stderr)

    @needs_pre_commit
    def test_pre_commit_uninstall_sticks(self) -> None:
        self._use_pre_commit_config()
        self._hooks("ensure")
        self.assertTrue((self.hooks / "pre-commit").exists())
        self._pre_commit("uninstall")
        self.assertNotIn("installed guardword checks", self._hooks("ensure").stderr)
        self.assertFalse((self.hooks / "pre-commit").exists())

    # -- status and setup -----------------------------------------------------

    def test_status_reports_what_is_missing(self) -> None:
        missing = self._hooks("status")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("pre-push hook is not installed", missing.stderr)
        self.assertIn("scanner is not cloned", missing.stderr)

        self._set_up()
        ready = self._hooks("status")
        self.assertEqual(ready.returncode, 0, ready.stderr)
        self.assertIn("Guardword checks are active", ready.stderr)
        self.assertIn("predates the commit check", ready.stderr)

    @needs_pre_commit
    def test_status_notes_when_commits_are_not_scanned(self) -> None:
        self._use_pre_commit_config()
        self._set_up()
        self.assertNotIn("Note", self._hooks("status").stderr)
        self._pre_commit("uninstall")
        self.assertIn("pre-commit install", self._hooks("status").stderr)
        self._git(self.work, "config", "flashinfer.guardwords.scanCommits", "false")
        self.assertIn("scanCommits is false", self._hooks("status").stderr)

    def test_status_includes_the_runner_report(self) -> None:
        self._set_up()
        note = self._hooks("status", FAKE_RUNNER_CHECK="note: the rules are old")
        self.assertEqual(note.returncode, 0, note.stderr)
        self.assertIn("Note: the rules are old", note.stderr)
        problem = self._hooks("status", FAKE_RUNNER_CHECK="problem: it does not run")
        self.assertEqual(problem.returncode, 1)
        self.assertIn("it does not run", problem.stderr)

    def test_setup_without_a_terminal_points_to_non_interactive(self) -> None:
        result = self._hooks("setup")
        self.assertEqual(result.returncode, 1)
        self.assertIn("--non-interactive", result.stderr)

    def test_non_interactive_setup_for_agents(self) -> None:
        self._git(self.work, "config", "--unset", "flashinfer.internal")
        self.scan_url = ""
        self.assertEqual(self._hooks("status").returncode, 1)

        no_location = self._hooks("setup", "--non-interactive")
        self.assertEqual(no_location.returncode, 1)
        self.assertIn("FLASHINFER_SCAN_REPO_URL", no_location.stderr)
        # Incomplete, but GitHub pushes are blocked until it is finished.
        self.assertTrue(self._is_shim(self.hooks / "pre-push"))
        self.assertNotEqual(self._push().returncode, 0)

        bad = self._hooks(
            "setup",
            "--non-interactive",
            FLASHINFER_SCAN_REPO_URL=str(self.tmp / "missing"),
        )
        self.assertEqual(bad.returncode, 1)
        self.assertIn("cloning it needs the right location", bad.stderr)

        ready = self._hooks(
            "setup", "--non-interactive", FLASHINFER_SCAN_REPO_URL=str(self.scanner)
        )
        self.assertEqual(ready.returncode, 0, ready.stderr)
        self.assertIn("Guardword checks are active", ready.stderr)
        self.assertEqual(self._global_config("flashinfer.internal"), "true")
        self._add("leak.txt", "CANARY\n")
        self._commit_all(self.work, "leak")
        self.assertNotEqual(self._push().returncode, 0)

    def test_setup_installs_hooks_even_when_ci_is_set(self) -> None:
        result = self._hooks("setup", "--non-interactive", CI="true")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self._is_shim(self.hooks / "pre-push"))

    def test_setup_reinstalls_hooks_from_an_existing_checkout(self) -> None:
        # As after a devcontainer rebuild: the checkout survives in its volume,
        # and no location is configured anywhere.
        self._git(self.tmp, "clone", "-q", str(self.scanner), str(self.cache))
        self.scan_url = ""
        self._set_up()
        self.assertTrue(self._is_shim(self.hooks / "pre-push"))

    def test_scanner_is_cloned_into_an_empty_mounted_directory(self) -> None:
        self.cache.mkdir()
        self._set_up()
        self.assertTrue((self.cache / "flashinfer_hooks.py").is_file())

    def test_a_checkout_that_predates_the_runner_is_updated(self) -> None:
        self._git(self.scanner, "rm", "-q", "flashinfer_hooks.py")
        self._commit_all(self.scanner, "before the runner")
        self._git(self.tmp, "clone", "-q", str(self.scanner), str(self.cache))
        self.assertIn("predates the hook runner", self._hooks("ensure").stderr)
        (self.scanner / "flashinfer_hooks.py").write_text(FAKE_RUNNER)
        self._commit_all(self.scanner, "runner")
        self.assertEqual(self._push().returncode, 0)
        self.assertTrue((self.cache / "flashinfer_hooks.py").is_file())
        self.assertEqual(self._hooks("status").returncode, 0)


if __name__ == "__main__":
    unittest.main()
