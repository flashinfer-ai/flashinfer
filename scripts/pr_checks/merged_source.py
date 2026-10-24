"""Construct and inspect a pinned merge without running repository code."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

# All mutable source preparation stays here; change this location if needed.
MERGE_WORK_DIRECTORY = Path(tempfile.gettempdir())


class MergedSource:
    def __init__(self, repository, base_sha, head_sha, reference=None, timeout=240):
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repository
        ):
            raise ValueError("Invalid source repository")
        if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (base_sha, head_sha)):
            raise ValueError("Invalid source commit")
        self.directory = tempfile.TemporaryDirectory(
            prefix="ci-bot-merge-", dir=MERGE_WORK_DIRECTORY
        )
        self.root = Path(self.directory.name)
        self.deadline = time.monotonic() + timeout
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
        # PR attributes cannot activate configured filters, merge drivers or hooks.
        self.env.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_TERMINAL_PROMPT="0",
            GIT_LFS_SKIP_SMUDGE="1",
            GIT_AUTHOR_NAME="FlashInfer CI",
            GIT_AUTHOR_EMAIL="ci@example.invalid",
            GIT_COMMITTER_NAME="FlashInfer CI",
            GIT_COMMITTER_EMAIL="ci@example.invalid",
            GIT_AUTHOR_DATE="2000-01-01T00:00:00+0000",
            GIT_COMMITTER_DATE="2000-01-01T00:00:00+0000",
        )
        try:
            self.git("init", "--quiet")
            # Reuse only immutable objects, never the reference's config or worktree.
            if reference:
                reference = Path(reference).resolve()
                if reference.is_dir():
                    shallow = subprocess.run(
                        [
                            "git",
                            "-C",
                            str(reference),
                            "rev-parse",
                            "--is-shallow-repository",
                        ],
                        env=self.env,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    # Borrowed commit objects do not prove their ancestry is present.
                    # A shallow cache must not be advertised as complete history.
                    if shallow.returncode == 0 and shallow.stdout.strip() == "true":
                        reference = None
                if reference is not None and reference.is_dir():
                    objects = subprocess.run(
                        [
                            "git",
                            "-C",
                            str(reference),
                            "rev-parse",
                            "--git-path",
                            "objects",
                        ],
                        env=self.env,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if objects.returncode == 0:
                        path = Path(objects.stdout.strip())
                        if not path.is_absolute():
                            path = reference / path
                        if path.is_dir():
                            (self.root / ".git/objects/info/alternates").write_text(
                                str(path.resolve()) + "\n"
                            )
            self.git("remote", "add", "origin", f"https://github.com/{repository}.git")
            for sha in (base_sha, head_sha):
                if self.git(
                    "cat-file", "-e", sha + "^{commit}", check=False
                ).returncode:
                    self.git("fetch", "--no-tags", "--filter=blob:none", "origin", sha)
            self.git("checkout", "--detach", base_sha)
            self.git("merge", "--no-ff", "--no-commit", "--no-edit", head_sha)
            self.tree_sha = self.git("write-tree").stdout.strip()
            self.merge_sha = self.git(
                "commit-tree",
                self.tree_sha,
                "-p",
                base_sha,
                "-p",
                head_sha,
                input="AI-selected merge snapshot\n",
            ).stdout.strip()
            self.commit = {
                "sha": self.merge_sha,
                "tree": {"sha": self.tree_sha},
                "parents": [{"sha": base_sha}, {"sha": head_sha}],
            }
        except BaseException:
            self.close()
            raise

    def git(self, *args, check=True, input=None):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Merged source preparation timed out")
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "rerere.enabled=false",
                "-c",
                "submodule.recurse=false",
                "-c",
                "protocol.ext.allow=never",
                *args,
            ],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            input=input,
            timeout=min(remaining, 90),
        )
        if check and result.returncode:
            # Git diagnostics can contain remote URLs; callers get a bounded public error.
            raise ValueError(
                "Could not prepare pinned merged source (fetch failure or merge conflict)"
            )
        return result

    def tree(self, sha):
        if sha != self.tree_sha:
            raise ValueError("Unexpected merged tree")
        entries = []
        for record in self.git("ls-tree", "-r", "-l", "-z", sha).stdout.split("\0"):
            if not record:
                continue
            metadata, path = record.split("\t", 1)
            mode, kind, blob, size = metadata.split()
            entries.append(
                {
                    "path": path,
                    "mode": mode,
                    "type": kind,
                    "sha": blob,
                    "size": int(size) if size != "-" else 0,
                }
            )
        return {"truncated": False, "tree": entries}

    def blob(self, sha, max_bytes):
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("Invalid source blob")
        if int(self.git("cat-file", "-s", sha).stdout) > max_bytes:
            raise ValueError("Repository blob exceeds the evidence size limit")
        return self.git("cat-file", "blob", sha).stdout

    def close(self):
        self.directory.cleanup()
