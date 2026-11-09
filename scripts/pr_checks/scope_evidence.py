"""Budget complete evidence objects before either scope-selection request.

The public Actions handler and internal ci-bot deploy identical copies separately.
"""

import json
from pathlib import PurePosixPath

CONFIGURATION_PATHS = (
    "pytest.ini",
    "pyproject.toml",
    "tests/conftest.py",
    "scripts/task_run_unit_tests.sh",
)


class EvidenceContext:
    """Account for complete evidence and a separate bounded omission map."""

    def __init__(self, pr, snapshot, changed, inventory, candidates, ci_files, limit):
        self.limit = limit
        self.diagnostic_limit = min(4096, limit // 10)
        self.content_limit = limit - self.diagnostic_limit
        self.context = {
            "pr": {
                "title": (pr.get("title") or "")[:1000],
                "body": (pr.get("body") or "")[:8000],
                "head_sha": snapshot["head_sha"],
                "merged_snapshot": snapshot,
            },
            "changed_files": [],
            "inventory_at_merged_tree": [],
            "candidate_test_files": [],
            "configuration_files": [
                path for path in CONFIGURATION_PATHS if path in inventory
            ],
            "source_files": {},
            "ci_files": ci_files,
            "unavailable_evidence": {},
            "omitted_evidence": {
                "pr_metadata": len(pr.get("title") or "") > 1000
                or len(pr.get("body") or "") > 8000,
                "changed_files": len(changed),
                "patches": sum(bool(item.get("patch")) for item in changed),
                "inventory_at_merged_tree": len(inventory) - len(candidates),
                "candidate_test_files": len(candidates),
                "unavailable_paths": 0,
            },
        }
        self.chars = len(json.dumps(self.context))
        if self.chars > self.content_limit:
            raise ValueError(
                f"Mandatory CI evidence exceeds the context budget ({self.chars}/{limit} characters)"
            )

        # Reserve a third of the request for complete configuration and test bodies.
        initial_limit = self.content_limit - min(100_000, limit // 3)
        for item in changed:
            record = {
                key: item.get(key)
                for key in ("filename", "status", "previous_filename")
            }
            record["patch"] = None
            if item.get("patch"):
                record["patch_omitted"] = True
            if self._append(self.context["changed_files"], record, initial_limit):
                self._count("changed_files", -1)

        changed_paths = {item["filename"] for item in changed}
        changed_dirs = {
            str(parent)
            for path in changed_paths
            for parent in PurePosixPath(path).parents
            if str(parent) != "."
        }

        def priority(path):
            return (
                path not in changed_paths,
                str(PurePosixPath(path).parent) not in changed_dirs,
                path,
            )

        # Prefer changed tests and their neighbours over an exhaustive global index.
        # Keep some room for patches even when the repository inventory is enormous.
        manifest_limit = max(self.chars, initial_limit - min(40_000, limit // 8))
        candidate_limit = self.chars + (manifest_limit - self.chars) * 2 // 3
        for path in sorted(candidates, key=priority):
            if self._append(
                self.context["candidate_test_files"], path, candidate_limit
            ):
                self._count("candidate_test_files", -1)
        candidate_set = set(candidates)
        for path in sorted(
            (path for path in inventory if path not in candidate_set), key=priority
        ):
            if self._append(
                self.context["inventory_at_merged_tree"], path, manifest_limit
            ):
                self._count("inventory_at_merged_tree", -1)

        patches = {item["filename"]: item.get("patch") for item in changed}
        for record in self.context["changed_files"]:
            patch = patches[record["filename"]]
            # Retain complete patches only. Omitted patches can be replaced by
            # complete source evidence in the model's file-request phase.
            if (
                patch
                and len(patch) <= 8000
                and self._set(record, "patch", patch, initial_limit)
            ):
                self._remove(record, "patch_omitted")
                self._count("patches", -1)

    def _append(self, items, value, limit):
        delta = len(json.dumps(value)) + (2 if items else 0)
        if self.chars + delta > limit:
            return False
        items.append(value)
        self.chars += delta
        return True

    def _set(self, mapping, key, value, limit):
        if key in mapping:
            delta = len(json.dumps(value)) - len(json.dumps(mapping[key]))
        else:
            delta = len(json.dumps({key: value})) - 2 + (2 if mapping else 0)
        if self.chars + delta > limit:
            return False
        mapping[key] = value
        self.chars += delta
        return True

    def _remove(self, mapping, key):
        value = mapping.pop(key)
        self.chars -= len(json.dumps({key: value})) - 2 + (2 if mapping else 0)

    def _count(self, key, change):
        counts = self.context["omitted_evidence"]
        self._set(counts, key, counts[key] + change, self.limit)

    def unavailable(self, path, reason):
        """Bound per-path diagnostics separately; count entries that cannot fit."""
        diagnostic_ceiling = (
            self.chars + self.diagnostic_limit - self._diagnostic_chars()
        )
        if not self._set(
            self.context["unavailable_evidence"],
            path,
            reason,
            min(diagnostic_ceiling, self.limit - 128),
        ):
            self._count("unavailable_paths", 1)

    def _diagnostic_chars(self):
        return len(json.dumps(self.context["unavailable_evidence"])) - 2

    def add_source(self, path, text, reserve=0):
        """Reserve space for later bodies without charging omission records to it."""
        unavailable = self.context["unavailable_evidence"]
        if path in unavailable:
            self._remove(unavailable, path)
        # Keep headroom for omission counters even when requested bodies fill the budget.
        if not self._set(
            self.context["source_files"],
            path,
            text,
            min(
                self.content_limit + self._diagnostic_chars() - reserve,
                self.limit - 128,
            ),
        ):
            self.unavailable(path, "context budget")

    def diagnostics(self):
        return {
            "chars": len(json.dumps(self.context)),
            "limit": self.limit,
            "sections": {
                key: len(json.dumps(value)) for key, value in self.context.items()
            },
            "omitted": self.context["omitted_evidence"],
            "unavailable_files": len(self.context["unavailable_evidence"]),
        }
