#!/usr/bin/env python3
"""Print the basenames of test files carrying a given pytest marker.

Used by ``scripts/test_utils.sh`` to derive its CI scheduling buckets
(``@pytest.mark.long_running`` -> front-load in the parallel queue,
``@pytest.mark.solo`` -> run alone/sequentially) directly from the test
sources. Keeping the property on the test file means it travels with the file
across renames/splits instead of being duplicated in a shell array that can
silently drift out of sync (see issue #3762, where a rename left the shell
array pointing at a file that no longer existed, dropping a heavy MoE file from
the front of the queue and causing GB200 walltime timeouts).

Implementation note: this is a pure-``ast`` scan -- it never imports the test
modules. That keeps it fast and, crucially, immune to a single module that
fails to import (a torch/CUDA import error in one test must not break CI
scheduling for every other test).

Usage:
    find_marked_tests.py MARKER [ROOT ...]

MARKER is the marker attribute name, e.g. ``long_running`` or ``solo``.
ROOT defaults to ``tests``. Output is one unique basename per line, sorted.
"""

import ast
import os
import sys


def _file_has_marker(path: str, marker: str) -> bool:
    """Return True if ``path`` references ``pytest.mark.<marker>`` anywhere.

    Matches any ``<obj>.mark.<marker>`` attribute access, which covers the
    common forms without importing the module:
      * module-level ``pytestmark = pytest.mark.<marker>``
      * module-level ``pytestmark = [pytest.mark.<marker>, ...]``
      * per-test/-class ``@pytest.mark.<marker>`` decorators
      * an aliased import (``import pytest as pt`` -> ``pt.mark.<marker>``)
    A file is treated as marked if *any* item in it carries the marker, which
    is the right granularity for whole-file CI scheduling.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except Exception:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == marker
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
        ):
            return True
    return False


def _is_test_file(name: str) -> bool:
    return name.startswith("test_") and name.endswith(".py")


def find_marked(marker: str, roots) -> list:
    hits = set()
    for root in roots:
        if os.path.isfile(root):
            base = os.path.basename(root)
            if _is_test_file(base) and _file_has_marker(root, marker):
                hits.add(base)
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if _is_test_file(name) and _file_has_marker(
                    os.path.join(dirpath, name), marker
                ):
                    hits.add(name)
    return sorted(hits)


def main(argv) -> int:
    if len(argv) < 2:
        sys.stderr.write("usage: find_marked_tests.py MARKER [ROOT ...]\n")
        return 2
    marker = argv[1]
    roots = argv[2:] or ["tests"]
    for name in find_marked(marker, roots):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
