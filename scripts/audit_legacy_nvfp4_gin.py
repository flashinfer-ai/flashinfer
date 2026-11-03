# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Reproduce the fixed-source inventory without importing GPU dependencies.

Run from the receiving checkout: python3 scripts/audit_legacy_nvfp4_gin.py
The vendor and MR objects must already exist in this checkout's object store.
This reads Git objects, writes evidence, and never applies a source diff.
"""

import ast
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

F_BASE = "e74a1677ab95a1b97aa1c6b6a80b3cf5c0ebac34"
VENDOR = "fa662f345980f75776ae0c85d2e5c188918ded26"
MR_BASE = "41a45039cf81c150f969907dffdd0ba2f10661ca"
MR = "f863d7540614bd546a42867c21adbf51d573061a"
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe"
OUT = ROOT / "docs/design_docs/legacy_nvfp4_gin/evidence"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def source(rev, path):
    exists = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{rev}:{path}"],
        capture_output=True,
    )
    return git("show", f"{rev}:{path}") if exists.returncode == 0 else None


def destination(path):
    return f"{PACKAGE}/src/" + path.replace("pre_unification/comm/", "src/").replace(
        "pre_unification/moe_nvfp4_swapab/", "moe_nvfp4_swapab/"
    )


def vendor_path(path):
    return path.replace("pre_unification/comm/", "src/").replace(
        "pre_unification/moe_nvfp4_swapab/", "moe_nvfp4_swapab/"
    )


def normalized(text):
    if text is None:
        return ""
    return text.replace("pre_unification.comm", "src").replace(
        "pre_unification.moe_nvfp4_swapab", "moe_nvfp4_swapab"
    )


def imports(text, path):
    """Enumerate static imports, including function-local and relative imports."""
    result = set()
    package = path.removesuffix(".py").split("/")[:-1]
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = package[: len(package) - node.level + 1]
                module = ".".join(prefix + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            result.add(module)
            result.update(f"{module}.{alias.name}" for alias in node.names)
    return sorted(result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_paths = git("ls-tree", "-r", "--name-only", MR).splitlines()
    modules = {
        p.removesuffix(".py").replace("/", ".").removesuffix(".__init__"): p
        for p in all_paths
        if p.endswith(".py")
    }
    changed = git(
        "diff",
        "--name-only",
        MR_BASE,
        MR,
        "--",
        "pre_unification/comm",
        "pre_unification/moe_nvfp4_swapab",
        "common",
    ).splitlines()
    rows = []
    for path in changed:
        donor = source(MR, path)
        target = destination(path)
        versions = {
            "vendor": source(VENDOR, vendor_path(path)),
            "mr_base": source(MR_BASE, path),
            "donor": donor,
            "fi_base": source(F_BASE, target),
        }
        row = {"donor_path": path, "target_path": target}
        row["sha256"] = {
            name: hashlib.sha256(text.encode()).hexdigest()
            if text is not None
            else None
            for name, text in versions.items()
        }
        row["fi_equals_vendor_after_import_normalization"] = (
            versions["fi_base"] is not None
            and versions["vendor"] is not None
            and normalized(versions["fi_base"]) == normalized(versions["vendor"])
        )
        row["imports"] = imports(donor, path) if donor else []
        rows.append(row)
        # Pairwise comparisons preserve FI deltas and upstream preconditions
        # separately. They do not pretend MR_BASE is the vendor baseline.
        for old, new in (
            ("vendor", "fi_base"),
            ("vendor", "mr_base"),
            ("mr_base", "donor"),
        ):
            patch = "".join(
                difflib.unified_diff(
                    normalized(versions[old]).splitlines(True),
                    normalized(versions[new]).splitlines(True),
                    fromfile=f"{old}/{path}",
                    tofile=f"{new}/{path}",
                )
            )
            directory = OUT / "comparisons" / f"{old}-to-{new}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / (path.replace("/", "__") + ".patch")).write_text(patch)

    # Conservative dependency closure: include imports in diagnostic branches.
    pending = ["pre_unification/moe_nvfp4_swapab/megamoe_kernel.py"]
    closure = {}
    external = set()
    while pending:
        path = pending.pop()
        if path in closure:
            continue
        names = imports(source(MR, path), path)
        children = sorted({modules[n] for n in names if n in modules})
        closure[path] = children
        pending.extend(children)
        external.update(
            n.split(".")[0]
            for n in names
            if n.split(".")[0] not in {"common", "pre_unification"}
        )
    abi = {}
    for name, rev, path in (
        ("fi", F_BASE, f"{PACKAGE}/src/moe_nvfp4_swapab/megamoe_kernel.py"),
        ("donor", MR, "pre_unification/moe_nvfp4_swapab/megamoe_kernel.py"),
    ):
        abi[name] = {}
        for cls in ast.parse(source(rev, path)).body:
            if isinstance(cls, ast.ClassDef) and cls.name == "Sm100MegaMoEKernel":
                for fn in cls.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name in (
                        "__init__",
                        "__call__",
                    ):
                        abi[name][fn.name] = [
                            a.arg
                            for a in (
                                *fn.args.posonlyargs,
                                *fn.args.args,
                                *fn.args.kwonlyargs,
                            )
                        ]
    report = {
        "fi_base": F_BASE,
        "vendor": VENDOR,
        "mr_base": MR_BASE,
        "donor": MR,
        "vendor_mr_merge_base": git("merge-base", VENDOR, MR).strip(),
        "normalization": [
            "pre_unification.comm -> src",
            "pre_unification.moe_nvfp4_swapab -> moe_nvfp4_swapab",
        ],
        "files": rows,
        "static_kernel_import_closure": closure,
        "external_import_roots": sorted(external),
        "kernel_abi": abi,
        "limitations": "Static imports are a conservative starting point; symbol hooks, dynamic imports and runtime layouts still need semantic audit.",
    }
    (OUT / "source-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{len(rows)} changed candidates; {len(closure)} modules in static kernel closure"
    )


if __name__ == "__main__":
    main()
