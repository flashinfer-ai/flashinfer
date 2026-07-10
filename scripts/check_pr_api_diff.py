#!/usr/bin/env python3
"""Check PR public-API changes for accompanying documentation updates.

The checker is deliberately dependency-free.  It reuses the public-API model
used by the release API diff tooling: a public callable is a Python function
decorated with ``@flashinfer_api``.  Unlike a release comparison, this is
scoped to one pull request's base and head commits.

By default findings are GitHub Actions warnings so the check is safe to roll
out without blocking contributors.  ``--strict`` makes findings fail the job.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pr_checks.inspect_sources import is_decorated_with
from pr_checks.write_reports import PrFinding, emit_finding, write_report


@dataclass(frozen=True)
class ApiFunction:
    qualified_name: str
    module: str
    path: str
    line: int
    signature: str
    docstring: str


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, errors="replace", stderr=subprocess.DEVNULL
    )


def git_file(rev: str, path: str) -> str | None:
    try:
        return git("show", f"{rev}:{path}")
    except subprocess.CalledProcessError:
        return None


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"


def extract_public_apis(path: str, source: str | None) -> dict[str, ApiFunction]:
    if source is None:
        return {}
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return {}

    module_parts = list(PurePosixPath(path).with_suffix("").parts)
    if module_parts and module_parts[-1] == "__init__":
        module_parts.pop()
    module = ".".join(module_parts)
    result: dict[str, ApiFunction] = {}

    def visit(parent: ast.AST, class_prefix: str = "") -> None:
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{class_prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if is_decorated_with(child, "flashinfer_api"):
                    name = f"{class_prefix}{child.name}"
                    result[name] = ApiFunction(
                        qualified_name=name,
                        module=module,
                        path=path,
                        line=child.lineno,
                        signature=signature(child),
                        docstring=ast.get_docstring(child, clean=False) or "",
                    )
                visit(child, class_prefix)

    visit(tree)
    return result


def exported_names(source: str | None) -> dict[str, str]:
    if source is None:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    exports: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level > 0:
            module = ".".join(
                part for part in ("flashinfer", node.module or "") if part
            )
        elif node.module and (
            node.module == "flashinfer" or node.module.startswith("flashinfer.")
        ):
            module = node.module
        else:
            continue
        if module:
            for alias in node.names:
                if alias.name != "*":
                    exports[alias.asname or alias.name] = module
    return exports


def changed_paths(base: str, head: str) -> set[str]:
    return {p for p in git("diff", "--name-only", f"{base}...{head}").splitlines() if p}


def changed_docs_contain(
    base: str, head: str, paths: set[str], api: ApiFunction
) -> bool:
    """Return whether a relevant documentation source changed in this PR."""
    needles = (api.qualified_name, api.qualified_name.rsplit(".", 1)[-1], api.module)
    for path in paths:
        if not (path.startswith("docs/") or path in {"README.md", "CONTRIBUTING.md"}):
            continue
        content = git_file(head, path)
        if content and any(needle in content for needle in needles):
            return True
    return False


def check(base: str, head: str) -> list[PrFinding]:
    paths = changed_paths(base, head)
    findings: list[PrFinding] = []

    for path in sorted(
        p for p in paths if p.startswith("flashinfer/") and p.endswith(".py")
    ):
        old = extract_public_apis(path, git_file(base, path))
        new = extract_public_apis(path, git_file(head, path))

        for name in sorted(set(old) - set(new)):
            api = old[name]
            findings.append(
                PrFinding(
                    "public_api_removed",
                    path,
                    api.line,
                    f"Public API `{api.module}.{name}` was removed; update deprecation and API documentation.",
                )
            )

        for name in sorted(set(old) & set(new)):
            before, after = old[name], new[name]
            if before.signature == after.signature:
                continue
            docs_changed = before.docstring != after.docstring or changed_docs_contain(
                base, head, paths, after
            )
            if not docs_changed:
                findings.append(
                    PrFinding(
                        "public_api_signature_changed",
                        after.path,
                        after.line,
                        f"Public API `{after.module}.{name}` signature changed without an updated docstring or relevant documentation file. "
                        f"Before: `{before.signature}`; after: `{after.signature}`.",
                    )
                )

    old_exports = exported_names(git_file(base, "flashinfer/__init__.py"))
    new_exports = exported_names(git_file(head, "flashinfer/__init__.py"))
    for name in sorted(set(old_exports) - set(new_exports)):
        findings.append(
            PrFinding(
                "public_export_removed",
                "flashinfer/__init__.py",
                1,
                f"Public top-level export `{name}` was removed.",
                level="error",
            )
        )
    for name in sorted(set(old_exports) & set(new_exports)):
        if old_exports[name] != new_exports[name]:
            findings.append(
                PrFinding(
                    "public_export_moved",
                    "flashinfer/__init__.py",
                    1,
                    f"Public top-level export `{name}` moved from `{old_exports[name]}` to `{new_exports[name]}`; "
                    "confirm compatibility for direct submodule imports.",
                )
            )
    for path in git(
        "diff", "--diff-filter=D", "--name-only", f"{base}...{head}"
    ).splitlines():
        parts = PurePosixPath(path).parts
        if (
            len(parts) == 2
            and parts[0] == "flashinfer"
            and path.endswith(".py")
            and not parts[1].startswith("_")
        ):
            module = path[:-3].replace("/", ".")
        elif (
            len(parts) == 3
            and parts[0] == "flashinfer"
            and parts[2] == "__init__.py"
            and not parts[1].startswith("_")
        ):
            module = f"flashinfer.{parts[1]}"
        else:
            continue
        findings.append(
            PrFinding(
                "public_module_deleted",
                path,
                1,
                f"Public Python submodule `{module}` was deleted.",
            )
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base commit SHA")
    parser.add_argument("--head", required=True, help="Head commit SHA")
    parser.add_argument(
        "--github-actions", action="store_true", help="Emit GitHub workflow annotations"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Return non-zero when findings exist"
    )
    parser.add_argument("--report-json", type=Path, help="Write findings as JSON")
    args = parser.parse_args()

    findings = check(args.base, args.head)
    print(f"Public API documentation check: {len(findings)} finding(s)")
    for finding in findings:
        emit_finding(finding, args.github_actions)
    if not findings:
        print("No public API/documentation drift introduced by this PR.")
    if args.report_json:
        write_report(args.report_json, "public_api", args.base, args.head, findings)
    return 1 if args.strict and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
