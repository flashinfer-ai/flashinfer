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

from pr_checks.git_compare import resolve_merge_base
from pr_checks.inspect_sources import is_decorated_with
from pr_checks.write_reports import PrFinding, emit_finding, write_report


@dataclass(frozen=True)
class ApiParameter:
    name: str
    annotation: str | None
    default: str | None


@dataclass(frozen=True)
class ApiFunction:
    qualified_name: str
    module: str
    path: str
    line: int
    signature: str
    docstring: str
    is_async: bool
    positional_only: tuple[ApiParameter, ...]
    positional_or_keyword: tuple[ApiParameter, ...]
    vararg: ApiParameter | None
    keyword_only: tuple[ApiParameter, ...]
    kwarg: ApiParameter | None
    return_annotation: str | None


@dataclass(frozen=True)
class ChangedFile:
    status: str
    old_path: str | None
    new_path: str | None


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


def parameter(arg: ast.arg, default: ast.expr | None = None) -> ApiParameter:
    return ApiParameter(
        name=arg.arg,
        annotation=ast.unparse(arg.annotation) if arg.annotation else None,
        default=ast.unparse(default) if default is not None else None,
    )


def parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[
    tuple[ApiParameter, ...],
    tuple[ApiParameter, ...],
    ApiParameter | None,
    tuple[ApiParameter, ...],
    ApiParameter | None,
]:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    positional_defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)
    positional_parameters = tuple(
        parameter(arg, default)
        for arg, default in zip(positional, positional_defaults, strict=True)
    )
    posonly_count = len(args.posonlyargs)
    keyword_only = tuple(
        parameter(arg, default)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    )
    return (
        positional_parameters[:posonly_count],
        positional_parameters[posonly_count:],
        parameter(args.vararg) if args.vararg else None,
        keyword_only,
        parameter(args.kwarg) if args.kwarg else None,
    )


def is_none_annotation(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def typing_annotation_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
    ):
        return node.attr
    return None


def optional_annotation_payload(node: ast.expr) -> ast.expr | None:
    if isinstance(node, ast.Subscript):
        annotation_name = typing_annotation_name(node.value)
        if annotation_name == "Optional":
            return node.slice
        if annotation_name == "Union" and isinstance(node.slice, ast.Tuple):
            elements = node.slice.elts
            if len(elements) == 2:
                if is_none_annotation(elements[0]):
                    return elements[1]
                if is_none_annotation(elements[1]):
                    return elements[0]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        if is_none_annotation(node.left):
            return node.right
        if is_none_annotation(node.right):
            return node.left
    return None


def is_compatible_annotation(before: str | None, after: str | None) -> bool:
    if before == after:
        return True
    if before is None or after is None:
        return False
    try:
        before_node = ast.parse(before, mode="eval").body
        after_node = ast.parse(after, mode="eval").body
    except SyntaxError:
        return False
    payload = optional_annotation_payload(after_node)
    return payload is not None and ast.dump(before_node) == ast.dump(payload)


def is_compatible_parameter(before: ApiParameter, after: ApiParameter) -> bool:
    return (
        before.name == after.name
        and is_compatible_annotation(before.annotation, after.annotation)
        and (before.default is None or before.default == after.default)
    )


def is_compatible_signature_extension(before: ApiFunction, after: ApiFunction) -> bool:
    """Return whether *after* is a narrowly compatible input widening.

    Existing ``*args``/``**kwargs`` APIs are kept conservative: a new named
    parameter can consume arguments that the old implementation forwarded.
    """
    if (
        before.is_async != after.is_async
        or before.return_annotation != after.return_annotation
        or before.vararg != after.vararg
        or before.kwarg != after.kwarg
        or before.vararg is not None
        or before.kwarg is not None
    ):
        return False
    if len(before.positional_only) != len(after.positional_only) or not all(
        is_compatible_parameter(old, new)
        for old, new in zip(
            before.positional_only,
            after.positional_only,
            strict=True,
        )
    ):
        return False

    old_positional = before.positional_or_keyword
    new_positional = after.positional_or_keyword
    if len(new_positional) < len(old_positional) or not all(
        is_compatible_parameter(old, new)
        for old, new in zip(
            old_positional,
            new_positional[: len(old_positional)],
            strict=True,
        )
    ):
        return False
    if any(item.default is None for item in new_positional[len(old_positional) :]):
        return False

    old_keyword_only = {item.name: item for item in before.keyword_only}
    new_keyword_only = {item.name: item for item in after.keyword_only}
    if not old_keyword_only.keys() <= new_keyword_only.keys():
        return False
    if any(
        not is_compatible_parameter(item, new_keyword_only[name])
        for name, item in old_keyword_only.items()
    ):
        return False
    return all(
        item.default is not None
        for name, item in new_keyword_only.items()
        if name not in old_keyword_only
    )


def is_compatible_api(before: ApiFunction, after: ApiFunction) -> bool:
    return before.signature == after.signature or is_compatible_signature_extension(
        before, after
    )


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
                    (
                        positional_only,
                        positional_or_keyword,
                        vararg,
                        keyword_only,
                        kwarg,
                    ) = parameters(child)
                    result[name] = ApiFunction(
                        qualified_name=name,
                        module=module,
                        path=path,
                        line=child.lineno,
                        signature=signature(child),
                        docstring=ast.get_docstring(child, clean=False) or "",
                        is_async=isinstance(child, ast.AsyncFunctionDef),
                        positional_only=positional_only,
                        positional_or_keyword=positional_or_keyword,
                        vararg=vararg,
                        keyword_only=keyword_only,
                        kwarg=kwarg,
                        return_annotation=(
                            ast.unparse(child.returns) if child.returns else None
                        ),
                    )
                visit(child, class_prefix)

    visit(tree)
    return result


class ModuleScopeImportFromVisitor(ast.NodeVisitor):
    """Collect module-scope imports that can be visible at runtime."""

    def __init__(self) -> None:
        self.imports: list[ast.ImportFrom] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append(node)

    def visit_If(self, node: ast.If) -> None:
        is_type_checking = (
            isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        )
        is_main_guard = (
            isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"
        )
        if is_type_checking or is_main_guard:
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def module_scope_imports(tree: ast.AST) -> list[ast.ImportFrom]:
    visitor = ModuleScopeImportFromVisitor()
    visitor.visit(tree)
    return visitor.imports


def module_name_from_path(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_import_module(node: ast.ImportFrom, path: str) -> str | None:
    """Resolve a relative or absolute import to a FlashInfer module."""
    if node.level == 0:
        if not node.module or not (
            node.module == "flashinfer" or node.module.startswith("flashinfer.")
        ):
            return None
        return node.module

    current_module = module_name_from_path(path)
    current_package = (
        current_module
        if path.endswith("/__init__.py")
        else current_module.rpartition(".")[0]
    )
    package_parts = current_package.split(".") if current_package else []
    ascend = node.level - 1
    if ascend > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - ascend]
    if node.module:
        base_parts.extend(node.module.split("."))
    module = ".".join(base_parts)
    if not (module == "flashinfer" or module.startswith("flashinfer.")):
        return None
    return module


ReexportTarget = tuple[str, str]


def module_reexports(
    path: str, source: str | None
) -> dict[str, tuple[ReexportTarget, ...]]:
    """Return public aliases and every direct target imported for each name."""
    if source is None:
        return {}
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return {}

    result: dict[str, set[ReexportTarget]] = {}
    for node in module_scope_imports(tree):
        module = resolve_import_module(node, path)
        if not module:
            continue
        for alias in node.names:
            exported_name = alias.asname or alias.name
            if exported_name != "*":
                result.setdefault(exported_name, set()).add((module, alias.name))
    return {
        exported_name: tuple(sorted(targets))
        for exported_name, targets in result.items()
    }


def resolve_reexported_api(
    qualified_name: str,
    reexports: dict[str, tuple[ReexportTarget, ...]],
) -> ReexportTarget | None:
    """Resolve an exact API or one direct ``ExportedClass.member`` alias."""
    exported_name, separator, member_name = qualified_name.partition(".")
    if separator and (not member_name or "." in member_name):
        return None
    targets = reexports.get(exported_name, ())
    if len(targets) != 1:
        return None
    target_module, target_name = targets[0]
    if member_name:
        target_name = f"{target_name}.{member_name}"
    return target_module, target_name


def module_apis(
    revision: str,
    module: str,
    cache: dict[str, dict[str, ApiFunction]],
) -> dict[str, ApiFunction]:
    if module in cache:
        return cache[module]

    module_path = module.replace(".", "/")
    for candidate in (f"{module_path}.py", f"{module_path}/__init__.py"):
        source = git_file(revision, candidate)
        if source is not None:
            cache[module] = extract_public_apis(candidate, source)
            return cache[module]
    cache[module] = {}
    return cache[module]


def exported_names(source: str | None) -> dict[str, str]:
    if source is None:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    exports: dict[str, str] = {}
    for node in module_scope_imports(tree):
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


def changed_files(base: str, head: str) -> list[ChangedFile]:
    """Return add/modify/delete/rename records without losing either rename path."""
    fields = git("diff", "--name-status", "-z", "-M", f"{base}..{head}").split("\0")
    if fields and not fields[-1]:
        fields.pop()

    result: list[ChangedFile] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if status.startswith(("R", "C")):
            old_path, new_path = fields[index : index + 2]
            index += 2
        else:
            path = fields[index]
            index += 1
            old_path = None if status == "A" else path
            new_path = None if status == "D" else path
        result.append(ChangedFile(status, old_path, new_path))
    return result


def public_module(path: str | None, *, has_decorated_api: bool = False) -> str | None:
    if path is None:
        return None
    parts = list(PurePosixPath(path).parts)
    if not path.endswith(".py") or not parts or parts[0] != "flashinfer":
        return None

    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts.pop()
    if len(parts) < 2 or any(part.startswith("_") for part in parts[1:]):
        return None

    # Top-level modules and packages are established import paths.  For deeper
    # modules, require direct evidence that the old file defines public APIs so
    # internal implementation-file renames do not generate noise.
    if len(parts) > 2 and not has_decorated_api:
        return None
    return ".".join(parts)


def check(base: str, head: str) -> list[PrFinding]:
    base = resolve_merge_base(base, head)
    changes = changed_files(base, head)
    findings: list[PrFinding] = []
    target_cache: dict[str, dict[str, ApiFunction]] = {}
    api_changes: dict[
        ChangedFile, tuple[dict[str, ApiFunction], dict[str, ApiFunction]]
    ] = {}

    python_changes = sorted(
        (
            change
            for change in changes
            if any(
                path and path.startswith("flashinfer/") and path.endswith(".py")
                for path in (change.old_path, change.new_path)
            )
        ),
        key=lambda change: (change.new_path or change.old_path or ""),
    )
    for change in python_changes:
        old_path, new_path = change.old_path, change.new_path
        old = (
            extract_public_apis(old_path or "", git_file(base, old_path))
            if old_path
            else {}
        )
        new = (
            extract_public_apis(new_path or "", git_file(head, new_path))
            if new_path
            else {}
        )
        api_changes[change] = (old, new)

        reexports = (
            module_reexports(old_path, git_file(head, old_path)) if old_path else {}
        )
        for name in sorted(set(old) - set(new)):
            api = old[name]
            target = resolve_reexported_api(name, reexports)
            if target:
                target_module, target_name = target
                target_api = module_apis(head, target_module, target_cache).get(
                    target_name
                )
                if target_api:
                    if is_compatible_api(api, target_api):
                        continue
                    findings.append(
                        PrFinding(
                            "public_api_signature_changed",
                            target_api.path,
                            target_api.line,
                            f"Public API `{api.module}.{name}` remains re-exported from "
                            f"`{target_api.module}`, but its signature changed in a potentially breaking way. "
                            f"Before: `{api.signature}`; after: `{target_api.signature}`.",
                        )
                    )
                    continue
            findings.append(
                PrFinding(
                    "public_api_removed",
                    old_path or new_path or "flashinfer",
                    api.line,
                    f"Public API `{api.module}.{name}` was removed; update deprecation and API documentation.",
                )
            )

        for name in sorted(set(old) & set(new)):
            before, after = old[name], new[name]
            if is_compatible_api(before, after):
                continue
            findings.append(
                PrFinding(
                    "public_api_signature_changed",
                    after.path,
                    after.line,
                    f"Public API `{after.module}.{name}` signature changed in a potentially breaking way. "
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
    for change in changes:
        old_apis, new_apis = api_changes.get(change, ({}, {}))
        old_module = public_module(change.old_path, has_decorated_api=bool(old_apis))
        if not old_module:
            continue
        if change.status == "D":
            findings.append(
                PrFinding(
                    "public_module_deleted",
                    change.old_path or "flashinfer",
                    1,
                    f"Public Python submodule `{old_module}` was deleted.",
                )
            )
        elif change.status.startswith("R"):
            new_module = public_module(
                change.new_path, has_decorated_api=bool(new_apis)
            )
            findings.append(
                PrFinding(
                    "public_module_moved",
                    change.new_path or change.old_path or "flashinfer",
                    1,
                    f"Public Python submodule `{old_module}` moved to `{new_module or change.new_path}`; preserve the old import path or document the breaking move.",
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

    base = resolve_merge_base(args.base, args.head)
    findings = check(base, args.head)
    print(f"Public API documentation check: {len(findings)} finding(s)")
    for finding in findings:
        emit_finding(finding, args.github_actions)
    if not findings:
        print("No public API/documentation drift introduced by this PR.")
    if args.report_json:
        write_report(args.report_json, "public_api", base, args.head, findings)
    return 1 if args.strict and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
