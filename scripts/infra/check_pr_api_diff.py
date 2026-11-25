#!/usr/bin/env python3
"""Check PR public-API changes for accompanying documentation updates.

The checker is deliberately dependency-free.  It reuses the public-API model
used by the release API diff tooling: a public callable is a Python function
decorated with ``@flashinfer_api``.  Unlike a release comparison, this is
scoped to one pull request's base and head commits.

The experimental track is out of scope: it carries no compatibility guarantees,
so it is not part of the surface this check defends (see ``EXPERIMENTAL_PACKAGE``).

Findings carry BREAKING/error, LIKELY_BREAKING/warning, or NON_BREAKING/notice.
``--fail-on error`` fails on blocking changes; legacy ``--strict`` fails on any
alert. Git/parse failures return 2 and cannot be treated as a clean scan.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
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
    type_aliases: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ChangedFile:
    status: str
    old_path: str | None
    new_path: str | None


# Experimental track, excluded from the public surface on two independent axes:
#
# - path — everything under ``flashinfer/experimental/`` is an experimental
#   backend by construction (see ``flashinfer/experimental/README.md``);
# - decorator — experimental APIs live in core modules but are marked
#   ``@flashinfer_experimental_api``, which ``is_decorated_with(..., "flashinfer_api")``
#   deliberately does not match (suffix matching requires a dot before the name).
#
# Only the path axis needs code here; the decorator axis falls out of the
# literal name.  Excluding an experimental *file* rather than the whole change
# keeps a core -> experimental move visible: the API disappears from the stable
# side and is still reported as a removal, while experimental -> core (a
# graduation) reports nothing.
EXPERIMENTAL_PACKAGE = ("flashinfer", "experimental")


def is_experimental_path(path: str | None) -> bool:
    """True for files in the experimental track, which has no API guarantees."""
    if path is None:
        return False
    return tuple(PurePosixPath(path).parts[: len(EXPERIMENTAL_PACKAGE)]) == (
        EXPERIMENTAL_PACKAGE
    )


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, errors="replace", stderr=subprocess.DEVNULL
    )


def git_file(rev: str, path: str) -> str | None:
    # Distinguish an absent path from an unreadable git object/revision.
    if not git("ls-tree", rev, "--", path).strip():
        return None
    return git("show", f"{rev}:{path}")


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


def literal_annotation_values(node: ast.expr) -> frozenset[str] | None:
    """Return the member dumps of a ``Literal[...]`` annotation, else ``None``."""
    if not isinstance(node, ast.Subscript):
        return None
    if typing_annotation_name(node.value) != "Literal":
        return None
    members = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
    return frozenset(ast.dump(member) for member in members)


def annotation_aliases(api: ApiFunction) -> dict[str, str]:
    """Return the module-local aliases visible to an extracted API."""
    return dict(api.type_aliases)


def expand_annotation_alias(
    node: ast.expr,
    aliases: dict[str, str],
    seen: frozenset[str] = frozenset(),
) -> ast.expr:
    """Expand a module-local type alias used as a complete annotation."""
    if not isinstance(node, ast.Name) or node.id not in aliases or node.id in seen:
        return node
    try:
        expanded = ast.parse(aliases[node.id], mode="eval").body
    except SyntaxError:
        return node
    return expand_annotation_alias(expanded, aliases, seen | {node.id})


def union_annotation_members(
    node: ast.expr, aliases: dict[str, str]
) -> tuple[ast.expr, ...] | None:
    """Flatten one supported union annotation into its member expressions."""
    node = expand_annotation_alias(node, aliases)
    if isinstance(node, ast.Subscript):
        annotation_name = typing_annotation_name(node.value)
        if annotation_name == "Optional":
            return (node.slice, ast.Constant(value=None))
        if annotation_name == "Union":
            elements = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
            )
            return tuple(elements)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = union_annotation_members(node.left, aliases) or (node.left,)
        right = union_annotation_members(node.right, aliases) or (node.right,)
        return (*left, *right)
    return None


def subscript_arguments(node: ast.Subscript) -> tuple[ast.expr, ...]:
    """Return a subscript's arguments as a tuple for uniform comparison."""
    return node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)


def is_tuple_annotation(node: ast.expr) -> bool:
    """Return whether *node* is a built-in or typing tuple annotation."""
    return isinstance(node, ast.Subscript) and (
        typing_annotation_name(node.value) == "Tuple"
        or (isinstance(node.value, ast.Name) and node.value.id == "tuple")
    )


def annotation_accepts(
    before: ast.expr,
    after: ast.expr,
    before_aliases: dict[str, str],
    after_aliases: dict[str, str],
) -> bool:
    """Return whether every value described by *before* is accepted by *after*."""
    before = expand_annotation_alias(before, before_aliases)
    after = expand_annotation_alias(after, after_aliases)
    if ast.dump(before) == ast.dump(after) and (
        before_aliases == after_aliases
        or not any(
            isinstance(node, ast.Name)
            and node.id in (before_aliases.keys() | after_aliases.keys())
            for node in ast.walk(before)
        )
    ):
        return True

    before_members = union_annotation_members(before, before_aliases)
    after_members = union_annotation_members(after, after_aliases)
    if before_members is not None or after_members is not None:
        old_members = before_members or (before,)
        new_members = after_members or (after,)
        return all(
            any(
                annotation_accepts(
                    old_member,
                    new_member,
                    before_aliases,
                    after_aliases,
                )
                for new_member in new_members
            )
            for old_member in old_members
        )

    before_literal = literal_annotation_values(before)
    after_literal = literal_annotation_values(after)
    if before_literal is not None or after_literal is not None:
        return (
            before_literal is not None
            and after_literal is not None
            and before_literal <= after_literal
        )

    if is_tuple_annotation(before) and is_tuple_annotation(after):
        assert isinstance(before, ast.Subscript)
        assert isinstance(after, ast.Subscript)
        old_items = subscript_arguments(before)
        new_items = subscript_arguments(after)
        if (
            len(new_items) == 2
            and isinstance(new_items[1], ast.Constant)
            and new_items[1].value is Ellipsis
        ):
            if (
                len(old_items) == 2
                and isinstance(old_items[1], ast.Constant)
                and old_items[1].value is Ellipsis
            ):
                old_items = old_items[:1]
            return all(
                annotation_accepts(item, new_items[0], before_aliases, after_aliases)
                for item in old_items
            )
        return len(old_items) == len(new_items) and all(
            annotation_accepts(old, new, before_aliases, after_aliases)
            for old, new in zip(old_items, new_items, strict=True)
        )
    return False


def is_compatible_annotation(
    before: str | None,
    after: str | None,
    before_aliases: dict[str, str] | None = None,
    after_aliases: dict[str, str] | None = None,
) -> bool:
    """Return whether *after* accepts every value admitted by *before*."""
    if before == after and (before is None or before_aliases == after_aliases):
        return True
    if before is None or after is None:
        return False
    try:
        before_node = ast.parse(before, mode="eval").body
        after_node = ast.parse(after, mode="eval").body
    except SyntaxError:
        return False
    return annotation_accepts(
        before_node,
        after_node,
        before_aliases or {},
        after_aliases or {},
    )


def classify_signature(before: ApiFunction, after: ApiFunction) -> PrFinding:
    """Classify structural changes using the explicit optional-binding policy."""
    errors: list[str] = []
    warnings: list[str] = []
    old_aliases, new_aliases = annotation_aliases(before), annotation_aliases(after)

    def indexed(api: ApiFunction) -> dict[str, tuple[ApiParameter, str, int]]:
        result = {}
        for kind, items in (
            ("posonly", api.positional_only),
            ("positional", api.positional_or_keyword),
            ("keyword", api.keyword_only),
        ):
            for index, item in enumerate(items):
                position = index + (
                    len(api.positional_only) if kind == "positional" else 0
                )
                result[item.name] = (item, kind, position)
        return result

    old, new = indexed(before), indexed(after)
    old_pos = (*before.positional_only, *before.positional_or_keyword)
    new_pos = (*after.positional_only, *after.positional_or_keyword)
    renamed_posonly = set()
    for name, (item, kind, position) in old.items():
        target = new.get(name)
        if (
            target is None
            and kind == "posonly"
            and position < len(after.positional_only)
        ):
            candidate = after.positional_only[position]
            if candidate.name not in old:
                target = (candidate, "posonly", position)
                renamed_posonly.add(candidate.name)
        if target is None:
            if after.kwarg is not None or (
                kind == "posonly" and after.vararg is not None
            ):
                warnings.append(
                    f"Compatibility handling for removed parameter `{name}` needs review."
                )
            else:
                errors.append(
                    f"Parameter `{name}` was removed or renamed; old calls passing it no longer bind."
                )
            continue
        updated, new_kind, new_position = target
        if item.default is not None and updated.default is None:
            errors.append(f"Optional parameter `{name}` became required.")
        elif item.default is not None and item.default != updated.default:
            warnings.append(
                f"Default for `{name}` changed from `{item.default}` to `{updated.default}`."
            )
        loses_keyword = kind != "posonly" and new_kind == "posonly"
        loses_position = kind != "keyword" and (
            new_kind == "keyword" or position != new_position
        )
        if loses_keyword or loses_position:
            reason = f"Parameter `{name}` changed its accepted calling convention or positional binding."
            (errors if item.default is None else warnings).append(reason)
        if not is_compatible_annotation(
            item.annotation, updated.annotation, old_aliases, new_aliases
        ):
            warnings.append(
                f"Input annotation for `{name}` changed without a recognized widening."
            )

    for name, (item, _, _) in new.items():
        if name not in old and name not in renamed_posonly and item.default is None:
            errors.append(
                f"Required parameter `{name}` was added; old calls omitting it fail."
            )
    if before.is_async != after.is_async:
        errors.append(
            "The callable changed between synchronous and asynchronous execution."
        )
    for label, previous, current in (
        ("*args", before.vararg, after.vararg),
        ("**kwargs", before.kwarg, after.kwarg),
    ):
        if previous is not None and current is None:
            errors.append(f"Support for `{label}` was removed.")
        elif (
            previous is not None
            and current is not None
            and not is_compatible_annotation(
                previous.annotation, current.annotation, old_aliases, new_aliases
            )
        ):
            warnings.append(
                f"The annotation of `{label}` changed without a recognized widening."
            )
    if before.vararg is not None and len(new_pos) > len(old_pos):
        warnings.append(
            "New positional parameters may consume values previously forwarded through `*args`."
        )
    if before.kwarg is not None and set(new) - set(old):
        warnings.append(
            "New named parameters may consume keywords previously forwarded through `**kwargs`."
        )
    if not is_compatible_annotation(
        before.return_annotation, after.return_annotation, old_aliases, new_aliases
    ):
        warnings.append("Return annotation changed without a recognized widening.")
    classification = (
        "BREAKING" if errors else "LIKELY_BREAKING" if warnings else "NON_BREAKING"
    )
    return PrFinding(
        "public_api_signature_changed",
        after.path,
        after.line,
        " ".join(errors + warnings) or "Recognized compatible signature extension.",
        level={
            "BREAKING": "error",
            "LIKELY_BREAKING": "warning",
            "NON_BREAKING": "notice",
        }[classification],
        classification=classification,
        api=f"{before.module}.{before.qualified_name}",
        before=before.signature,
        after=after.signature,
        rule_id="signature_changed",
    )


def is_compatible_signature_extension(before: ApiFunction, after: ApiFunction) -> bool:
    return classify_signature(before, after).classification == "NON_BREAKING"


def is_compatible_api(before: ApiFunction, after: ApiFunction) -> bool:
    return is_compatible_signature_extension(before, after)


def module_type_aliases(tree: ast.Module) -> tuple[tuple[str, str], ...]:
    """Collect simple module-scope aliases that can appear in annotations."""
    aliases: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            aliases[node.targets[0].id] = ast.unparse(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            aliases[node.target.id] = ast.unparse(node.value)
    return tuple(sorted(aliases.items()))


def extract_public_apis(
    path: str, source: str | None, *, decorated_only: bool = True
) -> dict[str, ApiFunction]:
    if source is None:
        return {}
    tree = ast.parse(source, filename=path)

    module_parts = list(PurePosixPath(path).with_suffix("").parts)
    if module_parts and module_parts[-1] == "__init__":
        module_parts.pop()
    module = ".".join(module_parts)
    result: dict[str, ApiFunction] = {}
    type_aliases = module_type_aliases(tree)

    def visit(parent: ast.AST, class_prefix: str = "") -> None:
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{class_prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not decorated_only or is_decorated_with(child, "flashinfer_api"):
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
                        type_aliases=type_aliases,
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
        ) or (
            isinstance(node.test, ast.Attribute)
            and isinstance(node.test.value, ast.Name)
            and node.test.value.id == "typing"
            and node.test.attr == "TYPE_CHECKING"
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
    tree = ast.parse(source, filename=path)

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
    if is_experimental_path(f"{module_path}.py"):
        # A stable API re-exported from the experimental track is no longer
        # covered by the stable contract; report it as removed rather than as
        # still re-exported.
        cache[module] = {}
        return cache[module]
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
    tree = ast.parse(source)
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
    exports.update(lazy_exported_names(tree))
    return exports


def lazy_exported_names(tree: ast.Module) -> dict[str, str]:
    """Names served by the module-level ``_LAZY_EXPORTS`` table.

    A row ``"name": (".module", "attribute")`` counts like
    ``from .module import attribute as name``; a module-valued row
    ``"name": (".package.module", None)`` counts like
    ``from .package import module as name``. Rows spread in with ``**`` are
    computed at runtime and are not read.
    """
    exports: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not isinstance(value, ast.Dict) or not any(
            isinstance(target, ast.Name) and target.id == "_LAZY_EXPORTS"
            for target in targets
        ):
            continue
        for key, row in zip(value.keys, value.values, strict=True):
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(row, ast.Tuple)
                and len(row.elts) == 2
                and all(isinstance(part, ast.Constant) for part in row.elts)
            ):
                continue
            module, attribute = (part.value for part in row.elts)
            if not isinstance(module, str) or not (
                attribute is None or isinstance(attribute, str)
            ):
                continue
            if module.startswith("."):
                try:
                    module = importlib.util.resolve_name(module, "flashinfer")
                except ImportError:  # "..x" climbs above flashinfer
                    continue
            if module != "flashinfer" and not module.startswith("flashinfer."):
                continue
            if attribute is None:
                module = module.rpartition(".")[0] or module
            exports[key.value] = module
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
    if is_experimental_path(path):
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


def dynamic_namespace(source: str | None) -> bool:
    if source is None:
        return False
    return any(
        (isinstance(node, ast.FunctionDef) and node.name == "__getattr__")
        or (
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "*" for alias in node.names)
        )
        for node in ast.walk(ast.parse(source))
    )


def check(base: str, head: str, *, include_compatible: bool = False) -> list[PrFinding]:
    base = resolve_merge_base(base, head)
    changes = changed_files(base, head)
    findings: list[PrFinding] = []

    def add_signature(before: ApiFunction, after: ApiFunction) -> None:
        finding = classify_signature(before, after)
        if finding.classification != "NON_BREAKING" or include_compatible:
            findings.append(finding)

    def surface(
        check_id: str,
        path: str,
        line: int,
        name: str,
        before: str,
        after: str,
        message: str,
        *,
        uncertain: bool = False,
    ) -> None:
        findings.append(
            PrFinding(
                check_id,
                path,
                line,
                message,
                level="warning" if uncertain else "error",
                classification="LIKELY_BREAKING" if uncertain else "BREAKING",
                api=name,
                before=before,
                after=after,
                rule_id=check_id,
            )
        )

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
            if old_path and not is_experimental_path(old_path)
            else {}
        )
        new = (
            extract_public_apis(new_path or "", git_file(head, new_path))
            if new_path and not is_experimental_path(new_path)
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
                    add_signature(api, target_api)
                    continue
            remaining_source = git_file(head, old_path) if old_path else None
            remaining = extract_public_apis(
                old_path or "",
                remaining_source,
                decorated_only=False,
            ).get(name)
            dynamic = dynamic_namespace(remaining_source)
            if remaining is not None:
                add_signature(api, remaining)

            surface(
                "public_api_removed",
                old_path or new_path or "flashinfer",
                api.line,
                f"{api.module}.{name}",
                api.signature,
                remaining.signature if remaining else "(removed)",
                "Public API decorator was removed; review tracing and registration."
                if remaining
                else "Public API was removed; old imports or calls may fail.",
                uncertain=bool(
                    remaining
                    or target
                    or dynamic
                    or name.partition(".")[0] in reexports
                ),
            )

        for name in sorted(set(old) & set(new)):
            before, after = old[name], new[name]
            if (
                before.signature != after.signature
                or before.type_aliases != after.type_aliases
            ):
                add_signature(before, after)
        if include_compatible:
            for name in sorted(set(new) - set(old)):
                api = new[name]
                findings.append(
                    PrFinding(
                        "public_api_added",
                        api.path,
                        api.line,
                        "New public API.",
                        level="notice",
                        classification="NON_BREAKING",
                        api=f"{api.module}.{name}",
                        before="(absent)",
                        after=api.signature,
                        rule_id="public_api_added",
                    )
                )

    old_exports = exported_names(git_file(base, "flashinfer/__init__.py"))
    new_init = git_file(head, "flashinfer/__init__.py")
    new_exports = exported_names(new_init)
    for name in sorted(set(old_exports) - set(new_exports)):
        surface(
            "public_export_removed",
            "flashinfer/__init__.py",
            1,
            f"flashinfer.{name}",
            f"from flashinfer import {name}",
            "(removed)",
            f"Public top-level export `{name}` was removed."
            if not dynamic_namespace(new_init)
            else f"Top-level export `{name}` requires dynamic namespace review.",
            uncertain=dynamic_namespace(new_init),
        )
    for name in sorted(set(old_exports) & set(new_exports)):
        if old_exports[name] != new_exports[name]:
            # A moved top-level export is compatible if the old module still
            # re-exports a compatible target. Otherwise direct imports need review.
            old_module = old_exports[name]
            old_path = old_module.replace(".", "/") + ".py"
            reexports = module_reexports(old_path, git_file(head, old_path))
            if (
                name not in module_apis(head, old_module, target_cache)
                and name not in reexports
            ):
                surface(
                    "public_export_moved",
                    "flashinfer/__init__.py",
                    1,
                    f"flashinfer.{name}",
                    old_exports[name],
                    new_exports[name],
                    "Top-level export moved; review the old direct submodule import.",
                    uncertain=True,
                )
    for change in changes:
        old_apis, new_apis = api_changes.get(change, ({}, {}))
        old_module = public_module(change.old_path, has_decorated_api=bool(old_apis))
        if not old_module:
            continue
        if change.status == "D":
            surface(
                "public_module_deleted",
                change.old_path or "flashinfer",
                1,
                old_module,
                f"import {old_module}",
                "(removed)",
                f"Public module `{old_module}` was deleted; its old import fails.",
            )
        elif change.status.startswith("R"):
            new_module = public_module(
                change.new_path, has_decorated_api=bool(new_apis)
            )
            if git_file(head, change.old_path or "") is None:
                surface(
                    "public_module_moved",
                    change.new_path or "flashinfer",
                    1,
                    old_module,
                    f"import {old_module}",
                    f"import {new_module or change.new_path}",
                    "Public module moved without preserving the old import path.",
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
    parser.add_argument("--fail-on", choices=["error"], help="Fail only on API errors")
    args = parser.parse_args()

    try:
        base = resolve_merge_base(args.base, args.head)
        findings = check(base, args.head, include_compatible=True)
    except (
        RuntimeError,
        subprocess.CalledProcessError,
        SyntaxError,
        ValueError,
    ) as exc:
        if args.report_json:
            write_report(
                args.report_json,
                "public_api",
                args.base,
                args.head,
                [],
                execution_status="FAILED",
                error=str(exc),
            )
        print(f"API scan incomplete: {exc}")
        return 2
    alerts = [finding for finding in findings if finding.level != "notice"]
    print(f"Public API compatibility check: {len(alerts)} finding(s)")
    for finding in alerts:
        emit_finding(finding, args.github_actions)
    if args.report_json:
        write_report(args.report_json, "public_api", base, args.head, findings)
    return int(
        bool(alerts)
        if args.strict
        else bool(args.fail_on and any(f.level == "error" for f in alerts))
    )


if __name__ == "__main__":
    raise SystemExit(main())
