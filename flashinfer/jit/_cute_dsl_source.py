"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

CuTe-DSL source dependency analysis and hashing for persistent kernel caches.

Callers supply kernel source files through ``extra_key_files``, usually their
own ``__file__``. ``_hash_source_files`` hashes those explicit files by default.
With ``FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY=1``, it also keys the FlashInfer
helpers those files can use. This protects long-lived CI caches when a shared
helper changes without editing the kernel's own source file.

Python's AST parser supplies the syntax trees; this module resolves imports,
package exports, attribute uses, and initializer code without executing source
or importing GPU dependencies. Initializers are keyed by import-time code and
used exports, so adding an unrelated public reexport need not rebuild kernels.
Evaluated annotations are included because their expressions can execute setup
code; postponed annotations do not run at import time.

This is a limited static dependency analysis, not a general Python interpreter.
Nonliteral dynamic imports and runtime file inputs require explicit key files.
Side effects of unused initializer imports, namespace deletion/mutation, and
overloaded operators are not fully modeled. Parsed sources and closures are
cached per process; source dependency graphs are assumed stable
within that process, and tests clear the closure cache when changing imports.
"""

import ast
import functools
import hashlib
import os
from pathlib import Path
from typing import Optional, Sequence


def cute_dsl_strict_cache_key() -> bool:
    """Whether cache keys also cover the in-package import closure of key files."""
    return os.environ.get("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY", "0") == "1"


# The flashinfer package directory; only imports resolving inside it are followed.
_PACKAGE_DIR = Path(__file__).resolve().parents[1]
# Never keyed by the strict key: ``version.py`` / ``_build_meta.py`` carry the
# build's git commit, which differs on every CI checkout. The package version
# is already part of the cache directory path (``FLASHINFER_JIT_DIR``).
_UNKEYED_PACKAGE_FILES = ("version.py", "_build_meta.py")
_TRY_STATEMENTS = (ast.Try, getattr(ast, "TryStar", ast.Try))


def _resolve_package_module(parts: Sequence[str]) -> Optional[Path]:
    """File of the dotted flashinfer module ``parts`` (first part ``flashinfer``)."""
    if not parts or parts[0] != _PACKAGE_DIR.name:
        return None
    base = _PACKAGE_DIR.joinpath(*parts[1:])
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _bound_name(statement: ast.AST, alias: ast.alias) -> str:
    """The name an import alias binds: ``import a.b`` binds ``a``."""
    if alias.asname:
        return alias.asname
    if isinstance(statement, ast.Import):
        return alias.name.split(".")[0]
    return alias.name


def _attribute_uses(scope: ast.AST) -> dict:
    """How each name loaded in ``scope`` is used.

    Maps a name to the attribute chains read from it (``env.JIT_DIR`` gives
    ``("JIT_DIR",)``), or to ``None`` when the bare value escapes (passed to
    ``getattr``, returned, called...), in which case every attribute counts.
    """
    parents: dict = {}
    for node in ast.walk(scope):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    uses: dict = {}
    for node in ast.walk(scope):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
            continue
        chain = []
        current: ast.AST = node
        while True:
            parent = parents.get(current)
            if not (isinstance(parent, ast.Attribute) and parent.value is current):
                break
            chain.append(parent.attr)
            current = parent
        if not chain:
            uses[node.id] = None
        elif uses.get(node.id, frozenset()) is not None:
            uses[node.id] = uses.get(node.id, frozenset()) | {tuple(chain)}
    return uses


class _Source:
    """A parsed flashinfer source file and what its module scope binds and runs."""

    def __init__(self, path: Path, data: bytes):
        self.path = path
        self.tree = ast.parse(data, filename=str(path))
        self.future_annotations = any(
            isinstance(stmt, ast.ImportFrom)
            and stmt.module == "__future__"
            and any(alias.name == "annotations" for alias in stmt.names)
            for stmt in self.tree.body
        )
        rel = path.relative_to(_PACKAGE_DIR.parent).with_suffix("")
        self.package = list(rel.parts[:-1])  # same for ``__init__`` and plain modules
        self.is_initializer = path.name == "__init__.py"
        # Name -> (statement, alias) pairs binding it at module scope, including
        # inside module-level if/try/with blocks; alias is None for non-imports.
        self.bindings: dict = {}
        self.star_imports: list = []
        # Module-scope code run at import, other than import statements and
        # function bodies (which run only when called).
        self.executed: list = []
        self._scan(self.tree.body)
        top_level = set(self.tree.body)
        self.definite_bindings = {
            name
            for name, bindings in self.bindings.items()
            if any(
                statement in top_level
                and not (
                    isinstance(statement, ast.AnnAssign) and statement.value is None
                )
                for statement, _ in bindings
            )
        }
        self._uses: dict = {}
        self.importlib_names = {"importlib"}
        self.import_module_names: set = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                self.importlib_names.update(
                    _bound_name(node, alias)
                    for alias in node.names
                    if alias.name == "importlib"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
                self.import_module_names.update(
                    _bound_name(node, alias)
                    for alias in node.names
                    if alias.name == "import_module"
                )

    def _bind(self, name: str, statement: ast.AST, alias=None) -> None:
        self.bindings.setdefault(name, []).append((statement, alias))

    def _bind_targets(self, target: ast.AST, statement: ast.AST) -> None:
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self._bind(node.id, statement)

    def _scan(self, body: list) -> None:
        for stmt in body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for alias in stmt.names:
                    if alias.name == "*":
                        self.star_imports.append(stmt)
                    else:
                        self._bind(_bound_name(stmt, alias), stmt, alias)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._bind(stmt.name, stmt)
                self.executed += stmt.decorator_list + stmt.args.defaults
                self.executed += [d for d in stmt.args.kw_defaults if d is not None]
                if not self.future_annotations:
                    # Annotations can call setup code even for unused functions.
                    self.executed += [
                        arg.annotation
                        for arg in [
                            *stmt.args.posonlyargs,
                            *stmt.args.args,
                            *stmt.args.kwonlyargs,
                            stmt.args.vararg,
                            stmt.args.kwarg,
                        ]
                        if arg is not None and arg.annotation is not None
                    ]
                    if stmt.returns is not None:
                        self.executed.append(stmt.returns)
            elif isinstance(stmt, ast.ClassDef):
                self._bind(stmt.name, stmt)
                self.executed.append(stmt)  # the class body runs at import
            elif isinstance(stmt, (ast.If, ast.While)):
                self.executed.append(stmt.test)
                self._scan(stmt.body)
                self._scan(stmt.orelse)
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                self.executed.append(stmt.iter)
                self._bind_targets(stmt.target, stmt)
                self._scan(stmt.body)
                self._scan(stmt.orelse)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    self.executed.append(item.context_expr)
                    if item.optional_vars is not None:
                        self._bind_targets(item.optional_vars, stmt)
                self._scan(stmt.body)
            elif isinstance(stmt, _TRY_STATEMENTS):
                self._scan(stmt.body)
                for handler in stmt.handlers:
                    if handler.type is not None:
                        self.executed.append(handler.type)
                    if handler.name:
                        self._bind(handler.name, handler)
                    self._scan(handler.body)
                self._scan(stmt.orelse)
                self._scan(stmt.finalbody)
            else:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        self._bind_targets(target, stmt)
                elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
                    self._bind_targets(stmt.target, stmt)
                self.executed.append(stmt)

    def uses(self, scope: ast.AST, name: str):
        if id(scope) not in self._uses:
            self._uses[id(scope)] = _attribute_uses(scope)
        return self._uses[id(scope)].get(name, frozenset())

    def scope_imports(self, scope: ast.AST) -> tuple:
        """Import statements, module-level names loaded, and literal dynamic
        imports in ``scope``; cached since every kernel's closure revisits
        shared helpers."""
        key = ("imports", id(scope))
        if key not in self._uses:
            imports = []
            loaded = set()
            for node in ast.walk(scope):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        imports.append((node, alias))
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id in self.bindings:
                        loaded.add(node.id)
            dynamic = list(self.dynamic_imports(scope))
            self._uses[key] = (imports, sorted(loaded), dynamic)
        return self._uses[key]

    def import_target(self, statement: ast.ImportFrom) -> Optional[Path]:
        """File of the module a ``from ... import`` statement imports from."""
        if statement.level:
            if statement.level - 1 > len(self.package):
                return None
            base = self.package[: len(self.package) - (statement.level - 1)]
        else:
            base = []
        module = statement.module.split(".") if statement.module else []
        return _resolve_package_module(base + module)

    def dynamic_imports(self, scope: ast.AST):
        """Dotted names of literal ``importlib.import_module`` / ``__import__`` calls."""
        from importlib.util import resolve_name

        package = self.package

        def import_string(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.Name) and node.id == "__package__":
                return ".".join(package)
            if isinstance(node, ast.JoinedStr):
                if any(
                    isinstance(part, ast.FormattedValue)
                    and (part.conversion != -1 or part.format_spec is not None)
                    for part in node.values
                ):
                    return None
                parts = [
                    import_string(part.value)
                    if isinstance(part, ast.FormattedValue)
                    else import_string(part)
                    for part in node.values
                ]
                if all(part is not None for part in parts):
                    return "".join(part for part in parts if part is not None)
            return None

        def argument(node, index, keyword, default=None):
            if len(node.args) > index:
                return node.args[index]
            return next(
                (kw.value for kw in node.keywords if kw.arg == keyword), default
            )

        for node in ast.walk(scope):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            builtin_import = (
                isinstance(function, ast.Name) and function.id == "__import__"
            )
            dynamic_import = (
                builtin_import
                or (
                    isinstance(function, ast.Name)
                    and function.id in self.import_module_names
                )
                or (
                    isinstance(function, ast.Attribute)
                    and function.attr == "import_module"
                    and isinstance(function.value, ast.Name)
                    and function.value.id in self.importlib_names
                )
            )
            if not dynamic_import:
                continue
            name_node = argument(node, 0, "name")
            name = import_string(name_node) if name_node is not None else None
            if name is None:
                continue  # Caller must list nonliteral dependencies explicitly.
            if builtin_import:
                level = argument(node, 4, "level", ast.Constant(0))
                if not isinstance(level, ast.Constant) or not isinstance(
                    level.value, int
                ):
                    continue
                if level.value:
                    name = resolve_name("." * level.value + name, ".".join(package))
            if name.startswith("."):
                context_node = argument(node, 1, "package")
                context = (
                    import_string(context_node) if context_node is not None else None
                )
                if context is None:
                    raise ValueError(f"Unresolved relative import in {self.path}")
                name = resolve_name(name, context)
            yield name
            if builtin_import:
                fromlist = argument(node, 3, "fromlist")
                if isinstance(fromlist, (ast.List, ast.Tuple)):
                    for child in fromlist.elts:
                        child_name = import_string(child)
                        if child_name is not None:
                            yield f"{name}.{child_name}"


class _ImportClosure:
    """Flashinfer sources that key files can run, resolved name by name.

    Plain modules are keyed by content, and everything they import is
    followed. Package initializers mostly re-export, so they are followed
    one name at a time: a lookup of ``name`` in a package resolves to the
    binding that defines it (a re-export leads to its source, a function to
    its body), plus the submodule of that name. Code an initializer runs at
    import is followed whenever a file in its package is keyed. Each
    initializer is then keyed by that code and the import lines that bind a
    used name, so adding an unrelated export changes no key.

    Not keyed: import-time side effects of modules that keyed code neither
    imports nor uses (an initializer importing an unused sibling), and
    nonliteral dynamic imports or file reads, which callers must list in
    ``extra_key_files``.
    """

    def __init__(self):
        self.files: set = set()  # keyed by content
        self.initializers: dict = {}  # path -> names whose import lines are keyed
        self._sources: dict = {}
        self._seen: set = set()
        self._unkeyed = {_PACKAGE_DIR / name for name in _UNKEYED_PACKAGE_FILES}

    def _source(self, path: Path) -> _Source:
        if path not in self._sources:
            self._sources[path] = _parse_source(path, path.read_bytes())
        return self._sources[path]

    def _once(self, *key) -> bool:
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def _add_parents(self, path: Path) -> None:
        for directory in path.parents:
            if not directory.is_relative_to(_PACKAGE_DIR):
                break
            initializer = directory / "__init__.py"
            if initializer != path and initializer.is_file():
                self.add_initializer(initializer)

    def add_file(self, path: Path) -> None:
        """A file whose every binding may be used: key it and all it imports."""
        if path in self._unkeyed or not self._once("file", path):
            return
        self.files.add(path)
        self._add_parents(path)
        source = self._source(path)
        self._add_imports(source, source.tree)

    def add_initializer(self, path: Path) -> None:
        """A package initializer that runs because something below it is imported."""
        if not self._once("initializer", path):
            return
        self.initializers.setdefault(path, set())
        self._add_parents(path)
        source = self._source(path)
        self._add_code(source, source.executed)

    def add_module(self, path: Optional[Path], uses) -> None:
        """A module object read through attribute chains ``uses`` (``None``: any)."""
        if path is None or path in self._unkeyed:
            return
        source = self._source(path)
        if not source.is_initializer:
            self.add_file(path)
            return
        self.add_initializer(path)
        if uses is None:
            # Any attribute may be read: every binding, and every submodule,
            # which becomes an attribute once anything imports it.
            names = set(source.bindings)
            names.update(
                child.stem if child.suffix == ".py" else child.name
                for child in path.parent.iterdir()
                if child.suffix == ".py"
                and child.name != "__init__.py"
                or (child / "__init__.py").is_file()
            )
            for name in sorted(names):
                self.add_symbol(path, name, None)
            for statement in source.star_imports:
                self.add_module(source.import_target(statement), None)
            return
        for chain in uses:
            self.add_symbol(
                path, chain[0], frozenset({chain[1:]}) if len(chain) > 1 else None
            )

    def add_symbol(self, path: Path, name: str, uses, *, attribute_lookup=True) -> None:
        """``name`` looked up in the package initializer ``path``."""
        if not self._once("symbol", path, name, uses, attribute_lookup):
            return
        self.add_initializer(path)
        self.initializers[path].add(name)
        source = self._source(path)
        # ``from pkg import name`` returns pkg's attribute when it binds one,
        # else the submodule; key both rather than model import order.
        self.add_module(_resolve_package_module(source.package + [name]), uses)
        bindings = source.bindings.get(name, [])
        for statement, alias in bindings:
            if alias is not None:
                self._add_alias(source, statement, alias, uses)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._add_code(source, [statement])
            # Other bindings take their value from code already run at import.
        # A later star import can replace even an unconditional binding.
        for statement in source.star_imports:
            target = source.import_target(statement)
            if target is not None and self._source(target).is_initializer:
                self.add_symbol(target, name, uses)
            elif target is not None:
                self.add_file(target)
        # A conditional or annotation-only binding may leave the name absent.
        if (
            attribute_lookup
            and name not in source.definite_bindings
            and "__getattr__" in source.bindings
        ):
            self.add_symbol(path, "__getattr__", None)

    def _add_alias(self, source: _Source, statement, alias: ast.alias, uses) -> None:
        if isinstance(statement, ast.Import):
            parts = alias.name.split(".")
            # Every package along the dotted path is imported.
            for depth in range(1, len(parts) + 1):
                self.add_module(_resolve_package_module(parts[:depth]), frozenset())
            bound = parts if alias.asname else parts[:1]
            self.add_module(_resolve_package_module(bound), uses)
            return
        target = source.import_target(statement)
        if target is None:
            return
        if self._source(target).is_initializer:
            # A self re-export resolves its own binding, rather than looking
            # up an unrelated lazy export in the same initializer.
            self.add_symbol(
                target,
                alias.name,
                uses,
                attribute_lookup=(
                    target != source.path or _bound_name(statement, alias) != alias.name
                ),
            )
        else:
            self.add_file(target)

    def _add_imports(self, source: _Source, scope: ast.AST) -> None:
        """Follow the imports in ``scope`` as far as ``scope`` uses their names."""
        imports, _, dynamic = source.scope_imports(scope)
        for node, alias in imports:
            if alias.name == "*" and isinstance(node, ast.ImportFrom):
                self.add_module(source.import_target(node), None)
            else:
                uses = source.uses(scope, _bound_name(node, alias))
                self._add_alias(source, node, alias, uses)
        for name in dynamic:
            self.add_module(_resolve_package_module(name.split(".")), None)

    def _add_code(self, source: _Source, nodes: list) -> None:
        """Follow the module-level names and imports that ``nodes`` use."""
        for node in nodes:
            self._add_imports(source, node)
            for name in source.scope_imports(node)[1]:
                # Global-name reads do not invoke a module's __getattr__.
                self.add_symbol(
                    source.path,
                    name,
                    source.uses(node, name),
                    attribute_lookup=False,
                )


@functools.lru_cache(maxsize=None)
def _parse_source(path: Path, data: bytes) -> _Source:
    # Reuse parsed ASTs by content when constructing a fresh closure.
    return _Source(path, data)


# Syntax treated as export-list construction, assuming ordinary builtin values,
# e.g. ``__all__ = [...] + _optional_exports``. Overloaded operators are not modeled.
# Only ``__all__`` assignments are dropped: any other module global may be read
# dynamically (``sys.modules[...]``) by code the strict key cannot see.
_PURE_NODES = (
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.Starred,
    ast.Name,
    ast.BinOp,
    ast.UnaryOp,
    ast.operator,
    ast.unaryop,
    ast.expr_context,
)


def _is_unused_literal(stmt: ast.AST, used: set) -> bool:
    """A docstring, or a pure ``__all__`` assignment no keyed code reads."""
    if isinstance(stmt, ast.Expr):
        return isinstance(stmt.value, ast.Constant)
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        targets = [stmt.target]
    else:
        return False
    return (
        stmt.value is not None
        and all(isinstance(node, _PURE_NODES) for node in ast.walk(stmt.value))
        and all(isinstance(t, ast.Name) and t.id == "__all__" for t in targets)
        and "__all__" not in used
    )


def _initializer_digest(path: Path, used: set) -> str:
    """An initializer's code without what no keyed code can observe.

    Drops the import lines of names nothing uses, docstrings, and ``__all__``
    unless something reads it, so adding an export (and its ``__all__`` entry)
    to a package changes no key. A name read dynamically or through a star
    import counts as used.
    """

    def keep(body: list) -> list:
        kept = []
        for stmt in body:
            if _is_unused_literal(stmt, used):
                continue
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                stmt.names = [
                    alias
                    for alias in stmt.names
                    if alias.name == "*" or _bound_name(stmt, alias) in used
                ]
                if not stmt.names:
                    continue
            elif isinstance(stmt, _TRY_STATEMENTS):
                stmt.body = keep(stmt.body)
                for handler in stmt.handlers:
                    handler.body = keep(handler.body)
                stmt.orelse = keep(stmt.orelse)
                stmt.finalbody = keep(stmt.finalbody)
            elif isinstance(
                stmt,
                (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith),
            ):
                stmt.body = keep(stmt.body)
                if hasattr(stmt, "orelse"):
                    stmt.orelse = keep(stmt.orelse)
            kept.append(stmt)
        return kept

    tree = ast.parse(path.read_bytes(), filename=str(path))
    tree.body = keep(tree.body)
    return ast.dump(tree)


@functools.cache
def _import_closure(path: Path) -> _ImportClosure:
    closure = _ImportClosure()
    closure.add_file(path)
    return closure


def _package_import_closure(path: Path) -> tuple:
    """``path`` plus every flashinfer source file keyed with it."""
    closure = _import_closure(path)
    return tuple(sorted(closure.files | set(closure.initializers)))


def _key_sources(paths: tuple) -> tuple:
    """Files keyed by content, and initializers keyed by their used bindings.

    Callers list the files whose content should invalidate their kernels,
    typically just ``__file__``; helpers those files import are easy to
    miss. Strict mode keeps a long-lived shared cache (CI) from serving a
    kernel compiled before such a helper changed.
    """
    initializers: dict = {}
    if not cute_dsl_strict_cache_key():
        # As given, duplicates included: default keys must not change.
        return [str(p) for p in paths], initializers
    files = {str(p) for p in paths}
    for p in paths:
        resolved = Path(p).resolve()
        if resolved.suffix == ".py" and resolved.is_relative_to(_PACKAGE_DIR):
            closure = _import_closure(resolved)
            files.update(str(dep) for dep in closure.files)
            for initializer, used in closure.initializers.items():
                initializers.setdefault(str(initializer), set()).update(used)
    for path in files:
        initializers.pop(path, None)  # keyed by content already
    return files, initializers


def _expand_key_files(paths: tuple) -> tuple:
    """Key files plus, in strict mode, every source in their import closures."""
    files, initializers = _key_sources(paths)
    return tuple(set(files) | set(initializers))


def _hash_source_files(paths: tuple) -> str:
    files, initializers = _key_sources(paths)
    sha = hashlib.sha256()
    for path in sorted(files):
        sha.update(path.encode())
        with open(path, "rb") as f:
            sha.update(f.read())
    for path in sorted(initializers):
        sha.update(f"{path} (initializer)".encode())
        sha.update(_initializer_digest(Path(path), initializers[path]).encode())
    return sha.hexdigest()
