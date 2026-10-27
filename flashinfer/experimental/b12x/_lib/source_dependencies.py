"""Process-local, file-granular source dependencies for compiled CuTe programs."""

from __future__ import annotations

import ast
import hashlib
import inspect
from functools import lru_cache
from importlib.util import resolve_name
from pathlib import Path
from types import ModuleType
import sys


def canonical_module(name: str) -> str:
    prefix = "flashinfer.experimental.b12x"
    return (
        "b12x" + name[len(prefix) :]
        if name == prefix or name.startswith(prefix + ".")
        else name
    )


def callable_source(target):
    if inspect.ismethod(target):
        target = target.__func__
    elif not inspect.isfunction(target) and not inspect.isclass(target):
        target = type(target).__call__
    target = inspect.unwrap(target)
    path = inspect.getsourcefile(target)
    if path is None:
        raise ValueError(f"CuTe cache requires Python source for {target!r}")
    return canonical_module(target.__module__), Path(path).resolve()


def callable_sources(target, root: Path):
    """Include local wrappers, inherited methods and callable defaults/closures."""
    return source_dependencies(root).callable_sources(target)


def _discover_callable_sources(target, root: Path):
    pending = [target]
    seen = set()
    roots = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if inspect.ismethod(value):
            owner = (
                value.__self__
                if inspect.isclass(value.__self__)
                else type(value.__self__)
            )
            pending.extend((value.__func__, owner))
            continue
        if not inspect.isfunction(value) and not inspect.isclass(value):
            if callable(value):
                pending.append(type(value))
            continue
        module = canonical_module(value.__module__)
        if module != "b12x" and not module.startswith("b12x."):
            wrapped = getattr(value, "__wrapped__", None)
            if wrapped is not None:
                pending.append(wrapped)
            continue
        path = inspect.getsourcefile(value)
        if path is None:
            raise ValueError(f"CuTe cache requires Python source for {value!r}")
        path = Path(path).resolve()
        if path.is_relative_to(root):
            relative = path.relative_to(root).with_suffix("")
            parts = relative.parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            module = "b12x" + ("." + ".".join(parts) if parts else "")
            roots.add((module, path))
        if inspect.isclass(value):
            pending.extend(value.__bases__)
            pending.extend(
                member.__func__
                if isinstance(member, (staticmethod, classmethod))
                else member
                for member in vars(value).values()
                if inspect.isfunction(member)
                or isinstance(member, (staticmethod, classmethod))
            )
        else:
            wrapped = getattr(value, "__wrapped__", None)
            if wrapped is not None:
                pending.append(wrapped)
            pending.extend(value.__defaults__ or ())
            pending.extend((value.__kwdefaults__ or {}).values())
            for cell in value.__closure__ or ():
                try:
                    pending.append(cell.cell_contents)
                except ValueError:
                    pass
    return tuple(sorted(roots)) if roots else (callable_source(target),)


class _Imports(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
        self._calls = []
        self._dynamic_names = {"__import__", "import_module"}
        self._lazy_export = False

    @property
    def dynamic(self):
        return [
            (node, lazy)
            for node, lazy in self._calls
            if (isinstance(node.func, ast.Name) and node.func.id in self._dynamic_names)
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"import_module", "__import__"}
            )
        ]

    def visit_FunctionDef(self, node):
        enclosing = self._lazy_export
        self._lazy_export = node.name == "__getattr__"
        self.generic_visit(node)
        self._lazy_export = enclosing

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node):
        test = node.test
        type_checking = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        for item in node.orelse if type_checking else (*node.body, *node.orelse):
            self.visit(item)

    def visit_Import(self, node):
        self.imports.append(node)

    def visit_ImportFrom(self, node):
        self.imports.append(node)
        if node.module in {"importlib", "builtins"}:
            self._dynamic_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in {"import_module", "__import__"}
            )

    def visit_Call(self, node):
        self._calls.append((node, self._lazy_export))
        self.generic_visit(node)


class SourceDependencies:
    """Hash local import closures without importing modules or executing code.

    Package initializers contribute their source and selected re-exports, not
    their entire lazy API registry. Computed imports in implementation modules
    declare their possible local targets in ``__b12x_source_dependencies__``.
    Source and callable structure are immutable for the lifetime of this object.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._files = {}
        self._closures = {}
        self._digests = {}
        self._callables = {}

    def callable_sources(self, target):
        if inspect.ismethod(target):
            owner = (
                target.__self__
                if inspect.isclass(target.__self__)
                else type(target.__self__)
            )
            key = (target.__func__, owner)
        elif inspect.isfunction(target) or inspect.isclass(target):
            key = (target,)
        else:
            # Instance state specializes compilation, not the source dependency set.
            key = (None, type(target))
        if key not in self._callables:
            self._callables[key] = _discover_callable_sources(target, self.root)
        return self._callables[key]

    def module_path(self, module: str) -> Path | None:
        module = canonical_module(module)
        if module == "b12x":
            api = self.root / "_api.py"
            return api if api.is_file() else self.root / "__init__.py"
        if not module.startswith("b12x."):
            return None
        path = self.root.joinpath(*module.split(".")[1:])
        if path.with_suffix(".py").is_file():
            return path.with_suffix(".py")
        if (path / "__init__.py").is_file():
            return path / "__init__.py"
        return None

    def _read(self, path):
        if path not in self._files:
            data = path.read_bytes()
            tree = ast.parse(data, filename=str(path))
            imports = _Imports()
            imports.visit(tree)
            declared = None
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "__b12x_source_dependencies__"
                    for t in node.targets
                ):
                    declared = ast.literal_eval(node.value)
                    if not isinstance(declared, (tuple, list)) or not all(
                        isinstance(v, str) for v in declared
                    ):
                        raise ValueError(
                            f"{path}: source dependencies must be literal module names"
                        )
            self._files[path] = (hashlib.sha256(data).digest(), imports, declared)
        return self._files[path]

    def _bound_module(self, module, name):
        loaded = sys.modules.get(module) or sys.modules.get(
            "flashinfer.experimental.b12x" + module[len("b12x") :]
        )
        value = vars(loaded).get(name) if loaded is not None else None
        if isinstance(value, ModuleType):
            return canonical_module(value.__name__)
        return canonical_module(getattr(value, "__module__", ""))

    def closure(self, module: str, path: Path) -> tuple[tuple[str, Path], ...]:
        module = canonical_module(module)
        key = (module, path)
        if key in self._closures:
            return self._closures[key]
        pending = [(module, path)]
        found = {}
        visited = set()

        def add(name, *, required=True):
            name = canonical_module(name)
            if name != "b12x" and not name.startswith("b12x."):
                return
            source = self.module_path(name)
            if source is None:
                if required:
                    raise ValueError(f"Cannot resolve CuTe source dependency {name!r}")
                return
            pending.append((name, source))

        while pending:
            name, source = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            found[name] = source
            _, imports, declared = self._read(source)
            package = (
                name
                if source.name == "__init__.py" or name == "b12x"
                else name.rpartition(".")[0]
            )
            # Import infrastructure participates without expanding lazy registries.
            for i in range(1, len(name.split("."))):
                parent = ".".join(name.split(".")[:i])
                parent_path = self.module_path(parent)
                if parent_path is not None:
                    pending.append((parent, parent_path))
            for node in imports.imports:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        add(alias.name)
                    continue
                base = "." * node.level + (node.module or "")
                base = canonical_module(
                    resolve_name(base, package) if node.level else base
                )
                add(base)
                for alias in node.names:
                    add(base + "." + alias.name, required=False)
                    bound = self._bound_module(base, alias.name)
                    if bound:
                        add(bound)
                    bound = self._bound_module(name, alias.asname or alias.name)
                    if bound:
                        add(bound)
            for dependency in declared or ():
                add(
                    resolve_name(dependency, package)
                    if dependency.startswith(".")
                    else dependency
                )
            for node, lazy_export in imports.dynamic:
                if lazy_export and (source.name == "__init__.py" or name == "b12x"):
                    continue
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    dependency = node.args[0].value
                    if dependency.startswith("."):
                        if len(node.args) < 2 or not isinstance(
                            node.args[1], ast.Constant
                        ):
                            if declared is None:
                                raise ValueError(
                                    f"{source}:{node.lineno}: declare __b12x_source_dependencies__ for relative dynamic imports"
                                )
                            continue
                        dependency = resolve_name(dependency, node.args[1].value)
                    add(dependency)
                elif declared is None:
                    raise ValueError(
                        f"{source}:{node.lineno}: declare __b12x_source_dependencies__ for computed imports"
                    )
        result = tuple(sorted(found.items()))
        self._closures[key] = result
        return result

    def fingerprint(self, roots: tuple[tuple[str, Path], ...]) -> str:
        if roots not in self._digests:
            sources = dict(item for root in roots for item in self.closure(*root))
            digest = hashlib.sha256()
            for name, path in sorted(sources.items()):
                digest.update(name.encode())
                digest.update(b"\0")
                digest.update(self._read(path)[0])
            self._digests[roots] = digest.hexdigest()
        return self._digests[roots]


@lru_cache(maxsize=1)
def source_dependencies(root: Path) -> SourceDependencies:
    return SourceDependencies(root)
