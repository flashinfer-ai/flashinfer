#!/usr/bin/env python3
"""Select the unit-test files a change can affect, from a static dependency graph.

For every test file under ``tests/`` the graph records what the test loads when it
runs: the Python files it reaches through imports (including imports inside
functions, module names held in strings such as ``import_module("flashinfer.x")``,
``flashinfer.<name>`` uses resolved through package export lists, and files named
by path such as ``"scripts/tool.py"``), the kernel sources those Python files name,
the headers those sources ``#include``, and data files next to the Python files. A
change selects every test whose dependencies it touches. A change to a global input
(Python dependencies, CI images, ``3rdparty``, the root conftest, the scripts the
test lanes run) selects every test.

An absolute import of a name outside the repository roots resolves to every module
of that name that a ``sys.path`` entry can expose, meaning modules whose directory
is not a package (``benchmarks/``, a kernel package under ``src/``), and through
``MODULE_ALIASES``. Python files that no test reaches through imports may still be
loaded by path, for example generated sources named by a manifest, so like data
files they belong to the reached Python files in their directory or its nearest
ancestor.

Importing a module runs every parent package ``__init__.py`` first, so an
``__init__`` that eagerly imports its whole package makes every test that imports
from the package depend on all of it. The graph follows that behavior, so selection
is broad. It cannot see imports and file paths computed at run time (f-strings,
``getattr`` on computed names) except through the directory rule above.

Only the standard library is used, so the script runs on a plain CI runner without
torch or a GPU. Any error selects every test.

Usage::

    select_tests.py --base origin/main                  # merge base .. working tree
    select_tests.py --base BASE_SHA --head HEAD_SHA --format markdown
    select_tests.py --base origin/main --explain tests/utils/test_norm.py
"""

from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# Changes that can affect every test, including the scripts the GPU test lanes run
# or source.
GLOBAL_INPUTS = re.compile(
    r"^(?:requirements[^/]*\.txt|pyproject\.toml|setup\.py|setup\.cfg|\.python-version"
    r"|version\.txt|build_backend\.py|build_utils\.py|pytest\.ini|docker/|ci/"
    r"|3rdparty(?:/|$)|3rdparty_patches/|tests/conftest\.py"
    r"|scripts/(?:task_run_unit_tests\.sh|task_jit_run_tests_[^/]*\.sh"
    r"|unit_test_runner\.py|find_marked_tests\.py|test_utils\.sh|test_path_scope\.sh"
    r"|test_run_status\.(?:sh|py)|setup_test_env\.sh|jit_test_logging\.sh"
    r"|run_with_test_sccache\.sh|jit_cache_build_common\.sh|install_cutlass_dsl\.sh"
    r"|test_sharding/))"
)
PYTHON_ROOTS = ("flashinfer/", "tests/", "scripts/", "benchmarks/")
MODULE_ROOTS = tuple(r.rstrip("/") for r in PYTHON_ROOTS)
# Top-level packages whose submodules are served from another directory at run time:
# b12x is installed from flashinfer/experimental/b12x/_compat, and its loader finds
# b12x.<name> under flashinfer/experimental/b12x.
MODULE_ALIASES = {"b12x": "flashinfer.experimental.b12x"}
# File names that, without a directory, do not identify one file.
GENERIC_FILE_NAMES = frozenset({"__init__.py", "__main__.py", "conftest.py"})
KERNEL_ROOTS = ("csrc/", "include/")
KERNEL_EXTS = (
    ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".inc", ".inl", ".jinja", ".ptx",
)  # fmt: skip
# Bare string literals with these names are too generic to identify a kernel directory.
GENERIC_DIR_NAMES = frozenset(
    {
        "include", "csrc", "flashinfer", "src", "common", "utils", "kernels", "cpp",
        "generated", "experimental", "cuda", "host",
    }
)  # fmt: skip
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.MULTILINE)
INCLUDE_DIRS = (
    "include",
    "csrc",
    "csrc/nv_internal",
    "csrc/nv_internal/include",
    "csrc/nv_internal/tensorrt_llm",
)
TEST_FILE_RE = re.compile(r"^tests/(?:.*/)?test_[^/]*\.py$")
MODULE_NAME_RE = re.compile(r"\.*[A-Za-z_][A-Za-z0-9_.]*")
# String literals that could name a module, a kernel file or a kernel directory.
CANDIDATE_RE = re.compile(r"[A-Za-z0-9_./-]+")


def git(*args: str, cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def _str_value(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _module_name(path: str) -> str:
    parts = path[: -len(".py")].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _parent_dirs(path: str) -> Iterable[str]:
    """Ancestor directories of ``path``, nearest first, excluding the repo root."""
    directory = PurePosixPath(path).parent
    while str(directory) not in (".", ""):
        yield str(directory)
        directory = directory.parent


def conftest_chain(test: str, tracked: Set[str]) -> List[str]:
    """conftest.py files pytest loads for ``test``."""
    return [
        f"{d}/conftest.py"
        for d in _parent_dirs(test)
        if d.startswith("tests") and f"{d}/conftest.py" in tracked
    ]


# ------------------------------------------------------------------------ file scan


@dataclass
class FileFacts:
    """What one Python file says about its dependencies, before name resolution."""

    path: str
    imports: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    from_imports: List[Tuple[int, Optional[str], List[Tuple[str, Optional[str]]]]] = (
        field(default_factory=list)
    )
    strings: Set[str] = field(default_factory=set)
    chains: Set[Tuple[str, ...]] = field(default_factory=set)
    export_pairs: List[Tuple[str, str]] = field(default_factory=list)
    plugins: List[str] = field(default_factory=list)


def scan_file(repo: str, path: str) -> FileFacts:
    """Parse one file and walk it once. Runs in worker processes."""
    facts = FileFacts(path)
    try:
        tree = ast.parse(Path(repo, path).read_bytes(), filename=path)
    except (SyntaxError, ValueError, OSError):
        return facts  # still a dependency of whatever imports it
    is_init = path.endswith("__init__.py")
    lazy_values: Set[int] = set()
    for node in ast.walk(tree):  # breadth-first: a Dict is seen before its values
        if isinstance(node, ast.Import):
            facts.imports.extend((a.name, a.asname) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names = [(a.name, a.asname) for a in node.names]
            facts.from_imports.append((node.level, node.module, names))
        elif isinstance(node, ast.Attribute):
            chain = []
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                chain.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                facts.chains.add((current.id, *reversed(chain)))
        elif isinstance(node, ast.Dict) and is_init:
            # {"name": ".module"} maps consumed by a lazy __getattr__: an export list,
            # not an import, so the values must not count as string imports.
            for key, value in zip(node.keys, node.values, strict=True):
                name, module = _str_value(key), _str_value(value)
                if name and module:
                    facts.export_pairs.append((name, module))
                    lazy_values.add(id(value))
        elif isinstance(node, ast.Assign) and path == "tests/conftest.py":
            if any(
                isinstance(t, ast.Name) and t.id == "pytest_plugins"
                for t in node.targets
            ):
                facts.plugins.extend(
                    v for v in map(_str_value, ast.walk(node.value)) if v
                )
        elif id(node) not in lazy_values:
            value = _str_value(node)
            if (
                value
                and len(value) <= 200
                and CANDIDATE_RE.fullmatch(value.strip().strip("/"))
            ):
                facts.strings.add(value)
    bound = {(a or n.split(".")[0]) for n, a in facts.imports}
    bound |= {(a or n) for _, _, names in facts.from_imports for n, a in names}
    facts.chains = {c for c in facts.chains if c[0] in bound}
    return facts


def scan_files(repo: Path, paths: Sequence[str], jobs: int) -> Dict[str, FileFacts]:
    if jobs <= 1 or len(paths) < 200:
        return {p: scan_file(str(repo), p) for p in paths}
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        results = pool.map(scan_file, [str(repo)] * len(paths), paths, chunksize=32)
        return {f.path: f for f in results}


# ----------------------------------------------------------------------------- graph


@dataclass
class Graph:
    tracked: Set[str]
    test_files: List[str]
    mod_to_path: Dict[str, str]
    path_to_mod: Dict[str, str]
    imports: Dict[str, Set[str]]  # Python file -> Python files it imports
    owners: Dict[str, Set[str]]  # tracked file -> Python files that depend on it
    pytest_plugins: Set[str]
    _neighbors: Dict[str, List[str]] = field(default_factory=dict, repr=False)

    def neighbors(self, path: str) -> List[str]:
        """Files loaded when ``path`` is imported: parent packages, then its imports."""
        if path not in self._neighbors:
            parts = self.path_to_mod[path].split(".")
            parents = (
                self.mod_to_path.get(".".join(parts[:i])) for i in range(1, len(parts))
            )
            self._neighbors[path] = [p for p in parents if p] + sorted(
                self.imports.get(path, ())
            )
        return self._neighbors[path]

    def reachable(self, starts: Iterable[str]) -> Set[str]:
        seen: Set[str] = set()
        stack = list(starts)
        while stack:
            path = stack.pop()
            if path not in seen:
                seen.add(path)
                stack.extend(self.neighbors(path))
        return seen

    def closure(self, test: str) -> Set[str]:
        """Python files a test loads, including the conftest files pytest loads."""
        return self.reachable([test, *conftest_chain(test, self.tracked)])


class _Resolver:
    """Turns scanned import facts into repository files."""

    def __init__(self, mod_to_path: Dict[str, str], path_to_mod: Dict[str, str]):
        self.mod_to_path = mod_to_path
        self.path_to_mod = path_to_mod
        self.exports: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
        inits = [p for p in path_to_mod if p.endswith("/__init__.py")]
        packages = {str(PurePosixPath(p).parent) for p in inits}
        # Top-level name -> modules a sys.path entry can expose under that name.
        self.path_entries: Dict[str, List[str]] = collections.defaultdict(list)
        self.by_basename: Dict[str, List[str]] = collections.defaultdict(list)
        for path, module in path_to_mod.items():
            self.by_basename[PurePosixPath(path).name].append(path)
            directory = PurePosixPath(path).parent
            container = directory.parent if path.endswith("/__init__.py") else directory
            name = module.rpartition(".")[2]
            if str(container) not in packages and name not in sys.stdlib_module_names:
                self.path_entries[name].append(module)

    def absolute_names(self, name: str) -> List[str]:
        """Modules an absolute import of ``name`` can load."""
        root, dot, rest = name.partition(".")
        if root in MODULE_ROOTS:
            return [name]
        names = [entry + dot + rest for entry in self.path_entries.get(root, ())]
        if root in MODULE_ALIASES:
            names.append(MODULE_ALIASES[root] + dot + rest)
        return names

    def sources(self, path: str, level: int, module: Optional[str]) -> List[str]:
        """Modules a ``from module import ...`` statement in ``path`` reads from."""
        if level == 0:
            return self.absolute_names(module) if module else []
        target = self.absolute(path, level, module)
        return [target] if target else []

    def absolute(self, path: str, level: int, module: Optional[str]) -> Optional[str]:
        """The module a relative name such as ``..jit`` names, seen from ``path``."""
        own = self.path_to_mod[path]
        package = own if path.endswith("__init__.py") else own.rpartition(".")[0]
        parts = package.split(".")
        if level - 1 >= len(parts):
            return None
        base = parts[: len(parts) - (level - 1)]
        return ".".join(base + (module.split(".") if module else []))

    def known(self, module: Optional[str]) -> List[str]:
        """``module`` and its parent packages, where they exist in the repository."""
        if not module:
            return []
        parts = module.split(".")
        names = (".".join(parts[:i]) for i in range(1, len(parts) + 1))
        return [n for n in names if n in self.mod_to_path]

    def string_modules(self, path: str, value: str) -> List[str]:
        """Modules a string such as ``"flashinfer.norm"`` or ``".kernels"`` names."""
        if not MODULE_NAME_RE.fullmatch(value):
            return []
        level = len(value) - len(value.lstrip("."))
        if level:
            targets = [self.absolute(path, level, value[level:] or None)]
        elif value.split(".")[0] in (*MODULE_ROOTS, *MODULE_ALIASES):
            targets = self.absolute_names(value)
        else:
            return []
        return [t for t in targets if t in self.mod_to_path]

    def files_named(self, value: str) -> List[str]:
        """Python files a path literal such as ``"scripts/tool.py"`` names."""
        parts = [p for p in PurePosixPath(value.strip()).parts if p.strip("./")]
        text = "/".join(parts)
        if not text.endswith(".py") or text in GENERIC_FILE_NAMES:
            return []
        return [
            p
            for p in self.by_basename.get(parts[-1], ())
            if len(parts) == 1 or p == text or p.endswith("/" + text)
        ]

    def add_exports(self, facts: FileFacts) -> None:
        """Names a package ``__init__`` exports, mapped to the module providing each."""
        package = self.path_to_mod[facts.path]
        for level, module, names in facts.from_imports:
            sources = self.sources(facts.path, level, module)
            for source in [s for s in sources if s in self.mod_to_path][:1]:
                for name, alias in names:
                    if name != "*":
                        sub = f"{source}.{name}"
                        self.exports[package][alias or name] = (
                            sub if sub in self.mod_to_path else source
                        )
        for name, value in facts.export_pairs:
            for target in self.string_modules(facts.path, value)[:1]:
                self.exports[package][name] = target

    def follow_reexport_chains(self) -> None:
        # Package A may re-export a name from package B, which re-exports it from
        # module C. Resolve the name to C, as the chain of lookups does at runtime.
        for _ in range(10):
            changed = False
            for package, names in self.exports.items():
                for name, target in list(names.items()):
                    deeper = self.exports.get(target, {}).get(name)
                    if deeper and deeper not in (target, package):
                        names[name] = deeper
                        changed = True
            if not changed:
                return

    def imports_of(self, facts: FileFacts) -> Set[str]:
        modules: Set[str] = set()
        aliases: Dict[str, List[str]] = collections.defaultdict(list)
        for name, alias in facts.imports:
            for target in self.absolute_names(name):
                modules.update(self.known(target))
            root = name.split(".")[0]
            aliases[alias or root] += self.absolute_names(name if alias else root)
        for level, module, names in facts.from_imports:
            for source in self.sources(facts.path, level, module):
                modules.update(self.known(source))
                exported = self.exports.get(source, {})
                for name, alias in names:
                    sub = f"{source}.{name}"
                    if sub in self.mod_to_path:
                        modules.add(sub)
                        aliases[alias or name].append(sub)
                    elif name in exported:
                        modules.add(exported[name])
        for value in facts.strings:
            for target in self.string_modules(facts.path, value):
                modules.update(self.known(target))
        # Attribute chains on imported modules, e.g. ``flashinfer.norm.rmsnorm``.
        for root, *attrs in facts.chains:
            for current in aliases.get(root, ()):
                for attr in attrs:
                    if f"{current}.{attr}" in self.mod_to_path:
                        current = f"{current}.{attr}"
                        modules.add(current)
                        continue
                    if attr in self.exports.get(current, {}):
                        modules.add(self.exports[current][attr])
                    break
        found = {self.mod_to_path[m] for m in modules}
        found.update(f for value in facts.strings for f in self.files_named(value))
        found.discard(facts.path)
        return found


class _KernelIndex:
    """Maps string literals in Python code to kernel sources, closed over #include."""

    def __init__(self, repo: Path, tracked: Iterable[str]):
        self.repo = repo
        files = sorted(
            p for p in tracked if p.startswith(KERNEL_ROOTS) and p.endswith(KERNEL_EXTS)
        )
        self.file_set = set(files)
        self.by_basename: Dict[str, List[str]] = collections.defaultdict(list)
        self.under_dir: Dict[str, List[str]] = collections.defaultdict(list)
        for path in files:
            self.by_basename[PurePosixPath(path).name].append(path)
            for directory in _parent_dirs(path):
                if directory in ("csrc", "include", "include/flashinfer"):
                    break
                self.under_dir[directory].append(path)
        self.by_dir_suffix: Dict[str, Set[str]] = collections.defaultdict(set)
        self.top_level_dirs: Dict[str, Set[str]] = collections.defaultdict(set)
        for directory in self.under_dir:
            parts = directory.split("/")
            for i in range(len(parts)):
                self.by_dir_suffix["/".join(parts[i:])].add(directory)
            if str(PurePosixPath(directory).parent) in ("csrc", "include/flashinfer"):
                self.top_level_dirs[parts[-1]].add(directory)
        self._direct: Dict[str, List[str]] = {}
        self._closure: Dict[str, Set[str]] = {}

    def for_literal(self, value: str, directories: bool = True) -> Set[str]:
        text = value.strip().strip("/")
        if text.endswith(KERNEL_EXTS):
            found: Set[str] = set()
            for candidate in self.by_basename.get(PurePosixPath(text).name, ()):
                if "/" not in text or candidate.endswith(text):
                    found |= self.include_closure(candidate)
            return found
        if not directories or text in GENERIC_DIR_NAMES:
            return set()
        # Path-like literals match any directory suffix; bare names match only
        # top-level component directories such as csrc/<name>.
        index = self.by_dir_suffix if "/" in text else self.top_level_dirs
        return {f for d in index.get(text, ()) for f in self.under_dir[d]}

    def _includes(self, path: str) -> List[str]:
        if path not in self._direct:
            try:
                text = (self.repo / path).read_text(errors="ignore")
            except OSError:
                text = ""
            resolved = []
            for name in INCLUDE_RE.findall(text):
                for base in (str(PurePosixPath(path).parent), *INCLUDE_DIRS):
                    candidate = os.path.normpath(f"{base}/{name}")
                    if candidate in self.file_set:
                        resolved.append(candidate)
                        break
            self._direct[path] = resolved
        return self._direct[path]

    def include_closure(self, path: str) -> Set[str]:
        if path not in self._closure:
            seen: Set[str] = set()
            stack = [path]
            while stack:
                current = stack.pop()
                if current not in seen:
                    seen.add(current)
                    stack.extend(self._includes(current))
            self._closure[path] = seen
        return self._closure[path]


def build_graph(repo: Path = REPO_ROOT, jobs: int = os.cpu_count() or 1) -> Graph:
    tracked = set(git("ls-files", "-z", cwd=repo).split("\0")) - {""}
    py_files = sorted(
        p for p in tracked if p.endswith(".py") and p.startswith(PYTHON_ROOTS)
    )
    mod_to_path = {_module_name(p): p for p in py_files}
    path_to_mod = {p: m for m, p in mod_to_path.items()}
    facts = scan_files(repo, list(path_to_mod), jobs)

    resolver = _Resolver(mod_to_path, path_to_mod)
    for path, f in facts.items():
        if path.endswith("__init__.py"):
            resolver.add_exports(f)
    resolver.follow_reexport_chains()
    imports = {p: resolver.imports_of(f) for p, f in facts.items()}
    conftest = facts.get("tests/conftest.py", FileFacts(""))
    plugins = {mod_to_path[p] for p in conftest.plugins if p in mod_to_path}
    test_files = sorted(p for p in path_to_mod if TEST_FILE_RE.match(p))
    owners: Dict[str, Set[str]] = collections.defaultdict(set)

    def graph() -> Graph:
        return Graph(
            tracked, test_files, mod_to_path, path_to_mod, imports, owners, plugins
        )

    # Files under flashinfer/ and tests/ that no test reaches through imports (data,
    # manifests, sources loaded by path) belong to the reached Python files in their
    # directory or the nearest ancestor that has some. Python files become imports of
    # those files, so what they import is followed as well. Under tests/, pytest loads
    # conftest.py and __init__.py for every test below them, and a stray script there
    # (a repro, a smoke run) is loaded by a test, if by anything.
    roots = {c for t in test_files for c in (t, *conftest_chain(t, tracked))}
    reached = graph().reachable(roots)
    loaders = _by_dir(
        p
        for p in reached
        if not p.startswith("tests/") or not p.endswith(("conftest.py", "__init__.py"))
    )
    for path in path_to_mod:
        if path not in reached and path.startswith(("flashinfer/", "tests/")):
            for loader in _nearest(loaders, path):
                imports[loader].add(path)
    loaders = _by_dir(graph().reachable(roots))
    for path in tracked:
        if not path.endswith(".py") and path.startswith(("flashinfer/", "tests/")):
            owners[path].update(_nearest(loaders, path))

    for path in path_to_mod:
        owners[path].add(path)
    kernels = _KernelIndex(repo, tracked)
    for path, f in facts.items():
        # Bare names outside the package are mostly options such as backend names,
        # which would match whole kernel directories; only file names count there.
        directories = path.startswith("flashinfer/")
        for value in f.strings:
            for kernel in kernels.for_literal(value, directories):
                owners[kernel].add(path)
    # Headers included by kernel files matched through a directory literal.
    for path in [p for p in owners if p in kernels.file_set]:
        for header in kernels.include_closure(path):
            owners[header] |= owners[path]
    return graph()


def _by_dir(paths: Iterable[str]) -> Dict[str, List[str]]:
    by_dir: Dict[str, List[str]] = collections.defaultdict(list)
    for path in paths:
        by_dir[str(PurePosixPath(path).parent)].append(path)
    return by_dir


def _nearest(by_dir: Dict[str, List[str]], path: str) -> List[str]:
    """Files in the nearest ancestor directory of ``path`` that has any."""
    for directory in _parent_dirs(path):
        if directory in by_dir:
            return by_dir[directory]
    return []


# ------------------------------------------------------------------------- selection


@dataclass
class Selection:
    changed: List[str]
    tests: List[str]
    total_tests: int
    global_inputs: List[str] = field(default_factory=list)
    reasons: Dict[str, int] = field(default_factory=dict)  # changed file -> tests
    unreached: List[str] = field(default_factory=list)  # changed, but no test loads it
    error: Optional[str] = None

    @property
    def full_suite(self) -> bool:
        return bool(self.global_inputs) or self.error is not None


class Selector:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.tests_by_file: Dict[str, Set[str]] = collections.defaultdict(set)
        for test in graph.test_files:
            for path in graph.closure(test):
                self.tests_by_file[path].add(test)
        # Owners of everything under each directory, for files that no longer exist.
        self.owners_under: Dict[str, Set[str]] = collections.defaultdict(set)
        for path, owners in graph.owners.items():
            for directory in _parent_dirs(path):
                self.owners_under[directory] |= owners

    def owners_of(self, path: str) -> Set[str]:
        if path in self.graph.owners:
            return self.graph.owners[path]
        if path in self.graph.tracked or not path.startswith(
            PYTHON_ROOTS + KERNEL_ROOTS
        ):
            return set()
        # Deleted or renamed away: everything under the nearest surviving directory
        # below the top level, or everything if no such directory survives.
        for directory in _parent_dirs(path):
            if "/" in directory and directory in self.owners_under:
                return self.owners_under[directory]
        return set(self.graph.path_to_mod)

    def select(self, changed: Sequence[str]) -> Selection:
        tests = self.graph.test_files
        global_inputs = sorted(
            f
            for f in changed
            if GLOBAL_INPUTS.match(f) or f in self.graph.pytest_plugins
        )
        if global_inputs:
            return Selection(list(changed), list(tests), len(tests), global_inputs)
        selected: Set[str] = set()
        reasons: Dict[str, int] = {}
        unreached: List[str] = []
        for path in changed:
            hits = {
                t for o in self.owners_of(path) for t in self.tests_by_file.get(o, ())
            }
            if hits:
                reasons[path] = len(hits)
                selected |= hits
            elif path.startswith(("flashinfer/",) + KERNEL_ROOTS):
                unreached.append(path)
        return Selection(
            list(changed), sorted(selected), len(tests), [], reasons, sorted(unreached)
        )

    def explain(self, test: str, changed: Sequence[str]) -> Optional[List[str]]:
        """Shortest chain of loads from ``test`` to a changed file."""
        target_of: Dict[str, str] = {}
        for path in changed:
            for owner in self.owners_of(path):
                target_of.setdefault(owner, path)
        start = [test, *conftest_chain(test, self.graph.tracked)]
        parent: Dict[str, Optional[str]] = {s: None for s in start}
        queue = collections.deque(start)
        while queue:
            node = queue.popleft()
            if node in target_of:
                chain = [] if target_of[node] == node else [target_of[node]]
                cursor: Optional[str] = node
                while cursor is not None:
                    chain.append(cursor)
                    cursor = parent[cursor]
                return chain[::-1]
            for nxt in self.graph.neighbors(node):
                if nxt not in parent:
                    parent[nxt] = node
                    queue.append(nxt)
        return None


def changed_files(base: str, head: Optional[str], repo: Path = REPO_ROOT) -> List[str]:
    merge_base = git("merge-base", base, head or "HEAD", cwd=repo).strip()
    args = ["diff", "--name-only", "--no-renames", merge_base] + (
        [head] if head else []
    )
    return sorted({line for line in git(*args, cwd=repo).splitlines() if line})


def scope_covers(scope: Sequence[str], test: str) -> bool:
    return any(
        test == t.rstrip("/") or test.startswith(t.rstrip("/") + "/") for t in scope
    )


# --------------------------------------------------------------------------- reports


def _examples(items: Sequence[str], limit: int = 5) -> str:
    return ", ".join(f"`{i}`" for i in items[:limit]) + (
        ", ..." if len(items) > limit else ""
    )


def render_markdown(sel: Selection, requested_scope: Sequence[str]) -> str:
    lines = [
        "### Dependency-based test selection (shadow mode)",
        "",
        "Informational only: this does not change which tests run.",
        "",
    ]
    if sel.error:
        lines.append(
            f"The selector failed, so it would select every test: `{sel.error}`"
        )
    elif sel.global_inputs:
        lines.append(
            f"Would run all {sel.total_tests} test files: the change touches global "
            f"inputs ({_examples(sel.global_inputs)})."
        )
    else:
        lines.append(
            f"Would run **{len(sel.tests)} of {sel.total_tests}** test files for "
            f"{len(sel.changed)} changed files."
        )
        if sel.reasons:
            lines += ["", "| Changed file | Test files it selects |", "| --- | ---: |"]
            top = sorted(sel.reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            lines += [f"| `{path}` | {count} |" for path, count in top]
        if sel.unreached:
            lines += [
                "",
                f"No test loads {len(sel.unreached)} changed package or kernel files: "
                f"{_examples(sel.unreached)}.",
            ]
    if requested_scope and not sel.error:
        missed = [t for t in sel.tests if not scope_covers(requested_scope, t)]
        lines.append("")
        lines.append(
            f"The requested scope leaves out {len(missed)} selected test files: "
            f"{_examples(missed)}."
            if missed
            else "The requested scope covers every selected test file."
        )
    if sel.tests and not sel.full_suite:
        lines += ["", "<details><summary>Selected test files</summary>", ""]
        lines += [f"- `{t}`" for t in sel.tests]
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--base", required=True, help="base revision; the diff starts at its merge base"
    )
    parser.add_argument(
        "--head",
        help="head revision (default: the working tree); the dependency graph is "
        "always read from the working tree, so check out the head first",
    )
    parser.add_argument(
        "--format", choices=("paths", "json", "markdown"), default="paths"
    )
    parser.add_argument(
        "--requested-scope",
        default="",
        help="space-separated TEST_PATH targets to compare with the selection",
    )
    parser.add_argument("--explain", metavar="TEST", help="print why TEST is selected")
    parser.add_argument(
        "--jobs", type=int, default=os.cpu_count() or 1, help=argparse.SUPPRESS
    )
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    selector: Optional[Selector] = None
    try:
        changed = changed_files(args.base, args.head, args.repo)
        selector = Selector(build_graph(args.repo, args.jobs))
        selection = selector.select(changed)
    except Exception as exc:  # any failure selects every test
        print(f"select_tests: {exc!r}; selecting every test", file=sys.stderr)
        listed = git("ls-files", "tests", cwd=args.repo).split()
        tests = sorted(p for p in listed if TEST_FILE_RE.match(p))
        selection = Selection([], tests, len(tests), error=repr(exc))

    if args.explain:
        test = os.path.normpath(args.explain)
        if selector is not None and test not in selector.graph.path_to_mod:
            print(f"select_tests: {test} is not a tracked Python file", file=sys.stderr)
            return 2
        if selection.error:
            print(f"{test} is selected because the selector failed: {selection.error}")
        elif selection.global_inputs:
            inputs = ", ".join(selection.global_inputs)
            print(f"{test} is selected because global inputs changed: {inputs}")
        else:
            chain = selector.explain(test, selection.changed) if selector else None
            print(" -> ".join(chain) if chain else f"{test} is not selected")
    elif args.format == "paths":
        print("\n".join(selection.tests))
    elif args.format == "json":
        print(
            json.dumps(
                dict(asdict(selection), full_suite=selection.full_suite), indent=1
            )
        )
    else:
        sys.stdout.write(render_markdown(selection, args.requested_scope.split()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
