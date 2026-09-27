"""Check the APIs used by frozen cuDNN Frost sources without compiling or executing them."""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any


@functools.cache
def _source_requirements(path: Path, digest: str):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != digest:
        raise RuntimeError(f"cuDNN Frost generated source digest mismatch: {path}")
    tree = ast.parse(source, filename=str(path))
    imports: dict[str, set[str]] = {}
    aliases: dict[str, str] = {}
    symbols: set[str] = set()
    calls: set[tuple[str, int, tuple[str, ...]]] = set()
    # Frozen sources use module-level imports, including their inlined helpers.
    # Derive requirements from those sources so new kernels cannot silently
    # outgrow a manually maintained list of compiler functions and enum members.
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name.startswith(("cutlass", "cuda.")):
                    imports.setdefault(alias.name, set())
                    if alias.asname is None:
                        imports.setdefault(alias.name.split(".")[0], set())
                    aliases[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
        elif isinstance(statement, ast.ImportFrom) and (
            statement.module or ""
        ).startswith(("cutlass", "cuda.")):
            assert statement.module is not None
            for alias in statement.names:
                imports.setdefault(statement.module, set()).add(alias.name)
                name = f"{statement.module}.{alias.name}"
                aliases[alias.asname or alias.name] = name
                symbols.add(name)

    def qualified_name(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name) and node.id in aliases:
            return ".".join([aliases[node.id], *reversed(parts)])
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = qualified_name(node)
            if name is not None:
                symbols.add(name)
        elif isinstance(node, ast.Call):
            name = qualified_name(node.func)
            if name is not None:
                symbols.add(name)
                # Dynamic argument expansion cannot be checked with Python's
                # signature binder. The callable itself is still checked.
                if not any(isinstance(arg, ast.Starred) for arg in node.args) and all(
                    kw.arg is not None for kw in node.keywords
                ):
                    calls.add(
                        (
                            name,
                            len(node.args),
                            tuple(sorted(kw.arg for kw in node.keywords)),
                        )
                    )
                else:
                    calls.add((name, -1, ()))
    return imports, symbols, calls


@functools.cache
def _compiler_error(arch: str, sources: tuple[tuple[Path, str], ...]) -> str | None:
    imports: dict[str, set[str]] = {
        "cutlass.cute": set(),
        "cutlass.cute.runtime": set(),
    }
    symbols = {
        "cutlass.cute.compile",
        "cutlass.cute.GPUArch",
        "cutlass.cute.EnableTVMFFI",
        "cutlass.cute.runtime.load_module",
    }
    calls = {("cutlass.cute.runtime.load_module", 1, ("enable_tvm_ffi",))}
    seen = set()
    for path, digest in sources:
        if digest in seen:
            continue
        seen.add(digest)
        required_imports, required_symbols, required_calls = _source_requirements(
            path, digest
        )
        for module, names in required_imports.items():
            imports.setdefault(module, set()).update(names)
        symbols.update(required_symbols)
        calls.update(required_calls)

    modules = {}
    for name, names in sorted(imports.items()):
        try:
            # fromlist also imports submodules such as cuda.bindings.driver and
            # cutlass._mlir.dialects.llvm, matching the generated Python imports.
            modules[name] = __import__(name, fromlist=tuple(names) or ("__name__",))
        except (ImportError, AttributeError, OSError) as exc:
            return f"cannot import {name}: {exc}"

    resolved: dict[str, Any] = {}
    for name in sorted(symbols):
        module_name = max(
            (
                module
                for module in modules
                if name == module or name.startswith(module + ".")
            ),
            key=len,
        )
        value = modules[module_name]
        try:
            for part in (
                name[len(module_name) + 1 :].split(".") if name != module_name else ()
            ):
                value = getattr(value, part)
            if value is None:
                raise AttributeError(name)
        except (ImportError, AttributeError) as exc:
            return f"missing API {name}: {exc}"
        resolved[name] = value

    for name, positional, keywords in sorted(calls):
        function = resolved[name]
        if not callable(function):
            return f"API {name} is not callable"
        if positional < 0:
            continue
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            # Some extension/DSL builtins do not expose Python signatures.
            continue
        try:
            signature.bind(*([None] * positional), **dict.fromkeys(keywords))
        except TypeError as exc:
            return f"incompatible signature for {name}: {exc}"

    try:
        resolved["cutlass.cute.GPUArch"](arch)
        resolved["cutlass.cute.EnableTVMFFI"]()
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        return f"compiler cannot target {arch} with TVM-FFI: {exc}"
    return None


def require_compiler(arch: str, sources: tuple[tuple[Path, str], ...]) -> None:
    """Raise NotImplementedError for unsupported compiler capabilities.

    Positive and negative probes are cached by target and source identity.
    Environment overrides are checked on every probe. This is an API and
    compile-option probe, not a guarantee against later compiler/runtime bugs.
    """
    target = os.environ.get("CUTE_DSL_ARCH", arch)
    if target.replace("_", "") != arch.replace("_", ""):
        raise NotImplementedError(
            f"cuDNN Frost source JIT requires {arch}; CUTE_DSL_ARCH={target} conflicts"
        )
    reason = _compiler_error(arch, sources)
    if reason is not None:
        raise NotImplementedError(
            f"cuDNN Frost source JIT requires compatible compiler APIs: {reason}"
        )
