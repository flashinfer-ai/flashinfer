"""In-memory CuTe-DSL compile cache for the SM12x W4A16 fused MoE path.

FlashInfer-local stand-in for sparkinfer's ``_lib/compiler.py``. Upstream layers
a spec-keyed memory cache, an on-disk object cache, and compile-progress
telemetry over ``cute.compile``; the W4A16 kernel call sites only rely on the
call surface (``KernelCompileSpec.from_key`` / ``from_facts`` plus
``compile(kernel, *fakes, compile_spec=..., dsl_compile_options=...)``) and on
memoization per compile spec. This module provides exactly that surface so the
ported kernel code stays byte-comparable with upstream, backed by a plain
process-local dict (FlashInfer's existing caching model for these kernels).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from threading import RLock
from typing import Any


def _json_pod(value: Any, *, path: str = "value") -> Any:
    """Coerce a compile-spec fact into a JSON-stable plain value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_pod(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {
            str(k): _json_pod(v, path=f"{path}.{k}") for k, v in sorted(value.items())
        }
    raise TypeError(
        f"compile spec fact {path} of type {type(value).__name__} is not a "
        "JSON-stable plain value"
    )


@dataclass(frozen=True)
class KernelCompileSpec:
    kernel_id: str
    version: int
    json_key: str

    @staticmethod
    def from_facts(kernel_id: str, version: int, *facts: object) -> "KernelCompileSpec":
        json_key = json.dumps(
            [str(kernel_id), int(version), _json_pod(list(facts), path="facts")],
            separators=(",", ":"),
        )
        return KernelCompileSpec(
            kernel_id=str(kernel_id), version=int(version), json_key=json_key
        )

    @staticmethod
    def from_key(
        kernel_id: str,
        version: int,
        key: tuple[object, ...],
        *,
        labels: tuple[str, ...] | None = None,
    ) -> "KernelCompileSpec":
        if labels is not None and len(labels) != len(key):
            raise ValueError(
                f"compile spec labels length {len(labels)} does not match "
                f"key length {len(key)}"
            )
        facts: tuple[object, ...]
        if labels is not None:
            facts = tuple((str(labels[idx]), value) for idx, value in enumerate(key))
        else:
            facts = tuple(key)
        return KernelCompileSpec.from_facts(kernel_id, version, *facts)


_COMPILE_CACHE: dict[tuple[str, str], Any] = {}
_COMPILE_CACHE_LOCK = RLock()


def compile(
    func: Any,
    *args: Any,
    compile_spec: KernelCompileSpec | None = None,
    dsl_compile_options: Any = None,
    **kwargs: Any,
) -> Any:
    """Memoizing wrapper over ``cute.compile`` keyed on the compile spec.

    Every W4A16 call site passes a ``compile_spec`` that fully identifies the
    kernel specialization (upstream relies on the same property for its disk
    cache), so the spec plus the DSL compile options is a sufficient cache key.
    """
    import cutlass.cute as cute

    if compile_spec is None:
        raise ValueError("compile() requires a compile_spec")
    cache_key = (compile_spec.json_key, repr(dsl_compile_options))
    with _COMPILE_CACHE_LOCK:
        compiled = _COMPILE_CACHE.get(cache_key)
    if compiled is not None:
        return compiled

    compile_callable = cute.compile
    if dsl_compile_options is not None:
        compile_callable = compile_callable[dsl_compile_options]
    compiled = compile_callable(func, *args, **kwargs)
    with _COMPILE_CACHE_LOCK:
        _COMPILE_CACHE[cache_key] = compiled
    return compiled


def clear_compile_cache() -> None:
    with _COMPILE_CACHE_LOCK:
        _COMPILE_CACHE.clear()
