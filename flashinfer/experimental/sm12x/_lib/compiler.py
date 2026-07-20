# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/cute/compiler.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager, suppress
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any

from .env import sync_legacy_env
from .runtime_patches import apply_cutlass_runtime_patches

sync_legacy_env()
apply_cutlass_runtime_patches()

# The package fingerprint hashes every source file under this root; it is
# part of every disk-cache key.  Must resolve to flashinfer/experimental/sm12x.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_MEMORY_CACHE: OrderedDict[object, Any] = OrderedDict()
_MEMORY_CACHE_LOCK = RLock()
_SPEC_MEMO: OrderedDict[tuple[object, ...], Any] = OrderedDict()
_SPEC_MEMO_LOCK = RLock()
_SPEC_MEMO_MAX = 8192
_SPEC_MEMO_HITS = 0
_SPEC_MEMO_MISSES = 0
_MEMORY_CACHE_HITS = 0
_MEMORY_CACHE_MISSES = 0
_DISK_CACHE_HITS = 0
_COMPILE_MISSES = 0
_COMPILE_PROGRESS_LOCK = RLock()
_COMPILE_PROGRESS_COUNT = 0
_COMPILE_PROGRESS_TOTAL_SECONDS = 0.0
_EXECUTOR_CACHE_LOCK = RLock()
_EXECUTOR_CACHE_ATTR = "_sm12x_cached_default_executor"
_VLLM_ENGINE_STARTED_ENV = "FLASHINFER_EXP_SM12X_ENGINE_STARTED"
_POST_ENGINE_START_LOG_ENV = "FLASHINFER_EXP_SM12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START"
_PRINT_COMPILE_PROGRESS_ENV = "FLASHINFER_EXP_SM12X_PRINT_COMPILE_PROGRESS"


@dataclass(frozen=True)
class DimKey:
    kind: str
    value: object = None

    @staticmethod
    def exact(value: object) -> "DimKey":
        return DimKey("exact", value)

    @staticmethod
    def capacity(value: object) -> "DimKey":
        return DimKey("capacity", value)

    @staticmethod
    def bucket(value: object) -> "DimKey":
        return DimKey("bucket", value)

    @staticmethod
    def dynamic() -> "DimKey":
        return DimKey("dynamic")

    @staticmethod
    def ignored() -> "DimKey":
        return DimKey("ignored")


@dataclass(frozen=True)
class TensorKey:
    name: str
    dtype: str
    rank: int
    dims: tuple[DimKey, ...]
    stride: tuple[int, ...]
    device: tuple[str, int | None]
    align: int | None = None
    layout: object = None

    @staticmethod
    def from_tensor(
        name: str,
        tensor: Any,
        *,
        dims: tuple[DimKey, ...] | None = None,
        align: int | None = None,
        layout: object = None,
    ) -> "TensorKey":
        shape = tuple(int(dim) for dim in tensor.shape)
        if dims is None:
            dims = tuple(DimKey.exact(dim) for dim in shape)
        if len(dims) != len(shape):
            raise ValueError(
                f"tensor key {name!r} dim policy rank {len(dims)} "
                f"does not match tensor rank {len(shape)}"
            )
        device = tensor.device
        return TensorKey(
            name=name,
            dtype=str(tensor.dtype),
            rank=len(shape),
            dims=dims,
            stride=tuple(int(stride) for stride in tensor.stride()),
            device=(device.type, device.index),
            align=align,
            layout=layout,
        )


def _spec_memo_enabled() -> bool:
    raw = os.environ.get("FLASHINFER_EXP_SM12X_COMPILE_SPEC_MEMO", "1")
    return raw.lower() not in {"0", "false", "no", ""}


def _spec_memo_key(
    kind: str,
    kernel_id: str,
    version: int,
    payload: object,
    extra: object = None,
) -> tuple[object, ...] | None:
    # repr() of a JSON-POD payload is content-deterministic and preserves
    # type distinctions (True vs 1, tuple vs list), so it is a safe memo key.
    # Non-POD payloads never get stored (see _spec_memo_put callers), which
    # keeps id-based reprs out of the cache.
    if not _spec_memo_enabled():
        return None
    try:
        memo_key = (kind, str(kernel_id), int(version), repr(payload), extra)
        hash(memo_key)
    except Exception:
        return None
    return memo_key


def _spec_memo_get(memo_key: tuple[object, ...] | None) -> Any | None:
    global _SPEC_MEMO_HITS
    global _SPEC_MEMO_MISSES
    if memo_key is None:
        return None
    with _SPEC_MEMO_LOCK:
        spec = _SPEC_MEMO.get(memo_key)
        if spec is None:
            _SPEC_MEMO_MISSES += 1
            return None
        _SPEC_MEMO_HITS += 1
        _SPEC_MEMO.move_to_end(memo_key)
        return spec


def _spec_memo_put(memo_key: tuple[object, ...] | None, spec: Any) -> None:
    if memo_key is None:
        return
    with _SPEC_MEMO_LOCK:
        _SPEC_MEMO[memo_key] = spec
        _SPEC_MEMO.move_to_end(memo_key)
        while len(_SPEC_MEMO) > _SPEC_MEMO_MAX:
            _SPEC_MEMO.popitem(last=False)


@dataclass(frozen=True)
class KeyField:
    name: str
    value: object


@dataclass(frozen=True)
class KernelCompileSpec:
    kernel_id: str
    version: int
    fields: tuple[KeyField, ...] = ()
    json_key: str = ""
    hash_key: str = ""
    legacy: bool = False

    def __post_init__(self) -> None:
        if self.json_key and self.hash_key:
            return
        json_key = _compile_spec_json(
            self.kernel_id,
            self.version,
            _legacy_compile_spec_facts(self.kernel_id, self.version, self.fields),
        )
        object.__setattr__(self, "json_key", json_key)
        object.__setattr__(self, "hash_key", _hash_json_key(json_key))
        object.__setattr__(self, "legacy", True)

    @staticmethod
    def from_facts(
        kernel_id: str,
        version: int,
        *facts: object,
    ) -> "KernelCompileSpec":
        memo_key = _spec_memo_key("facts", kernel_id, version, facts)
        spec = _spec_memo_get(memo_key)
        if spec is not None:
            return spec
        json_key = _compile_spec_json(kernel_id, version, facts)
        spec = KernelCompileSpec(
            kernel_id=str(kernel_id),
            version=int(version),
            fields=(),
            json_key=json_key,
            hash_key=_hash_json_key(json_key),
            legacy=False,
        )
        # _compile_spec_json raises for non-POD facts, so only POD-validated
        # specs reach the memo.
        _spec_memo_put(memo_key, spec)
        return spec

    @staticmethod
    def from_fields(
        kernel_id: str,
        version: int,
        *fields: KeyField | tuple[str, object],
    ) -> "KernelCompileSpec":
        memo_key = _spec_memo_key("fields", kernel_id, version, fields)
        spec = _spec_memo_get(memo_key)
        if spec is not None:
            return spec
        coerced_fields = tuple(_coerce_key_field(field) for field in fields)
        if all(_is_json_pod(field.value) for field in coerced_fields):
            spec = KernelCompileSpec.from_facts(
                kernel_id,
                version,
                *((field.name, field.value) for field in coerced_fields),
            )
            _spec_memo_put(memo_key, spec)
            return spec
        return KernelCompileSpec(
            kernel_id=kernel_id,
            version=int(version),
            fields=coerced_fields,
            legacy=True,
        )

    @staticmethod
    def from_legacy_fields(
        kernel_id: str,
        version: int,
        *fields: KeyField | tuple[str, object],
    ) -> "KernelCompileSpec":
        return KernelCompileSpec(
            kernel_id=kernel_id,
            version=int(version),
            fields=tuple(_coerce_key_field(field) for field in fields),
            legacy=True,
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
        memo_key = _spec_memo_key("key", kernel_id, version, key, extra=labels)
        spec = _spec_memo_get(memo_key)
        if spec is not None:
            return spec
        if _is_json_pod(key):
            facts = (
                tuple((str(labels[idx]), value) for idx, value in enumerate(key))
                if labels is not None
                else key
            )
            spec = KernelCompileSpec.from_facts(kernel_id, version, *facts)
            _spec_memo_put(memo_key, spec)
            return spec
        return KernelCompileSpec(
            kernel_id=kernel_id,
            version=int(version),
            fields=tuple(
                KeyField(labels[idx] if labels is not None else f"arg{idx}", value)
                for idx, value in enumerate(key)
            ),
            legacy=True,
        )

    @staticmethod
    def from_legacy_key(
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
        return KernelCompileSpec(
            kernel_id=kernel_id,
            version=int(version),
            fields=tuple(
                KeyField(labels[idx] if labels is not None else f"arg{idx}", value)
                for idx, value in enumerate(key)
            ),
            legacy=True,
        )


def _coerce_key_field(field: KeyField | tuple[str, object]) -> KeyField:
    if isinstance(field, KeyField):
        return field
    name, value = field
    return KeyField(str(name), value)


def key_field(name: str, value: object) -> KeyField:
    return KeyField(name, value)


def tensor_key(
    name: str,
    tensor: Any | None,
    *,
    dims: tuple[DimKey, ...] | None = None,
    align: int | None = None,
    layout: object = None,
) -> KeyField:
    if tensor is None:
        value = None
    else:
        try:
            value = tensor_compile_fact(
                name,
                tensor,
                dims=dims,
                align=align,
                layout=layout,
            )
        except TypeError:
            value = TensorKey.from_tensor(
                name,
                tensor,
                dims=dims,
                align=align,
                layout=layout,
            )
    return KeyField(name, value)


def dim_compile_fact(dim: DimKey | object) -> tuple[str, object]:
    return _dim_policy_fact(dim)


def _dim_policy_fact(dim: DimKey | object) -> tuple[str, object]:
    if isinstance(dim, DimKey):
        return ("dim", str(dim.kind), _json_pod(dim.value, path="dim.value"))
    return ("dim", "exact", _json_pod(dim, path="dim.value"))


def tensor_compile_fact(
    name: str,
    tensor: Any,
    *,
    dims: tuple[DimKey | object, ...] | None = None,
    dynamic_dims: tuple[int, ...] = (),
    strides: tuple[DimKey | object, ...] | None = None,
    dynamic_strides: tuple[int, ...] = (),
    align: int | None = None,
    layout: object = None,
) -> tuple[object, ...]:
    shape = tuple(int(dim) for dim in tensor.shape)
    if dims is None:
        dynamic_dim_set = set(dynamic_dims)
        dims = tuple(
            DimKey.dynamic() if idx in dynamic_dim_set else DimKey.exact(dim)
            for idx, dim in enumerate(shape)
        )
    if len(dims) != len(shape):
        raise ValueError(
            f"tensor key {name!r} dim policy rank {len(dims)} "
            f"does not match tensor rank {len(shape)}"
        )

    raw_strides = tuple(int(stride) for stride in tensor.stride())
    if strides is None:
        dynamic_stride_set = set(dynamic_strides)
        strides = tuple(
            DimKey.dynamic() if idx in dynamic_stride_set else DimKey.exact(stride)
            for idx, stride in enumerate(raw_strides)
        )
    if len(strides) != len(raw_strides):
        raise ValueError(
            f"tensor key {name!r} stride policy rank {len(strides)} "
            f"does not match tensor rank {len(raw_strides)}"
        )

    device = tensor.device
    layout_fact = None if layout is None else _json_pod(layout, path="layout")
    return (
        "tensor",
        str(name),
        str(tensor.dtype),
        len(shape),
        tuple(_dim_policy_fact(dim) for dim in dims),
        tuple(_dim_policy_fact(stride) for stride in strides),
        (str(device.type), device.index),
        None if align is None else int(align),
        layout_fact,
    )


def _is_json_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _is_json_pod(value: Any) -> bool:
    if _is_json_scalar(value):
        return True
    if isinstance(value, DimKey):
        return _is_json_pod(value.value)
    if isinstance(value, KeyField):
        return _is_json_pod(value.value)
    if isinstance(value, (TensorKey, KernelCompileSpec)):
        return False
    if isinstance(value, (tuple, list)):
        return all(_is_json_pod(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_pod(item) for key, item in value.items()
        )
    return False


def _json_pod(value: Any, *, path: str = "value") -> Any:
    if _is_json_scalar(value):
        return value
    if isinstance(value, float):
        raise TypeError(f"{path} must be finite for JSON compile specs")
    if isinstance(value, DimKey):
        return [
            "dim",
            str(value.kind),
            _json_pod(value.value, path=f"{path}.value"),
        ]
    if isinstance(value, KeyField):
        return [
            "field",
            str(value.name),
            _json_pod(value.value, path=f"{path}.{value.name}"),
        ]
    if isinstance(value, (TensorKey, KernelCompileSpec)):
        raise TypeError(
            f"{path} contains legacy compile-key object {type(value).__name__}; "
            "use explicit POD facts instead"
        )
    if isinstance(value, tuple):
        return [
            _json_pod(item, path=f"{path}[{idx}]") for idx, item in enumerate(value)
        ]
    if isinstance(value, list):
        return [
            _json_pod(item, path=f"{path}[{idx}]") for idx, item in enumerate(value)
        ]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} dict key {key!r} is not a string")
            out[key] = _json_pod(item, path=f"{path}.{key}")
        return out
    raise TypeError(
        f"{path} contains non-POD compile fact "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _json_dumps_pod(value: Any) -> str:
    return json.dumps(
        _json_pod(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _hash_json_key(json_key: str) -> str:
    return hashlib.sha256(json_key.encode("utf-8")).hexdigest()


def _compile_spec_json(
    kernel_id: str,
    version: int,
    facts: object,
) -> str:
    return _json_dumps_pod(
        {
            "facts": facts,
            "kernel": str(kernel_id),
            "version": int(version),
        }
    )


def _legacy_compile_spec_facts(
    kernel_id: str,
    version: int,
    fields: tuple[KeyField, ...],
) -> tuple[object, ...]:
    return (
        "legacy",
        str(kernel_id),
        int(version),
        tuple(_compile_spec_shape_key(field) for field in fields),
    )


def _compile_kwargs_json_key(kwargs: dict[str, Any]) -> tuple[str, str]:
    if not kwargs:
        return "", ""
    try:
        json_key = _json_dumps_pod(kwargs)
    except TypeError:
        json_key = _json_dumps_pod(_compile_spec_shape_key(kwargs))
    return json_key, _hash_json_key(json_key)


def _compile_spec_shape_key(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return ("bytes", value)
    if isinstance(value, Path):
        return ("path", str(value))
    if isinstance(value, DimKey):
        return ("dim", value.kind, _compile_spec_shape_key(value.value))
    if isinstance(value, TensorKey):
        return (
            "tensor",
            value.name,
            value.dtype,
            value.rank,
            tuple(_compile_spec_shape_key(dim) for dim in value.dims),
            value.stride,
            value.device,
            value.align,
            _compile_spec_shape_key(value.layout),
        )
    if isinstance(value, KeyField):
        return ("field", value.name, _compile_spec_shape_key(value.value))
    if isinstance(value, KernelCompileSpec):
        return (
            "kernel_spec_json" if not value.legacy else "kernel_spec_legacy_json",
            value.kernel_id,
            value.version,
            value.hash_key,
            value.json_key,
        )
    if isinstance(value, (tuple, list)):
        return tuple(_compile_spec_shape_key(item) for item in value)
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (
                    _compile_spec_shape_key(key),
                    _compile_spec_shape_key(item_value),
                )
                for key, item_value in value.items()
            ),
        )
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    cache_key_attr = getattr(value, "__cache_key__", None)
    if cache_key_attr is not None:
        return (
            "cache_key",
            type(value).__module__,
            type(value).__qualname__,
            _compile_spec_shape_key(cache_key_attr),
        )
    if is_dataclass(value):
        return (
            "dataclass",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (
                    field.name,
                    _compile_spec_shape_key(getattr(value, field.name)),
                )
                for field in fields(value)
            ),
        )
    return (
        "repr",
        type(value).__module__,
        type(value).__qualname__,
        repr(value),
    )


def _compile_memory_cache_key(
    compile_callable: Any,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    compile_spec: KernelCompileSpec | None,
) -> object:
    if compile_spec is not None:
        kwargs_json_key, kwargs_hash_key = _compile_kwargs_json_key(kwargs)
        key: tuple[object, ...] = (
            "sm12x_cute_memory_cache_v2_explicit_spec",
            compile_spec.hash_key,
        )
        if kwargs_hash_key:
            key += (kwargs_hash_key, kwargs_json_key)
        return key

    return hashlib.sha256(
        repr(_compile_disk_cache_payload(compile_callable, func, args, kwargs)).encode(
            "utf-8"
        )
    ).hexdigest()


def _cute_compile_memory_cache_enabled() -> bool:
    raw = os.environ.get("FLASHINFER_EXP_SM12X_COMPILE_MEMORY_CACHE", "1")
    return raw.lower() not in {"0", "false", "no", ""}


def _cute_compile_memory_cache_size() -> int:
    raw = os.environ.get("FLASHINFER_EXP_SM12X_COMPILE_MEMORY_CACHE_SIZE", "1024")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1024


def _cute_compile_disk_cache_enabled() -> bool:
    raw = os.environ.get("FLASHINFER_EXP_SM12X_COMPILE_DISK_CACHE", "1")
    return raw.lower() not in {"0", "false", "no", ""}


def _cute_compile_cache_dir() -> Path:
    root = os.environ.get("FLASHINFER_EXP_SM12X_COMPILE_CACHE_DIR")
    if root:
        return Path(root)
    flashinfer_cache_dir = os.environ.get("FLASHINFER_CACHE_DIR")
    if flashinfer_cache_dir:
        return Path(flashinfer_cache_dir) / "experimental" / "sm12x" / "compile"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return (
            Path(xdg_cache_home) / "flashinfer" / "experimental" / "sm12x" / "compile"
        )
    return Path.home() / ".cache" / "flashinfer" / "experimental" / "sm12x" / "compile"


def _cute_compile_log_enabled() -> bool:
    raw = os.environ.get("FLASHINFER_EXP_SM12X_LOG_CUTE_COMPILES", "")
    return raw.lower() not in {"", "0", "false", "no", "off"}


def _cute_compile_progress_enabled() -> bool:
    raw = os.environ.get(_PRINT_COMPILE_PROGRESS_ENV, "")
    return raw.lower() not in {"", "0", "false", "no", "off"}


def _cute_compile_post_engine_start_log_enabled() -> bool:
    marker = os.environ.get(_VLLM_ENGINE_STARTED_ENV, "")
    if marker.lower() in {"", "0", "false", "no", "off"}:
        return False
    raw = os.environ.get(_POST_ENGINE_START_LOG_ENV)
    if raw is None:
        return True
    return raw.lower() not in {"", "0", "false", "no", "off"}


def _cute_compile_stack_log_enabled() -> bool:
    raw = os.environ.get("FLASHINFER_EXP_SM12X_LOG_CUTE_COMPILE_STACK", "")
    if raw:
        return raw.lower() not in {"0", "false", "no", "off"}
    raw = os.environ.get("FLASHINFER_EXP_SM12X_LOG_CUTE_COMPILES", "")
    if raw.lower() in {"stack", "trace", "traceback", "full"}:
        return True
    raw = os.environ.get(_POST_ENGINE_START_LOG_ENV, "")
    return raw.lower() in {"stack", "trace", "traceback", "full"}


def _cute_compile_stack_log_depth() -> int:
    raw = os.environ.get("FLASHINFER_EXP_SM12X_LOG_CUTE_COMPILE_STACK_DEPTH", "")
    if not raw:
        return 48
    try:
        return max(1, int(raw))
    except ValueError:
        return 48


def _short_repr(value: Any, *, max_len: int = 160) -> str:
    try:
        if isinstance(value, type):
            text = f"{value.__module__}.{value.__qualname__}"
        else:
            text = repr(value)
    except Exception:
        text = f"<{type(value).__module__}.{type(value).__qualname__}>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _compile_target_name(func: Any) -> str:
    unwrapped = inspect.unwrap(func)
    if inspect.ismethod(unwrapped):
        module = getattr(unwrapped.__func__, "__module__", "")
        qualname = getattr(
            unwrapped.__func__,
            "__qualname__",
            getattr(unwrapped.__func__, "__name__", ""),
        )
        return f"{module}.{qualname}" if module else qualname
    if inspect.isfunction(unwrapped):
        module = getattr(unwrapped, "__module__", "")
        qualname = getattr(
            unwrapped, "__qualname__", getattr(unwrapped, "__name__", "")
        )
        return f"{module}.{qualname}" if module else qualname
    target_type = type(func)
    module = getattr(target_type, "__module__", "")
    qualname = getattr(target_type, "__qualname__", target_type.__name__)
    return f"{module}.{qualname}" if module else qualname


def _simple_log_value(value: Any) -> Any | None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, (tuple, list)) and len(value) <= 8:
        items = []
        for item in value:
            simple = _simple_log_value(item)
            if simple is None:
                return None
            items.append(simple)
        return tuple(items) if isinstance(value, tuple) else items
    return None


def _compile_target_attrs(func: Any) -> dict[str, Any]:
    if not hasattr(func, "__dict__"):
        return {}
    attrs = {}
    for name, value in sorted(vars(func).items()):
        if name.startswith("_"):
            continue
        simple = _simple_log_value(value)
        if simple is None:
            continue
        attrs[name] = simple
        if len(attrs) >= 48:
            attrs["..."] = "truncated"
            break
    return attrs


def _compile_arg_shape_summary(value: Any) -> dict[str, Any] | None:
    shape = _first_present_attr(value, "_shape", "shape")
    if shape is None:
        return None
    try:
        shape_value = tuple(shape)
    except TypeError:
        shape_value = shape
    summary: dict[str, Any] = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "shape": _short_repr(shape_value, max_len=96),
    }
    stride = _first_present_attr(value, "_stride", "stride")
    if stride is not None:
        try:
            stride_value = tuple(stride)
        except TypeError:
            stride_value = stride
        summary["stride"] = _short_repr(stride_value, max_len=96)
    stride_order = _first_present_attr(value, "_stride_order", "stride_order")
    if stride_order is not None:
        try:
            stride_order_value = tuple(stride_order)
        except TypeError:
            stride_order_value = stride_order
        summary["stride_order"] = _short_repr(stride_order_value, max_len=96)
    dtype = _first_present_attr(value, "_dtype", "dtype", "element_type")
    if dtype is not None:
        summary["dtype"] = _short_repr(dtype, max_len=80)
    memspace = _first_present_attr(value, "_memspace", "memspace")
    if memspace is not None:
        summary["memspace"] = _short_repr(memspace, max_len=80)
    assumed_align = _first_present_attr(value, "_assumed_align", "assumed_align")
    if assumed_align is not None:
        summary["align"] = assumed_align
    return summary


def _compile_args_shape_summary(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for idx, value in enumerate(args):
        shaped = _compile_arg_shape_summary(value)
        if shaped is not None:
            summary[f"arg{idx}"] = shaped
        elif value is None or isinstance(value, (bool, int, float, str)):
            summary[f"arg{idx}"] = value
        if len(summary) >= 24:
            summary["..."] = "truncated"
            return summary
    for name, value in sorted(kwargs.items()):
        shaped = _compile_arg_shape_summary(value)
        if shaped is not None:
            summary[f"kw:{name}"] = shaped
        elif value is None or isinstance(value, (bool, int, float, str)):
            summary[f"kw:{name}"] = value
        if len(summary) >= 24:
            summary["..."] = "truncated"
            return summary
    return summary


def _type_log_name(module: str, qualname: str) -> str:
    return f"{module}.{qualname}" if module else qualname


def _function_fingerprint_log_value(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 3:
        module, qualname, _fingerprint = value
        if isinstance(module, str) and isinstance(qualname, str):
            return _type_log_name(module, qualname)
    return _cache_key_log_value(value, max_depth=2, max_items=8)


def _object_state_log_value(
    value: Any, *, max_depth: int, max_items: int
) -> dict[str, Any] | None:
    if not (isinstance(value, tuple) and len(value) == 4 and value[0] == "object"):
        return None
    _tag, module, qualname, attrs = value
    if not isinstance(attrs, tuple):
        return None

    state: dict[str, Any] = {}
    for idx, item in enumerate(attrs):
        if idx >= max_items:
            state["..."] = f"{len(attrs) - idx} more"
            break
        if not (isinstance(item, tuple) and len(item) == 2):
            continue
        name, attr_value = item
        state[str(name)] = _cache_key_log_value(
            attr_value, max_depth=max_depth - 1, max_items=max_items
        )
    return {"type": _type_log_name(str(module), str(qualname)), "attrs": state}


def _tensor_key_log_value(
    names: tuple[str, ...], values: tuple[Any, ...], *, max_depth: int, max_items: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in zip(names, values, strict=False):
        if value is None:
            continue
        out[name] = _cache_key_log_value(
            value, max_depth=max_depth - 1, max_items=max_items
        )
    return out


def _cache_key_log_value(value: Any, *, max_depth: int = 5, max_items: int = 32) -> Any:
    if max_depth <= 0:
        return _short_repr(value, max_len=120)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, type):
        return _type_log_name(value.__module__, value.__qualname__)

    object_state = _object_state_log_value(
        value, max_depth=max_depth, max_items=max_items
    )
    if object_state is not None:
        return object_state

    if (
        isinstance(value, tuple)
        and value
        and all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        )
    ):
        out: dict[str, Any] = {}
        for idx, (key, item_value) in enumerate(value):
            if idx >= max_items:
                out["..."] = f"{len(value) - idx} more"
                break
            out[key] = _cache_key_log_value(
                item_value, max_depth=max_depth - 1, max_items=max_items
            )
        return out

    if isinstance(value, tuple) and value:
        tag = value[0]
        if tag == "type" and len(value) == 3:
            return _type_log_name(str(value[1]), str(value[2]))
        if tag == "function" and len(value) == 2:
            return _function_fingerprint_log_value(value[1])
        if tag == "method" and len(value) == 3:
            return {
                "kind": "method",
                "function": _function_fingerprint_log_value(value[1]),
                "self": _cache_key_log_value(
                    value[2], max_depth=max_depth - 1, max_items=max_items
                ),
            }
        if tag == "callable_instance" and len(value) == 5:
            return {
                "kind": "callable_instance",
                "type": _type_log_name(str(value[1]), str(value[2])),
                "call": _function_fingerprint_log_value(value[3]),
                "state": _cache_key_log_value(
                    value[4], max_depth=max_depth - 1, max_items=max_items
                ),
            }
        if tag == "callable" and len(value) == 4:
            return {
                "kind": "callable",
                "type": _type_log_name(str(value[1]), str(value[2])),
                "repr": _short_repr(value[3], max_len=160),
            }
        if tag == "cache_key" and len(value) == 4:
            return {
                "type": _type_log_name(str(value[1]), str(value[2])),
                "cache_key": _cache_key_log_value(
                    value[3], max_depth=max_depth - 1, max_items=max_items
                ),
            }
        if tag == "fake_tensor" and len(value) == 12:
            return _tensor_key_log_value(
                (
                    "kind",
                    "type",
                    "dtype",
                    "shape",
                    "stride",
                    "stride_order",
                    "device",
                    "layout",
                    "memspace",
                    "align",
                    "use_32bit_stride",
                ),
                (
                    "fake_tensor",
                    _type_log_name(str(value[1]), str(value[2])),
                    *value[3:],
                ),
                max_depth=max_depth,
                max_items=max_items,
            )
        if tag == "runtime_tensor" and len(value) == 8:
            return _tensor_key_log_value(
                (
                    "kind",
                    "dtype",
                    "shape",
                    "stride",
                    "memspace",
                    "align",
                    "is_dynamic",
                    "use_32bit_stride",
                ),
                value,
                max_depth=max_depth,
                max_items=max_items,
            )
        if tag == "fake_compact_tensor" and len(value) == 7:
            return _tensor_key_log_value(
                (
                    "kind",
                    "dtype",
                    "shape",
                    "stride_order",
                    "memspace",
                    "align",
                    "use_32bit_stride",
                ),
                value,
                max_depth=max_depth,
                max_items=max_items,
            )
        if tag == "cuda_stream":
            return "cuda_stream"
        if tag == "symbolic_dim" and len(value) == 4:
            return value[3]
        if tag == "bytes" and len(value) == 2 and isinstance(value[1], str):
            return f"bytes[{len(value[1]) // 2}]"
        if tag == "path" and len(value) == 2:
            return value[1]
        if tag == "repr" and len(value) == 4:
            return {
                "type": _type_log_name(str(value[1]), str(value[2])),
                "repr": _short_repr(value[3], max_len=160),
            }
        if tag == "cycle" and len(value) == 3:
            return {"cycle": _type_log_name(str(value[1]), str(value[2]))}

    if isinstance(value, dict):
        out = {}
        for idx, (key, item_value) in enumerate(
            sorted(value.items(), key=lambda kv: str(kv[0]))
        ):
            if idx >= max_items:
                out["..."] = f"{len(value) - idx} more"
                break
            out[str(key)] = _cache_key_log_value(
                item_value, max_depth=max_depth - 1, max_items=max_items
            )
        return out

    if isinstance(value, (tuple, list)):
        items = [
            _cache_key_log_value(item, max_depth=max_depth - 1, max_items=max_items)
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append(f"... {len(value) - max_items} more")
        return tuple(items) if isinstance(value, tuple) else items

    return _short_repr(value, max_len=160)


def _toolchain_log_value(toolchain: tuple[object, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for entry in toolchain:
        if not (isinstance(entry, tuple) and entry):
            continue
        name = entry[0]
        if name == "python" and len(entry) >= 3:
            summary[str(name)] = f"{entry[1]} {'.'.join(str(v) for v in entry[2])}"
        elif len(entry) >= 2 and entry[1]:
            summary[str(name)] = entry[1]
    return summary


def _environment_log_value(env: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return {name: value for name, value in env if value}


def _compile_cache_payload_log_value(
    payload: tuple[object, ...] | None,
) -> dict[str, Any]:
    if payload is None:
        return {}
    if len(payload) == 10 and payload[0] == "sm12x_cute_compile_cache_v5_explicit_spec":
        (
            _version,
            target_key,
            _sm12x_fingerprint,
            toolchain_key,
            spec_hash,
            spec_json,
            kwargs_hash,
            kwargs_json,
            options_key,
            env_key,
        ) = payload

        summary: dict[str, Any] = {
            "target": _cache_key_log_value(target_key, max_depth=7, max_items=80),
            "spec_hash": spec_hash,
            "spec": spec_json,
        }
        if kwargs_hash:
            summary["kwargs_hash"] = kwargs_hash
            summary["kwargs"] = kwargs_json
        if options_key:
            summary["options"] = _cache_key_log_value(
                options_key, max_depth=4, max_items=32
            )
        env_summary = (
            _environment_log_value(env_key) if isinstance(env_key, tuple) else {}
        )
        if env_summary:
            summary["env"] = env_summary
        if isinstance(toolchain_key, tuple):
            toolchain_summary = _toolchain_log_value(toolchain_key)
            if toolchain_summary:
                summary["toolchain"] = toolchain_summary
        return summary

    if len(payload) == 8 and payload[0] in {
        "sm12x_cute_compile_cache_v3_explicit_spec",
        "sm12x_cute_compile_cache_v4_explicit_spec",
    }:
        (
            _version,
            target_key,
            _sm12x_fingerprint,
            toolchain_key,
            spec_key,
            kwargs_key,
            options_key,
            env_key,
        ) = payload

        summary: dict[str, Any] = {
            "target": _cache_key_log_value(target_key, max_depth=7, max_items=80),
            "spec": _cache_key_log_value(spec_key, max_depth=7, max_items=80),
        }
        if options_key:
            summary["options"] = _cache_key_log_value(
                options_key, max_depth=4, max_items=32
            )
        kwargs_summary = _cache_key_log_value(kwargs_key, max_depth=5, max_items=32)
        if kwargs_summary:
            summary["kwargs"] = kwargs_summary
        env_summary = (
            _environment_log_value(env_key) if isinstance(env_key, tuple) else {}
        )
        if env_summary:
            summary["env"] = env_summary
        if isinstance(toolchain_key, tuple):
            toolchain_summary = _toolchain_log_value(toolchain_key)
            if toolchain_summary:
                summary["toolchain"] = toolchain_summary
        return summary

    if len(payload) != 8:
        return {}
    (
        _version,
        target_key,
        _sm12x_fingerprint,
        toolchain_key,
        args_key,
        kwargs_key,
        options_key,
        env_key,
    ) = payload

    summary: dict[str, Any] = {
        "target": _cache_key_log_value(target_key, max_depth=7, max_items=80),
        "args": _cache_key_log_value(args_key, max_depth=5, max_items=32),
    }
    kwargs_summary = _cache_key_log_value(kwargs_key, max_depth=5, max_items=32)
    if kwargs_summary:
        summary["kwargs"] = kwargs_summary
    if options_key:
        summary["options"] = _cache_key_log_value(
            options_key, max_depth=4, max_items=32
        )
    env_summary = _environment_log_value(env_key) if isinstance(env_key, tuple) else {}
    if env_summary:
        summary["env"] = env_summary
    if isinstance(toolchain_key, tuple):
        toolchain_summary = _toolchain_log_value(toolchain_key)
        if toolchain_summary:
            summary["toolchain"] = toolchain_summary
    return summary


def _is_explicit_spec_payload(payload: tuple[object, ...] | None) -> bool:
    return (
        payload is not None
        and len(payload) == 10
        and payload[0] == "sm12x_cute_compile_cache_v5_explicit_spec"
    )


def _cute_compile_arg_log_enabled() -> bool:
    raw = os.environ.get("FLASHINFER_EXP_SM12X_LOG_CUTE_COMPILE_ARGS", "")
    return raw.lower() not in {"", "0", "false", "no", "off"}


def _format_explicit_spec_log_details(summary: dict[str, Any]) -> str:
    parts = []
    if "spec_hash" in summary:
        parts.append(f"spec_hash={summary['spec_hash']}")
    if "spec" in summary:
        parts.append(f"spec={summary['spec']}")
    if "kwargs_hash" in summary:
        parts.append(f"kwargs_hash={summary['kwargs_hash']}")
    if "kwargs" in summary:
        parts.append(f"kwargs={summary['kwargs']}")
    if "options" in summary:
        parts.append(f"options={_short_repr(summary['options'], max_len=1200)}")
    if "env" in summary:
        parts.append(f"env={_short_repr(summary['env'], max_len=1200)}")
    if "toolchain" in summary:
        parts.append(f"toolchain={_short_repr(summary['toolchain'], max_len=1200)}")
    return " ".join(parts)


def _format_cute_compile_stack() -> str:
    depth = _cute_compile_stack_log_depth()
    frames = traceback.extract_stack()[:-2]
    runtime_patch_path = str(Path(__file__).resolve())
    visible_frames = [
        frame
        for frame in frames
        if str(Path(frame.filename).resolve()) != runtime_patch_path
    ]
    if depth:
        visible_frames = visible_frames[-depth:]

    lines = ["[sm12x cute.compile] python_stack (most recent call last):"]
    for frame in visible_frames:
        lines.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
        if frame.line:
            lines.append(f"    {frame.line.strip()}")
    return "\n".join(lines)


def _log_cute_compile_event(
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    event: str,
    cache_status: str,
    cache_payload: tuple[object, ...] | None = None,
    reason: str | None = None,
    cache_key: str | None = None,
) -> None:
    key_inputs = _compile_cache_payload_log_value(cache_payload)
    reason_text = "" if reason is None else f"reason={reason} "
    cache_key_text = "" if cache_key is None else f"cache_key={cache_key[:16]} "
    if _is_explicit_spec_payload(cache_payload):
        attrs_text = _short_repr(_compile_target_attrs(func), max_len=1200)
        args_text = ""
        if _cute_compile_arg_log_enabled():
            args_text = f"args={_short_repr(_compile_args_shape_summary(args, kwargs), max_len=1600)} "
        details = _format_explicit_spec_log_details(key_inputs)
        print(
            f"[sm12x cute.compile] {event} "
            f"{reason_text}"
            f"target={_compile_target_name(func)} "
            f"status={cache_status} "
            f"{cache_key_text}"
            f"attrs={attrs_text} "
            f"{args_text}"
            f"{details}",
            flush=True,
        )
        if _cute_compile_stack_log_enabled():
            print(_format_cute_compile_stack(), flush=True)
        return

    print(
        f"[sm12x cute.compile] {event} "
        f"{reason_text}"
        f"target={_compile_target_name(func)} "
        f"status={cache_status} "
        f"{cache_key_text}"
        f"attrs={_short_repr(_compile_target_attrs(func), max_len=1200)} "
        f"args={_short_repr(_compile_args_shape_summary(args, kwargs), max_len=1600)} "
        f"key_inputs={_short_repr(key_inputs, max_len=4000)}",
        flush=True,
    )
    if _cute_compile_stack_log_enabled():
        print(_format_cute_compile_stack(), flush=True)


def _log_cute_compile_miss(
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    cache_status: str,
    cache_payload: tuple[object, ...] | None = None,
    reason: str | None = None,
    cache_key: str | None = None,
) -> None:
    _log_cute_compile_event(
        func,
        args,
        kwargs,
        event="miss",
        cache_status=cache_status,
        cache_payload=cache_payload,
        reason=reason,
        cache_key=cache_key,
    )


def _compile_progress_details(
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    compile_spec: KernelCompileSpec | None,
    cache_key: str,
) -> str:
    parts = [
        f"target={_compile_target_name(func)}",
        f"cache_key={cache_key[:16]}",
    ]
    if compile_spec is not None:
        try:
            params = json.loads(compile_spec.json_key)["facts"]
        except Exception:
            params = compile_spec.json_key
        parts.extend(
            (
                f"kernel={compile_spec.kernel_id}",
                f"version={compile_spec.version}",
                f"params={_short_repr(params, max_len=2400)}",
            )
        )
    else:
        attrs = _compile_target_attrs(func)
        if attrs:
            parts.append(f"attrs={_short_repr(attrs, max_len=1200)}")
        parts.append(
            f"args={_short_repr(_compile_args_shape_summary(args, kwargs), max_len=2000)}"
        )
    return " ".join(parts)


def _call_cute_compile(
    compile_callable: Any,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    compile_spec: KernelCompileSpec | None,
    cache_key: str,
) -> Any:
    if not _cute_compile_progress_enabled():
        return compile_callable(func, *args, **kwargs)

    global _COMPILE_PROGRESS_COUNT
    global _COMPILE_PROGRESS_TOTAL_SECONDS
    with _COMPILE_PROGRESS_LOCK:
        _COMPILE_PROGRESS_COUNT += 1
        compile_number = _COMPILE_PROGRESS_COUNT

    try:
        details = _compile_progress_details(
            func,
            args,
            kwargs,
            compile_spec=compile_spec,
            cache_key=cache_key,
        )
    except Exception as exc:
        details = (
            f"target={_short_repr(func)} cache_key={cache_key[:16]} "
            f"details_error={type(exc).__name__}"
        )
    print(
        f"[sm12x cute.compile] compile-start number={compile_number} {details}",
        flush=True,
    )
    started = time.perf_counter()
    try:
        compiled = compile_callable(func, *args, **kwargs)
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        with _COMPILE_PROGRESS_LOCK:
            _COMPILE_PROGRESS_TOTAL_SECONDS += elapsed
            total = _COMPILE_PROGRESS_TOTAL_SECONDS
        print(
            f"[sm12x cute.compile] compile-failed number={compile_number} "
            f"duration_s={elapsed:.3f} total_compile_s={total:.3f} {details} "
            f"error={type(exc).__name__}: {_short_repr(exc, max_len=500)}",
            flush=True,
        )
        raise

    elapsed = time.perf_counter() - started
    with _COMPILE_PROGRESS_LOCK:
        _COMPILE_PROGRESS_TOTAL_SECONDS += elapsed
        total = _COMPILE_PROGRESS_TOTAL_SECONDS
    print(
        f"[sm12x cute.compile] compile-done number={compile_number} "
        f"duration_s={elapsed:.3f} total_compile_s={total:.3f} {details}",
        flush=True,
    )
    return compiled


def _iter_fingerprint_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    files.sort()
    return files


def _compute_sm12x_package_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in _iter_fingerprint_files(_PACKAGE_ROOT):
        rel_path = str(path.relative_to(_PACKAGE_ROOT))
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _sm12x_package_fingerprint() -> str:
    return _compute_sm12x_package_fingerprint()


def sm12x_package_fingerprint() -> str:
    """Return the exact content fingerprint used by the CuTe object cache."""
    return _sm12x_package_fingerprint()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


@lru_cache(maxsize=1)
def _runtime_toolchain_key() -> tuple[object, ...]:
    from .runtime_patches import cutlass_runtime_patch_status

    torch_version = _distribution_version("torch")
    torch_cuda_version = ""
    try:
        import torch

        if not torch_version:
            torch_version = getattr(torch, "__version__", "")
        torch_cuda_version = getattr(torch.version, "cuda", "") or ""
    except Exception:
        pass

    cutlass_version = _distribution_version("nvidia-cutlass-dsl")
    if not cutlass_version:
        cutlass_version = _distribution_version("cutlass")
    if not cutlass_version:
        try:
            import cutlass

            cutlass_version = getattr(cutlass, "__version__", "")
        except Exception:
            cutlass_version = ""
    cutlass_version = cutlass_version or "missing"

    return (
        ("python", sys.implementation.name, sys.version_info[:3]),
        ("torch", torch_version),
        ("torch_cuda", torch_cuda_version),
        ("cutlass_dsl", cutlass_version),
        (
            "cutlass_dsl_libs_base",
            _distribution_version("nvidia-cutlass-dsl-libs-base") or "missing",
        ),
        (
            "cutlass_dsl_libs_core",
            _distribution_version("nvidia-cutlass-dsl-libs-core") or "missing",
        ),
        (
            "cutlass_dsl_libs_cu12",
            _distribution_version("nvidia-cutlass-dsl-libs-cu12") or "missing",
        ),
        (
            "cutlass_dsl_libs_cu13",
            _distribution_version("nvidia-cutlass-dsl-libs-cu13") or "missing",
        ),
        ("cuda_python", _distribution_version("cuda-python")),
        ("cuda_bindings", _distribution_version("cuda-bindings")),
        ("sm12x_runtime_patches", cutlass_runtime_patch_status()),
    )


@lru_cache(maxsize=1)
def _compile_environment_key() -> tuple[tuple[str, str], ...]:
    compile_env_vars = {
        "CC",
        "CXX",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_TOOLKIT_PATH",
        "CUDACXX",
        "CUTE_DSL_ARCH",
        "NVCC_APPEND_FLAGS",
        "NVCC_PREPEND_FLAGS",
    }
    operational_env_vars = {
        "FLASHINFER_EXP_SM12X_COMPILE_CACHE_DIR",
        "FLASHINFER_EXP_SM12X_COMPILE_DISK_CACHE",
        "FLASHINFER_EXP_SM12X_COMPILE_MEMORY_CACHE",
        "FLASHINFER_EXP_SM12X_COMPILE_MEMORY_CACHE_SIZE",
        "FLASHINFER_EXP_SM12X_COMPILE_SPEC_MEMO",
        "FLASHINFER_EXP_SM12X_LOG_CUTE_COMPILES",
        "FLASHINFER_EXP_SM12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START",
        "FLASHINFER_EXP_SM12X_LOG_CUTE_COMPILE_ARGS",
        "FLASHINFER_EXP_SM12X_LOG_CUTE_COMPILE_STACK",
        "FLASHINFER_EXP_SM12X_LOG_CUTE_COMPILE_STACK_DEPTH",
        "FLASHINFER_EXP_SM12X_PRINT_COMPILE_PROGRESS",
        "FLASHINFER_EXP_SM12X_TIMING",
        "FLASHINFER_EXP_SM12X_TIMING_THRESHOLD_MS",
        "CUTE_DSL_CACHE_DIR",
    }
    for name in os.environ:
        if name.startswith(("FLASHINFER_EXP_SM12X_", "CUTE_", "CUTLASS_")):
            if name not in operational_env_vars:
                compile_env_vars.add(name)
    return tuple((name, os.environ.get(name, "")) for name in sorted(compile_env_vars))


@lru_cache(maxsize=16)
def _static_compile_cache_context(compile_callable: Any) -> tuple[object, ...]:
    return (
        _sm12x_package_fingerprint(),
        _runtime_toolchain_key(),
        _compile_options_cache_key(compile_callable),
        _compile_environment_key(),
    )


def _function_fingerprint(func: Any) -> tuple[str, str, str]:
    func = inspect.unwrap(func)
    module = getattr(func, "__module__", "")
    qualname = getattr(
        func, "__qualname__", getattr(func, "__name__", type(func).__qualname__)
    )
    if module.startswith("flashinfer.experimental.sm12x"):
        return module, qualname, f"sm12x:{_sm12x_package_fingerprint()}"
    try:
        source = inspect.getsource(func)
        payload = source.encode("utf-8")
    except (OSError, TypeError):
        code = getattr(func, "__code__", None)
        if code is None:
            payload = repr(func).encode("utf-8")
        else:
            payload = repr(
                (
                    code.co_code,
                    code.co_consts,
                    code.co_names,
                    code.co_varnames,
                    code.co_argcount,
                    code.co_kwonlyargcount,
                )
            ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return module, qualname, digest


def _normalize_compile_target(func: Any, visited: set[int]) -> Any:
    if inspect.ismethod(func):
        return (
            "method",
            _function_fingerprint(func.__func__),
            _structural_cache_key(func.__self__, visited),
        )
    if inspect.isfunction(func):
        return ("function", _function_fingerprint(func))
    if callable(func) and hasattr(func.__call__, "__func__"):
        state = vars(func) if hasattr(func, "__dict__") else None
        return (
            "callable_instance",
            type(func).__module__,
            type(func).__qualname__,
            _function_fingerprint(func.__call__.__func__),
            _structural_cache_key(state, visited),
        )
    return ("callable", type(func).__module__, type(func).__qualname__, repr(func))


def _explicit_spec_compile_target(func: Any) -> Any:
    if inspect.ismethod(func):
        return ("method", _function_fingerprint(func.__func__))
    if inspect.isfunction(func):
        return ("function", _function_fingerprint(func))
    if callable(func) and hasattr(func.__call__, "__func__"):
        return (
            "callable_instance",
            type(func).__module__,
            type(func).__qualname__,
            _function_fingerprint(func.__call__.__func__),
        )
    return ("callable", type(func).__module__, type(func).__qualname__)


def _structural_dim_key(dim: Any, visited: set[int]) -> Any:
    if dim is None or isinstance(dim, (bool, int, float, str)):
        return dim
    try:
        return int(dim)
    except (TypeError, ValueError):
        pass
    label = None
    for attr in ("symbol", "_symbol", "name", "_name"):
        value = getattr(dim, attr, None)
        if isinstance(value, str) and value:
            label = value
            break
    if label is None:
        node = getattr(dim, "node", None)
        expr = getattr(node, "expr", getattr(node, "_expr", None))
        if expr is not None:
            label = str(expr)
    if label is None:
        text = str(dim)
        if text and not text.startswith("?{"):
            label = text
    if label is not None:
        return (
            "symbolic_dim",
            type(dim).__module__,
            type(dim).__qualname__,
            label,
        )
    return (
        "symbolic_dim",
        type(dim).__module__,
        type(dim).__qualname__,
        id(getattr(dim, "node", dim)),
    )


def _maybe_call_zero_arg(value: Any) -> Any:
    if callable(value):
        try:
            return value()
        except TypeError:
            return None
    return value


def _first_present_attr(value: Any, *names: str) -> Any:
    for name in names:
        try:
            attr = getattr(value, name)
        except Exception:
            continue
        attr = _maybe_call_zero_arg(attr)
        if attr is not None:
            return attr
    return None


def _tensor_like_cache_key(value: Any, visited: set[int]) -> Any | None:
    shape = _first_present_attr(value, "_shape", "shape")
    if shape is None:
        return None
    stride = _first_present_attr(value, "_stride", "stride")
    stride_order = _first_present_attr(value, "_stride_order", "stride_order")
    dtype = _first_present_attr(value, "_dtype", "dtype", "element_type")
    device = _first_present_attr(value, "fake_device", "device")
    layout = _first_present_attr(value, "layout")
    memspace = _first_present_attr(value, "memspace", "_memspace")
    assumed_align = _first_present_attr(value, "_assumed_align")
    use_32bit_stride = _first_present_attr(value, "_use_32bit_stride")
    shape_key = tuple(_structural_dim_key(dim, visited) for dim in shape)
    stride_key = (
        None
        if stride is None
        else tuple(_structural_dim_key(dim, visited) for dim in stride)
    )
    stride_order_key = (
        None
        if stride_order is None
        else tuple(_structural_dim_key(dim, visited) for dim in stride_order)
    )
    return (
        "fake_tensor",
        type(value).__module__,
        type(value).__qualname__,
        _structural_cache_key(dtype, visited),
        shape_key,
        stride_key,
        stride_order_key,
        _structural_cache_key(device, visited),
        _structural_cache_key(layout, visited),
        _structural_cache_key(memspace, visited),
        assumed_align,
        use_32bit_stride,
    )


def _structural_cache_key(value: Any, visited: set[int] | None = None) -> Any:
    if visited is None:
        visited = set()

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, Path):
        return ("path", str(value))
    if inspect.isfunction(value) or inspect.ismethod(value):
        return _normalize_compile_target(value, visited)
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if isinstance(value, SimpleNamespace):
        return (
            "namespace",
            tuple(
                sorted(
                    (k, _structural_cache_key(v, visited))
                    for k, v in vars(value).items()
                )
            ),
        )
    if isinstance(value, dict):
        return tuple(
            sorted(
                (_structural_cache_key(k, visited), _structural_cache_key(v, visited))
                for k, v in value.items()
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_structural_cache_key(v, visited) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_structural_cache_key(v, visited) for v in value))

    type_name = type(value).__name__
    type_module = type(value).__module__
    if type_name == "CUstream" and type_module.startswith("cuda.bindings"):
        return ("cuda_stream",)
    if type_module == "cutlass.cute.runtime" and type_name == "_Tensor":
        dtype = getattr(value, "_dtype", getattr(value, "element_type", None))
        shape = tuple(_structural_dim_key(dim, visited) for dim in value.shape)
        stride = tuple(_structural_dim_key(dim, visited) for dim in value.stride)
        memspace = getattr(value, "memspace", getattr(value, "_memspace", None))
        assumed_align = getattr(value, "_assumed_align", None)
        is_dynamic = getattr(value, "_is_dynamic", None)
        use_32bit_stride = getattr(value, "_use_32bit_stride", None)
        return (
            "runtime_tensor",
            dtype,
            shape,
            stride,
            memspace,
            assumed_align,
            is_dynamic,
            use_32bit_stride,
        )
    if type_module == "cutlass.cute.runtime" and type_name == "_FakeCompactTensor":
        dtype = getattr(value, "_dtype", None)
        shape = tuple(
            _structural_dim_key(dim, visited) for dim in getattr(value, "_shape", ())
        )
        stride_order = tuple(
            _structural_dim_key(dim, visited)
            for dim in getattr(value, "_stride_order", ())
        )
        memspace = getattr(value, "_memspace", None)
        assumed_align = getattr(value, "_assumed_align", None)
        use_32bit_stride = getattr(value, "_use_32bit_stride", None)
        return (
            "fake_compact_tensor",
            dtype,
            shape,
            stride_order,
            memspace,
            assumed_align,
            use_32bit_stride,
        )
    if "FakeTensor" in type_name:
        fake_tensor_key = _tensor_like_cache_key(value, visited)
        if fake_tensor_key is not None:
            return fake_tensor_key

    cache_key_attr = getattr(value, "__cache_key__", None)
    if cache_key_attr is not None:
        return (
            "cache_key",
            type_module,
            type_name,
            _structural_cache_key(cache_key_attr, visited),
        )

    object_id = id(value)
    if object_id in visited:
        return ("cycle", type_module, type_name)

    if hasattr(value, "__dict__"):
        visited.add(object_id)
        try:
            return (
                "object",
                type_module,
                type_name,
                tuple(
                    sorted(
                        (
                            k,
                            _structural_cache_key(v, visited),
                        )
                        for k, v in vars(value).items()
                    )
                ),
            )
        finally:
            visited.remove(object_id)

    return ("repr", type_module, type_name, repr(value))


def _compile_options_cache_key(compile_callable: Any) -> tuple[str, ...]:
    compile_options = getattr(compile_callable, "_compile_options", None)
    if compile_options is None:
        return ()
    options = getattr(compile_options, "options", {})
    serialized = []
    for option in options.values():
        value = option.serialize()
        if value:
            serialized.append(value)
    return tuple(serialized)


def _compile_disk_cache_payload(
    compile_callable: Any,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    compile_spec: KernelCompileSpec | None = None,
) -> tuple[object, ...]:
    (
        package_fingerprint,
        runtime_toolchain,
        compile_options,
        compile_environment,
    ) = _static_compile_cache_context(compile_callable)
    if compile_spec is not None:
        kwargs_json_key, kwargs_hash_key = _compile_kwargs_json_key(kwargs)
        return (
            "sm12x_cute_compile_cache_v5_explicit_spec",
            _explicit_spec_compile_target(func),
            package_fingerprint,
            runtime_toolchain,
            compile_spec.hash_key,
            compile_spec.json_key,
            kwargs_hash_key,
            kwargs_json_key,
            compile_options,
            compile_environment,
        )
    return (
        "sm12x_cute_compile_cache_v2",
        _normalize_compile_target(func, set()),
        package_fingerprint,
        runtime_toolchain,
        _structural_cache_key(args),
        _structural_cache_key(kwargs),
        compile_options,
        compile_environment,
    )


def _build_compile_disk_cache_key(
    compile_callable: Any,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    compile_spec: KernelCompileSpec | None = None,
) -> str:
    payload = _compile_disk_cache_payload(
        compile_callable, func, args, kwargs, compile_spec
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _cache_prefix(cache_key: str) -> str:
    return f"sm12x_cute_{cache_key}"


def _cache_object_path(cache_key: str) -> Path:
    return _cute_compile_cache_dir() / cache_key[:2] / f"{cache_key}.o"


def _cache_manifest_path(cache_key: str) -> Path:
    return _cache_object_path(cache_key).with_suffix(".json")


def _cache_lock_path(cache_key: str) -> Path:
    return _cache_object_path(cache_key).with_suffix(".lock")


def _manifest_json_value(value: Any) -> Any:
    """Return a lossless-enough JSON view of an already-normalized cache value.

    Compile-cache payloads consist almost entirely of tuples and JSON scalar
    values.  Keep an explicit representation for uncommon values instead of
    allowing a sidecar write failure to discard the useful semantic fields.
    The exact payload used to form ``cache_key`` is also stored as ``repr`` in
    the manifest.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"kind": "float", "repr": repr(value)}
    if isinstance(value, bytes):
        return {"kind": "bytes", "hex": value.hex()}
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    if isinstance(value, (tuple, list)):
        return [_manifest_json_value(item) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {
                key: _manifest_json_value(item) for key, item in sorted(value.items())
            }
        return {
            "kind": "mapping",
            "items": [
                [_manifest_json_value(key), _manifest_json_value(item)]
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            ],
        }
    return {
        "kind": "repr",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": repr(value),
    }


def _semantic_target_key(target_key: Any) -> Any:
    """Remove source fingerprints while retaining callable semantic state."""

    if not isinstance(target_key, tuple) or not target_key:
        return _manifest_json_value(target_key)

    tag = target_key[0]
    if tag in {"method", "function"} and len(target_key) >= 2:
        fingerprint = target_key[1]
        if isinstance(fingerprint, tuple) and len(fingerprint) >= 2:
            result: dict[str, Any] = {
                "kind": tag,
                "module": str(fingerprint[0]),
                "qualname": str(fingerprint[1]),
            }
            if len(target_key) >= 3:
                result["state"] = _semantic_structural_key(target_key[2])
            return result
    if tag == "callable_instance" and len(target_key) >= 4:
        call_fingerprint = target_key[3]
        result = {
            "kind": tag,
            "type": f"{target_key[1]}.{target_key[2]}",
        }
        if isinstance(call_fingerprint, tuple) and len(call_fingerprint) >= 2:
            result["call"] = f"{call_fingerprint[0]}.{call_fingerprint[1]}"
        else:
            result["call"] = _semantic_structural_key(call_fingerprint)
        if len(target_key) >= 5:
            result["state"] = _semantic_structural_key(target_key[4])
        return result
    if tag == "callable" and len(target_key) >= 3:
        return {
            "kind": tag,
            "type": f"{target_key[1]}.{target_key[2]}",
        }
    return _semantic_structural_key(target_key)


def _semantic_structural_key(value: Any) -> Any:
    if isinstance(value, tuple):
        if value and value[0] in {
            "method",
            "function",
            "callable_instance",
            "callable",
        }:
            return _semantic_target_key(value)
        return [_semantic_structural_key(item) for item in value]
    if isinstance(value, list):
        return [_semantic_structural_key(item) for item in value]
    return _manifest_json_value(value)


def _semantic_compile_manifest_payload(
    cache_payload: tuple[object, ...],
) -> dict[str, Any]:
    cache_format = str(cache_payload[0]) if cache_payload else "unknown"
    semantic: dict[str, Any] = {
        "cache_format": cache_format,
        "target": _semantic_target_key(cache_payload[1]),
    }
    if cache_format == "sm12x_cute_compile_cache_v5_explicit_spec":
        semantic["compile_spec_hash"] = cache_payload[4]
        try:
            semantic["compile_spec"] = json.loads(str(cache_payload[5]))
        except (TypeError, ValueError, json.JSONDecodeError):
            semantic["compile_spec"] = str(cache_payload[5])
        if cache_payload[6]:
            semantic["compile_kwargs_hash"] = cache_payload[6]
            try:
                semantic["compile_kwargs"] = json.loads(str(cache_payload[7]))
            except (TypeError, ValueError, json.JSONDecodeError):
                semantic["compile_kwargs"] = str(cache_payload[7])
        semantic["compile_options"] = _manifest_json_value(cache_payload[8])
        semantic["compile_environment"] = _manifest_json_value(cache_payload[9])
    else:
        semantic["args"] = _semantic_structural_key(cache_payload[4])
        semantic["kwargs"] = _semantic_structural_key(cache_payload[5])
        semantic["compile_options"] = _manifest_json_value(cache_payload[6])
        semantic["compile_environment"] = _manifest_json_value(cache_payload[7])
    return semantic


_LLVM_SSA_NAME = r"%[-a-zA-Z$._0-9]+"
_LLVM_I64_CONSTANT_RE = re.compile(
    rf"^\s*(?P<result>{_LLVM_SSA_NAME}) = "
    r"llvm\.mlir\.constant\((?P<value>[0-9]+) : i64\) : i64\s*$"
)
_LLVM_GEP_RE = re.compile(
    rf"^\s*(?P<result>{_LLVM_SSA_NAME}) = llvm\.getelementptr "
    rf"(?P<base>{_LLVM_SSA_NAME})\[(?P<indices>[^]]+)\]"
)
_LLVM_LOAD_RE = re.compile(
    rf"^\s*(?P<result>{_LLVM_SSA_NAME}) = llvm\.load "
    rf"(?P<pointer>{_LLVM_SSA_NAME})(?:\s|:)"
)
_LLVM_STORE_RE = re.compile(
    rf"^\s*llvm\.store (?P<value>{_LLVM_SSA_NAME}), "
    rf"(?P<pointer>{_LLVM_SSA_NAME})(?:\s|:)"
)
_LLVM_KERNEL_ADDRESS_RE = re.compile(
    rf"^\s*(?P<result>{_LLVM_SSA_NAME}) = llvm\.mlir\.addressof "
    r"@kernels_(?P<kernel>[^\s:]+)\s*:"
)
_LLVM_LAUNCH_EX_RE = re.compile(
    r"llvm\.call @_cudaLaunchKernelEx\((?P<arguments>[^)]*)\)"
)


def _final_llvm_function_bodies(module_text: str) -> list[list[str]]:
    """Return lowered LLVM function bodies without parsing the embedded cubin.

    CUTLASS' final module contains the cubin as one very large LLVM global.
    Launch configuration is emitted in host ``llvm.func`` bodies, whose closing
    brace is at module indentation.  Keeping the scan line-oriented avoids
    retaining or reparsing a second MLIR module during every compilation.
    """

    bodies: list[list[str]] = []
    current: list[str] | None = None
    for line in module_text.splitlines():
        if current is None:
            if line.startswith("  llvm.func @") and line.rstrip().endswith("{"):
                current = [line]
            continue
        current.append(line)
        if line == "  }":
            bodies.append(current)
            current = None
    return bodies


def _parse_launch_dynamic_smem_from_final_llvm(
    module_text: str, expected_kernels: set[str]
) -> dict[str, Any]:
    """Extract exact resolved CUDA launch SMEM from CUTLASS final LLVM IR.

    CUTLASS 4.5 and 4.6 lower ``cute.kernel_smem_size`` to an i64 constant
    stored in field 2 of the CUDA launch-config struct.  The exported object
    retains the host launcher but not the MLIR dataflow needed to identify that
    constant, so this must run while the freshly compiled function still owns
    ``ir_module``.  Any unrecognised lowering is reported as unknown rather
    than guessed from the cubin's shared sections.
    """

    source = "cutlass-final-llvm-launch-config-field-2"
    if not expected_kernels:
        return {
            "status": "unknown",
            "source": source,
            "reason": "kernel-info-unavailable",
            "launch_dynamic_smem_bytes": {},
        }

    launches: dict[str, list[int]] = {kernel: [] for kernel in expected_kernels}
    parsed_launches = 0
    unparsed_launches = 0

    for lines in _final_llvm_function_bodies(module_text):
        if not any("llvm.call @_cudaLaunchKernelEx" in line for line in lines):
            continue

        constants: dict[str, int] = {}
        geps: dict[str, tuple[str, tuple[str, ...]]] = {}
        loads: dict[str, tuple[str, int]] = {}
        stores: list[tuple[int, str, str]] = []
        kernel_addresses: dict[str, str] = {}

        def pointer_key(pointer: str) -> tuple[str, tuple[str, ...]]:
            return geps.get(pointer, (pointer, ()))

        def stored_value(pointer: str, before: int) -> str | None:
            key = pointer_key(pointer)
            for line_number, value, destination in reversed(stores):
                if line_number < before and pointer_key(destination) == key:
                    return value
            return None

        for line_number, line in enumerate(lines):
            if match := _LLVM_I64_CONSTANT_RE.match(line):
                constants[match.group("result")] = int(match.group("value"))
                continue
            if match := _LLVM_GEP_RE.match(line):
                indices = tuple(
                    item.strip() for item in match.group("indices").split(",")
                )
                geps[match.group("result")] = (match.group("base"), indices)
                continue
            if match := _LLVM_LOAD_RE.match(line):
                loads[match.group("result")] = (
                    match.group("pointer"),
                    line_number,
                )
                continue
            if match := _LLVM_STORE_RE.match(line):
                stores.append(
                    (line_number, match.group("value"), match.group("pointer"))
                )
                continue
            if match := _LLVM_KERNEL_ADDRESS_RE.match(line):
                kernel_addresses[match.group("result")] = match.group("kernel")
                continue
            match = _LLVM_LAUNCH_EX_RE.search(line)
            if match is None:
                continue

            arguments = [
                argument.strip() for argument in match.group("arguments").split(",")
            ]
            if len(arguments) < 2:
                unparsed_launches += 1
                continue
            config_operand, kernel_operand = arguments[:2]

            kernel_load = loads.get(kernel_operand)
            kernel = (
                kernel_addresses.get(kernel_load[0])
                if kernel_load is not None
                else kernel_addresses.get(kernel_operand)
            )

            config = config_operand
            if config_load := loads.get(config_operand):
                config = stored_value(config_load[0], config_load[1]) or ""

            smem_values: list[int] = []
            if config:
                for pointer, (base, indices) in geps.items():
                    if base != config or indices != ("0", "2"):
                        continue
                    value_name = stored_value(pointer, line_number)
                    if value_name in constants:
                        smem_values.append(constants[value_name])

            if (
                kernel not in expected_kernels
                or len(smem_values) != 1
                or smem_values[0] < 0
            ):
                unparsed_launches += 1
                continue
            launches[kernel].append(smem_values[0])
            parsed_launches += 1

    missing_kernels = sorted(
        kernel for kernel, values in launches.items() if not values
    )
    if unparsed_launches or missing_kernels or parsed_launches == 0:
        reason_parts = []
        if unparsed_launches:
            reason_parts.append(f"unparsed-launches={unparsed_launches}")
        if missing_kernels:
            reason_parts.append("missing-kernels=" + ",".join(missing_kernels))
        if parsed_launches == 0 and not reason_parts:
            reason_parts.append("no-launches")
        return {
            "status": "unknown",
            "source": source,
            "reason": ";".join(reason_parts),
            "launch_dynamic_smem_bytes": {},
        }
    return {
        "status": "exact",
        "source": source,
        "launch_dynamic_smem_bytes": {
            kernel: values for kernel, values in sorted(launches.items())
        },
    }


def _extract_launch_dynamic_smem_bytes(compiled: Any) -> dict[str, Any]:
    module = getattr(compiled, "ir_module", None)
    kernel_info = getattr(compiled, "kernel_info", None)
    source = "cutlass-final-llvm-launch-config-field-2"
    if module is None:
        return {
            "status": "unknown",
            "source": source,
            "reason": "compiled-ir-unavailable",
            "launch_dynamic_smem_bytes": {},
        }
    expected_kernels = (
        {str(kernel) for kernel in kernel_info}
        if isinstance(kernel_info, dict)
        else set()
    )
    try:
        return _parse_launch_dynamic_smem_from_final_llvm(str(module), expected_kernels)
    except Exception as exc:
        return {
            "status": "unknown",
            "source": source,
            "reason": f"extractor-error:{type(exc).__name__}",
            "launch_dynamic_smem_bytes": {},
        }


def _build_compile_manifest(
    cache_key: str,
    cache_payload: tuple[object, ...],
    func: Any,
    object_bytes: bytes,
    compiled: Any = None,
) -> dict[str, Any]:
    semantic_payload = _semantic_compile_manifest_payload(cache_payload)
    semantic_json = json.dumps(
        semantic_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    cache_format = str(cache_payload[0]) if cache_payload else "unknown"
    explicit = cache_format == "sm12x_cute_compile_cache_v5_explicit_spec"
    options_index = 8 if explicit else 6
    environment_index = 9 if explicit else 7
    launch_metadata = (
        _extract_launch_dynamic_smem_bytes(compiled)
        if compiled is not None
        else {
            "status": "unknown",
            "source": "cutlass-final-llvm-launch-config-field-2",
            "reason": "compiled-ir-unavailable-on-object-reload",
            "launch_dynamic_smem_bytes": {},
        }
    )
    object_sha256 = hashlib.sha256(object_bytes).hexdigest()
    artifact_evidence = {
        "cache_key": cache_key,
        "object_sha256": object_sha256,
        "launch_metadata": launch_metadata,
    }
    artifact_evidence_json = json.dumps(
        artifact_evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    manifest: dict[str, Any] = {
        "schema": "flashinfer.experimental.sm12x._lib.compile_manifest.v3",
        "cache_key": cache_key,
        "cache_format": cache_format,
        "cache_payload_repr": repr(cache_payload),
        "cache_payload": _manifest_json_value(cache_payload),
        "object_sha256": object_sha256,
        "object_bytes": len(object_bytes),
        "semantic_key": hashlib.sha256(semantic_json.encode("utf-8")).hexdigest(),
        "semantic_payload": semantic_payload,
        "target": _compile_target_name(func),
        "target_identity": _semantic_target_key(cache_payload[1]),
        "package_fingerprint": str(cache_payload[2]),
        "toolchain": _manifest_json_value(cache_payload[3]),
        "compile_options": _manifest_json_value(cache_payload[options_index]),
        "compile_environment": _manifest_json_value(cache_payload[environment_index]),
        "launch_metadata": launch_metadata,
        "artifact_evidence_sha256": hashlib.sha256(
            artifact_evidence_json.encode("utf-8")
        ).hexdigest(),
    }
    if explicit:
        manifest["compile_spec_hash"] = str(cache_payload[4])
        manifest["compile_spec_json"] = str(cache_payload[5])
        manifest["compile_kwargs_hash"] = str(cache_payload[6])
        manifest["compile_kwargs_json"] = str(cache_payload[7])
        try:
            spec = json.loads(str(cache_payload[5]))
        except (TypeError, ValueError, json.JSONDecodeError):
            spec = None
        if isinstance(spec, dict):
            manifest["kernel_id"] = str(spec.get("kernel", ""))
            manifest["compile_spec_version"] = spec.get("version", "")
    return manifest


def _write_compile_manifest(
    cache_key: str,
    cache_payload: tuple[object, ...],
    func: Any,
    object_bytes: bytes,
    compiled: Any = None,
) -> None:
    manifest_path = _cache_manifest_path(cache_key)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_compile_manifest(
        cache_key, cache_payload, func, object_bytes, compiled=compiled
    )
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_name = tmp_file.name
            json.dump(
                manifest,
                tmp_file,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            tmp_file.write("\n")
        os.replace(tmp_name, manifest_path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            with suppress(OSError):
                os.unlink(tmp_name)


def _ensure_cute_compile_manifest(
    cache_key: str,
    cache_payload: tuple[object, ...],
    func: Any,
) -> None:
    manifest_path = _cache_manifest_path(cache_key)
    if manifest_path.exists():
        return
    object_bytes = _cache_object_path(cache_key).read_bytes()
    _write_compile_manifest(cache_key, cache_payload, func, object_bytes)


@contextmanager
def _disk_cache_key_lock(cache_key: str):
    try:
        import fcntl
    except ImportError:
        yield
        return

    lock_path = _cache_lock_path(cache_key)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_cute_compile_from_disk(cache_key: str):
    from cutlass.base_dsl.export.external_binary_module import ExternalBinaryModule

    object_path = _cache_object_path(cache_key)
    if not object_path.exists():
        return None
    try:
        # CUTLASS may finalize or patch the ELF while loading it.  The cache
        # object is content-addressed and its digest is recorded in the compile
        # manifest, so never expose that canonical object to the loader.
        with tempfile.TemporaryDirectory(prefix="sm12x-cute-cache-load-") as raw_stage:
            staged_object = Path(raw_stage) / object_path.name
            shutil.copy2(object_path, staged_object)
            module = ExternalBinaryModule(str(staged_object))
            return getattr(module, _cache_prefix(cache_key))
    except Exception:
        return None


def _store_cute_compile_to_disk(
    cache_key: str,
    compiled: Any,
    *,
    cache_payload: tuple[object, ...] | None = None,
    func: Any = None,
) -> None:
    if not hasattr(compiled, "dump_to_object"):
        return

    object_path = _cache_object_path(cache_key)
    object_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = object_path.with_suffix(".tmp")
    object_bytes = compiled.dump_to_object(_cache_prefix(cache_key))
    with open(tmp_path, "wb") as f:
        f.write(object_bytes)
    os.replace(tmp_path, object_path)
    if cache_payload is not None and func is not None:
        _write_compile_manifest(
            cache_key, cache_payload, func, object_bytes, compiled=compiled
        )


def _memory_cache_get(cache_key: object) -> Any | None:
    global _MEMORY_CACHE_HITS
    global _MEMORY_CACHE_MISSES
    if not _cute_compile_memory_cache_enabled():
        return None
    with _MEMORY_CACHE_LOCK:
        compiled = _MEMORY_CACHE.get(cache_key)
        if compiled is None:
            _MEMORY_CACHE_MISSES += 1
            return None
        _MEMORY_CACHE_HITS += 1
        _MEMORY_CACHE.move_to_end(cache_key)
        return compiled


def _memory_cache_put(cache_key: object, compiled: Any) -> None:
    if not _cute_compile_memory_cache_enabled():
        return
    with _MEMORY_CACHE_LOCK:
        _MEMORY_CACHE[cache_key] = compiled
        _MEMORY_CACHE.move_to_end(cache_key)
        while len(_MEMORY_CACHE) > _cute_compile_memory_cache_size():
            _MEMORY_CACHE.popitem(last=False)


def clear_compile_cache() -> None:
    global _MEMORY_CACHE_HITS
    global _MEMORY_CACHE_MISSES
    global _DISK_CACHE_HITS
    global _COMPILE_MISSES
    global _SPEC_MEMO_HITS
    global _SPEC_MEMO_MISSES
    global _COMPILE_PROGRESS_COUNT
    global _COMPILE_PROGRESS_TOTAL_SECONDS
    _compile_environment_key.cache_clear()
    _static_compile_cache_context.cache_clear()
    with _MEMORY_CACHE_LOCK:
        _MEMORY_CACHE.clear()
        _MEMORY_CACHE_HITS = 0
        _MEMORY_CACHE_MISSES = 0
        _DISK_CACHE_HITS = 0
        _COMPILE_MISSES = 0
    with _SPEC_MEMO_LOCK:
        _SPEC_MEMO.clear()
        _SPEC_MEMO_HITS = 0
        _SPEC_MEMO_MISSES = 0
    with _COMPILE_PROGRESS_LOCK:
        _COMPILE_PROGRESS_COUNT = 0
        _COMPILE_PROGRESS_TOTAL_SECONDS = 0.0


def compile_cache_info() -> dict[str, int | bool]:
    with _MEMORY_CACHE_LOCK:
        info: dict[str, int | bool] = {
            "memory_cache_enabled": _cute_compile_memory_cache_enabled(),
            "memory_cache_size": len(_MEMORY_CACHE),
            "memory_cache_max_size": _cute_compile_memory_cache_size(),
            "memory_cache_hits": _MEMORY_CACHE_HITS,
            "memory_cache_misses": _MEMORY_CACHE_MISSES,
            "disk_cache_enabled": _cute_compile_disk_cache_enabled(),
            "disk_cache_hits": _DISK_CACHE_HITS,
            "compile_misses": _COMPILE_MISSES,
        }
    with _SPEC_MEMO_LOCK:
        info["spec_memo_enabled"] = _spec_memo_enabled()
        info["spec_memo_size"] = len(_SPEC_MEMO)
        info["spec_memo_hits"] = _SPEC_MEMO_HITS
        info["spec_memo_misses"] = _SPEC_MEMO_MISSES
    return info


def compile(
    func: Any,
    *args: Any,
    compile_spec: KernelCompileSpec | None = None,
    dsl_compile_options: Any = None,
    **kwargs: Any,
) -> Any:
    import cutlass.cute as cute

    global _DISK_CACHE_HITS
    global _COMPILE_MISSES
    compile_callable = cute.compile
    if dsl_compile_options is not None:
        # Subscript-style DSL compile options (e.g. OptLevel(2): ptxas -O3's
        # scheduler register-starves some scalar-heavy kernels; see the w4a8
        # dynamic MoE recipe).
        if hasattr(compile_callable, "__getitem__"):
            compile_callable = compile_callable[dsl_compile_options]
        else:
            # Some embedded runtimes expose cutlass.cute.compile as a plain
            # function instead of the CompileCallable instance installed by the
            # top-level module import.  Recreate the callable explicitly so DSL
            # options still take effect instead of crashing or silently falling
            # back to the default compiler options.
            from cutlass.base_dsl.compiler import CompileCallable

            compile_callable = CompileCallable(dsl_compile_options)
        kwargs = dict(kwargs)
        kwargs["__dsl_compile_options_key"] = _structural_cache_key(dsl_compile_options)
    memory_cache_key = _compile_memory_cache_key(
        compile_callable, func, args, kwargs, compile_spec
    )
    compiled = _memory_cache_get(memory_cache_key)
    if compiled is not None:
        return compiled

    post_engine_start_log = _cute_compile_post_engine_start_log_enabled()
    payload = _compile_disk_cache_payload(
        compile_callable, func, args, kwargs, compile_spec
    )
    cache_key = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    if _cute_compile_disk_cache_enabled():
        compiled = _load_cute_compile_from_disk(cache_key)
        if compiled is not None:
            with suppress(Exception):
                _ensure_cute_compile_manifest(cache_key, payload, func)
            with _MEMORY_CACHE_LOCK:
                _DISK_CACHE_HITS += 1
            if post_engine_start_log:
                with suppress(Exception):
                    _log_cute_compile_event(
                        func,
                        args,
                        kwargs,
                        event="disk-hit",
                        cache_status="disk-cache-hit",
                        cache_payload=payload,
                        reason="post-engine-start",
                        cache_key=cache_key,
                    )
            _memory_cache_put(memory_cache_key, compiled)
            return compiled

        with _disk_cache_key_lock(cache_key):
            compiled = _memory_cache_get(memory_cache_key)
            if compiled is not None:
                return compiled

            compiled = _load_cute_compile_from_disk(cache_key)
            if compiled is not None:
                with suppress(Exception):
                    _ensure_cute_compile_manifest(cache_key, payload, func)
                with _MEMORY_CACHE_LOCK:
                    _DISK_CACHE_HITS += 1
                if post_engine_start_log:
                    with suppress(Exception):
                        _log_cute_compile_event(
                            func,
                            args,
                            kwargs,
                            event="disk-hit-after-wait",
                            cache_status="disk-cache-hit-after-wait",
                            cache_payload=payload,
                            reason="post-engine-start",
                            cache_key=cache_key,
                        )
                _memory_cache_put(memory_cache_key, compiled)
                return compiled

            cache_status = "disk-cache-miss"

            if _cute_compile_log_enabled() or post_engine_start_log:
                with suppress(Exception):
                    _log_cute_compile_miss(
                        func,
                        args,
                        kwargs,
                        cache_status=cache_status,
                        cache_payload=payload,
                        reason="post-engine-start" if post_engine_start_log else None,
                        cache_key=cache_key,
                    )

            with _MEMORY_CACHE_LOCK:
                _COMPILE_MISSES += 1
            from flashinfer.experimental.sm12x._lib.runtime_control import (
                raise_if_kernel_resolution_frozen,
            )

            raise_if_kernel_resolution_frozen(
                "cute.compile",
                target=func,
                cache_key=compile_spec if compile_spec is not None else payload,
            )
            call_kwargs = {
                k: v for k, v in kwargs.items() if k != "__dsl_compile_options_key"
            }
            compiled = _call_cute_compile(
                compile_callable,
                func,
                args,
                call_kwargs,
                compile_spec=compile_spec,
                cache_key=cache_key,
            )
            with suppress(Exception):
                _store_cute_compile_to_disk(
                    cache_key,
                    compiled,
                    cache_payload=payload,
                    func=func,
                )
            _memory_cache_put(memory_cache_key, compiled)
            return compiled
    else:
        cache_status = "disk-cache-disabled"

    if _cute_compile_log_enabled() or post_engine_start_log:
        with suppress(Exception):
            _log_cute_compile_miss(
                func,
                args,
                kwargs,
                cache_status=cache_status,
                cache_payload=payload,
                reason="post-engine-start" if post_engine_start_log else None,
                cache_key=cache_key,
            )

    with _MEMORY_CACHE_LOCK:
        _COMPILE_MISSES += 1
    from flashinfer.experimental.sm12x._lib.runtime_control import (
        raise_if_kernel_resolution_frozen,
    )

    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=func,
        cache_key=compile_spec if compile_spec is not None else payload,
    )
    call_kwargs = {k: v for k, v in kwargs.items() if k != "__dsl_compile_options_key"}
    compiled = _call_cute_compile(
        compile_callable,
        func,
        args,
        call_kwargs,
        compile_spec=compile_spec,
        cache_key=cache_key,
    )
    if _cute_compile_disk_cache_enabled():
        with suppress(Exception):
            _store_cute_compile_to_disk(
                cache_key,
                compiled,
                cache_payload=payload,
                func=func,
            )
    _memory_cache_put(memory_cache_key, compiled)
    return compiled


def _cached_default_executor(compiled: Any) -> Any | None:
    if not hasattr(compiled, "_default_executor"):
        return None

    executor = getattr(compiled, _EXECUTOR_CACHE_ATTR, None)
    if executor is not None:
        return executor

    with _EXECUTOR_CACHE_LOCK:
        executor = getattr(compiled, _EXECUTOR_CACHE_ATTR, None)
        if executor is not None:
            return executor
        executor = getattr(compiled, "_default_executor", None)
        if executor is None:
            to_executor = getattr(compiled, "to", None)
            if to_executor is None:
                return None
            executor = to_executor(None)
            with suppress(Exception):
                compiled._default_executor = executor
        with suppress(Exception):
            setattr(compiled, _EXECUTOR_CACHE_ATTR, executor)
        return executor


def run_compiled(compiled: Any, args: tuple[Any, ...]) -> Any:
    if hasattr(compiled, "generate_execution_args") and hasattr(
        compiled, "run_compiled_program"
    ):
        execution_args, _ = compiled.generate_execution_args(*args)
        executor = _cached_default_executor(compiled)
        if executor is not None and hasattr(executor, "run_compiled_program"):
            return executor.run_compiled_program(execution_args)
        return compiled.run_compiled_program(execution_args)
    return compiled(*args)


def launch(
    func: Any,
    *,
    compile_spec: KernelCompileSpec,
    compile_args: tuple[Any, ...],
    runtime_args: tuple[Any, ...],
    compile_kwargs: dict[str, Any] | None = None,
) -> Any:
    compiled = compile(
        func,
        *compile_args,
        compile_spec=compile_spec,
        **(compile_kwargs or {}),
    )
    return run_compiled(compiled, runtime_args)
