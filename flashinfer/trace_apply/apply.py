# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""First-class runtime kernel substitution for FlashInfer APIs.

This module provides a small, explicit apply runtime keyed by fi_trace
definition names.  It intentionally does not select kernels from benchmark
results; callers register the solution they want for each definition.

The design is derived from flashinfer-bench
(https://github.com/flashinfer-ai/flashinfer-bench) apply system.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

import torch

from ..trace.solution import Solution
from ..trace.template import Scalar, Tensor, TraceTemplate
from .config import ApplyConfig
from .solution_builder.python import load_solution_object


class ApplyError(RuntimeError):
    """Raised when an apply solution fails during execution."""


@dataclass
class _ApplySolution:
    """Callable solution registered for one fi_trace definition."""

    fn: Callable[..., Any]
    definition: Optional[str] = None
    destination_passing_style: bool = False

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError("ApplySolution callable must be callable")

    def __call__(self, *args: Any) -> Any:
        return self.fn(*args)


def _resolve_param(json_key: str, descriptor: Union[Tensor, Scalar]) -> str:
    return descriptor.param if descriptor.param is not None else json_key


def _extract_argument(
    bound_kwargs: Mapping[str, Any],
    json_key: str,
    descriptor: Union[Tensor, Scalar],
) -> tuple[bool, Any]:
    """Return ``(found, value)`` for a template input descriptor."""

    param = _resolve_param(json_key, descriptor)
    if param not in bound_kwargs or bound_kwargs[param] is None:
        return False, None

    value = bound_kwargs[param]
    if isinstance(descriptor, Tensor) and descriptor.tuple_idx is not None:
        if not isinstance(value, (tuple, list)) or len(value) <= descriptor.tuple_idx:
            return False, None
        value = value[descriptor.tuple_idx]

    return True, value


def build_solution_args(
    template: TraceTemplate,
    bound_kwargs: Mapping[str, Any],
) -> list[Any]:
    """Build positional solution args from ``TraceTemplate.inputs``.

    ``Tensor(param=...)`` and tuple inputs are resolved against ``bound_kwargs``.
    """

    args: list[Any] = []
    missing: list[str] = []
    for json_key, descriptor in template.inputs.items():
        found, value = _extract_argument(bound_kwargs, json_key, descriptor)
        if found:
            args.append(value)
        elif not descriptor.optional:
            missing.append(json_key)

    if missing:
        raise ApplyError(
            "Cannot build apply solution inputs; missing required input(s): "
            + ", ".join(missing)
        )
    return args


def _resolve_output_buffer(
    template: TraceTemplate,
    bound_kwargs: Mapping[str, Any],
    output_name: str,
) -> Any:
    if output_name in bound_kwargs and bound_kwargs[output_name] is not None:
        return bound_kwargs[output_name]

    # Common value-returning APIs use an optional Python parameter named
    # ``out`` while the trace output is named semantically (for example
    # ``"output"``). Allow DPS solutions to use that caller-provided buffer.
    if len(template.outputs) == 1 and isinstance(bound_kwargs.get("out"), torch.Tensor):
        return bound_kwargs["out"]

    descriptor = template.outputs.get(output_name)
    if (
        isinstance(descriptor, Tensor)
        and descriptor.param is not None
        and descriptor.param in bound_kwargs
        and bound_kwargs[descriptor.param] is not None
    ):
        return bound_kwargs[descriptor.param]

    raise ApplyError(
        f"Cannot call DPS apply solution; missing output buffer '{output_name}'"
    )


def _build_positional_output_args(
    template: TraceTemplate,
    bound_kwargs: Mapping[str, Any],
) -> list[Any]:
    return [
        _resolve_output_buffer(template, bound_kwargs, output_name)
        for output_name in template.outputs
    ]


def _adapt_dps_outputs(
    template: TraceTemplate,
    output_args: list[Any],
    bound_kwargs: Mapping[str, Any],
) -> Any:
    if len(output_args) == 1:
        output_name = next(iter(template.outputs))
        if output_name == "out":
            return None
        if output_args[0] is bound_kwargs.get("out"):
            return bound_kwargs["out"]
        return output_args[0]
    return tuple(output_args)


def _copy_tensor(dst: Any, src: Any, *, name: str) -> None:
    if not isinstance(dst, torch.Tensor) or not isinstance(src, torch.Tensor):
        raise ApplyError(
            f"Cannot copy apply output '{name}': expected tensor destination and source"
        )
    if tuple(dst.shape) != tuple(src.shape):
        raise ApplyError(
            f"Cannot copy apply output '{name}': shape mismatch "
            f"dst={tuple(dst.shape)} src={tuple(src.shape)}"
        )
    dst.copy_(src)


def _flatten_outputs(template: TraceTemplate, result: Any) -> Dict[str, Any]:
    output_names = list(template.outputs)
    if isinstance(result, dict):
        return result
    if len(output_names) == 1:
        return {output_names[0]: result}
    if isinstance(result, (tuple, list)):
        if len(result) != len(output_names):
            raise ApplyError(
                f"Apply solution returned {len(result)} outputs, expected {len(output_names)}"
            )
        return dict(zip(output_names, result, strict=True))
    raise ApplyError(
        f"Apply solution returned a single value for {len(output_names)} outputs"
    )


def _adapt_solution_result(
    result: Any,
    template: TraceTemplate,
    bound_kwargs: Mapping[str, Any],
) -> Any:
    """Copy destination-style outputs back when the original call provides buffers."""

    if result is None:
        return None

    output_values = _flatten_outputs(template, result)
    copied_output_names: set[str] = set()

    for output_name, output_value in output_values.items():
        if output_name in bound_kwargs and bound_kwargs[output_name] is not None:
            _copy_tensor(bound_kwargs[output_name], output_value, name=output_name)
            copied_output_names.add(output_name)
            continue

        descriptor = template.outputs.get(output_name)
        if (
            isinstance(descriptor, Tensor)
            and descriptor.param is not None
            and descriptor.param in bound_kwargs
            and bound_kwargs[descriptor.param] is not None
        ):
            _copy_tensor(bound_kwargs[descriptor.param], output_value, name=output_name)
            copied_output_names.add(output_name)

    # Many value-returning FlashInfer APIs accept an optional ``out`` buffer
    # even though the trace output is named semantically (for example
    # ``"output"``). Preserve the API contract by copying the single solution
    # output into that buffer and returning the buffer.
    explicit_out = bound_kwargs.get("out")
    if (
        isinstance(explicit_out, torch.Tensor)
        and "out" not in template.outputs
        and len(output_values) == 1
    ):
        output_name, output_value = next(iter(output_values.items()))
        if output_name not in copied_output_names:
            _copy_tensor(explicit_out, output_value, name="out")
        return explicit_out

    # Common destination-passing FlashInfer APIs use an output named "out" and
    # return None after mutating the caller-provided buffer.
    if len(output_values) == 1 and "out" in copied_output_names:
        return None

    return result


@dataclass
class _Patch:
    owner: Any
    attr: str
    original: Callable[..., Any]


def _warn_once(
    warned: set[tuple[str, str]],
    definition_name: str,
    exc: BaseException,
) -> None:
    key = (definition_name, type(exc).__name__)
    if key in warned:
        return
    warned.add(key)
    warnings.warn(
        f"[flashinfer] apply failed for '{definition_name}': "
        f"{type(exc).__name__}: {exc}. Falling back to the original API.",
        stacklevel=3,
    )


def _build_solution_map(
    config: Union[ApplyConfig, Mapping[str, Union[Callable[..., Any], Solution]]],
) -> Dict[str, _ApplySolution]:
    solution_map: Dict[str, _ApplySolution] = {}
    if isinstance(config, ApplyConfig):
        for definition_name, solution in config.solutions.items():
            solution_map[definition_name] = _build_apply_solution(solution)
        return solution_map

    if isinstance(config, Mapping):
        for definition_name, solution in config.items():
            solution_map[definition_name] = _build_apply_solution(solution)
        return solution_map

    raise TypeError("enable_apply expects an ApplyConfig or definition-to-solution mapping")


def _build_apply_solution(
    solution: Union[Callable[..., Any], Solution],
) -> _ApplySolution:
    if isinstance(solution, Solution):
        built_solution = load_solution_object(solution)
        return _ApplySolution(
            built_solution.fn,
            definition=built_solution.definition,
            destination_passing_style=built_solution.destination_passing_style,
        )
    elif callable(solution):
        return _ApplySolution(solution)
    raise TypeError("apply solutions must be callables or Solution objects")


def _resolve_target(dotted: str) -> tuple[Any, str] | None:
    parts = dotted.split(".")
    if len(parts) < 2:
        return None

    module = None
    module_end = 0
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        try:
            module = importlib.import_module(candidate)
            module_end = i
            break
        except ImportError:
            continue
    if module is None:
        return None

    owner: Any = module
    for part in parts[module_end : len(parts) - 1]:
        owner = getattr(owner, part, None)
        if owner is None:
            return None

    attr = parts[-1]
    if not hasattr(owner, attr):
        return None
    return owner, attr


def _resolve_patch_targets(dotted: str) -> list[tuple[Any, str]]:
    resolved = _resolve_target(dotted)
    if resolved is None:
        return []

    targets = [resolved]
    owner, attr = resolved
    original = getattr(owner, attr)

    flashinfer_module = sys.modules.get("flashinfer")
    if flashinfer_module is not None:
        for alias, value in vars(flashinfer_module).items():
            if value is original:
                target = (flashinfer_module, alias)
                if target not in targets:
                    targets.append(target)

    return targets


def _resolve_template(trace_spec: Any, bound_kwargs: Mapping[str, Any]) -> TraceTemplate | None:
    if isinstance(trace_spec, TraceTemplate):
        return trace_spec
    template = trace_spec(**bound_kwargs)
    if template is None:
        return None
    if not isinstance(template, TraceTemplate):
        raise ApplyError("trace dispatch callable did not return a TraceTemplate")
    return template


def _bind_call(
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _build_definition(
    *,
    fi_api: str,
    trace_spec: Any,
    definition_fns: dict[int, Callable[..., dict]],
    bound_kwargs: Mapping[str, Any],
) -> tuple[TraceTemplate | None, dict | None]:
    template = _resolve_template(trace_spec, bound_kwargs)
    if template is None:
        return None, None
    template_id = id(template)
    definition_fn = definition_fns.get(template_id)
    if definition_fn is None:
        definition_fn = template.build_fi_trace_fn(fi_api)
        definition_fns[template_id] = definition_fn
    return template, definition_fn(_write=False, **bound_kwargs)


def _resolve_solution(
    *,
    solutions: Mapping[str, _ApplySolution],
    warned: set[tuple[str, str]],
    definition: dict,
) -> _ApplySolution | None:
    definition_name = definition.get("name")
    if not isinstance(definition_name, str) or not definition_name:
        return None

    solution = solutions.get(definition_name)
    if solution is None:
        return None

    if solution.definition is not None and solution.definition != definition_name:
        _warn_once(
            warned,
            definition_name,
            ApplyError(
                f"Apply solution declares definition '{solution.definition}', "
                f"but runtime definition is '{definition_name}'"
            ),
        )
        return None
    return solution


def _dispatch_solution(
    *,
    solutions: Mapping[str, _ApplySolution],
    warned: set[tuple[str, str]],
    definition: dict,
    template: TraceTemplate,
    bound_kwargs: Mapping[str, Any],
    fallback: Callable[[], Any],
) -> Any:
    solution = _resolve_solution(
        solutions=solutions,
        warned=warned,
        definition=definition,
    )
    if solution is None:
        return fallback()

    try:
        input_args = build_solution_args(template, bound_kwargs)
        if solution.destination_passing_style:
            output_args = _build_positional_output_args(template, bound_kwargs)
            solution(*input_args, *output_args)
            return _adapt_dps_outputs(template, output_args, bound_kwargs)
        result = solution(*input_args)
        return _adapt_solution_result(result, template, bound_kwargs)
    except Exception as exc:
        definition_name = definition.get("name")
        if not isinstance(definition_name, str):
            definition_name = "<unknown>"
        _warn_once(warned, definition_name, exc)
        return fallback()


def _make_wrapper(
    *,
    registration: Mapping[str, Any],
    original: Callable[..., Any],
    solutions: Mapping[str, _ApplySolution],
    warned: set[tuple[str, str]],
) -> Callable[..., Any]:
    fi_api = registration["fi_api"]
    plan_fi_api = _PLAN_RUN_PAIRS.get(fi_api)
    if plan_fi_api is not None:
        _patch_plan(plan_fi_api, warned)
        return _patch_run(
            registration=registration,
            original=original,
            solutions=solutions,
            warned=warned,
        )

    trace_spec = registration["trace"]
    signature = inspect.signature(registration["original"])
    definition_fns: dict[int, Callable[..., dict]] = {}

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            bound_kwargs = _bind_call(signature, args, kwargs)
            template, definition = _build_definition(
                fi_api=fi_api,
                trace_spec=trace_spec,
                definition_fns=definition_fns,
                bound_kwargs=bound_kwargs,
            )
            if template is None or definition is None:
                return original(*args, **kwargs)
            return _dispatch_solution(
                solutions=solutions,
                warned=warned,
                definition=definition,
                template=template,
                bound_kwargs=bound_kwargs,
                fallback=lambda: original(*args, **kwargs),
            )
        except Exception as exc:
            _warn_once(warned, fi_api, exc)
            return original(*args, **kwargs)

    return wrapper


def _patch_plan(plan_fi_api: str, warned: set[tuple[str, str]]) -> None:
    def make_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                bound_kwargs = _bind_call(signature, args, kwargs)
            except Exception as exc:
                _warn_once(warned, plan_fi_api, exc)
                return original(*args, **kwargs)

            result = original(*args, **kwargs)
            instance = bound_kwargs.get("self")
            if instance is not None:
                plan_kwargs = dict(bound_kwargs)
                plan_kwargs.pop("self", None)
                setattr(instance, _PLAN_RUN_CACHE_ATTR, plan_kwargs)
            return result

        return wrapper

    for owner, attr in _resolve_patch_targets(plan_fi_api):
        patch = _find_patch(owner, attr)
        if patch is None:
            original = getattr(owner, attr)
            _patches[(id(owner), attr)] = _Patch(
                owner=owner, attr=attr, original=original
            )
        else:
            original = patch.original

        setattr(owner, attr, make_wrapper(original))


def _patch_run(
    *,
    registration: Mapping[str, Any],
    original: Callable[..., Any],
    solutions: Mapping[str, _ApplySolution],
    warned: set[tuple[str, str]],
) -> Callable[..., Any]:
    fi_api = registration["fi_api"]
    trace_spec = registration["trace"]
    signature = inspect.signature(registration["original"])
    definition_fns: dict[int, Callable[..., dict]] = {}

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            run_kwargs = _bind_call(signature, args, kwargs)
            instance = run_kwargs.get("self")
            plan_kwargs = getattr(instance, _PLAN_RUN_CACHE_ATTR, None)
            if not isinstance(plan_kwargs, dict):
                return original(*args, **kwargs)

            bound_kwargs = {**plan_kwargs, **run_kwargs}
            bound_kwargs.pop("self", None)
            template, definition = _build_definition(
                fi_api=fi_api,
                trace_spec=trace_spec,
                definition_fns=definition_fns,
                bound_kwargs=bound_kwargs,
            )
            if template is None or definition is None:
                return original(*args, **kwargs)
            solution = _resolve_solution(
                solutions=solutions,
                warned=warned,
                definition=definition,
            )
            if solution is None:
                return original(*args, **kwargs)

            result = solution.fn(**bound_kwargs)
            return _adapt_solution_result(result, template, bound_kwargs)
        except Exception as exc:
            _warn_once(warned, fi_api, exc)
            return original(*args, **kwargs)

    return wrapper


_patches: dict[tuple[int, str], _Patch] = {}
_PLAN_RUN_PAIRS: dict[str, str] = {}
_PLAN_RUN_CACHE_ATTR = "_flashinfer_apply_plan_kwargs"
_APPLY_ENV = "FLASHINFER_APPLY"
_APPLY_CONFIG_ENV = "FLASHINFER_APPLY_CONFIG"


def _find_patch(owner: Any, attr: str) -> _Patch | None:
    return _patches.get((id(owner), attr))


def register_plan_run(
    *,
    plan_fi_api: str,
    run_fi_api: str,
) -> None:
    """Register a plan/run method pair for trace apply."""

    _PLAN_RUN_PAIRS[run_fi_api] = plan_fi_api


def enable_apply(
    config: Union[ApplyConfig, Mapping[str, Union[Callable[..., Any], Solution]]],
) -> None:
    """Enable apply wrappers for imported FlashInfer APIs.

    The wrappers are installed independently of ``@flashinfer_api`` logging:
    trace decorators only register metadata, while this function performs the
    runtime monkey-patching.
    """

    solution_map = _build_solution_map(config)
    if not solution_map:
        return

    from flashinfer.api_logging import _TRACE_APPLY_REGISTRY  # noqa: PLC0415

    warned: set[tuple[str, str]] = set()
    seen_targets: set[str] = set()
    for registration in _TRACE_APPLY_REGISTRY:
        fi_api = registration.get("fi_api")
        if not isinstance(fi_api, str) or fi_api in seen_targets:
            continue
        seen_targets.add(fi_api)
        targets = _resolve_patch_targets(fi_api)
        if not targets:
            continue
        for owner, attr in targets:
            patch = _find_patch(owner, attr)
            if patch is None:
                original = getattr(owner, attr)
                _patches[(id(owner), attr)] = _Patch(
                    owner=owner, attr=attr, original=original
                )
            else:
                original = patch.original
            wrapper = _make_wrapper(
                registration=registration,
                original=original,
                solutions=solution_map,
                warned=warned,
            )
            setattr(owner, attr, wrapper)


def disable_apply() -> None:
    """Disable trace apply and restore patched APIs."""

    while _patches:
        _, patch = _patches.popitem()
        setattr(patch.owner, patch.attr, patch.original)


def enable_apply_from_env() -> None:
    enabled = os.environ.get(_APPLY_ENV, "0")
    if enabled in ("", "0"):
        return
    if enabled != "1":
        warnings.warn(
            f"{_APPLY_ENV} must be '0' or '1', got {enabled!r}; trace apply is disabled.",
            stacklevel=2,
        )
        return

    config_path = os.environ.get(_APPLY_CONFIG_ENV)
    if not config_path:
        warnings.warn(
            f"{_APPLY_ENV}=1 but {_APPLY_CONFIG_ENV} is not set; trace apply is disabled.",
            stacklevel=2,
        )
        return

    try:
        enable_apply(_load_apply_config_path(config_path))
    except Exception as exc:
        warnings.warn(
            f"Failed to enable trace apply from {_APPLY_CONFIG_ENV}={config_path!r}: "
            f"{type(exc).__name__}: {exc}",
            stacklevel=2,
        )


def _load_apply_config_path(path: str) -> ApplyConfig:
    config_path = Path(path).expanduser()
    with config_path.open() as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise TypeError(f"{config_path} must contain a JSON object")
    return ApplyConfig.from_dict(data)
