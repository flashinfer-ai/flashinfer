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
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Sequence

from . import env as jit_env
from ._kda_jit_common import (
    gen_kda_jit_spec,
    get_flashinfer_include_dir,
    get_kda_csrc_dir,
)
from .core import JitSpec, logger
from .utils import write_if_different

FusedKDADecodeGeneratedTarget = Literal["sm100a"]
FusedKDADecodeGeneratedSlotClass = Literal[
    "positive_unique",
    "unique_or_null",
    "repeated_positive",
]

_MANIFEST_FILENAME = "fused_kda_decode_generated_manifest.json"
_BINDING_HEADER = "fused_kda_decode_generated_binding.cuh"
_SCHEMA_VERSION = 1
_TARGETS: tuple[FusedKDADecodeGeneratedTarget, ...] = ("sm100a",)
_TARGET_DEFINE = "-DFLASHINFER_FUSED_KDA_DECODE_TARGET_MINOR=0"
_SLUG = re.compile(r"[a-z0-9][a-z0-9_]*")
_C_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAXRREGCOUNT = re.compile(r"--maxrregcount=([1-9][0-9]{1,2})")
_FIXED_GENERATED_CUDA_CFLAGS = frozenset({"--use_fast_math"})

# These are the two physical parameter orders of the generated CUDA entry
# points. The repeated-slot-safe schedule also consumes the row count. The host
# binding derives all layout scalars from the tensor views.
_COMMON_BUFFER_ABI: tuple[tuple[str, str, str], ...] = (
    ("buffer", "x", "bfloat16"),
    ("buffer", "weight", "float32"),
    ("buffer", "conv_state", "bfloat16"),
    ("buffer", "raw_gate", "bfloat16"),
    ("buffer", "raw_beta", "bfloat16"),
    ("buffer", "A_log", "float32"),
    ("buffer", "dt_bias", "float32"),
    ("buffer", "state_indices", "int32"),
    ("buffer", "state", "state_dtype"),
    ("buffer", "output_gate", "bfloat16"),
    ("buffer", "norm_weight", "float32"),
    ("buffer", "output", "bfloat16"),
)
_COMMON_SCALAR_ABI: tuple[tuple[str, str, str], ...] = (
    ("parameter", "x_row_stride", "int32"),
    ("parameter", "conv_slot_stride", "int32"),
    ("parameter", "beta_row_stride", "int32"),
    ("parameter", "state_slot_stride", "int32"),
    ("parameter", "output_gate_row_stride", "int32"),
    ("parameter", "H", "int32"),
)
_RUNTIME_CONFIG_ABI: tuple[tuple[str, str, str], ...] = (
    ("parameter", "use_lower_bound", "int32"),
    ("parameter", "lower_bound_log2", "float32_scalar"),
    ("parameter", "norm_eps", "float32_scalar"),
)
FUSED_KDA_DECODE_GENERATED_ABIS: dict[
    str, tuple[tuple[str, str, str], ...]
] = {
    "standard": _COMMON_BUFFER_ABI + _COMMON_SCALAR_ABI + _RUNTIME_CONFIG_ABI,
    "repeated_safe": (
        _COMMON_BUFFER_ABI
        + _COMMON_SCALAR_ABI
        + (("parameter", "rows", "int32"),)
        + _RUNTIME_CONFIG_ABI
    ),
}

_CONTRACT = {
    "head_dim": 128,
    "convolution_width": 4,
    "activation_dtype": "bfloat16",
    "state_dtypes": ["bfloat16", "float32"],
    "lower_bound": "runtime_negative_or_null",
    "norm_eps": "runtime_nonnegative",
    "supported_heads": [12, 24, 32, 48, 96],
    "kernel_abis": {
        name: [list(argument) for argument in arguments]
        for name, arguments in FUSED_KDA_DECODE_GENERATED_ABIS.items()
    },
}
_ARG_PLAN_SHA256 = {
    name: hashlib.sha256(
        json.dumps(arguments, separators=(",", ":")).encode()
    ).hexdigest()
    for name, arguments in _CONTRACT["kernel_abis"].items()
}

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "contract",
        "status",
        "variants",
        "remaining_generated_inputs",
    }
)
_VARIANT_KEYS = frozenset(
    {
        "name",
        "target",
        "body",
        "source_sha256",
        "kernel_symbol",
        "abi_kind",
        "state_dtype",
        "extra_cuda_cflags",
        "launch",
        "eligibility",
    }
)
_LAUNCH_KEYS = frozenset({"threads", "dynamic_smem_bytes"})
_ELIGIBILITY_KEYS = frozenset(
    {
        "heads",
        "minimum_rows",
        "maximum_rows",
        "slot_classes",
        "lower_bound_values",
        "norm_eps_values",
        "strides",
    }
)
_STRIDE_KEYS = frozenset(
    {
        "x_row_stride",
        "conv_slot_stride",
        "beta_row_stride",
        "state_slot_stride",
        "output_gate_row_stride",
    }
)
_SLOT_CLASSES: tuple[FusedKDADecodeGeneratedSlotClass, ...] = (
    "positive_unique",
    "unique_or_null",
    "repeated_positive",
)

_SOURCE_PARAMETER_TYPES = {
    "bfloat16": {"__nv_bfloat16*"},
    "float32": {"float*"},
    "int32": {"int", "int32_t"},
    "float32_scalar": {"float"},
}


class FusedKDADecodeGeneratedManifestError(ValueError):
    """Raised when a frozen-source manifest or source body is not exact."""


@dataclass(frozen=True)
class FusedKDADecodeGeneratedEligibility:
    """One conjunction of runtime facts selecting a frozen CUDA body."""

    heads: tuple[int, ...]
    minimum_rows: int
    maximum_rows: int | None
    slot_classes: tuple[FusedKDADecodeGeneratedSlotClass, ...]
    lower_bound_values: tuple[float | None, ...] | None
    norm_eps_values: tuple[float, ...] | None
    strides: tuple[tuple[str, int | None], ...]


@dataclass(frozen=True)
class FusedKDADecodeGeneratedVariant:
    """One verified CUDA source and launch description."""

    name: str
    target: FusedKDADecodeGeneratedTarget
    body_path: Path
    source_sha256: str
    kernel_symbol: str
    abi_kind: str
    state_dtype: str
    extra_cuda_cflags: tuple[str, ...]
    threads: int
    dynamic_smem_bytes: int
    eligibility: tuple[FusedKDADecodeGeneratedEligibility, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FusedKDADecodeGeneratedManifestError(
            f"invalid fused KDA decode generated manifest: {message}"
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FusedKDADecodeGeneratedManifestError(
                f"invalid fused KDA decode generated manifest: duplicate field {key!r}"
            )
        result[key] = value
    return result


def _exact_keys(value: object, expected: frozenset[str], label: str) -> None:
    _require(isinstance(value, dict), f"{label} must be an object")
    assert isinstance(value, dict)
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    _require(not missing and not unknown, f"{label} fields: missing={missing}, unknown={unknown}")


def _source_parameter_declarations(source: str, kernel_symbol: str) -> list[tuple[str, str]]:
    signature = re.search(
        rf"\b__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s+)?void\s+"
        rf"{re.escape(kernel_symbol)}\s*\((?P<parameters>.*?)\)\s*\{{",
        source,
        flags=re.DOTALL,
    )
    _require(signature is not None, f"body does not define CUDA kernel {kernel_symbol!r}")
    assert signature is not None
    declarations: list[tuple[str, str]] = []
    for index, raw in enumerate(signature.group("parameters").split(",")):
        declaration = re.sub(r"\b(?:const|volatile|__restrict__)\b", "", raw)
        declaration = " ".join(declaration.split())
        match = re.fullmatch(r"(?P<type>.*?\*?)\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)", declaration)
        _require(match is not None, f"kernel parameter {index} is not a simple C declaration")
        assert match is not None
        normalized_type = re.sub(r"\s*\*\s*", "*", match.group("type").strip())
        declarations.append((match.group("name"), normalized_type))
    return declarations


def _verify_source_abi(
    source: str, kernel_symbol: str, *, abi_kind: str, state_dtype: str
) -> None:
    _require('extern "C"' in source, "body must expose its kernel with C linkage")
    declarations = _source_parameter_declarations(source, kernel_symbol)
    expected_abi = FUSED_KDA_DECODE_GENERATED_ABIS[abi_kind]
    _require(
        len(declarations) == len(expected_abi),
        f"kernel has {len(declarations)} parameters; expected {len(expected_abi)}",
    )
    for index, ((actual_name, actual_type), (_, expected_name, dtype)) in enumerate(
        zip(declarations, expected_abi, strict=True)
    ):
        _require(
            actual_name == expected_name,
            f"kernel parameter {index} must be named {expected_name!r}",
        )
        source_dtype = state_dtype if expected_name == "state" else dtype
        expected_types = _SOURCE_PARAMETER_TYPES[source_dtype]
        if expected_name == "state_indices":
            expected_types = {"int*", "int32_t*"}
        _require(
            actual_type in expected_types,
            f"kernel parameter {expected_name!r} has type {actual_type!r}; "
            f"expected one of {sorted(expected_types)}",
        )


def _verify_source_launch(
    source: str,
    kernel_symbol: str,
    *,
    threads: int,
    dynamic_smem_bytes: int,
) -> None:
    launch_bounds = re.search(
        rf"\b__global__\s+__launch_bounds__\s*\(\s*(?P<threads>[0-9]+)"
        rf"(?:\s*,[^)]*)?\)\s+void\s+{re.escape(kernel_symbol)}\s*\(",
        source,
        flags=re.DOTALL,
    )
    _require(
        launch_bounds is not None,
        f"body does not declare launch bounds for {kernel_symbol!r}",
    )
    assert launch_bounds is not None
    _require(
        int(launch_bounds.group("threads")) == threads,
        f"body launch bounds do not match manifest threads={threads}",
    )
    smem_defines = re.findall(r"(?m)^\s*#define\s+SMEM_TOTAL\s+([0-9]+)\s*$", source)
    _require(
        smem_defines == [str(dynamic_smem_bytes)],
        "body must define exactly one SMEM_TOTAL matching "
        f"dynamic_smem_bytes={dynamic_smem_bytes}",
    )


def _resolve_body(csrc_dir: Path, relative_value: object, label: str) -> Path:
    _require(isinstance(relative_value, str) and bool(relative_value), f"{label} is missing")
    assert isinstance(relative_value, str)
    relative = PurePosixPath(relative_value)
    _require(
        not relative.is_absolute()
        and relative.as_posix() == relative_value
        and len(relative.parts) == 1
        and ".." not in relative.parts
        and relative.name.startswith("fused_kda_decode_generated_")
        and relative.suffix == ".cu",
        f"{label} must name one fused_kda_decode_generated_*.cu file",
    )
    body = csrc_dir / relative.name
    _require(body.is_file(), f"{label} does not exist: {body}")
    return body


def _validate_extra_cuda_cflags(value: object, label: str) -> tuple[str, ...]:
    _require(isinstance(value, list), f"{label} must be a list")
    assert isinstance(value, list)
    flags: list[str] = []
    for index, flag in enumerate(value):
        _require(isinstance(flag, str), f"{label}[{index}] must be a string")
        assert isinstance(flag, str)
        if flag in _FIXED_GENERATED_CUDA_CFLAGS:
            _require(flag not in flags, f"{label} contains duplicate flag {flag!r}")
            flags.append(flag)
            continue
        match = _MAXRREGCOUNT.fullmatch(flag)
        _require(match is not None, f"{label}[{index}] is not an allowed generated-kernel flag")
        assert match is not None
        count = int(match.group(1))
        _require(16 <= count <= 255, f"{label}[{index}] max register count is outside [16, 255]")
        _require(flag not in flags, f"{label} contains duplicate flag {flag!r}")
        flags.append(flag)
    return tuple(flags)


def _positive_int(value: object, label: str, *, maximum: int) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and 0 < value <= maximum,
        f"{label} must be an integer in [1, {maximum}]",
    )
    assert isinstance(value, int)
    return value


def _nonnegative_int(value: object, label: str, *, maximum: int) -> int:
    _require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= maximum,
        f"{label} must be an integer in [0, {maximum}]",
    )
    assert isinstance(value, int)
    return value


def _finite_float(value: object, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float("-inf") < float(value) < float("inf"),
        f"{label} must be finite",
    )
    return float(value)


def _parse_exact_values(
    value: object,
    label: str,
    *,
    allow_null: bool,
    require_negative: bool,
) -> tuple[float | None, ...] | None:
    if value == "any":
        return None
    _require(isinstance(value, list) and bool(value), f"{label} must be 'any' or a non-empty list")
    assert isinstance(value, list)
    result: list[float | None] = []
    for index, item in enumerate(value):
        if item is None:
            _require(allow_null, f"{label}[{index}] must not be null")
            parsed = None
        else:
            parsed = _finite_float(item, f"{label}[{index}]")
            if require_negative:
                _require(parsed < 0.0, f"{label}[{index}] must be negative or null")
            else:
                _require(parsed >= 0.0, f"{label}[{index}] must be non-negative")
        _require(parsed not in result, f"{label} contains duplicate value {parsed!r}")
        result.append(parsed)
    return tuple(result)


def _parse_eligibility(
    value: object,
    label: str,
    *,
    abi_kind: str,
) -> tuple[FusedKDADecodeGeneratedEligibility, ...]:
    _require(isinstance(value, list) and bool(value), f"{label} must be a non-empty list")
    assert isinstance(value, list)
    result: list[FusedKDADecodeGeneratedEligibility] = []
    for index, item in enumerate(value):
        rule_label = f"{label}[{index}]"
        _exact_keys(item, _ELIGIBILITY_KEYS, rule_label)
        assert isinstance(item, dict)

        heads_value = item["heads"]
        _require(
            isinstance(heads_value, list) and bool(heads_value),
            f"{rule_label}.heads must be a non-empty list",
        )
        assert isinstance(heads_value, list)
        heads: list[int] = []
        for head_index, head in enumerate(heads_value):
            _require(
                isinstance(head, int)
                and not isinstance(head, bool)
                and head in _CONTRACT["supported_heads"],
                f"{rule_label}.heads[{head_index}] is unsupported",
            )
            _require(head not in heads, f"{rule_label}.heads contains duplicate {head}")
            heads.append(head)
        _require(heads == sorted(heads), f"{rule_label}.heads must be sorted")

        minimum_rows = _positive_int(
            item["minimum_rows"], f"{rule_label}.minimum_rows", maximum=2**31 - 1
        )
        maximum_rows_value = item["maximum_rows"]
        if maximum_rows_value is None:
            maximum_rows = None
        else:
            maximum_rows = _positive_int(
                maximum_rows_value,
                f"{rule_label}.maximum_rows",
                maximum=2**31 - 1,
            )
            _require(
                maximum_rows >= minimum_rows,
                f"{rule_label}.maximum_rows must be at least minimum_rows",
            )

        slot_classes_value = item["slot_classes"]
        _require(
            isinstance(slot_classes_value, list) and bool(slot_classes_value),
            f"{rule_label}.slot_classes must be a non-empty list",
        )
        assert isinstance(slot_classes_value, list)
        slot_classes: list[FusedKDADecodeGeneratedSlotClass] = []
        for slot_index, slot_class in enumerate(slot_classes_value):
            _require(
                slot_class in _SLOT_CLASSES,
                f"{rule_label}.slot_classes[{slot_index}] is unsupported",
            )
            _require(
                slot_class not in slot_classes,
                f"{rule_label}.slot_classes contains duplicate {slot_class!r}",
            )
            slot_classes.append(slot_class)  # type: ignore[arg-type]
        _require(
            abi_kind == "repeated_safe" or "repeated_positive" not in slot_classes,
            f"{rule_label} cannot send repeated positive slots to the standard ABI",
        )

        lower_bound_values = _parse_exact_values(
            item["lower_bound_values"],
            f"{rule_label}.lower_bound_values",
            allow_null=True,
            require_negative=True,
        )
        norm_eps_values = _parse_exact_values(
            item["norm_eps_values"],
            f"{rule_label}.norm_eps_values",
            allow_null=False,
            require_negative=False,
        )

        strides_value = item["strides"]
        _exact_keys(strides_value, _STRIDE_KEYS, f"{rule_label}.strides")
        assert isinstance(strides_value, dict)
        strides: list[tuple[str, int | None]] = []
        for stride_name in sorted(_STRIDE_KEYS):
            stride_value = strides_value[stride_name]
            if stride_value is None:
                strides.append((stride_name, None))
            else:
                strides.append(
                    (
                        stride_name,
                        _nonnegative_int(
                            stride_value,
                            f"{rule_label}.strides.{stride_name}",
                            maximum=2**31 - 1,
                        ),
                    )
                )

        result.append(
            FusedKDADecodeGeneratedEligibility(
                heads=tuple(heads),
                minimum_rows=minimum_rows,
                maximum_rows=maximum_rows,
                slot_classes=tuple(slot_classes),
                lower_bound_values=lower_bound_values,
                norm_eps_values=(
                    None
                    if norm_eps_values is None
                    else tuple(value for value in norm_eps_values if value is not None)
                ),
                strides=tuple(strides),
            )
        )
    return tuple(result)


@functools.cache
def load_fused_kda_decode_generated_variants(
    *,
    manifest_path: Path | None = None,
    csrc_dir: Path | None = None,
) -> tuple[FusedKDADecodeGeneratedVariant, ...]:
    """Load and verify the complete frozen-source manifest.

    A pending manifest returns an empty tuple. A complete manifest verifies the
    schema, launch contract, source SHA-256, and physical CUDA argument order
    before exposing any source to the JIT compiler.
    """

    source_dir = get_kda_csrc_dir() if csrc_dir is None else Path(csrc_dir)
    path = source_dir / _MANIFEST_FILENAME if manifest_path is None else Path(manifest_path)
    try:
        payload: Any = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except FileNotFoundError as exc:
        raise FusedKDADecodeGeneratedManifestError(
            f"fused KDA decode generated manifest not found: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise FusedKDADecodeGeneratedManifestError(
            f"invalid fused KDA decode generated manifest JSON at {path}: {exc}"
        ) from exc

    _exact_keys(payload, _ROOT_KEYS, "root")
    assert isinstance(payload, dict)
    _require(payload["schema_version"] == _SCHEMA_VERSION, "unsupported schema version")
    _require(payload["contract"] == _CONTRACT, "contract mismatch")
    variants = payload["variants"]
    remaining = payload["remaining_generated_inputs"]
    _require(isinstance(variants, list), "variants must be a list")
    _require(
        isinstance(remaining, list)
        and all(isinstance(item, str) and item for item in remaining),
        "remaining_generated_inputs must be a list of non-empty strings",
    )
    if payload["status"] == "pending_generated_sources":
        _require(not variants, "pending manifest must not list variants")
        _require(bool(remaining), "pending manifest must name its remaining generated inputs")
        return ()
    _require(payload["status"] == "complete", f"unsupported status {payload['status']!r}")
    _require(bool(variants), "complete manifest must list at least one variant")
    _require(not remaining, "complete manifest must not list remaining generated inputs")

    observed: set[tuple[str, str]] = set()
    result: list[FusedKDADecodeGeneratedVariant] = []
    for index, item in enumerate(variants):
        label = f"variants[{index}]"
        _exact_keys(item, _VARIANT_KEYS, label)
        assert isinstance(item, dict)
        name = item["name"]
        target = item["target"]
        kernel_symbol = item["kernel_symbol"]
        source_sha256 = item["source_sha256"]
        abi_kind = item["abi_kind"]
        state_dtype = item["state_dtype"]
        _require(isinstance(name, str) and _SLUG.fullmatch(name) is not None, f"{label}.name")
        _require(target in _TARGETS, f"{label}.target must be one of {_TARGETS}")
        _require(
            isinstance(kernel_symbol, str) and _C_IDENT.fullmatch(kernel_symbol) is not None,
            f"{label}.kernel_symbol must be a C identifier",
        )
        _require(
            isinstance(source_sha256, str) and _SHA256.fullmatch(source_sha256) is not None,
            f"{label}.source_sha256 must be one lowercase SHA-256",
        )
        _require(
            abi_kind in FUSED_KDA_DECODE_GENERATED_ABIS,
            f"{label}.abi_kind must be one of {tuple(FUSED_KDA_DECODE_GENERATED_ABIS)}",
        )
        _require(
            state_dtype in ("bfloat16", "float32"),
            f"{label}.state_dtype must be bfloat16 or float32",
        )
        assert isinstance(name, str)
        assert isinstance(target, str)
        assert isinstance(kernel_symbol, str)
        assert isinstance(source_sha256, str)
        assert isinstance(abi_kind, str)
        assert isinstance(state_dtype, str)
        key = (name, target)
        _require(key not in observed, f"duplicate variant {name}/{target}")
        observed.add(key)

        body = _resolve_body(source_dir, item["body"], f"{label}.body")
        source_bytes = body.read_bytes()
        actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
        _require(
            actual_sha256 == source_sha256,
            f"{label}.source_sha256 mismatch: {actual_sha256} != {source_sha256}",
        )
        try:
            source = source_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FusedKDADecodeGeneratedManifestError(
                f"invalid fused KDA decode generated manifest: {label}.body is not UTF-8"
            ) from exc
        launch = item["launch"]
        _exact_keys(launch, _LAUNCH_KEYS, f"{label}.launch")
        assert isinstance(launch, dict)
        threads = _positive_int(launch["threads"], f"{label}.launch.threads", maximum=1024)
        _require(threads % 32 == 0, f"{label}.launch.threads must be warp-aligned")
        dynamic_smem_bytes = _positive_int(
            launch["dynamic_smem_bytes"],
            f"{label}.launch.dynamic_smem_bytes",
            maximum=256 * 1024,
        )
        _verify_source_abi(
            source,
            kernel_symbol,
            abi_kind=abi_kind,
            state_dtype=state_dtype,
        )
        _verify_source_launch(
            source,
            kernel_symbol,
            threads=threads,
            dynamic_smem_bytes=dynamic_smem_bytes,
        )

        result.append(
            FusedKDADecodeGeneratedVariant(
                name=name,
                target=target,  # type: ignore[arg-type]
                body_path=body,
                source_sha256=source_sha256,
                kernel_symbol=kernel_symbol,
                abi_kind=abi_kind,
                state_dtype=state_dtype,
                extra_cuda_cflags=_validate_extra_cuda_cflags(
                    item["extra_cuda_cflags"], f"{label}.extra_cuda_cflags"
                ),
                threads=threads,
                dynamic_smem_bytes=dynamic_smem_bytes,
                eligibility=_parse_eligibility(
                    item["eligibility"],
                    f"{label}.eligibility",
                    abi_kind=abi_kind,
                ),
            )
        )
    return tuple(result)


def _eligibility_matches(
    rule: FusedKDADecodeGeneratedEligibility,
    *,
    num_heads: int,
    num_rows: int,
    slot_class: FusedKDADecodeGeneratedSlotClass,
    lower_bound: float | None,
    norm_eps: float,
    strides: dict[str, int],
) -> bool:
    if num_heads not in rule.heads or num_rows < rule.minimum_rows:
        return False
    if rule.maximum_rows is not None and num_rows > rule.maximum_rows:
        return False
    if slot_class not in rule.slot_classes:
        return False
    if (
        rule.lower_bound_values is not None
        and lower_bound not in rule.lower_bound_values
    ):
        return False
    if rule.norm_eps_values is not None and norm_eps not in rule.norm_eps_values:
        return False
    return all(
        expected is None or strides[name] == expected
        for name, expected in rule.strides
    )


def select_fused_kda_decode_generated_variant(
    *,
    target: FusedKDADecodeGeneratedTarget,
    num_heads: int,
    num_rows: int,
    state_dtype: str,
    slot_class: FusedKDADecodeGeneratedSlotClass,
    lower_bound: float | None,
    norm_eps: float,
    x_row_stride: int,
    conv_slot_stride: int,
    beta_row_stride: int,
    state_slot_stride: int,
    output_gate_row_stride: int,
    variants: Sequence[FusedKDADecodeGeneratedVariant] | None = None,
) -> FusedKDADecodeGeneratedVariant | None:
    """Select the first manifest route matching exact runtime facts.

    Manifest order is the dispatch order, allowing narrow measured routes to
    precede full-domain fallbacks without baking a winner-specific inventory
    into the Python API.
    """

    _require(target in _TARGETS, f"target must be one of {_TARGETS}")
    _require(
        num_heads in _CONTRACT["supported_heads"],
        f"num_heads must be one of {_CONTRACT['supported_heads']}",
    )
    _require(num_rows > 0, "num_rows must be positive")
    _require(state_dtype in _CONTRACT["state_dtypes"], "unsupported state dtype")
    _require(slot_class in _SLOT_CLASSES, "unsupported slot class")
    if lower_bound is not None:
        lower_bound = _finite_float(lower_bound, "lower_bound")
        _require(lower_bound < 0.0, "lower_bound must be negative or null")
    norm_eps = _finite_float(norm_eps, "norm_eps")
    _require(norm_eps >= 0.0, "norm_eps must be non-negative")
    strides = {
        "x_row_stride": x_row_stride,
        "conv_slot_stride": conv_slot_stride,
        "beta_row_stride": beta_row_stride,
        "state_slot_stride": state_slot_stride,
        "output_gate_row_stride": output_gate_row_stride,
    }
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) for value in strides.values()),
        "strides must be integers",
    )
    _require(all(value >= 0 for value in strides.values()), "strides must be non-negative")
    if num_rows > 2**31 - 1 or any(value > 2**31 - 1 for value in strides.values()):
        return None
    available = (
        load_fused_kda_decode_generated_variants()
        if variants is None
        else tuple(variants)
    )
    for variant in available:
        if variant.target != target or variant.state_dtype != state_dtype:
            continue
        for rule in variant.eligibility:
            if _eligibility_matches(
                rule,
                num_heads=num_heads,
                num_rows=num_rows,
                slot_class=slot_class,
                lower_bound=lower_bound,
                norm_eps=norm_eps,
                strides=strides,
            ):
                return variant
    return None


def fused_kda_decode_generated_is_available() -> bool:
    """Return whether a complete, verified generated-source manifest is installed."""

    return bool(load_fused_kda_decode_generated_variants())


def get_fused_kda_decode_generated_variant(
    name: str, target: FusedKDADecodeGeneratedTarget
) -> FusedKDADecodeGeneratedVariant:
    """Return one verified source record from the installed manifest."""

    for variant in load_fused_kda_decode_generated_variants():
        if variant.name == name and variant.target == target:
            return variant
    raise RuntimeError(f"fused KDA decode generated source is unavailable for {name}/{target}")


def get_fused_kda_decode_generated_uri(
    name: str, target: FusedKDADecodeGeneratedTarget
) -> str:
    """Return the source-identity-bearing JIT cache key for one variant."""

    variant = get_fused_kda_decode_generated_variant(name, target)
    return f"fused_kda_decode_generated_{name}_{variant.source_sha256[:16]}_{target}"


def _render_binding(variant: FusedKDADecodeGeneratedVariant) -> str:
    has_rows = int(variant.abi_kind == "repeated_safe")
    state_is_bfloat16 = int(variant.state_dtype == "bfloat16")
    return f"""\
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define FLASHINFER_FUSED_KDA_DECODE_BODY_FILE "{variant.body_path.name}"
#define FLASHINFER_FUSED_KDA_DECODE_KERNEL {variant.kernel_symbol}
#define FLASHINFER_FUSED_KDA_DECODE_THREADS {variant.threads}
#define FLASHINFER_FUSED_KDA_DECODE_SMEM_BYTES {variant.dynamic_smem_bytes}
#define FLASHINFER_FUSED_KDA_DECODE_HAS_ROWS {has_rows}
#define FLASHINFER_FUSED_KDA_DECODE_STATE_IS_BFLOAT16 {state_is_bfloat16}
#define FLASHINFER_FUSED_KDA_DECODE_ARG_PLAN_SHA256 "{_ARG_PLAN_SHA256[variant.abi_kind]}"

#include "{_BINDING_HEADER}"
"""


@functools.cache
def gen_fused_kda_decode_generated_module(
    name: str, target: FusedKDADecodeGeneratedTarget
) -> JitSpec:
    """Generate one JIT module from a verified frozen CUDA body."""

    variant = get_fused_kda_decode_generated_variant(name, target)
    csrc_dir = get_kda_csrc_dir()
    binding_header = csrc_dir / _BINDING_HEADER
    if not binding_header.is_file():
        raise FileNotFoundError(f"fused KDA decode generated binding not found: {binding_header}")
    uri = get_fused_kda_decode_generated_uri(name, target)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "fused_kda_decode_generated_binding.cu"
    write_if_different(binding, _render_binding(variant))
    spec = gen_kda_jit_spec(
        name=uri,
        sources=[binding],
        target=target,
        target_define=_TARGET_DEFINE,
        csrc_dir=csrc_dir,
        include_dir=get_flashinfer_include_dir(),
        extra_cuda_cflags=variant.extra_cuda_cflags,
    )
    logger.info(f"Generated fused KDA decode {name} {target} JIT spec: {spec.name}")
    return spec


@functools.cache
def load_fused_kda_decode_generated_module(
    name: str, target: FusedKDADecodeGeneratedTarget
):
    """Build or load one verified generated-source module."""

    module = gen_fused_kda_decode_generated_module(name, target).build_and_load()
    logger.info(f"Loaded fused KDA decode {name} {target} module")
    return module


__all__ = [
    "FUSED_KDA_DECODE_GENERATED_ABIS",
    "FusedKDADecodeGeneratedEligibility",
    "FusedKDADecodeGeneratedManifestError",
    "FusedKDADecodeGeneratedSlotClass",
    "FusedKDADecodeGeneratedTarget",
    "FusedKDADecodeGeneratedVariant",
    "fused_kda_decode_generated_is_available",
    "gen_fused_kda_decode_generated_module",
    "get_fused_kda_decode_generated_uri",
    "get_fused_kda_decode_generated_variant",
    "load_fused_kda_decode_generated_module",
    "load_fused_kda_decode_generated_variants",
    "select_fused_kda_decode_generated_variant",
]
