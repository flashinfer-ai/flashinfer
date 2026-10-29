# Copyright (c) 2026 by FlashInfer team.
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

"""Emit a proposed production config JSON from a selected menu prefix."""

from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .db import canonical_json
from .menu_config import runtime_twin


_ACTIVATION_FIELDS = ("fused_act", "act", "eltwise_act_type")
_ACTIVATION_KEY = ",".join(_ACTIVATION_FIELDS)
_PER_TOKEN_FIELDS = ("use_per_token_sf_b", "per_token_sf_dtype")
_DECODER = json.JSONDecoder()

_COMMENT_PREFIX = {
    "NvFP4xNvFP4": "FP4",
    "MxFP4xMxFP8": "MxFp4xMxFp8",
    "MxFP8xMxFP8": "MxFp8xMxFp8",
}

# Semantic order for both scalar keys and columns in grouped keys.
_KEY_ORDER = (
    "fused_act",
    "act",
    "eltwise_act_type",
    "dtype_a",
    "dtype_b",
    "dtype_c",
    "bias_type",
    "use_global_scales",
    "use_deepseek_fp8",
    "transpose_mma_output",
    "route_act",
    "route_sfs_act",
    "sf_layout_b",
    "sf_layout_c",
    "mma_m",
    "umma_cta_count",
    "mma_n",
    "mma_k",
    "tile_n",
    "epi_tile_m",
    "epi_tile_n",
    "tile_k",
    "cluster_dim_x",
    "cluster_dim_y",
    "fallback_cluster_dim_x",
    "fallback_cluster_dim_y",
    "cta_raster_order",
    "num_stages_a",
    "num_stages_b",
    "num_stages_smem_sfa",
    "num_stages_smem_sfb",
    "num_stages_tmem_sfa",
    "num_stages_tmem_sfb",
    "num_stages_tmem_acc",
    "num_stages_c_smem",
    "num_stages_workid",
    "num_load_b_warps",
    "num_load_sfb_warps",
    "mma_regs",
    "load_regs",
    "load_a_regs",
    "load_b_regs",
    "load_sf_regs",
    "load_sfa_regs",
    "load_sfb_regs",
    "copy_sf_regs",
    "padding_regs",
    "workid_regs",
    "epilogue_regs",
    "fuse_sf_copy_to_mma",
    "fuse_operand_sf_loads",
    "use_max_tmem_overlap",
    "use_unroll_loop_2x_for_mma",
    "use_tma_store",
    "use_tma_oob_opt",
    "use_early_exit",
    "tile_scheduler",
    "use_clc_fast_drain",
    "use_work_throttle",
    "use_per_token_sf_b",
    "per_token_sf_dtype",
    "weight_layout",
)
_KEY_RANK = {key: index for index, key in enumerate(_KEY_ORDER)}

# Emitted as one grouped key (a single row when constant) and never elided
# against the template, so each config shows its complete pipeline depth.
_STAGE_FIELDS = (
    "num_stages_a",
    "num_stages_b",
    "num_stages_smem_sfa",
    "num_stages_smem_sfb",
    "num_stages_tmem_sfa",
    "num_stages_tmem_sfb",
)
_TMEM_SF_STAGE_FIELDS = ("num_stages_tmem_sfa", "num_stages_tmem_sfb")
# Scalars when constant; join the stage block when they vary.
_EXTRA_STAGE_FIELDS = ("num_stages_tmem_acc", "num_stages_c_smem", "num_stages_workid")
_REGISTER_FIELDS = tuple(key for key in _KEY_ORDER if key.endswith("_regs"))

# Zero requests automatic resolution for these fields. Emit writes the resolved
# value instead, so the menu pins the kernel that was measured even if the
# heuristics or register fallbacks change later.
_AUTO_FIELDS = (
    "cluster_dim_x",
    "num_load_a_warps",
    "num_load_b_warps",
    "num_load_sfa_warps",
    "num_load_sfb_warps",
    "load_a_regs",
    "load_b_regs",
    "load_sfa_regs",
    "load_sfb_regs",
)


def _resolve_template(
    name: str,
    templates: Mapping[str, Mapping[str, object]],
    *,
    stack: tuple[str, ...] = (),
) -> dict[str, object]:
    if name in stack:
        raise ValueError(f"template cycle: {' -> '.join((*stack, name))}")
    raw = templates[name]
    result: dict[str, object] = {}
    parent = raw.get("_template")
    if parent is not None:
        result.update(_resolve_template(str(parent), templates, stack=(*stack, name)))
    result.update({key: value for key, value in raw.items() if key != "_template"})
    return result


def expand_config_rows(data: Mapping[str, object]) -> list[dict[str, object]]:
    """Expand grouped/list config rows into explicit mapper rows."""

    templates = data.get("templates", {})
    if not isinstance(templates, Mapping):
        raise ValueError("templates must be an object")
    result = []
    for raw_index, raw_value in enumerate(data.get("configs", ())):
        if not isinstance(raw_value, Mapping):
            raise ValueError(f"configs[{raw_index}] must be an object")
        raw = dict(raw_value)
        merged: dict[str, object] = {}
        parent = raw.get("_template")
        if parent is not None:
            merged.update(_resolve_template(str(parent), templates))
        merged.update(
            {
                key: value
                for key, value in raw.items()
                if key not in ("_template", "_comment")
            }
        )
        combinations: list[dict[str, object]] = [{}]
        for key, value in merged.items():
            if "," in key:
                keys = tuple(part.strip() for part in key.split(",") if part.strip())
                rows = value if isinstance(value, list) else [value]
                choices = []
                for row in rows:
                    if not isinstance(row, list) or len(row) != len(keys):
                        raise ValueError(
                            f"config {raw_index}: grouped key {key!r} has invalid row"
                        )
                    choices.append(dict(zip(keys, row, strict=True)))
            elif isinstance(value, list):
                choices = [{key: item} for item in value]
            else:
                choices = [{key: value}]
            combinations = [
                {**combination, **choice}
                for combination in combinations
                for choice in choices
            ]
        comment = str(raw.get("_comment", f"config_{raw_index}"))
        result.extend({"_comment": comment, **row} for row in combinations)
    return result


def _route_active(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.lower() not in ("", "false", "none", "0")
    return bool(value)


def _row_slice(row: Mapping[str, object]) -> tuple[str, str | None]:
    fc = "fc1" if _route_active(row.get("route_act")) else "fc2"
    if fc == "fc2":
        return fc, None
    if bool(row.get("fused_act", False)):
        return fc, "gated"
    if str(row.get("eltwise_act_type", "none")).lower() not in ("", "none"):
        return fc, "elementwise"
    return fc, None


def _matches_dtype(row: Mapping[str, object], dtype: str, fc: str) -> bool:
    signatures = {
        "NvFP4xNvFP4": (
            "e2m1",
            "e2m1",
            "e2m1" if fc == "fc1" else "bf16",
            False,
        ),
        "MxFP4xMxFP8": (
            "mxe2m1",
            "mxe4m3",
            "mxe4m3" if fc == "fc1" else "bf16",
            False,
        ),
        "MxFP8xMxFP8": (
            "mxe4m3",
            "mxe4m3",
            "mxe4m3" if fc == "fc1" else "bf16",
            False,
        ),
        "BF16": ("bf16", "bf16", "bf16", False),
        "Fp8-Block": (
            "e4m3",
            "e4m3",
            "e4m3" if fc == "fc1" else "bf16",
            True,
        ),
        "Fp8-Per-Tensor": (
            "e4m3",
            "e4m3",
            "e4m3" if fc == "fc1" else "bf16",
            False,
        ),
    }
    if dtype not in signatures:
        raise ValueError(f"emission for dtype {dtype!r} is not implemented")
    dtype_a, dtype_b, dtype_c, deepseek = signatures[dtype]
    row_dtype_a = str(row.get("dtype_a", "")).lower()
    return (
        row_dtype_a == dtype_a
        and str(row.get("dtype_b", row_dtype_a)).lower() == dtype_b
        and str(row.get("dtype_c", "")).lower() == dtype_c
        and bool(row.get("use_deepseek_fp8", False)) == deepseek
    )


def _quant_key(row: Mapping[str, object]) -> tuple[object, ...]:
    dtype_a = str(row.get("dtype_a", "")).lower()
    weight_layout = row.get("weight_layout", "major_k")
    if str(weight_layout).lower() in ("0", "majork", "major_k"):
        weight_layout = "major_k"
    return (
        dtype_a,
        str(row.get("dtype_b", dtype_a)).lower(),
        str(row.get("dtype_c", "")).lower(),
        bool(row.get("use_deepseek_fp8", False)),
        weight_layout,
    )


def _expand_raw_row(
    raw_value: Mapping[str, object],
    templates: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    raw = dict(raw_value)
    merged: dict[str, object] = {}
    parent = raw.get("_template")
    if parent is not None:
        merged.update(_resolve_template(str(parent), templates))
    merged.update(
        {
            key: value
            for key, value in raw.items()
            if key not in ("_template", "_comment")
        }
    )
    combinations: list[dict[str, object]] = [{}]
    for key, value in merged.items():
        if "," in key:
            keys = tuple(part.strip() for part in key.split(",") if part.strip())
            values = value if isinstance(value, list) else [value]
            choices = []
            for row in values:
                if not isinstance(row, list) or len(row) != len(keys):
                    raise ValueError(f"grouped key {key!r} has invalid row")
                choices.append(dict(zip(keys, row, strict=True)))
        elif isinstance(value, list):
            choices = [{key: item} for item in value]
        else:
            choices = [{key: value}]
        combinations = [
            {**combination, **choice}
            for combination in combinations
            for choice in choices
        ]
    return combinations


def _target_row_indices(
    base: Mapping[str, object],
    dtype: str,
) -> tuple[list[int], list[dict[str, object]]]:
    templates = base.get("templates", {})
    if not isinstance(templates, Mapping):
        raise ValueError("templates must be an object")
    indices = []
    expanded = []
    for index, raw in enumerate(base.get("configs", ())):
        if not isinstance(raw, Mapping):
            raise ValueError(f"configs[{index}] must be an object")
        rows = _expand_raw_row(raw, templates)
        matches = []
        for row in rows:
            fc, _ = _row_slice(row)
            matches.append(_matches_dtype(row, dtype, fc))
        if any(matches) and not all(matches):
            raise ValueError(f"configs[{index}] mixes {dtype} and non-{dtype} options")
        if matches and all(matches):
            indices.append(index)
            expanded.extend(rows)
    if indices and indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"{dtype} config rows must form one contiguous slice")
    return indices, expanded


def _activation_tuple(row: Mapping[str, object]) -> tuple[object, str, str]:
    return (
        bool(row.get("fused_act", False)),
        str(row.get("act", "none")).lower(),
        str(row.get("eltwise_act_type", "none")).lower(),
    )


def _activation_class(values: Sequence[object]) -> str | None:
    fused, act, eltwise = bool(values[0]), str(values[1]), str(values[2])
    if fused and act.lower() in ("swiglu", "geglu", "situ"):
        return "gated"
    if not fused and eltwise.lower() not in ("", "none", "false", "0"):
        return "elementwise"
    return None


def _stable_unique(values: Iterable[tuple[Any, ...]]) -> tuple[tuple[Any, ...], ...]:
    result = []
    seen = set()
    for value in values:
        key = canonical_json(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _base_variants(
    rows: Sequence[Mapping[str, object]],
) -> tuple[
    dict[str, tuple[tuple[object, str, str], ...]],
    dict[str, tuple[tuple[object | None, object | None], ...]],
]:
    activation_values = _stable_unique(
        _activation_tuple(row)
        for row in rows
        if _row_slice(row)[0] == "fc1"
        and _activation_class(_activation_tuple(row)) is not None
    )
    activations: dict[str, tuple[tuple[object, str, str], ...]] = {}
    for class_name in ("gated", "elementwise"):
        values = tuple(
            value
            for value in activation_values
            if _activation_class(value) == class_name
        )
        if values:
            activations[class_name] = values
    if not activations:
        raise ValueError("base dtype slice has no supported FC1 activations")

    per_token: dict[str, tuple[tuple[object | None, object | None], ...]] = {}
    for fc in ("fc1", "fc2"):
        fc_rows = [row for row in rows if _row_slice(row)[0] == fc]
        if not fc_rows:
            raise ValueError(f"base dtype slice has no {fc} rows")
        values = _stable_unique(
            (
                row.get("use_per_token_sf_b"),
                row.get("per_token_sf_dtype"),
            )
            for row in fc_rows
        )
        if fc == "fc1":
            by_class = {
                class_name: {
                    (
                        row.get("use_per_token_sf_b"),
                        row.get("per_token_sf_dtype"),
                    )
                    for row in fc_rows
                    if _row_slice(row)[1] == class_name
                }
                for class_name in activations
            }
            if any(set(values) != class_values for class_values in by_class.values()):
                raise ValueError(
                    "base FC1 per-token variants differ by activation class"
                )
        per_token[fc] = values
    return activations, per_token


def _geometry_options(
    options: Mapping[str, object],
    *,
    fc: str,
) -> dict[str, object]:
    result = dict(options)
    for field in _ACTIVATION_FIELDS:
        result.pop(field, None)
    for field in _PER_TOKEN_FIELDS:
        result.pop(field, None)
    if fc == "fc2":
        result.pop("route_act", None)
    return result


def _selected_geometries(
    selected: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for entry in selected:
        fc = str(entry["fc"])
        if fc not in ("fc1", "fc2"):
            raise ValueError(f"invalid selected FC {fc!r}")
        options = _geometry_options(entry["options"], fc=fc)
        key = (fc, canonical_json(options))
        value = grouped.setdefault(
            key,
            {
                "fc": fc,
                "options": options,
                "classes": set(),
                "rank": int(entry["rank"]),
            },
        )
        class_name = entry.get("activation_class")
        if fc == "fc1":
            if class_name not in ("gated", "elementwise"):
                raise ValueError("selected FC1 entry is missing its activation class")
            value["classes"].add(str(class_name))
        value["rank"] = min(int(value["rank"]), int(entry["rank"]))

    # A persistent base receives this exact sibling from the runtime.  Avoid
    # emitting the sibling explicitly when its base is already selected.
    redundant_twins = set()
    for _key, value in tuple(grouped.items()):
        twin = runtime_twin(value["options"])
        if twin is None:
            continue
        twin_key = (str(value["fc"]), canonical_json(twin))
        explicit_twin = grouped.get(twin_key)
        if explicit_twin is None:
            continue
        value["classes"].update(explicit_twin["classes"])
        value["rank"] = min(int(value["rank"]), int(explicit_twin["rank"]))
        redundant_twins.add(twin_key)
    return [value for key, value in grouped.items() if key not in redundant_twins]


def _auto_field_value(cfg: Any, field: str) -> int:
    if field == "cluster_dim_x":
        return int(cfg.preferred_cluster_shape[0])
    return int(getattr(cfg.warp_layout, field))


def _row_configs(
    options: Mapping[str, object],
    *,
    fc: str,
    sm_archs: Sequence[int],
) -> dict[tuple[int, bool], Any]:
    """Build the configs the production mapper derives from one expanded row."""

    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        make_config,
        validate_config,
    )
    from flashinfer.prims_ts.moe import config_mapper

    row = dict(options)
    act_kind = config_mapper._json_act_kind(row)
    activation_type = next(
        activation
        for activation, kind in config_mapper._ACTIVATION_TO_ACT_KIND.items()
        if kind == act_kind
    )
    dtype_a = config_mapper._json_dtype_value(row.get("dtype_a"))
    json_cfg = config_mapper._JsonBatchedGemmConfig(
        global_index=0, raw_index=0, comment="", combo_index=0, options=row
    )
    configs = {}
    for sm_arch in sm_archs:
        for has_bias in (False, True):
            try:
                cfg = make_config(
                    **config_mapper._json_config_kwargs(
                        json_cfg,
                        sm_arch=sm_arch,
                        fc=fc,
                        activation_type=activation_type,
                        has_bias=has_bias,
                        default_dtype_a=dtype_a,
                        default_dtype_b=dtype_a,
                        default_dtype_c=config_mapper._json_dtype_value(
                            row.get("dtype_c")
                        ),
                    )
                )
                validate_config(cfg)
            except ValueError:
                continue
            configs[(sm_arch, has_bias)] = cfg
    return configs


def _runtime_rows(
    options: Mapping[str, object],
    geometry: Mapping[str, object],
    *,
    mode: str,
    activations: Mapping[str, Sequence[tuple[object, str, str]]],
    per_token: Mapping[str, Sequence[tuple[object | None, object | None]]],
) -> list[dict[str, object]]:
    fc = str(geometry["fc"])
    activation_values: Sequence[Sequence[object] | None] = (
        _activation_values_for_geometry(geometry, mode=mode, activations=activations)
        if fc == "fc1"
        else (None,)
    )
    rows = [
        _apply_variants(options, fc=fc, activation=activation, per_token=value)
        for activation in activation_values
        for value in per_token[fc]
    ]
    twins = [runtime_twin(row) for row in rows]
    return [*rows, *(twin for twin in twins if twin is not None)]


def _resolve_auto_fields(
    geometries: Sequence[Mapping[str, object]],
    *,
    mode: str,
    activations: Mapping[str, Sequence[tuple[object, str, str]]],
    per_token: Mapping[str, Sequence[tuple[object | None, object | None]]],
    sm_archs: Sequence[int],
) -> list[dict[str, object]]:
    """Replace automatic zeros with the values every runtime variant resolves to.

    A replacement is kept only when it leaves every variant's compile signature
    unchanged, so the emitted menu runs exactly the selected kernels.
    """

    def rows_for(options: Mapping[str, object], geometry: Mapping[str, object]):
        return _runtime_rows(
            options, geometry, mode=mode, activations=activations, per_token=per_token
        )

    def signatures(rows: Sequence[Mapping[str, object]], fc: str):
        return [
            {
                key: cfg.compile_signature()
                for key, cfg in _row_configs(row, fc=fc, sm_archs=sm_archs).items()
            }
            for row in rows
        ]

    merged: dict[tuple[str, str], dict[str, object]] = {}
    for geometry in geometries:
        options = dict(geometry["options"])
        fc = str(geometry["fc"])
        fields = [
            field
            for field in _AUTO_FIELDS
            if field in options
            and not isinstance(options[field], bool)
            and int(options[field]) == 0
        ]
        if fields:
            rows = rows_for(options, geometry)
            baseline_configs = [
                _row_configs(row, fc=fc, sm_archs=sm_archs) for row in rows
            ]
            baseline = [
                {key: cfg.compile_signature() for key, cfg in configs.items()}
                for configs in baseline_configs
            ]
            values = {}
            for field in fields:
                resolved = {
                    _auto_field_value(cfg, field)
                    for configs in baseline_configs
                    for cfg in configs.values()
                }
                if len(resolved) == 1 and (value := resolved.pop()) != 0:
                    values[field] = value
            replaced = rows_for({**options, **values}, geometry)
            if values and signatures(replaced, fc) != baseline:
                accepted: dict[str, int] = {}
                for field, value in values.items():
                    trial = {**options, **accepted, field: value}
                    if signatures(rows_for(trial, geometry), fc) == baseline:
                        accepted[field] = value
                values = accepted
            options.update(values)
        key = (fc, canonical_json(options))
        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                **geometry,
                "options": options,
                "classes": set(geometry["classes"]),
            }
        else:
            existing["classes"].update(geometry["classes"])
            existing["rank"] = min(int(existing["rank"]), int(geometry["rank"]))
    return list(merged.values())


def _resolve_activation_mode(
    mode: str,
    selected: Sequence[Mapping[str, object]],
) -> str:
    if mode not in ("auto", "per-class", "shared"):
        raise ValueError("activations must be auto, per-class, or shared")
    if mode != "auto":
        return mode
    return (
        "per-class"
        if any(
            entry.get("fc") == "fc1" and entry.get("activation_class") == "elementwise"
            for entry in selected
        )
        else "shared"
    )


def _activation_values_for_geometry(
    geometry: Mapping[str, object],
    *,
    mode: str,
    activations: Mapping[str, Sequence[tuple[object, str, str]]],
) -> tuple[tuple[object, str, str], ...]:
    if geometry["fc"] == "fc2":
        return ()
    if mode == "shared":
        return tuple(
            value
            for class_name in ("gated", "elementwise")
            for value in activations.get(class_name, ())
        )
    return tuple(
        value
        for class_name in ("gated", "elementwise")
        if class_name in geometry["classes"]
        for value in activations.get(class_name, ())
    )


def _apply_variants(
    options: Mapping[str, object],
    *,
    fc: str,
    activation: Sequence[object] | None,
    per_token: Sequence[object],
) -> dict[str, object]:
    result = dict(options)
    if fc == "fc1":
        if activation is None:
            raise ValueError("FC1 option requires an activation")
        result.update(dict(zip(_ACTIVATION_FIELDS, activation, strict=True)))
    for field, value in zip(_PER_TOKEN_FIELDS, per_token, strict=True):
        if value is None:
            result.pop(field, None)
        else:
            result[field] = value
    return result


def _runtime_options(options: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    result = list(options)
    seen = {canonical_json(value) for value in result}
    for value in tuple(result):
        twin = runtime_twin(value)
        if twin is not None and canonical_json(twin) not in seen:
            seen.add(canonical_json(twin))
            result.append(twin)
    return result


def _runtime_identity(options: Mapping[str, object], *, fc: str) -> str:
    normalized = dict(options)
    if fc == "fc2":
        for field in _ACTIVATION_FIELDS:
            normalized.pop(field, None)
    return canonical_json(normalized)


def _expected_sets(
    geometries: Sequence[Mapping[str, object]],
    *,
    mode: str,
    activations: Mapping[str, Sequence[tuple[object, str, str]]],
    per_token: Mapping[str, Sequence[tuple[object | None, object | None]]],
) -> dict[tuple[str, str, object | None], set[str]]:
    explicit: dict[tuple[str, str, object | None], list[dict[str, object]]] = (
        collections.defaultdict(list)
    )
    for geometry in geometries:
        fc = str(geometry["fc"])
        activation_values: Sequence[Sequence[object] | None] = (
            _activation_values_for_geometry(
                geometry, mode=mode, activations=activations
            )
            if fc == "fc1"
            else (None,)
        )
        for activation in activation_values:
            activation_name = (
                "none"
                if activation is None
                else (
                    str(activation[2])
                    if str(activation[2]) != "none"
                    else str(activation[1])
                )
            )
            for per_token_value in per_token[fc]:
                key = (fc, activation_name, per_token_value[0])
                explicit[key].append(
                    _apply_variants(
                        geometry["options"],
                        fc=fc,
                        activation=activation,
                        per_token=per_token_value,
                    )
                )
    return {
        key: {_runtime_identity(value, fc=key[0]) for value in _runtime_options(values)}
        for key, values in explicit.items()
    }


def _ordered_keys(keys: Iterable[str]) -> list[str]:
    return sorted(keys, key=lambda key: (_KEY_RANK.get(key, len(_KEY_RANK)), key))


def _template_candidates(
    base: Mapping[str, object],
    indices: Sequence[int],
) -> list[str]:
    result = []
    seen = set()
    for index in indices:
        raw = base["configs"][index]
        parent = raw.get("_template")
        if parent is not None and str(parent) not in seen:
            seen.add(str(parent))
            result.append(str(parent))
    if not result:
        raise ValueError("base dtype slice does not reference any templates")
    return result


def _choose_template(
    options: Mapping[str, object],
    *,
    templates: Mapping[str, Mapping[str, object]],
    candidates: Sequence[str],
) -> tuple[str, dict[str, object]]:
    ignored = {*_ACTIVATION_FIELDS, *_PER_TOKEN_FIELDS}
    choices = []
    for order, name in enumerate(candidates):
        inherited = _resolve_template(name, templates)
        required = {
            key: value for key, value in inherited.items() if key not in ignored
        }
        if any(key not in options for key in required):
            continue
        diff = {
            key: value
            for key, value in options.items()
            if key in ignored or inherited.get(key, object()) != value
        }
        choices.append((len(diff), order, name, diff))
    if not choices:
        raise ValueError(
            "no base-slice template is compatible with selected option "
            f"tile_n={options.get('tile_n')}"
        )
    _, _, name, diff = min(choices, key=lambda value: (value[0], value[1]))
    return name, diff


def _compact_variant_fields(
    variants: Sequence[tuple[object | None, object | None]],
) -> dict[str, object]:
    values = [value[0] for value in variants]
    dtypes = [value[1] for value in variants]
    result: dict[str, object] = {}
    if any(value is not None for value in values):
        if any(value is None for value in values):
            raise ValueError("cannot compact mixed absent and explicit per-token flags")
        result["use_per_token_sf_b"] = values[0] if len(values) == 1 else values
    present_dtypes = [value for value in dtypes if value is not None]
    if present_dtypes:
        if len(set(present_dtypes)) != 1 or len(present_dtypes) != len(dtypes):
            raise ValueError("base per-token scale dtype is not uniform")
        result["per_token_sf_dtype"] = present_dtypes[0]
    return result


def _prepared_rows(
    geometries: Sequence[Mapping[str, object]],
    *,
    mode: str,
    activations: Mapping[str, Sequence[tuple[object, str, str]]],
    per_token: Mapping[str, Sequence[tuple[object | None, object | None]]],
    templates: Mapping[str, Mapping[str, object]],
    template_candidates: Sequence[str],
) -> list[dict[str, object]]:
    rows = []
    for geometry in geometries:
        fc = str(geometry["fc"])
        template, _ = _choose_template(
            geometry["options"],
            templates=templates,
            candidates=template_candidates,
        )
        row = {
            "_template": template,
            **geometry["options"],
        }
        activation_values = _activation_values_for_geometry(
            geometry, mode=mode, activations=activations
        )
        if fc == "fc1":
            if not activation_values:
                raise ValueError("selected FC1 geometry has no emitted activation")
            row[_ACTIVATION_KEY] = [list(value) for value in activation_values]
        row.update(_compact_variant_fields(per_token[fc]))
        rows.append(
            {
                "fc": fc,
                "classes": frozenset(geometry["classes"]),
                "rank": int(geometry["rank"]),
                "row": row,
            }
        )
    return rows


def _structure_flags(
    resolved: Mapping[str, object],
) -> tuple[bool, bool, bool, bool, bool]:
    return (
        str(resolved.get("tile_scheduler", "static")).lower() == "static",
        bool(resolved.get("fuse_sf_copy_to_mma", False)),
        bool(resolved.get("use_max_tmem_overlap", False)),
        bool(resolved.get("use_unroll_loop_2x_for_mma", False)),
        str(resolved.get("route_sfs_act", "auto")).lower() == "tma",
    )


def _range_label(values: Iterable[object]) -> str:
    unique = sorted({int(value) for value in values})
    return str(unique[0]) if len(unique) == 1 else f"{unique[0]}-{unique[-1]}"


def _structural_name(
    dtype: str,
    fc: str,
    template: str,
    members: Sequence[Mapping[str, object]],
    *,
    classes: frozenset[str],
    mode: str,
    templates: Mapping[str, Mapping[str, object]],
) -> str:
    inherited = _resolve_template(template, templates)
    effective = [{**inherited, **member} for member in members]
    ctas = {int(member.get("umma_cta_count", 1)) for member in effective}
    if len(ctas) != 1:
        raise ValueError(f"structural group mixes CTA counts: {sorted(ctas)}")
    name = (
        f"{_COMMENT_PREFIX.get(dtype, dtype)}_{fc.upper()}_"
        f"{next(iter(ctas))}Cta"
        f"_N{_range_label(member['tile_n'] for member in effective)}"
        f"_K{_range_label(member['tile_k'] for member in effective)}"
    )
    tags = (
        (
            "Static",
            lambda row: str(row.get("tile_scheduler", "static")).lower() == "static",
        ),
        ("FusedSf", lambda row: bool(row.get("fuse_sf_copy_to_mma", False))),
        ("MaxOverlap", lambda row: bool(row.get("use_max_tmem_overlap", False))),
        ("Unroll2x", lambda row: bool(row.get("use_unroll_loop_2x_for_mma", False))),
        ("TmaSf", lambda row: str(row.get("route_sfs_act", "auto")).lower() == "tma"),
    )
    for tag, predicate in tags:
        if all(predicate(member) for member in effective):
            name += f"_{tag}"
    if mode == "per-class" and fc == "fc1" and len(classes) == 1:
        name += "_Gated" if "gated" in classes else "_Eltwise"
    return name


def _compact_members(
    keys: Sequence[str],
    members: Sequence[Mapping[str, object]],
    inherited: Mapping[str, object],
) -> dict[str, object]:
    """Lay out one structural group's members as scalars and grouped keys.

    Separate multi-row keys expand as a cartesian product, so every column
    that varies across members lands in a single joint key.
    """

    all_stage_fields = (*_STAGE_FIELDS, *_EXTRA_STAGE_FIELDS)
    filled = [key for key in all_stage_fields if key not in keys and key in inherited]
    members = [{**{key: inherited[key] for key in filled}, **m} for m in members]
    keys = _ordered_keys((*keys, *filled))
    varying = {
        key
        for key in keys
        if any(member[key] != members[0][key] for member in members[1:])
    }

    stage_block = [key for key in keys if key in _STAGE_FIELDS]
    if all(
        key in keys and key not in varying and members[0][key] == 1
        for key in _TMEM_SF_STAGE_FIELDS
    ):
        stage_block = [key for key in stage_block if key not in _TMEM_SF_STAGE_FIELDS]
    stage_block += [
        key for key in keys if key in _EXTRA_STAGE_FIELDS and key in varying
    ]
    register_block = [key for key in keys if key in _REGISTER_FIELDS]
    # A one-column grouped key would read as a list of choices, not a row.
    block_of = {
        key: tuple(block)
        for block in (stage_block, register_block)
        if len(block) > 1
        for key in block
    }
    joint = [
        key
        for key in keys
        if any(column in varying for column in block_of.get(key, (key,)))
    ]
    explicit = {*all_stage_fields, *register_block}

    compact: dict[str, object] = {}
    for key in keys:
        if key in joint:
            if len(joint) == 1:
                compact[key] = [member[key] for member in members]
            continue
        block = block_of.get(key)
        if block is not None:
            block_key = ",".join(block)
            if block_key not in compact:
                compact[block_key] = [[members[0][column] for column in block]]
            continue
        if key in explicit or key not in inherited or inherited[key] != members[0][key]:
            compact[key] = members[0][key]
    if len(joint) > 1:
        compact[",".join(joint)] = [
            [member[key] for key in joint] for member in members
        ]
    return compact


def _group_rows(
    prepared: Sequence[Mapping[str, object]],
    *,
    dtype: str,
    mode: str,
    templates: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    independent = {_ACTIVATION_KEY, *_PER_TOKEN_FIELDS}
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = (
        collections.defaultdict(list)
    )
    for value in prepared:
        row = value["row"]
        template = str(row["_template"])
        inherited = _resolve_template(template, templates)
        resolved = {**inherited, **row}
        member_keys = frozenset(
            key for key in row if key not in {"_template", *independent}
        )
        variant_values = tuple(
            (key, canonical_json(row[key]))
            for key in (_ACTIVATION_KEY, *_PER_TOKEN_FIELDS)
            if key in row
        )
        key = (
            value["fc"],
            template,
            _structure_flags(resolved),
            value["classes"],
            variant_values,
            member_keys,
        )
        groups[key].append(value)

    prepared_groups = []
    for key, values in groups.items():
        fc, template, flags, classes, _, member_keys = key
        members = [dict(value["row"]) for value in values]
        keys = _ordered_keys(member_keys)
        members.sort(
            key=lambda member: tuple(
                (type(member[key]).__name__, repr(member[key])) for key in keys
            )
        )
        prepared_groups.append(
            (
                fc,
                template,
                flags,
                classes,
                min(int(value["rank"]) for value in values),
                keys,
                members,
            )
        )

    def template_rank(name: str) -> int:
        if name.endswith("LowLatency"):
            return 0
        if name.endswith("Base"):
            return 1
        if name.endswith("HighThroughput"):
            return 2
        return 3

    prepared_groups.sort(
        key=lambda value: (
            0 if value[0] == "fc1" else 1,
            template_rank(value[1]),
            min(
                int(
                    member.get(
                        "tile_n",
                        _resolve_template(value[1], templates)["tile_n"],
                    )
                )
                for member in value[6]
            ),
            value[2],
            value[4],
        )
    )
    names = [
        _structural_name(
            dtype,
            fc,
            template,
            members,
            classes=classes,
            mode=mode,
            templates=templates,
        )
        for fc, template, _, classes, _, _, members in prepared_groups
    ]
    totals = collections.Counter(names)

    output = []
    for name, (
        _,
        template,
        _,
        _,
        _,
        keys,
        members,
    ) in zip(names, prepared_groups, strict=True):
        if totals[name] > 1:
            payload = sorted(canonical_json(member) for member in members)
            suffix = hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:4]
            name = f"{name}_{suffix}"
        inherited = _resolve_template(template, templates)
        variants = {
            key: members[0][key]
            for key in (_ACTIVATION_KEY, *_PER_TOKEN_FIELDS)
            if key in members[0]
        }
        compact: dict[str, object] = {
            "_comment": name,
            "_template": template,
            **_compact_members(keys, members, inherited),
            **variants,
        }
        expected = [
            {**inherited, **member_options, **variant}
            for member_options in (
                {key: member[key] for key in keys} for member in members
            )
            for variant in _expand_raw_row(variants, {})
        ]
        if _expand_raw_row(compact, templates) != expected:
            raise ValueError(f"{name}: compact layout changes the expanded configs")
        output.append(compact)
    return output


def _skip_separators(text: str, position: int) -> int:
    while position < len(text) and text[position] in " \t\r\n,":
        position += 1
    return position


def _array_spans(text: str, key: str) -> tuple[int, int, list[tuple[int, int]]]:
    marker = f'"{key}": ['
    open_position = text.index(marker) + len(marker) - 1
    spans = []
    position = open_position + 1
    while True:
        position = _skip_separators(text, position)
        if text[position] == "]":
            return open_position, position, spans
        _, end = _DECODER.raw_decode(text, position)
        spans.append((position, end))
        position = end


def _format_json(value: Any, level: int = 0) -> str:
    indent = " " * (4 * level)
    child_indent = " " * (4 * (level + 1))
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f"{child_indent}{json.dumps(key)}: {_format_json(item, level + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{indent}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(item, (dict, list)) for item in value):
            return json.dumps(value)
        items = [f"{child_indent}{_format_json(item, level + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + f"\n{indent}]"
    return json.dumps(value)


def _splice_rows(
    base_text: str,
    base: Mapping[str, object],
    *,
    indices: Sequence[int],
    emitted: Sequence[Mapping[str, object]],
) -> str:
    open_position, close_position, spans = _array_spans(base_text, "configs")
    base_rows = base.get("configs", ())
    if len(spans) != len(base_rows):
        raise ValueError("config text and parsed config row counts differ")
    selected = set(indices)
    insert_at = indices[0] if indices else len(spans)
    rendered = [_format_json(dict(row), 2) for row in emitted]
    rows = []
    for index, (start, end) in enumerate(spans):
        if index == insert_at:
            rows.extend(rendered)
        if index not in selected:
            rows.append(base_text[start:end])
    if insert_at == len(spans):
        rows.extend(rendered)
    configs_text = ",\n".join(f"        {row}" for row in rows)
    return (
        base_text[: open_position + 1]
        + f"\n{configs_text}\n    "
        + base_text[close_position:]
    )


def _actual_sets(
    rows: Sequence[Mapping[str, object]],
    dtype: str,
) -> tuple[
    dict[tuple[str, str, object | None], set[str]],
    collections.Counter[str],
]:
    explicit: dict[tuple[str, str, object | None], list[dict[str, object]]] = (
        collections.defaultdict(list)
    )
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        fc, _ = _row_slice(row)
        if not _matches_dtype(row, dtype, fc):
            continue
        activation = (
            "none"
            if fc == "fc2"
            else (
                str(row.get("eltwise_act_type"))
                if str(row.get("eltwise_act_type", "none")).lower() != "none"
                else str(row.get("act", "none"))
            )
        )
        per_token = row.get("use_per_token_sf_b")
        key = (fc, activation.lower(), per_token)
        value = {key: item for key, item in row.items() if key != "_comment"}
        identity = _runtime_identity(value, fc=fc)
        counts[identity] += 1
        explicit[key].append(value)
    actual = {
        key: {_runtime_identity(value, fc=key[0]) for value in _runtime_options(values)}
        for key, values in explicit.items()
    }
    return actual, counts


def _verify_tile_pairing(
    expected: Mapping[tuple[str, str, object | None], set[str]],
) -> None:
    fc2_sets = [values for (fc, _, _), values in expected.items() if fc == "fc2"]
    if not fc2_sets:
        raise ValueError("emitted menu has no FC2 runtime configs")
    fc2_tiles = {
        int(json.loads(value)["tile_n"]) for values in fc2_sets for value in values
    }
    for (fc, activation, per_token), values in expected.items():
        if fc != "fc1":
            continue
        fc1_tiles = {int(json.loads(value)["tile_n"]) for value in values}
        if fc1_tiles != fc2_tiles:
            raise ValueError(
                f"FC1/FC2 tile_n mismatch for {activation}, "
                f"per_token={per_token}: "
                f"fc1_only={sorted(fc1_tiles - fc2_tiles)}, "
                f"fc2_only={sorted(fc2_tiles - fc1_tiles)}"
            )


def _verify_output(
    base: Mapping[str, object],
    output: Mapping[str, object],
    *,
    dtype: str,
    expected: Mapping[tuple[str, str, object | None], set[str]],
) -> dict[tuple[str, str, object | None], set[str]]:
    old_rows = expand_config_rows(base)
    new_rows = expand_config_rows(output)
    actual, explicit_counts = _actual_sets(new_rows, dtype)
    if actual != expected:
        keys = sorted(set(actual) | set(expected), key=str)
        details = []
        for key in keys:
            if actual.get(key, set()) != expected.get(key, set()):
                details.append(
                    f"{key}: expected={len(expected.get(key, set()))}, "
                    f"actual={len(actual.get(key, set()))}"
                )
        raise ValueError("emitted runtime sets differ: " + "; ".join(details))
    duplicates = [key for key, count in explicit_counts.items() if count != 1]
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} emitted expanded configs collapse to duplicates"
        )

    old_other = collections.Counter(
        canonical_json(row)
        for row in old_rows
        if not _matches_dtype(row, dtype, _row_slice(row)[0])
    )
    new_other = collections.Counter(
        canonical_json(row)
        for row in new_rows
        if not _matches_dtype(row, dtype, _row_slice(row)[0])
    )
    if old_other != new_other:
        raise ValueError("non-target expanded options changed")
    _verify_tile_pairing(expected)
    return actual


def _print_pair_summary(
    actual: Mapping[tuple[str, str, object | None], set[str]],
) -> None:
    fc2 = next(values for (fc, _, _), values in actual.items() if fc == "fc2")
    fc2_by_tile = collections.Counter(int(json.loads(value)["tile_n"]) for value in fc2)
    print("Runtime autotune pairs by tile_n:")
    for (fc, activation, per_token), values in sorted(actual.items(), key=str):
        if fc != "fc1":
            continue
        fc1_by_tile = collections.Counter(
            int(json.loads(value)["tile_n"]) for value in values
        )
        label = activation
        if per_token is not None:
            label += f", use_per_token_sf_b={str(bool(per_token)).lower()}"
        total = 0
        rendered = []
        for tile in sorted(set(fc1_by_tile) | set(fc2_by_tile)):
            pairs = fc1_by_tile[tile] * fc2_by_tile[tile]
            total += pairs
            rendered.append(f"N{tile}:{fc1_by_tile[tile]}x{fc2_by_tile[tile]}={pairs}")
        print(f"  {label}: {', '.join(rendered)}; total={total}")


def _atomic_write_text(
    destination: Path,
    content: str,
    *,
    mode_from: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(raw_path)
    try:
        os.fchmod(descriptor, mode_from.stat().st_mode & 0o777)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            descriptor = -1
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def emit_menu_prefix(
    order_path: str | Path,
    *,
    n: int,
    base_path: str | Path,
    out_path: str | Path | None = None,
    dtype: str,
    activations: str = "auto",
    in_place: bool = False,
    dry_run: bool = False,
) -> Path:
    """Compactly replace one dtype slice and verify its exact runtime set."""

    order_file = Path(order_path)
    base_file = Path(base_path)
    if in_place == (out_path is not None):
        raise ValueError("provide exactly one of out_path or in_place=True")
    destination = base_file if in_place else Path(out_path)
    if not in_place and base_file.resolve() == destination.resolve():
        raise ValueError("the emitted menu must not overwrite the base config")
    order = json.loads(order_file.read_text(encoding="utf-8"))
    order_dtype = order.get("inputs", {}).get("target", {}).get("dtype")
    if order_dtype is not None and str(order_dtype) != dtype:
        raise ValueError(
            f"--dtype={dtype} conflicts with selection target dtype={order_dtype}"
        )
    floor_size = int(order.get("floor_size", 0))
    entries = list(order.get("entries", ()))
    if n < floor_size:
        raise ValueError(f"N={n} is below the mandatory floor of {floor_size} entries")
    if n > len(entries):
        raise ValueError(f"N={n} exceeds the selected order length {len(entries)}")
    selected = entries[:n]
    selected_quant_keys: dict[str, tuple[object, ...]] = {}
    for entry in selected:
        fc = str(entry["fc"])
        quant_key = _quant_key(entry["options"])
        prior = selected_quant_keys.setdefault(fc, quant_key)
        if quant_key != prior:
            raise ValueError(f"selected {fc} entries cross quantization boundaries")
        options = dict(entry["options"])
        if not _matches_dtype(options, dtype, fc):
            raise ValueError(f"selected {fc} entry does not match dtype {dtype}")

    base_text = base_file.read_text(encoding="utf-8")
    base = json.loads(base_text)
    indices, base_rows = _target_row_indices(base, dtype)
    if not indices:
        raise ValueError(f"base config has no {dtype} slice")
    base_quant_keys = {
        fc: {_quant_key(row) for row in base_rows if _row_slice(row)[0] == fc}
        for fc in ("fc1", "fc2")
    }
    for fc, quant_key in selected_quant_keys.items():
        if quant_key not in base_quant_keys.get(fc, set()):
            raise ValueError(f"selected {fc} quantization does not match base slice")

    activation_values, per_token = _base_variants(base_rows)
    activation_mode = _resolve_activation_mode(activations, selected)
    geometries = _resolve_auto_fields(
        _selected_geometries(selected),
        mode=activation_mode,
        activations=activation_values,
        per_token=per_token,
        sm_archs=tuple(int(arch) for arch in base.get("sm_archs", ())) or (100,),
    )
    expected = _expected_sets(
        geometries,
        mode=activation_mode,
        activations=activation_values,
        per_token=per_token,
    )
    templates = base.get("templates", {})
    candidates = _template_candidates(base, indices)
    prepared = _prepared_rows(
        geometries,
        mode=activation_mode,
        activations=activation_values,
        per_token=per_token,
        templates=templates,
        template_candidates=candidates,
    )
    emitted = _group_rows(
        prepared,
        dtype=dtype,
        mode=activation_mode,
        templates=templates,
    )
    output_text = _splice_rows(
        base_text,
        base,
        indices=indices,
        emitted=emitted,
    )
    output = json.loads(output_text)
    actual = _verify_output(base, output, dtype=dtype, expected=expected)
    _print_pair_summary(actual)
    if not dry_run:
        _atomic_write_text(destination, output_text, mode_from=base_file)
    return destination
