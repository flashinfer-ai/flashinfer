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
"""JIT loader for the generated Cake fused QK RMSNorm + NeoX RoPE + paged KV append (BF16) kernel.

One source-only manifest lives under ``csrc/cake_fused_qk_rope_append``
(``cake_fused_qk_rope_append_manifest.json``, schema ``cake.library_export.v5``):
for every architecture (``sm_90a`` H100/H200, ``sm_100a`` B200, ``sm_103a`` B300)
one stage per build the Cake host policy can select for the two head
configurations ``(Hq, Hkv) = (8, 1)`` / ``(64, 8)`` (``contract.stages_by_arch``;
the base build ``hq8_hkv1`` / ``hq64_hkv8`` plus lever builds such as
``hq8_hkv1_vsub3_d`` or ``hq64_hkv8_w20_vsub18_d``), each with its tvm-ffi
binding.  :func:`stage_for` mirrors the Cake policy (``contract.stage_policy``)
and picks the stage from the architecture and the problem shape; the host
wrapper lives in :mod:`flashinfer.cake_fused_qk_rope_append`.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm90a_nvcc_flags,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

_PACKAGE = "cake_fused_qk_rope_append"
_GENERATED_ROOT = f"csrc/{_PACKAGE}"
_MANIFEST_NAME = f"{_PACKAGE}_manifest.json"

#: Legacy single-build stage names (one per head configuration).  Since the round-2 export the manifest ships
#: one stage per build the Cake host policy can select (``contract.stages_by_arch``); :func:`stage_for` picks it.
STAGES = ("hq8_hkv1", "hq64_hkv8")
ARCHES = ("sm_90a", "sm_100a", "sm_103a")
HEAD_DIM = 128
_VEC = 8
_CLEAR_UNIT_CHUNKS = 256
#: Fallback for manifests without ``contract.stage_policy`` (mirrors Cake ``POLICY_EXPORT`` of 2026-10-03).
_DEFAULT_POLICY_CONSTANTS = {
    "direct_min_requests": 257,
    "sm100_decode_8_1_min_requests": 128,
    "clear_ctas_per_free_sm": 16,
    "w20": 20,
    "w20_v_sub_base": 18,
    "v_sub_base_8_1": 3,
    "sm100_u2_min_clear_ctas": 2048,
}
_NVCC_FLAGS = {
    "sm_90a": sm90a_nvcc_flags,
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def _base_stage(num_q_heads: int, num_kv_heads: int) -> str:
    if (num_q_heads, num_kv_heads) in ((8, 1), (64, 8)):
        return f"hq{num_q_heads}_hkv{num_kv_heads}"
    raise ValueError(
        f"cake_fused_qk_rope_append supports (num_q_heads, num_kv_heads) in {{(8, 1), (64, 8)}}, "
        f"got ({num_q_heads}, {num_kv_heads})"
    )


def clear_units_per_request(page_size: int, num_kv_heads: int) -> int:
    """4 KiB clearing units of one request's last page (K and V), as the Cake kernel counts them."""
    chunks = page_size * num_kv_heads * HEAD_DIM // _VEC
    return 2 * ((chunks + _CLEAR_UNIT_CHUNKS - 1) // _CLEAR_UNIT_CHUNKS)


def policy_constants() -> dict[str, int]:
    try:
        policy = _manifest().get("contract", {}).get("stage_policy") or {}
        constants = dict(policy.get("constants") or {})
    except (FileNotFoundError, RuntimeError):
        constants = {}
    return {**_DEFAULT_POLICY_CONSTANTS, **constants}


def policy_knobs(
    num_q_heads: int,
    num_kv_heads: int,
    *,
    arch: str,
    num_rows: int,
    num_requests: int,
    page_size: int,
    num_sms: int,
) -> dict[str, int]:
    """Lever selection of the Cake v4 host policy (``policy_knobs`` in the Cake kernel module), mirrored.

    Every entry is a paired-A/B winner on that card family; an empty result is the v3-identical build.
    """
    c = policy_constants()
    hopper = arch.startswith("sm_90")
    cfg = (num_q_heads, num_kv_heads)
    units = clear_units_per_request(page_size, num_kv_heads)
    clear_ctas = (num_requests * units + 3) // 4
    out: dict[str, int] = {}
    w20 = False
    if cfg == (64, 8):
        free_sms = num_sms - num_rows
        w20 = free_sms >= 0 and clear_ctas <= free_sms * c["clear_ctas_per_free_sm"]
        if w20:
            out["WARPS_PER_ROW"] = c["w20"]
            out["V_SUB_BASE"] = c["w20_v_sub_base"]
    decode = num_rows == num_requests
    direct = num_requests >= c["direct_min_requests"]
    if hopper:
        direct = direct or cfg == (8, 1) or w20
    else:
        direct = (
            direct
            or (cfg == (64, 8) and decode)
            or (
                cfg == (8, 1)
                and decode
                and num_requests >= c["sm100_decode_8_1_min_requests"]
            )
        )
    if direct:
        out["L_DIRECT_LOOKUP"] = 1
    if cfg == (8, 1) and (hopper or direct):
        out["V_SUB_BASE"] = c["v_sub_base_8_1"]
    if not hopper and cfg == (64, 8) and clear_ctas >= c["sm100_u2_min_clear_ctas"]:
        out["CLEAR_UNITS_PER_WARP"] = 2
    return out


def stage_name(num_q_heads: int, num_kv_heads: int, knobs: dict[str, int]) -> str:
    """Stage name of one policy outcome (same encoding as the Cake export: ``hq8_hkv1_vsub3_d`` ...)."""
    parts = [_base_stage(num_q_heads, num_kv_heads)]
    if "WARPS_PER_ROW" in knobs:
        parts.append(f"w{knobs['WARPS_PER_ROW']}")
    if "V_SUB_BASE" in knobs:
        parts.append(f"vsub{knobs['V_SUB_BASE']}")
    if knobs.get("L_DIRECT_LOOKUP"):
        parts.append("d")
    if knobs.get("CLEAR_UNITS_PER_WARP", 1) > 1:
        parts.append(f"u{knobs['CLEAR_UNITS_PER_WARP']}")
    return "_".join(parts)


def stages_for(arch: str) -> tuple[str, ...]:
    """Stage names the manifest ships for ``arch``."""
    contract = _manifest().get("contract", {})
    by_arch = contract.get("stages_by_arch")
    if isinstance(by_arch, dict) and by_arch.get(arch):
        return tuple(by_arch[arch])
    return tuple(contract.get("stages") or STAGES)


def stage_for(
    num_q_heads: int,
    num_kv_heads: int,
    *,
    arch: str | None = None,
    num_rows: int | None = None,
    num_requests: int | None = None,
    page_size: int | None = None,
    num_sms: int | None = None,
) -> str:
    """Stage (generated build) for one problem.

    Without the shape arguments this is the head configuration's base build (v3-identical code); with them the
    Cake host policy selects the build, and the result must be one of the stages shipped for ``arch``.
    """
    base = _base_stage(num_q_heads, num_kv_heads)
    if (
        arch is None
        or num_rows is None
        or num_requests is None
        or page_size is None
        or num_sms is None
    ):
        return base
    knobs = policy_knobs(
        num_q_heads,
        num_kv_heads,
        arch=arch,
        num_rows=num_rows,
        num_requests=num_requests,
        page_size=page_size,
        num_sms=num_sms,
    )
    name = stage_name(num_q_heads, num_kv_heads, knobs)
    shipped = stages_for(arch)
    if name not in shipped:
        raise RuntimeError(
            f"the Cake stage policy selected {name!r} for {arch}, which the manifest does not ship ({shipped})"
        )
    return name


def arch_for(compute_capability: tuple[int, int]) -> str:
    major, minor = compute_capability
    if (major, minor) == (9, 0):
        return "sm_90a"
    if (major, minor) == (10, 0):
        return "sm_100a"
    if (major, minor) == (10, 3):
        return "sm_103a"
    raise ValueError(
        f"cake_fused_qk_rope_append has no cubin for compute capability {major}.{minor}; "
        f"supported: sm_90a (9.0), sm_100a (10.0), sm_103a (10.3)"
    )


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / _PACKAGE
    if (installed / _MANIFEST_NAME).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / _PACKAGE
    if (checkout / _MANIFEST_NAME).is_file():
        return checkout
    raise FileNotFoundError("Cake fused QK RoPE append sources were not found")


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


@functools.cache
def _manifest() -> dict[str, Any]:
    path = _get_csrc_dir() / _MANIFEST_NAME
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "cake.library_export.v5"
        or value.get("producer") != "cake"
        or value.get("library") != "flashinfer"
        or value.get("name") != _PACKAGE
    ):
        raise RuntimeError("invalid Cake fused QK RoPE append manifest")
    modules = value.get("modules")
    if not isinstance(modules, list) or not modules:
        raise RuntimeError("empty Cake fused QK RoPE append module inventory")
    return value


def _record(stage: str, arch: str) -> dict[str, Any]:
    if arch not in ARCHES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHES}")
    shipped = stages_for(arch)
    if stage not in shipped:
        raise ValueError(
            f"unknown stage {stage!r} for {arch}; expected one of {shipped}"
        )
    matches = [
        item
        for item in _manifest()["modules"]
        if item.get("arch") == arch
        and dict(item.get("route", {})).get("stage") == stage
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one generated Cake module for ({arch}, {stage}), got {len(matches)}"
        )
    return matches[0]


def _source_path(relative_path: str) -> Path:
    manifest = _manifest()
    relative = Path(relative_path)
    try:
        suffix = relative.relative_to(Path(_GENERATED_ROOT))
    except ValueError as exc:
        raise RuntimeError(
            f"generated source escaped {_GENERATED_ROOT}: {relative_path}"
        ) from exc
    path = _get_csrc_dir() / suffix
    inventory = {item["path"]: item for item in manifest["files"]}
    receipt = inventory.get(relative.as_posix())
    if not isinstance(receipt, dict) or not path.is_file():
        raise FileNotFoundError(
            f"generated source is absent from the manifest: {relative_path}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != receipt.get("sha256"):
        raise RuntimeError(f"generated source hash mismatch: {relative_path}")
    return path


def cake_fused_qk_rope_append_record(stage: str, arch: str) -> dict[str, Any]:
    """Manifest record of one generated module (arg plan, launch block, specialization)."""
    return dict(_record(stage, arch))


@functools.cache
def gen_cake_fused_qk_rope_append_module(stage: str, arch: str) -> JitSpec:
    record = _record(stage, arch)
    units = record["translation_units"]
    spec = gen_jit_spec(
        name=f"{record['name']}_{arch}",
        sources=[_source_path(units["device"]), _source_path(units["binding"])],
        extra_cuda_cflags=[*_NVCC_FLAGS[arch], *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[_get_csrc_dir().parent, _get_include_dir()],
    )
    logger.info("Generated Cake fused QK RoPE append JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fused_qk_rope_append_module(stage: str, arch: str):
    """Build and load one module; return ``(module, manifest record)``."""
    spec = gen_cake_fused_qk_rope_append_module(stage, arch)
    module = spec.build_and_load()
    return module, _record(stage, arch)


def build_all_cake_fused_qk_rope_append_modules(arch: str) -> dict[str, str]:
    built = {}
    for stage in stages_for(arch):
        spec = gen_cake_fused_qk_rope_append_module(stage, arch)
        module = spec.build_and_load()
        record = _record(stage, arch)
        if not hasattr(module, str(record["ffi_entry"])):
            raise RuntimeError(f"built module {record['name']} lacks its FFI entry")
        built[stage] = spec.name
    return built
