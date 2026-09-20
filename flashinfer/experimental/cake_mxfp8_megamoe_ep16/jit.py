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

# JIT loader for the generated Cake MXFP8 MegaMoE EP16 route.

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from ...jit import env as jit_env
from ...jit.core import JitSpec, gen_jit_spec, logger, sm103a_nvcc_flags

_OPERATOR_DIR = "cake_mxfp8_megamoe_ep16"
_MANIFEST = "cake_mxfp8_megamoe_ep16_manifest.json"
_POLICIES = {
    f"{family}_{protocol}": (tile_n, protocol)
    for family, tile_n in (("mixed", "mixed"), ("n16", 16), ("n32", 32))
    for protocol in ("cta0", "all_cta")
}


def _get_csrc_root() -> Path:
    package_sources = Path(__file__).resolve().parent / "csrc"
    if (package_sources / _OPERATOR_DIR / _MANIFEST).is_file():
        return package_sources

    raise FileNotFoundError(
        "Cake MXFP8 MegaMoE sources were not found in the installed package or source checkout"
    )


def _get_flashinfer_header_dirs() -> list[Path]:
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer" / "layout.cuh"
    ).is_file():
        return installed

    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer" / "layout.cuh"
    ).is_file():
        return source

    raise FileNotFoundError("FlashInfer JIT headers were not found")


def _read_manifest(csrc_root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    csrc_root = _get_csrc_root() if csrc_root is None else Path(csrc_root)
    path = csrc_root / _OPERATOR_DIR / _MANIFEST
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "flashinfer.experimental.source_closure.v2":
        raise RuntimeError("Cake MXFP8 MegaMoE manifest has an unexpected schema")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or len(sequences) != len(_POLICIES):
        raise RuntimeError(
            "Cake MXFP8 MegaMoE manifest must contain six policy sequences"
        )
    if any(not isinstance(sequence, dict) for sequence in sequences):
        raise RuntimeError("Cake MXFP8 MegaMoE policy sequences must be mappings")
    ids = [sequence.get("policy_id") for sequence in sequences]
    if any(not isinstance(policy_id, str) for policy_id in ids) or set(ids) != set(
        _POLICIES
    ):
        raise RuntimeError(
            "Cake MXFP8 MegaMoE manifest has missing or duplicate policies"
        )
    reducers = set()
    devices = set()
    bindings = set()
    for sequence in sequences:
        _validate_sequence(csrc_root, sequence)
        units = sequence["translation_units"]
        devices.update(units["devices"])
        reducers.add(
            (
                units["devices"][1],
                sequence["closure"][1]["sha256"],
                sequence["reducer_symbol"],
            )
        )
        bindings.add(units["binding"])
    if len(reducers) != 1 or len(devices) != 7 or len(bindings) != 6:
        raise RuntimeError(
            "Cake MXFP8 MegaMoE policies must share exactly one ordered reducer"
        )
    return csrc_root, manifest


def _validate_sequence(csrc_root: Path, sequence: dict[str, Any]) -> None:
    tile_n, protocol = _POLICIES[sequence["policy_id"]]
    if (
        sequence.get("arch") != "sm_103a"
        or sequence.get("ffi_entry") != "run"
        or sequence.get("setup_ffi_entry") != "setup_tma"
        or sequence.get("tile_n") != tile_n
        or sequence.get("return_protocol") != protocol
        or sequence.get("row_segments")
        != ([32, 16, 16] if tile_n == "mixed" else [tile_n] * (64 // tile_n))
        or sequence.get("runtime_tokens_per_rank") != [1, 64]
        or sequence.get("launches_per_call") != 2
        or sequence.get("setup_launches") != 1
        or sequence.get("tma_descriptor_count") != 8
        or sequence.get("dynamic_smem_bytes") != (72704 if tile_n == 16 else 76800)
        or sequence.get("compile_flags") != ["--use_fast_math"]
    ):
        raise RuntimeError("Cake MXFP8 MegaMoE manifest has an unexpected ABI")

    closure = sequence.get("closure")
    if not isinstance(closure, list) or not closure:
        raise RuntimeError("Cake MXFP8 MegaMoE manifest has an invalid source closure")
    closure_paths: list[str] = []
    for artifact in closure:
        if not isinstance(artifact, dict):
            raise RuntimeError(
                "Cake MXFP8 MegaMoE source closure entries must be mappings"
            )
        raw_path = artifact.get("path")
        expected_digest = artifact.get("sha256")
        if not isinstance(raw_path, str):
            raise RuntimeError(
                "Cake MXFP8 MegaMoE source closure path must be a string"
            )
        relative = PurePosixPath(raw_path)
        if (
            relative.is_absolute()
            or relative.parts[:3] != ("csrc", _OPERATOR_DIR, "sm_103a")
            or len(relative.parts) != 4
            or relative.suffix != ".cu"
            or ".." in relative.parts
        ):
            raise RuntimeError(f"invalid generated source path: {relative}")
        if raw_path in closure_paths:
            raise RuntimeError(f"duplicate generated source path: {relative}")
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise RuntimeError(f"invalid generated source digest: {relative}")
        closure_paths.append(raw_path)

    translation_units = sequence.get("translation_units")
    if not isinstance(translation_units, dict):
        raise RuntimeError("Cake MXFP8 MegaMoE manifest has invalid translation units")
    devices = translation_units.get("devices")
    binding = translation_units.get("binding")
    if (
        not isinstance(devices, list)
        or len(devices) != 2
        or not all(isinstance(path, str) for path in devices)
        or not isinstance(binding, str)
        or closure_paths != [*devices, binding]
    ):
        raise RuntimeError(
            "Cake MXFP8 MegaMoE translation units must exactly equal the source closure"
        )

    expected_identity = sequence.get("closure_sha256")
    identity_input = {
        key: value for key, value in sequence.items() if key != "closure_sha256"
    }
    actual_identity = hashlib.sha256(
        json.dumps(identity_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_identity != actual_identity:
        raise RuntimeError(
            "Cake MXFP8 MegaMoE aggregate source-closure identity mismatch"
        )

    for artifact in closure:
        relative = PurePosixPath(artifact["path"])
        source = csrc_root.joinpath(*relative.parts[1:])
        if not source.is_file():
            raise FileNotFoundError(f"generated source not found: {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != artifact["sha256"]:
            raise RuntimeError(f"generated source digest mismatch: {source}")


@functools.cache
def gen_cake_mxfp8_megamoe_ep16_module(policy_id: str = "mixed_cta0") -> JitSpec:
    """Create the exact-SM103a JIT specification from the sealed closure."""

    csrc_root, manifest = _read_manifest()
    if policy_id not in _POLICIES:
        raise ValueError(f"unsupported Cake MXFP8 MegaMoE policy: {policy_id!r}")
    sequence = next(
        sequence
        for sequence in manifest["sequences"]
        if sequence["policy_id"] == policy_id
    )
    translation_units = sequence["translation_units"]
    relative_sources = [
        *translation_units["devices"],
        translation_units["binding"],
    ]
    sources = [csrc_root.joinpath(*Path(path).parts[1:]) for path in relative_sources]
    identity = sequence["closure_sha256"][:20]
    name = f"cake_mxfp8_megamoe_ep16_sm103a_{identity}"
    operator_dir = csrc_root / _OPERATOR_DIR
    spec = gen_jit_spec(
        name=name,
        sources=sources,
        extra_cuda_cflags=[*sm103a_nvcc_flags, "--use_fast_math"],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            csrc_root,
            operator_dir,
            operator_dir / "sm_103a",
            *_get_flashinfer_header_dirs(),
        ],
    )
    logger.info(f"Generated Cake MXFP8 MegaMoE JIT spec: {spec.name}")
    return spec


def _check_exact_sm103a(device: Any = None) -> None:
    import torch

    from ...utils import get_compute_capability

    resolved = torch.device("cuda") if device is None else torch.device(device)
    capability = get_compute_capability(resolved)
    if capability != (10, 3):
        raise RuntimeError(
            f"Cake MXFP8 MegaMoE EP16 requires compute capability 10.3, got {capability[0]}.{capability[1]}"
        )


@functools.cache
def _build_and_load(policy_id: str) -> Any:
    module = gen_cake_mxfp8_megamoe_ep16_module(policy_id).build_and_load()
    logger.info("Loaded Cake MXFP8 MegaMoE EP16 module")
    return module


def load_cake_mxfp8_megamoe_ep16_module(
    *, device: Any = None, policy_id: str = "mixed_cta0"
) -> Any:
    """Build or load the generated module for an exact SM103a device."""

    _check_exact_sm103a(device)
    return _build_and_load(policy_id)


__all__ = [
    "gen_cake_mxfp8_megamoe_ep16_module",
    "load_cake_mxfp8_megamoe_ep16_module",
]
