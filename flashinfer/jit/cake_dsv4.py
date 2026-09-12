"""JIT loader for CAKE-generated DeepSeek V4 sparse MLA kernels."""

from __future__ import annotations

import functools
import json
from pathlib import Path

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm103a_nvcc_flags


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_dsv4"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_dsv4"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "CAKE DSv4 CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


@functools.cache
def _module_metadata() -> dict:
    path = _get_csrc_dir() / "cake_dsv4_modules.json"
    return json.loads(path.read_text(encoding="utf-8"))["variants"]


def get_cake_dsv4_spec(variant: str) -> dict:
    """Return the generated physical build and argument contract."""
    try:
        return _module_metadata()[variant]
    except KeyError as exc:
        raise ValueError(
            f"CAKE DSv4 variant has no generated source contract: {variant}"
        ) from exc


@functools.cache
def gen_cake_dsv4_module(variant: str) -> JitSpec:
    contract = get_cake_dsv4_spec(variant)
    if contract["arch"] != "sm_103a":
        raise ValueError(f"unsupported CAKE DSv4 architecture: {contract['arch']}")
    csrc_dir = _get_csrc_dir()
    sources = [csrc_dir / name for name in contract["sources"]]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "CAKE DSv4 generated sources were not found: "
            + ", ".join(str(path) for path in missing)
        )
    spec = gen_jit_spec(
        name=f"cake_dsv4_{variant}_sm103a",
        sources=sources,
        extra_cuda_cflags=[*sm103a_nvcc_flags, *contract["compile_flags"]],
        # The generated contract owns the fast-math decision.
        use_fast_math=False,
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
        extra_ldflags=["-lcuda"],
    )
    logger.info(f"Generated CAKE DSv4 {variant} JIT spec: {spec.name}")
    return spec


@functools.cache
def get_cake_dsv4_module(variant: str):
    loaded = gen_cake_dsv4_module(variant).build_and_load()
    logger.info(f"Loaded CAKE DSv4 {variant} module")
    return loaded


@functools.cache
def _program_metadata() -> dict:
    path = _get_csrc_dir() / "cake_dsv4_modules.json"
    return json.loads(path.read_text(encoding="utf-8"))["programs"]


def get_cake_dsv4_program_spec(program_id: str) -> dict:
    """Return the generated family signature, routes, and build contract."""
    try:
        return _program_metadata()[program_id]
    except KeyError as exc:
        raise ValueError(
            f"CAKE DSv4 program has no generated source contract: {program_id}"
        ) from exc


@functools.cache
def get_cake_dsv4_program_for_variant(variant: str) -> tuple[str, dict] | None:
    """Find the generated family whose selected route starts with this variant."""
    matches = [
        (program_id, contract)
        for program_id, contract in _program_metadata().items()
        if any(route["stage_variants"][0] == variant for route in contract["routes"])
    ]
    if len(matches) > 1:
        raise ValueError(
            f"CAKE DSv4 variant belongs to multiple generated programs: {variant}"
        )
    return matches[0] if matches else None


@functools.cache
def gen_cake_dsv4_program(program_id: str) -> JitSpec:
    """Describe a family host library linked to separately compiled variants."""
    contract = get_cake_dsv4_program_spec(program_id)
    if contract["arch"] != "sm_103a":
        raise ValueError(f"unsupported CAKE DSv4 architecture: {contract['arch']}")
    csrc_dir = _get_csrc_dir()
    sources = [csrc_dir / name for name in contract["sources"]]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "CAKE DSv4 generated program sources were not found: "
            + ", ".join(str(path) for path in missing)
        )
    dependency_paths = [
        gen_cake_dsv4_module(variant).get_library_path().resolve()
        for variant in contract["dependencies"]
    ]
    # Device translation units retain their individual generated compiler flags.
    # This translation unit contains only the family selector and ordered calls.
    spec = gen_jit_spec(
        name=f"cake_dsv4_{program_id}_sm103a",
        sources=sources,
        extra_cuda_cflags=[*sm103a_nvcc_flags, *contract["compile_flags"]],
        # The generated contract owns the fast-math decision.
        use_fast_math=False,
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
        extra_ldflags=[
            "-Wl,--no-as-needed",
            *(str(path) for path in dependency_paths),
            "-Wl,--as-needed",
            "-lcuda",
            *(
                f"-Wl,-rpath,{path}"
                for path in sorted({p.parent for p in dependency_paths})
            ),
        ],
    )
    logger.info(f"Generated CAKE DSv4 {program_id} program JIT spec: {spec.name}")
    return spec


@functools.cache
def get_cake_dsv4_program(program_id: str):
    """Build variant dependencies, then load the compiled family entry points."""
    for variant in get_cake_dsv4_program_spec(program_id)["dependencies"]:
        get_cake_dsv4_module(variant)
    loaded = gen_cake_dsv4_program(program_id).build_and_load()
    logger.info(f"Loaded CAKE DSv4 {program_id} program")
    return loaded


__all__ = [
    "gen_cake_dsv4_module",
    "get_cake_dsv4_module",
    "get_cake_dsv4_spec",
    "gen_cake_dsv4_program",
    "get_cake_dsv4_program",
    "get_cake_dsv4_program_spec",
    "get_cake_dsv4_program_for_variant",
]
