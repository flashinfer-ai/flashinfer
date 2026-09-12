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

"""JIT loader for generated Blackwell NVFP4 SVDQuant kernels."""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
from pathlib import Path
from typing import Literal

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags

CakeNvfp4SvdquantTarget = Literal["sm100a", "sm103a"]
_TARGET_FLAGS = {"sm100a": sm100a_nvcc_flags, "sm103a": sm103a_nvcc_flags}
_CATALOG_MODULE = {
    "sm100a": ".cake_nvfp4_svdquant_sm100a_catalog",
    "sm103a": ".cake_nvfp4_svdquant_sm103a_catalog",
}
_UINT16_SCALE_TEMPLATES = {
    "tactic14_m128n64k128_cluster1x2",
    "tactic18_2sm_m256n128k256",
}
_BLOCK_K = {
    "tactic0_n128_exact_epilogue": 128,
    "tactic25_k12288_r32": 256,
    "tactic25_rank96_compact_tail": 256,
    "tactic0_rank64_bias_smem": 128,
    "tactic14_m128n64k128_cluster1x2": 128,
    "tactic18_2sm_m256n128k256": 256,
    "tactic25_rank96_persistent": 256,
}


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_nvfp4_svdquant_gemm"
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_nvfp4_svdquant_gemm"
    for candidate in (installed, checkout):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cake NVFP4 SVDQuant sources were not found. Checked:\n"
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


def cake_nvfp4_svdquant_target(device: torch.device) -> CakeNvfp4SvdquantTarget:
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (10, 0):
        return "sm100a"
    if (major, minor) == (10, 3):
        return "sm103a"
    raise RuntimeError(
        "the Cake NVFP4 SVDQuant backend requires exact compute capability "
        f"10.0 or 10.3, got {major}.{minor}"
    )


@functools.cache
def _catalog(target: CakeNvfp4SvdquantTarget):
    return importlib.import_module(_CATALOG_MODULE[target], __package__)


def _module_key(name: str, role: str) -> str:
    return f"{name}:{role}"


def _record(name: str, role: str, target: CakeNvfp4SvdquantTarget) -> dict:
    record = _catalog(target).MODULES.get(_module_key(name, role))
    if not isinstance(record, dict) or record.get("arch") != f"sm_{target[2:-1]}a":
        raise RuntimeError(f"missing generated module {name}:{role} for {target}")
    return record


def _verified_source(root: Path, relative: str, expected: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"generated Cake source not found: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"generated Cake source hash mismatch for {relative}")
    return path


@functools.cache
def gen_cake_nvfp4_svdquant_module(
    name: str,
    role: str,
    target: CakeNvfp4SvdquantTarget,
) -> JitSpec:
    record = _record(name, role, target)
    root = _get_csrc_dir()
    device = _verified_source(root, record["device"], record["device_sha256"])
    binding = _verified_source(root, record["binding"], record["binding_sha256"])
    identity = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    spec = gen_jit_spec(
        name=f"cake_nvfp4_svdquant_{target}_{identity}",
        sources=[device, binding],
        extra_cuda_cflags=[*_TARGET_FLAGS[target], *record["compile_flags"]],
        extra_include_paths=[root, root.parent, _get_include_dir()],
        needs_device_linking=True,
    )
    logger.info("Generated Cake NVFP4 SVDQuant JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_nvfp4_svdquant_module(
    name: str,
    role: str,
    target: CakeNvfp4SvdquantTarget,
):
    record = _record(name, role, target)
    module = gen_cake_nvfp4_svdquant_module(name, role, target).build_and_load()
    getattr(module, record["ffi_entry"])
    return module


def _artifact_identity(name: str, role: str, target: CakeNvfp4SvdquantTarget) -> dict:
    record = _record(name, role, target)
    spec = gen_cake_nvfp4_svdquant_module(name, role, target)
    load_cake_nvfp4_svdquant_module(name, role, target)
    return {
        "jit_spec": spec.name,
        "loaded_binary_sha256": hashlib.sha256(
            spec.get_library_path().read_bytes()
        ).hexdigest(),
        "ffi_entry": record["ffi_entry"],
    }


def build_all_cake_nvfp4_svdquant_modules(
    target: CakeNvfp4SvdquantTarget | None = None,
) -> dict[str, dict]:
    if target is None:
        target = cake_nvfp4_svdquant_target(torch.device("cuda"))
    result = {}
    for key, record in sorted(_catalog(target).MODULES.items()):
        result[f"module:{key}"] = _artifact_identity(
            record["name"], record["role"], target
        )
    return result


def cake_nvfp4_svdquant_dispatcher_identity(
    target: CakeNvfp4SvdquantTarget | None = None,
) -> dict:
    if target is None:
        target = cake_nvfp4_svdquant_target(torch.device("cuda"))
    root = Path(__file__).resolve().parents[1]
    paths = {
        "flashinfer/jit/cake_nvfp4_svdquant.py": Path(__file__).resolve(),
        f"flashinfer/jit/cake_nvfp4_svdquant_{target}_catalog.py": Path(
            _catalog(target).__file__
        ).resolve(),
        "flashinfer/gemm/gemm_svdquant.py": root / "gemm" / "gemm_svdquant.py",
    }
    return {
        "python_sources": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(paths.items())
        }
    }


def _desired_template(m: int, n: int, k: int, rank: int, has_bias: bool) -> str:
    key = (m, n, k, rank, has_bias)
    if key == (129, 3072, 3072, 32, False):
        return "tactic14_m128n64k128_cluster1x2"
    if key == (129, 3072, 12288, 32, False):
        return "tactic18_2sm_m256n128k256"
    if key == (129, 3072, 3072, 96, True):
        return "tactic25_rank96_persistent"
    if rank == 96:
        return "tactic25_rank96_compact_tail"
    if rank == 64 and has_bias:
        return "tactic0_rank64_bias_smem"
    if k == 12288 and rank == 32:
        return "tactic25_k12288_r32"
    return "tactic0_n128_exact_epilogue"


def select_cake_nvfp4_svdquant_route(
    *,
    m: int,
    n: int,
    k: int,
    rank: int,
    has_bias: bool,
    device: torch.device,
) -> dict:
    target = cake_nvfp4_svdquant_target(device)
    template = _desired_template(m, n, k, rank, has_bias)
    matches = []
    for route in _catalog(target).ROUTES.values():
        args = route["args"]
        if (args["M"], args["N"], args["K"], args["rank"], bool(args["bias"])) == (
            m,
            n,
            k,
            rank,
            has_bias,
        ) and route["stage"]["template"] == template:
            matches.append(route)
    identities = {
        (
            item["stage"]["module"]["name"],
            item["stage"]["module"]["role"],
            item["stage"]["pdl_enabled"],
        )
        for item in matches
    }
    if len(identities) != 1:
        raise ValueError(
            "no unique generated Cake NVFP4 SVDQuant route for "
            f"M={m}, N={n}, K={k}, rank={rank}, bias={has_bias}"
        )
    return matches[0]


def is_cake_nvfp4_svdquant_problem_supported(
    *,
    m: int,
    n: int,
    k: int,
    rank: int,
    has_bias: bool,
    device: torch.device,
) -> bool:
    try:
        select_cake_nvfp4_svdquant_route(
            m=m, n=n, k=k, rank=rank, has_bias=has_bias, device=device
        )
        return True
    except (FileNotFoundError, ImportError, RuntimeError, ValueError):
        return False


def cake_nvfp4_svdquant_workspace_size(
    *,
    m: int,
    n: int,
    k: int,
    rank: int,
    has_bias: bool,
    device: torch.device,
) -> int:
    return int(
        select_cake_nvfp4_svdquant_route(
            m=m, n=n, k=k, rank=rank, has_bias=has_bias, device=device
        )["stage"]["tma_workspace_bytes"]
    )


def _launch_route(
    route: dict,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    workspace: torch.Tensor | None,
    enable_pdl: bool | None,
) -> torch.Tensor:
    target = cake_nvfp4_svdquant_target(a.device)
    stage = route["stage"]
    module_ref = stage["module"]
    record = _record(module_ref["name"], module_ref["role"], target)
    baked_pdl = bool(stage["pdl_enabled"])
    if enable_pdl is not None and enable_pdl is not baked_pdl:
        raise ValueError(
            f"selected Cake route requires enable_pdl={baked_pdl}, got {enable_pdl}"
        )

    m, k_packed = (int(value) for value in a.shape)
    n = int(b.shape[0])
    k = k_packed * 2
    rank = int(d.shape[1])
    padded_m = ((m + 127) // 128) * 128
    padded_n = ((n + 127) // 128) * 128
    sf_quads = (k // 16) // 4
    # The public API permits larger contiguous backing buffers.
    a_sf = a_sf.reshape(-1)[: padded_m * (k // 16)]
    b_sf = b_sf.reshape(-1)[: padded_n * (k // 16)]
    template = stage["template"]
    if template in _UINT16_SCALE_TEMPLATES:
        a_sf_view = a_sf.view(torch.uint16).view(padded_m // 128, sf_quads, 4, 64)
        b_sf_view = b_sf.view(torch.uint16).view(padded_n // 128, sf_quads, 4, 64)
    else:
        a_sf_view = a_sf.view(padded_m // 128, sf_quads, 4, 8, 16)
        b_sf_view = b_sf.view(padded_n // 128, sf_quads, 4, 8, 16)
    block_n = 64 if template == "tactic14_m128n64k128_cluster1x2" else 128
    grid_m = padded_m // 128
    grid_n = n // block_n
    raster_along_m = grid_n > grid_m
    if template == "tactic14_m128n64k128_cluster1x2":
        grid = (grid_m, grid_n, 1)
    elif template == "tactic18_2sm_m256n128k256":
        grid = (2, grid_n, 1)
    elif template == "tactic25_rank96_persistent":
        grid = (grid_m, grid_n, 1) if raster_along_m else (grid_n, grid_m, 1)
    else:
        grid = (grid_m * grid_n, 1, 1)

    tensors = {
        "A": a,
        "B": b,
        "SFA": a_sf_view,
        "SFB": b_sf_view,
        "D": d,
        "L1": l1,
        "D_tail": d,
        "L1_tail": l1,
        "alpha": alpha,
        "bias": bias if bias is not None else out,
        "out": out,
    }
    params = {
        "M": m,
        "N": n,
        "K_tiles": k // _BLOCK_K[template],
        "rank_tiles": rank // 32,
        "rank": rank,
        "rank_chunks": (rank + 63) // 64,
        "grid_n": grid_n,
        "raster_along_m": 1 if raster_along_m else 0,
        "has_bias": 1 if bias is not None else 0,
    }
    workspace_bytes = int(stage["tma_workspace_bytes"])
    if workspace_bytes:
        if workspace is None or workspace.dtype != torch.uint8 or not workspace.is_cuda:
            raise ValueError(
                "selected Cake route requires caller-owned CUDA uint8 workspace"
            )
        if workspace.numel() < workspace_bytes or workspace.data_ptr() % 128:
            raise ValueError(
                f"Cake descriptor workspace requires {workspace_bytes} bytes and 128-byte alignment"
            )
    values = []
    for kind, key in record["arg_plan"]:
        if kind in {"buffer", "tma_buffer"}:
            values.append(tensors[key])
        elif kind == "parameter":
            values.append(params[key])
        elif kind == "workspace" and key == "tma_descriptor_workspace":
            values.append(workspace)
        elif kind == "grid":
            values.append(grid[{"grid_x": 0, "grid_y": 1, "grid_z": 2}[key]])
        else:
            raise RuntimeError(f"unsupported generated argument {kind}:{key}")
    module = load_cake_nvfp4_svdquant_module(
        module_ref["name"], module_ref["role"], target
    )
    getattr(module, record["ffi_entry"])(*values)
    return out


def launch_cake_nvfp4_svdquant_shape(
    shape_name: str,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    workspace: torch.Tensor | None,
) -> torch.Tensor:
    target = cake_nvfp4_svdquant_target(a.device)
    route = _catalog(target).ROUTES.get(shape_name)
    if route is None:
        raise ValueError(f"unknown generated Cake NVFP4 SVDQuant shape {shape_name!r}")
    return _launch_route(
        route,
        a=a,
        b=b,
        a_sf=a_sf,
        b_sf=b_sf,
        alpha=alpha,
        d=d,
        l1=l1,
        bias=bias,
        out=out,
        workspace=workspace,
        enable_pdl=None,
    )


def run_cake_nvfp4_svdquant(
    *,
    backend: Literal["cake"],
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    workspace: torch.Tensor | None,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    if backend != "cake":
        raise ValueError("run_cake_nvfp4_svdquant only accepts the cake backend")
    route = select_cake_nvfp4_svdquant_route(
        m=int(a.shape[0]),
        n=int(b.shape[0]),
        k=int(a.shape[1]) * 2,
        rank=int(d.shape[1]),
        has_bias=bias is not None,
        device=a.device,
    )
    return _launch_route(
        route,
        a=a,
        b=b,
        a_sf=a_sf,
        b_sf=b_sf,
        alpha=alpha,
        d=d,
        l1=l1,
        bias=bias,
        out=out,
        workspace=workspace,
        enable_pdl=enable_pdl,
    )


__all__ = [
    "CakeNvfp4SvdquantTarget",
    "build_all_cake_nvfp4_svdquant_modules",
    "cake_nvfp4_svdquant_dispatcher_identity",
    "cake_nvfp4_svdquant_target",
    "cake_nvfp4_svdquant_workspace_size",
    "gen_cake_nvfp4_svdquant_module",
    "is_cake_nvfp4_svdquant_problem_supported",
    "launch_cake_nvfp4_svdquant_shape",
    "load_cake_nvfp4_svdquant_module",
    "run_cake_nvfp4_svdquant",
    "select_cake_nvfp4_svdquant_route",
]
