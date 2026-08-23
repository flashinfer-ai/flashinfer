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
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch

from .api_logging import flashinfer_api
from .trace.templates.attention import (
    block_sparse_attention_run_trace,
    cake_vsa_plan_trace,
)


_MAX_COMPACT_BLOCKS = 64
_MAX_DIRECT_TOPK = 32
_MAX_LONGSEQ_BLOCKS = 192
_EXPECTED_PROFILES = {
    "blk128_compact",
    "blk64_persistent",
    "longseq",
    "ultrasparse_bsr",
    "gqa_mask",
    "head64_native",
    "head96_native",
    "blk128_fp16_compact",
    "fp16_direct",
}


def _source_dir() -> Path:
    from .jit import env as jit_env

    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_vsa"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[1] / "csrc" / "cake_vsa"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("Cake VSA source export is not installed")


@functools.cache
def _manifest() -> dict[str, Any]:
    path = _source_dir() / "cake_vsa_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "cake-vsa-block-sparse-source-export-v2":
        raise RuntimeError("unsupported Cake VSA source manifest")
    source_records = _manifest_source_records(manifest)
    content_identity = hashlib.sha256(
        "".join(
            f"{source_path}\0{digest}\n"
            for source_path, digest in sorted(source_records)
        ).encode("utf-8")
    ).hexdigest()
    if manifest.get("export_content_sha256") != content_identity:
        raise RuntimeError("Cake VSA source manifest content identity mismatch")
    return manifest


def _manifest_source_records(
    manifest: dict[str, Any],
) -> list[tuple[str, str]]:
    architectures = manifest.get("architectures")
    profiles = manifest.get("profiles")
    if (
        architectures != ["sm_100a", "sm_103a"]
        or not isinstance(profiles, list)
        or not profiles
    ):
        raise RuntimeError("invalid Cake VSA source manifest inventory")

    records: list[tuple[str, str]] = []
    profile_names: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict) or not isinstance(profile.get("profile"), str):
            raise RuntimeError("invalid Cake VSA profile record")
        profile_name = profile["profile"]
        if profile_name in profile_names:
            raise RuntimeError(f"duplicate Cake VSA profile: {profile_name}")
        profile_names.add(profile_name)
        host = profile.get("host")
        devices = profile.get("device")
        if not isinstance(host, dict) or not isinstance(devices, dict):
            raise RuntimeError(f"invalid Cake VSA source records for {profile_name}")
        if set(devices) != set(architectures):
            raise RuntimeError(
                f"Cake VSA profile {profile_name} does not cover every architecture"
            )
        for source in (host, *(devices[arch] for arch in architectures)):
            if not isinstance(source, dict):
                raise RuntimeError(f"invalid Cake VSA source record for {profile_name}")
            source_path = source.get("path")
            digest = source.get("sha256")
            size_bytes = source.get("size_bytes")
            if (
                not isinstance(source_path, str)
                or not source_path
                or "\0" in source_path
                or "\n" in source_path
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not isinstance(size_bytes, int)
                or size_bytes < 0
            ):
                raise RuntimeError(
                    f"invalid Cake VSA source identity for {profile_name}"
                )
            records.append((source_path, digest))
    if len({source_path for source_path, _ in records}) != len(records):
        raise RuntimeError("duplicate Cake VSA source path in manifest")
    if profile_names != _EXPECTED_PROFILES or len(records) != 27:
        raise RuntimeError("incomplete Cake VSA source manifest inventory")
    return records


def _profile_record(profile: str) -> dict[str, Any]:
    for record in _manifest()["profiles"]:
        if record["profile"] == profile:
            return record
    raise ValueError(f"unknown Cake VSA profile: {profile}")


def _arch_for_device(device: torch.device) -> str:
    properties = torch.cuda.get_device_properties(device)
    cc = (properties.major, properties.minor)
    if cc == (10, 0):
        return "sm_100a"
    if cc == (10, 3):
        return "sm_103a"
    raise RuntimeError(
        "Cake VSA requires an SM100 or SM103 GPU, "
        f"got compute capability {properties.major}.{properties.minor}"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nvcc() -> Path:
    executable = shutil.which("nvcc")
    if executable is None:
        raise RuntimeError("nvcc is required to build the Cake VSA source export")
    return Path(executable).resolve()


@functools.cache
def _load_module(profile: str, arch: str):
    from tvm_ffi import cpp

    from .jit import env as jit_env

    record = _profile_record(profile)
    device_record = record["device"][arch]
    root = _source_dir()
    device_source = root / device_record["path"]
    host_source = root / record["host"]["path"]
    if _sha256(device_source) != device_record["sha256"]:
        raise RuntimeError(f"Cake VSA device source hash mismatch: {device_source}")
    if _sha256(host_source) != record["host"]["sha256"]:
        raise RuntimeError(f"Cake VSA host source hash mismatch: {host_source}")

    identity = hashlib.sha256(
        (device_record["sha256"] + record["host"]["sha256"] + arch).encode("ascii")
    ).hexdigest()[:16]
    module_name = f"cake_vsa_{profile}_{arch}_{identity}"
    build_dir = jit_env.FLASHINFER_JIT_DIR / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = build_dir / f"{module_name}.cubin"
    if not cubin_path.exists():
        nvcc = _nvcc()
        with tempfile.NamedTemporaryFile(
            dir=build_dir, prefix=f".{module_name}.", suffix=".cubin", delete=False
        ) as handle:
            temporary = Path(handle.name)
        command = [
            str(nvcc),
            "-cubin",
            "--std=c++17",
            "--use_fast_math",
            f"-arch={arch}",
            str(device_source),
            "-o",
            str(temporary),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Cake VSA CUDA compilation failed for {profile}/{arch}:\n{result.stderr}"
            )
        temporary.replace(cubin_path)

    cuda_include = _nvcc().parent.parent / "include"
    return cpp.load_inline(
        module_name,
        cpp_sources=host_source.read_text(encoding="utf-8"),
        embed_cubin={record["module_ident"]: cubin_path.read_bytes()},
        extra_include_paths=[str(cuda_include)],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
    )


def _dense_mask(
    indptr: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    block_mask: Optional[torch.Tensor],
    *,
    mb: int,
    nb: int,
    num_qo_heads: int,
    num_kv_heads: int,
    device: torch.device,
) -> torch.Tensor:
    if block_mask is not None:
        source = block_mask.to(device=device, dtype=torch.bool).contiguous()
        if tuple(source.shape) == (num_kv_heads, mb, nb):
            source = source.repeat_interleave(
                num_qo_heads // num_kv_heads, dim=0
            ).contiguous()
        if tuple(source.shape) != (num_qo_heads, mb, nb):
            raise ValueError(
                "block_mask must have shape [num_qo_heads, MB, NB] or "
                "[num_kv_heads, MB, NB]"
            )
        return source
    if indptr is None or indices is None:
        raise ValueError("Cake VSA requires block_mask or BSR indptr/indices")
    row_offsets = indptr.to(device="cpu", dtype=torch.int64).tolist()
    columns = indices.to(device="cpu", dtype=torch.int64).tolist()
    if len(row_offsets) != mb + 1:
        raise ValueError("indptr must have MB + 1 entries")
    shared = torch.zeros((mb, nb), dtype=torch.bool, device=device)
    for row in range(mb):
        selected = columns[row_offsets[row] : row_offsets[row + 1]]
        if not selected:
            raise ValueError("every Cake VSA block row must select at least one block")
        if min(selected) < 0 or max(selected) >= nb:
            raise ValueError("BSR column index is out of range")
        shared[row, torch.tensor(selected, device=device)] = True
    return shared.unsqueeze(0).expand(num_qo_heads, -1, -1).contiguous()


def _shared_bsr(
    dense: torch.Tensor,
    indptr: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    *,
    trust_bsr: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.equal(dense, dense[:1].expand_as(dense)):
        raise ValueError(
            "Cake blk128 routes require one shared BSR pattern across heads"
        )
    shared = dense[0]
    counts = shared.sum(dim=1, dtype=torch.int32)
    ptr = torch.cat(
        [
            torch.zeros((1,), dtype=torch.int32, device=dense.device),
            counts.cumsum(0, dtype=torch.int32),
        ]
    )
    if trust_bsr and indptr is not None and indices is not None:
        raw_ptr = indptr.to(device=dense.device, dtype=torch.int32).contiguous()
        raw_cols = indices.to(device=dense.device, dtype=torch.int32).contiguous()
        if raw_cols.numel() == int(ptr[-1].item()) and torch.equal(raw_ptr, ptr):
            return raw_ptr, raw_cols

        # The ultrasparse kernel consumes six columns per row with a fixed
        # stride rather than consulting indptr. Canonicalize duplicate or
        # otherwise non-packed BSR rows before making that metadata launchable.
    cols = shared.nonzero(as_tuple=False)[:, 1].to(torch.int32).contiguous()
    return ptr, cols


@flashinfer_api(trace=cake_vsa_plan_trace)
def plan_cake_vsa(
    indptr: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    block_mask: Optional[torch.Tensor],
    kv_block_lens: Optional[torch.Tensor],
    q2k_indices: Optional[torch.Tensor],
    q2k_num: Optional[torch.Tensor],
    *,
    M: int,
    N: int,
    R: int,
    C: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_data_type: torch.dtype,
    sm_scale: Optional[float],
    device: torch.device,
) -> dict[str, Any]:
    """Create stable metadata and workspaces for the source-level backend.

    Parameters
    ----------
    indptr : Optional[torch.Tensor]
        CSR-style row pointers for a block pattern shared by all heads, with
        shape ``(M // R + 1,)``. Required with ``indices`` when neither
        ``block_mask`` nor ``q2k_indices`` is supplied.
    indices : Optional[torch.Tensor]
        Column indices corresponding to ``indptr``. Duplicate or non-packed
        rows are canonicalized before a fixed-stride source kernel can use
        them.
    block_mask : Optional[torch.Tensor]
        Boolean block mask with shape ``(num_qo_heads, M // R, N // C)`` or
        ``(num_kv_heads, M // R, N // C)``.
    kv_block_lens : Optional[torch.Tensor]
        Valid-token count for each KV block, with shape ``(N // C,)``. This is
        supported only for 64-token blocks.
    q2k_indices : Optional[torch.Tensor]
        Direct block-64 selections as contiguous int32 metadata with shape
        ``(num_qo_heads, M // R, topk)``.
    q2k_num : Optional[torch.Tensor]
        Number of active ``q2k_indices`` entries per row, as contiguous int32
        metadata with shape ``(num_qo_heads, M // R)``.
    M : int
        Query sequence length.
    N : int
        Key/value sequence length.
    R : int
        Query block size. Cake supports 64 and 128.
    C : int
        Key/value block size, which must equal ``R``.
    num_qo_heads : int
        Number of query/output heads.
    num_kv_heads : int
        Number of key/value heads.
    head_dim : int
        Per-head dimension. Cake supports 64, 96, and 128.
    q_data_type : torch.dtype
        Planned Q/K/V dtype, either ``torch.float16`` or ``torch.bfloat16``.
    sm_scale : Optional[float]
        Softmax scale. ``None`` selects ``1 / sqrt(head_dim)``.
    device : torch.device
        SM100 or SM103 CUDA device that will execute the plan.

    Returns
    -------
    dict[str, Any]
        Validated metadata and reusable workspaces consumed by
        :func:`run_cake_vsa`.
    """

    _arch_for_device(device)
    if R != C or R not in (64, 128):
        raise ValueError("Cake VSA supports square 64- or 128-token blocks")
    if kv_block_lens is not None and R != 64:
        raise ValueError("kv_block_lens is supported only by Cake blk64")
    if q2k_indices is not None and R != 64:
        raise ValueError("q2k_indices is supported only by Cake blk64")
    if q2k_num is not None and q2k_indices is None:
        raise ValueError("q2k_num requires q2k_indices")
    if M % R or N % C:
        raise ValueError("M and N must be divisible by the Cake VSA block size")
    if head_dim not in (64, 96, 128):
        raise ValueError("Cake VSA supports head dimensions 64, 96, and 128")
    if q_data_type not in (torch.float16, torch.bfloat16):
        raise ValueError("Cake VSA supports float16 and bfloat16 inputs")
    if num_qo_heads % num_kv_heads:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    if R == 64 and (
        head_dim != 128 or q_data_type != torch.bfloat16 or num_qo_heads != num_kv_heads
    ):
        raise ValueError("Cake blk64 supports native-head BF16 D128 only")
    if head_dim in (64, 96) and (
        q_data_type != torch.bfloat16 or num_qo_heads != num_kv_heads
    ):
        raise ValueError("Cake D64/D96 routes support native-head BF16 only")
    if (
        q_data_type == torch.bfloat16
        and num_qo_heads != num_kv_heads
        and (num_qo_heads != 8 or num_qo_heads // num_kv_heads not in (2, 4, 8))
    ):
        raise ValueError("Cake BF16 GQA routes require Hq=8 and group size 2, 4, or 8")
    mb, nb = M // R, N // C
    dense = None
    if q2k_indices is not None:
        if block_mask is not None or indptr is not None or indices is not None:
            raise ValueError(
                "q2k_indices is mutually exclusive with block_mask and BSR metadata"
            )
        if (
            q2k_indices.dtype != torch.int32
            or q2k_indices.device != device
            or not q2k_indices.is_contiguous()
            or q2k_indices.ndim != 3
            or tuple(q2k_indices.shape[:2]) != (num_qo_heads, mb)
        ):
            raise ValueError(
                "q2k_indices must be contiguous int32 [num_qo_heads, MB, topk] "
                "on the wrapper device"
            )
        max_selected_blocks = int(q2k_indices.shape[2])
        if max_selected_blocks <= 0 or max_selected_blocks > nb:
            raise ValueError("q2k_indices topk must be in [1, NB]")
        uniform_selected_blocks = q2k_num is None
        if q2k_num is None:
            q2k_num = torch.full(
                (num_qo_heads, mb),
                max_selected_blocks,
                dtype=torch.int32,
                device=device,
            )
        elif (
            q2k_num.dtype != torch.int32
            or q2k_num.device != device
            or not q2k_num.is_contiguous()
            or tuple(q2k_num.shape) != (num_qo_heads, mb)
        ):
            raise ValueError(
                "q2k_num must be contiguous int32 [num_qo_heads, MB] on the "
                "wrapper device"
            )
        row_counts = q2k_num
        if bool(torch.any((q2k_num < 1) | (q2k_num > max_selected_blocks)).item()):
            raise ValueError("q2k_num entries must be in [1, topk]")
        slots = torch.arange(max_selected_blocks, device=device)
        active_indices = q2k_indices[slots < q2k_num.unsqueeze(-1)]
        if bool(torch.any((active_indices < 0) | (active_indices >= nb)).item()):
            raise ValueError("active q2k_indices entries must be in [0, NB)")
    else:
        dense = _dense_mask(
            indptr,
            indices,
            block_mask,
            mb=mb,
            nb=nb,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            device=device,
        )
        row_counts = dense.sum(dim=-1, dtype=torch.int32)
        min_selected_blocks = int(row_counts.min().item())
        max_selected_blocks = int(row_counts.max().item())
        uniform_selected_blocks = bool(
            torch.all(row_counts == max_selected_blocks).item()
        )
        if min_selected_blocks <= 0:
            raise ValueError("every Cake VSA block row must select at least one block")
        if num_qo_heads > num_kv_heads:
            group_size = num_qo_heads // num_kv_heads
            grouped = dense.view(num_kv_heads, group_size, mb, nb)
            if not torch.equal(grouped, grouped[:, :1].expand_as(grouped)):
                raise ValueError(
                    "Cake GQA masks must be identical within each KV-head group"
                )
        if R == 64:
            q2k_num = row_counts.contiguous()
            q2k_indices = (
                torch.argsort(
                    dense.to(torch.int8),
                    dim=-1,
                    descending=True,
                    stable=True,
                )
                .to(torch.int32)
                .contiguous()
            )

    if head_dim in (64, 96) and max_selected_blocks > _MAX_COMPACT_BLOCKS:
        raise ValueError(
            "Cake D64/D96 routes support at most 64 selected blocks per row"
        )

    planned_kv_block_lens = None
    if R == 64:
        if kv_block_lens is None:
            planned_kv_block_lens = torch.full(
                (nb,), C, dtype=torch.int32, device=device
            )
        else:
            if tuple(kv_block_lens.shape) != (nb,):
                raise ValueError("kv_block_lens must have shape [NB]")
            planned_kv_block_lens = kv_block_lens.to(
                device=device, dtype=torch.int32
            ).contiguous()
            if bool(
                torch.any(
                    (planned_kv_block_lens < 1) | (planned_kv_block_lens > C)
                ).item()
            ):
                raise ValueError("kv_block_lens entries must be in [1, C]")
    shared_indptr = shared_indices = None
    if R != 64 and dense is not None and torch.equal(dense, dense[:1].expand_as(dense)):
        shared_indptr, shared_indices = _shared_bsr(
            dense,
            indptr,
            indices,
            trust_bsr=block_mask is None,
        )
    return {
        "M": M,
        "N": N,
        "R": R,
        "C": C,
        "mb": mb,
        "nb": nb,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": q_data_type,
        "sm_scale": sm_scale,
        "block_mask": dense,
        "row_counts": row_counts,
        "max_selected_blocks": max_selected_blocks,
        "uniform_selected_blocks": uniform_selected_blocks,
        "indptr": shared_indptr,
        "indices": shared_indices,
        "q2k_indices": q2k_indices,
        "q2k_num": q2k_num,
        "kv_block_lens": planned_kv_block_lens,
        "workspace": {},
    }


def _workspace_tensor(
    plan: dict[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    workspace = plan["workspace"]
    tensor = workspace.get(name)
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != shape
        or tensor.dtype != dtype
        or tensor.device != device
    ):
        tensor = torch.empty(shape, dtype=dtype, device=device)
        workspace[name] = tensor
    return tensor


def _outputs(
    plan: dict[str, Any],
    q: torch.Tensor,
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise ValueError("out must match the query shape, dtype, and device")
    stats_shape = (int(q.shape[0]), int(q.shape[1])) if return_lse else (1,)
    if lse is None:
        if return_lse:
            lse = torch.empty(stats_shape, dtype=torch.float32, device=q.device)
        else:
            lse = _workspace_tensor(
                plan, "stats_scratch", stats_shape, torch.float32, q.device
            )
    elif (
        not return_lse
        or tuple(lse.shape) != stats_shape
        or lse.dtype != torch.float32
        or lse.device != q.device
    ):
        raise ValueError("lse must be float32 [M, num_qo_heads] on the query device")
    return out, lse


def _check_inputs(
    plan: dict[str, Any], q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> None:
    expected_q = (plan["M"], plan["num_qo_heads"], plan["head_dim"])
    expected_kv = (plan["N"], plan["num_kv_heads"], plan["head_dim"])
    for name, tensor, shape in (
        ("q", q, expected_q),
        ("k", k, expected_kv),
        ("v", v, expected_kv),
    ):
        if (
            tensor.device.type != "cuda"
            or tensor.device != q.device
            or tensor.dtype != plan["dtype"]
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} does not match the Cake VSA plan")


def _run_standard(
    profile: str,
    plan: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    stats: torch.Tensor,
    *,
    return_lse: bool,
    selected_blocks: Optional[int] = None,
) -> None:
    import tvm_ffi

    module = _load_module(profile, _arch_for_device(q.device))
    args: list[Any] = [
        q,
        k,
        v,
        out,
        stats,
        stats,
        (
            plan["indices"]
            if profile == "ultrasparse_bsr"
            else plan["block_mask"].view(torch.uint8)
        ),
        plan["mb"],
        plan["nb"],
    ]
    if selected_blocks is not None:
        args.append(selected_blocks)
    if profile == "ultrasparse_bsr":
        if selected_blocks != 6:
            raise ValueError(
                "Cake ultrasparse route requires exactly six selected blocks"
            )
        total_tiles = plan["mb"] * plan["num_qo_heads"]
        args.append(total_tiles)
    args.extend(
        [
            plan["num_qo_heads"],
            plan["num_kv_heads"],
            float(plan["sm_scale"] or 1.0 / math.sqrt(plan["head_dim"]))
            / math.log(2.0),
            1.0,
            int(return_lse),
            0,
        ]
    )
    if profile == "gqa_mask":
        group_size = plan["num_qo_heads"] // plan["num_kv_heads"]
        tokens_per_tile = 2 * (64 // group_size)
        grid_x = (plan["M"] + tokens_per_tile - 1) // tokens_per_tile
        grid_y = plan["num_kv_heads"]
    elif profile == "ultrasparse_bsr":
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        grid_x, grid_y = min(total_tiles, sm_count), 1
    elif profile in {"head64_native", "head96_native"}:
        grid_x, grid_y = plan["mb"] * 2, plan["num_qo_heads"]
    else:
        grid_x, grid_y = plan["mb"], plan["num_qo_heads"]
    args.extend([grid_x, grid_y, 1])
    with tvm_ffi.use_torch_stream():
        module.run(*args)


def _run_blk64(
    plan: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    stats: torch.Tensor,
    return_lse: bool,
) -> None:
    import tvm_ffi

    module = _load_module("blk64_persistent", _arch_for_device(q.device))
    total_tiles = plan["mb"] * plan["num_qo_heads"]
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    persistent_ctas = min(total_tiles, sm_count)
    tiles_per_cta = (total_tiles + persistent_ctas - 1) // persistent_ctas
    scale = float(plan["sm_scale"] or 1.0 / math.sqrt(plan["head_dim"]))
    with tvm_ffi.use_torch_stream():
        module.run(
            q,
            k,
            v,
            out,
            stats,
            plan["q2k_indices"],
            plan["q2k_num"],
            plan["kv_block_lens"],
            plan["q2k_indices"].shape[-1],
            plan["M"],
            plan["mb"],
            total_tiles,
            tiles_per_cta,
            plan["num_qo_heads"],
            scale / math.log(2.0),
            int(return_lse),
            persistent_ctas,
            1,
            1,
        )


def _fp16_metadata(plan: dict[str, Any], q: torch.Tensor):
    cached = plan["workspace"].get("fp16_metadata")
    if cached is not None and cached[0].device == q.device:
        return cached
    group_size = plan["num_qo_heads"] // plan["num_kv_heads"]
    masks = plan["block_mask"][::group_size]
    counts = plan["row_counts"][::group_size]
    topk = int(counts.max().item())
    if topk > _MAX_DIRECT_TOPK or not torch.all(counts == topk):
        raise ValueError("Cake FP16 direct route requires fixed top-k <= 32")
    selected = masks.nonzero(as_tuple=False)
    per_block = selected[:, 2].view(plan["num_kv_heads"], plan["mb"], topk)
    q2k = (
        per_block.repeat_interleave(plan["R"], dim=1)
        .to(device=q.device, dtype=torch.int32)
        .contiguous()
    )
    device = q.device
    cached = (
        q2k,
        torch.tensor([0, plan["M"]], dtype=torch.int32, device=device),
        torch.tensor([0, plan["N"]], dtype=torch.int32, device=device),
        torch.zeros((1,), dtype=torch.int32, device=device),
        torch.tensor([plan["N"]], dtype=torch.int32, device=device),
        torch.zeros((1,), dtype=torch.int32, device=device),
        topk,
    )
    plan["workspace"]["fp16_metadata"] = cached
    return cached


def _run_fp16(
    plan: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    stats: torch.Tensor,
    return_lse: bool,
) -> None:
    import tvm_ffi

    module = _load_module("fp16_direct", _arch_for_device(q.device))
    q2k, cu_q, cu_k, q_offsets, kv_lens, page_table, topk = _fp16_metadata(plan, q)
    scale = float(plan["sm_scale"] or 1.0 / math.sqrt(plan["head_dim"]))
    with tvm_ffi.use_torch_stream():
        module.run(
            q,
            k,
            v,
            out,
            stats,
            stats,
            q2k,
            cu_q,
            cu_k,
            q_offsets,
            kv_lens,
            page_table,
            plan["M"],
            plan["num_qo_heads"],
            plan["num_kv_heads"],
            topk,
            1,
            0,
            0,
            0,
            scale / math.log(2.0),
            1.0,
            int(return_lse),
            0,
            (plan["M"] + 255) // 256,
            plan["num_qo_heads"],
            1,
        )


@flashinfer_api(trace=block_sparse_attention_run_trace)
def run_cake_vsa(
    plan: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    return_lse: bool,
    backend: str,
):
    """Run one explicit source-level route; no external fallback is available.

    Parameters
    ----------
    plan : dict[str, Any]
        Metadata returned by :func:`plan_cake_vsa`.
    q : torch.Tensor
        Contiguous query tensor with shape
        ``(M, num_qo_heads, head_dim)``.
    k : torch.Tensor
        Contiguous key tensor with shape
        ``(N, num_kv_heads, head_dim)``.
    v : torch.Tensor
        Contiguous value tensor with the same shape and dtype as ``k``.
    out : Optional[torch.Tensor]
        Optional output buffer matching ``q``.
    lse : Optional[torch.Tensor]
        Optional float32 log-sum-exp buffer with shape
        ``(M, num_qo_heads)``. It is accepted only when ``return_lse`` is true.
    return_lse : bool
        Return log-sum-exp values with the output. D64/D96, BF16 GQA,
        ultrasparse, and long-sequence routes do not support this option.
    backend : str
        Must be ``"cake"``.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Attention output, or ``(output, lse)`` when ``return_lse`` is true.
    """

    if backend != "cake":
        raise ValueError("run_cake_vsa requires backend='cake'")
    _check_inputs(plan, q, k, v)
    output, stats = _outputs(plan, q, out, lse, return_lse)
    if plan["head_dim"] in (64, 96):
        if return_lse:
            raise ValueError("Cake D64/D96 routes do not support return_lse")
        _run_standard(
            f"head{plan['head_dim']}_native",
            plan,
            q,
            k,
            v,
            output,
            stats,
            return_lse=False,
        )
        return output

    if plan["R"] == 64:
        _run_blk64(plan, q, k, v, output, stats, return_lse)
    elif q.dtype == torch.float16:
        if plan["num_qo_heads"] == plan["num_kv_heads"]:
            _run_standard(
                "blk128_fp16_compact",
                plan,
                q,
                k,
                v,
                output,
                stats,
                return_lse=return_lse,
            )
        else:
            _run_fp16(plan, q, k, v, output, stats, return_lse)
    elif plan["num_qo_heads"] != plan["num_kv_heads"]:
        if return_lse:
            raise ValueError("Cake BF16 GQA routes do not support return_lse")
        _run_standard("gqa_mask", plan, q, k, v, output, stats, return_lse=False)
    elif (
        plan["mb"] >= 625
        and plan["num_qo_heads"] == plan["num_kv_heads"] == 8
        and plan["indices"] is not None
    ):
        if return_lse:
            raise ValueError("Cake ultrasparse routes do not support return_lse")
        selected = plan["max_selected_blocks"]
        if selected != 6 or not plan["uniform_selected_blocks"]:
            raise ValueError(
                "Cake ultrasparse route requires exactly six selected blocks"
            )
        _run_standard(
            "ultrasparse_bsr",
            plan,
            q,
            k,
            v,
            output,
            stats,
            return_lse=False,
            selected_blocks=selected,
        )
    elif plan["N"] >= 16384 and plan["num_qo_heads"] == 8:
        if return_lse:
            raise ValueError("Cake long-sequence routes do not support return_lse")
        selected = plan["max_selected_blocks"]
        if selected > _MAX_LONGSEQ_BLOCKS or not plan["uniform_selected_blocks"]:
            raise ValueError("Cake long-sequence route requires fixed top-k <= 192")
        _run_standard(
            "longseq",
            plan,
            q,
            k,
            v,
            output,
            stats,
            return_lse=False,
            selected_blocks=selected,
        )
    else:
        if plan["max_selected_blocks"] > _MAX_COMPACT_BLOCKS:
            raise ValueError("Cake compact route supports at most 64 selected blocks")
        _run_standard(
            "blk128_compact",
            plan,
            q,
            k,
            v,
            output,
            stats,
            return_lse=return_lse,
        )
    return (output, stats) if return_lse else output


__all__ = ["plan_cake_vsa", "run_cake_vsa"]
