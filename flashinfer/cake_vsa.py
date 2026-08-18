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


_MAX_COMPACT_BLOCKS = 64
_MAX_DIRECT_TOPK = 32
_MAX_LONGSEQ_BLOCKS = 192


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
    if manifest.get("schema") != "cake-vsa-block-sparse-source-export-v1":
        raise RuntimeError("unsupported Cake VSA source manifest")
    return manifest


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
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.equal(dense, dense[:1].expand_as(dense)):
        raise ValueError("Cake blk64 requires one shared BSR pattern across heads")
    if indptr is not None and indices is not None:
        return (
            indptr.to(device=dense.device, dtype=torch.int32).contiguous(),
            indices.to(device=dense.device, dtype=torch.int32).contiguous(),
        )
    shared = dense[0]
    counts = shared.sum(dim=1, dtype=torch.int32)
    ptr = torch.cat(
        [
            torch.zeros((1,), dtype=torch.int32, device=dense.device),
            counts.cumsum(0, dtype=torch.int32),
        ]
    )
    cols = shared.nonzero(as_tuple=False)[:, 1].to(torch.int32).contiguous()
    return ptr, cols


def plan_cake_vsa(
    indptr: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    block_mask: Optional[torch.Tensor],
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
    """Create stable metadata and workspaces for the source-level backend."""

    _arch_for_device(device)
    if R != C or R not in (64, 128):
        raise ValueError("Cake VSA supports square 64- or 128-token blocks")
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
    if int(row_counts.min().item()) <= 0:
        raise ValueError("every Cake VSA block row must select at least one block")
    shared_indptr = shared_indices = None
    if R == 64 or torch.equal(dense, dense[:1].expand_as(dense)):
        shared_indptr, shared_indices = _shared_bsr(dense, indptr, indices)
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
        "indptr": shared_indptr,
        "indices": shared_indices,
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
            plan["indptr"],
            plan["indices"],
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
    if cached is not None:
        return cached
    counts = plan["row_counts"][0]
    topk = int(counts.max().item())
    if topk > _MAX_DIRECT_TOPK or not torch.all(counts == topk):
        raise ValueError("Cake FP16 direct route requires fixed top-k <= 32")
    selected = plan["block_mask"][0].nonzero(as_tuple=False)
    per_block = selected[:, 1].view(plan["mb"], topk).to(torch.int32)
    per_query = per_block.repeat_interleave(plan["R"], dim=0)
    q2k = (
        per_query.unsqueeze(0)
        .expand(plan["num_kv_heads"], plan["M"], topk)
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
    """Run one explicit source-level route; no external fallback is available."""

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
        counts = plan["row_counts"]
        selected = int(counts.max().item())
        if selected > _MAX_DIRECT_TOPK or not torch.all(counts == selected):
            raise ValueError("Cake ultrasparse route requires fixed top-k <= 32")
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
        counts = plan["row_counts"]
        selected = int(counts.max().item())
        if selected > _MAX_LONGSEQ_BLOCKS or not torch.all(counts == selected):
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
        if int(plan["row_counts"].max().item()) > _MAX_COMPACT_BLOCKS:
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
