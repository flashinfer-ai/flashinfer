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

"""
Tests for GEMM + Reduce-Scatter (flashinfer.comm.gemm_reduce_scatter).

Run with pytest:
    pytest tests/comm/test_gemm_reduce_scatter_blackwell.py -v

"""

import multiprocessing as mp
import os
import socket
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.distributed as dist

from flashinfer.comm.gemm_reduce_scatter.gemm_reduce_scatter_blackwell import (
    BlackwellGemmRSConfig,
    BlackwellGemmRSUnavailableError,
    BlackwellGemmRSWorkspace,
    _require_16b_alignment,
    _require_blackwell_deps,
    _require_blackwell_device,
    _require_world_group,
    _resolve_b_layout,
    _validate_blackwell_problem_shape,
    gemm_reduce_scatter_blackwell_cutlass,
)


# ---------------------------------------------------------------------------
# Utilities — matching pattern from test_vllm_custom_allreduce.py
# ---------------------------------------------------------------------------


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(world_size: int, target, target_args: tuple = ()) -> None:
    """Launch world_size processes, each calling target(world_size, rank, port, *target_args)."""
    mp.set_start_method("spawn", force=True)
    # Ensure the repo root is on PYTHONPATH so spawn'd processes can import
    # the tests package (tests/__init__.py exists but repo root may not be
    # on sys.path when pytest uses --import-mode=importlib).
    repo_root = str(Path(__file__).resolve().parents[2])
    existing = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in existing.split(os.pathsep) if entry != repo_root]
    os.environ["PYTHONPATH"] = os.pathsep.join([repo_root, *filter(None, entries)])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    port = get_open_port()
    procs = []
    for rank in range(world_size):
        proc = mp.Process(
            target=target,
            args=(world_size, rank, port) + target_args,
            name=f"Worker-{rank}",
        )
        proc.start()
        procs.append(proc)
    for rank, proc in enumerate(procs):
        proc.join()
        assert proc.exitcode == 0, (
            f"Process {rank} failed with exit code {proc.exitcode}"
        )


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _ref_gemm_reduce_scatter(X_local, W_local, group):
    """
    fp32 ground truth for correctness comparison.

    Uses fp32 matmul (not bf16) and fp32 all_gather+sum to avoid NCCL's
    bf16 ring-reduce rounding (which accumulates bf16 rounding error and
    would make a strict comparison unfair to our fp32-accumulating kernel).
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    M_local = X_local.shape[0] // world_size

    # fp32 partial K-sum, then all_gather + sum on each rank
    partial_fp32 = X_local.float() @ W_local.float()
    all_partials = [torch.zeros_like(partial_fp32) for _ in range(world_size)]
    dist.all_gather(all_partials, partial_fp32)
    gt_fp32 = sum(all_partials)[rank * M_local : (rank + 1) * M_local]
    return gt_fp32.to(X_local.dtype)


# ---------------------------------------------------------------------------
# Distributed initialisation
# ---------------------------------------------------------------------------


def _dist_init(world_size: int, rank: int, port: int):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    return device, group


# ---------------------------------------------------------------------------
# Module-level worker functions (must be at top-level for mp.Process)
# ---------------------------------------------------------------------------


def _blackwell_nvshmem_init(rank: int, world_size: int):
    try:
        from cuda.core import Device
    except ImportError:
        from cuda.core.experimental import Device
    import nvshmem.core

    dev = Device(rank)
    dev.set_current()
    uid = nvshmem.core.get_unique_id(empty=(rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).cuda()
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)
    nvshmem.core.init(
        device=dev,
        uid=uid,
        rank=rank,
        nranks=world_size,
        initializer_method="uid",
    )
    return nvshmem.core


def _blackwell_correctness_worker(
    world_size,
    rank,
    port,
    dtype_str,
    M,
    K,
    N,
    noncontiguous_weight,
    mma_tiler_mn,
    cluster_shape_mn,
    loops,
):
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    device, group = _dist_init(world_size, rank, port)
    nvshmem_core = _blackwell_nvshmem_init(rank, world_size)
    workspace = None
    try:
        config = BlackwellGemmRSConfig(
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
        )
        K_local = K // world_size
        workspace = BlackwellGemmRSWorkspace(
            M=M,
            N=N,
            K_local=K_local,
            group=group,
            dtype=dtype,
            device=device,
            config=config,
        )
        for loop in range(loops):
            generator = torch.Generator(device=device)
            generator.manual_seed(1009 * rank + 104729 * loop)
            X_local = torch.randn(
                M, K_local, device=device, dtype=dtype, generator=generator
            )
            if noncontiguous_weight:
                W_storage = torch.randn(
                    N, K_local, device=device, dtype=dtype, generator=generator
                )
                W_local = W_storage.transpose(0, 1)
                assert not W_local.is_contiguous()
            else:
                W_local = torch.randn(
                    K_local, N, device=device, dtype=dtype, generator=generator
                )

            ref = _ref_gemm_reduce_scatter(X_local, W_local, group)
            out = gemm_reduce_scatter_blackwell_cutlass(
                X_local,
                W_local,
                group,
                workspace=workspace,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(out, ref, atol=3.0, rtol=0.05)
            expected_layout = "staged" if noncontiguous_weight else "nocopy"
            assert workspace.last_b_layout == expected_layout
    finally:
        if workspace is not None:
            workspace.destroy()
        nvshmem_core.finalize()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Skip helpers — called in the main pytest process only, never inside workers
# ---------------------------------------------------------------------------


def _skip_if_insufficient_gpus(world_size: int):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, found {torch.cuda.device_count()}")


def _gpu_major_via_smi() -> int:
    """Return compute capability major of GPU 0 via nvidia-smi.

    Does NOT initialize the CUDA runtime, so it is safe to call before mp.spawn
    (spawn requires that CUDA is not yet initialized in the parent process).
    """
    import subprocess

    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=compute_cap",
                    "--format=csv,noheader",
                    "--id=0",
                ],
                timeout=5,
            )
            .decode()
            .strip()
        )
        return int(out.split(".")[0])
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------


def test_cutlass_blackwell_requires_world_group():
    subgroup = object()

    _require_world_group(dist.group.WORLD)
    with pytest.raises(NotImplementedError, match="supports only dist.group.WORLD"):
        _require_world_group(subgroup)


def _fake_cuda_weight(*, k=1024, n=2048, stride=None, ptr=0x1000):
    weight = Mock()
    weight.ndim = 2
    weight.is_cuda = True
    weight.dtype = torch.bfloat16
    weight.shape = (k, n)
    weight.stride.return_value = stride or (n, 1)
    weight.data_ptr.return_value = ptr
    weight.element_size.return_value = 2
    return weight


def test_blackwell_default_prefers_nocopy():
    config = BlackwellGemmRSConfig()
    assert config.b_layout == "nocopy"
    assert config.allow_staged_fallback
    assert _resolve_b_layout(_fake_cuda_weight(), config) == "nocopy"


def test_blackwell_noncontiguous_weight_uses_staged_fallback():
    config = BlackwellGemmRSConfig()
    weight = _fake_cuda_weight(stride=(1, 1024))
    assert _resolve_b_layout(weight, config) == "staged"


def test_blackwell_staged_fallback_can_be_disabled():
    config = BlackwellGemmRSConfig(allow_staged_fallback=False)
    weight = _fake_cuda_weight(stride=(1, 1024))
    with pytest.raises(ValueError, match="staged fallback is disabled"):
        _resolve_b_layout(weight, config)


def test_blackwell_explicit_staged_layout_is_preserved():
    config = BlackwellGemmRSConfig(b_layout="staged")
    assert _resolve_b_layout(_fake_cuda_weight(), config) == "staged"


def test_blackwell_problem_shape_accepts_supported_aligned_shape():
    _validate_blackwell_problem_shape(
        M=2048,
        N=1536,
        K_local=1024,
        world_size=4,
        config=BlackwellGemmRSConfig(),
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"M": 2176}, "M-tail tiles"),
        ({"M": 2048, "N": 2050}, "16-byte aligned"),
        ({"M": 2048, "K_local": 1026}, "16-byte aligned"),
        ({"M": 2048, "world_size": 16}, "world sizes"),
    ],
)
def test_blackwell_problem_shape_rejects_unsupported_cases(kwargs, match):
    values = {"M": 2048, "N": 2048, "K_local": 1024, "world_size": 4}
    values.update(kwargs)
    error = NotImplementedError if values["world_size"] == 16 else ValueError
    with pytest.raises(error, match=match):
        _validate_blackwell_problem_shape(**values, config=BlackwellGemmRSConfig())


def test_blackwell_config_validates_tiler_and_cluster():
    BlackwellGemmRSConfig(mma_tiler_mn=(256, 128), cluster_shape_mn=(4, 1))
    with pytest.raises(ValueError, match="MMA tile"):
        BlackwellGemmRSConfig(mma_tiler_mn=(192, 128))
    with pytest.raises(ValueError, match="cluster shape"):
        BlackwellGemmRSConfig(cluster_shape_mn=(3, 1))


def test_blackwell_dependency_error_is_actionable(monkeypatch):
    import importlib

    module = importlib.import_module(
        "flashinfer.comm.gemm_reduce_scatter.gemm_reduce_scatter_blackwell"
    )

    monkeypatch.setattr(module, "_CUTLASS_IMPORT_ERROR", ImportError("missing cutlass"))
    monkeypatch.setattr(module, "_NVSHMEM_IMPORT_ERROR", None)
    with pytest.raises(BlackwellGemmRSUnavailableError, match="nvidia-cutlass-dsl"):
        _require_blackwell_deps()


def test_blackwell_packed_store_alignment_guard():
    tensor = Mock()
    tensor.data_ptr.return_value = 0x1000
    _require_16b_alignment("test tensor", tensor)

    tensor.data_ptr.return_value = 0x1008
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        _require_16b_alignment("test tensor", tensor)


def test_blackwell_device_guard_rejects_cpu():
    with pytest.raises(BlackwellGemmRSUnavailableError, match="CUDA device"):
        _require_blackwell_device(torch.device("cpu"))


def test_blackwell_requires_explicit_workspace():
    with pytest.raises(ValueError, match="requires a BlackwellGemmRSWorkspace"):
        gemm_reduce_scatter_blackwell_cutlass(None, None, dist.group.WORLD)


def test_blackwell_kernel_source_has_release_acquire_protocol():
    source = (
        Path(__file__).resolve().parents[2]
        / "flashinfer/comm/gemm_reduce_scatter/cutlass_blackwell_gemm_rs.py"
    ).read_text()
    assert 'sem="acquire"' in source
    assert 'scope="sys"' in source
    assert "spin_lock_atom_cas_acquire_wait" in source
    assert "self.memory_protocol_level = 3" in source
    assert source.count('cute.arch.fence_proxy("alias")') >= 3
    assert "st.global.sys.relaxed.v4.f32" in source
    assert "multimem_ld_reduce_8xf16" in source
    assert "multimem_ld_reduce_8xbf16" in source


BLACKWELL_CORRECTNESS_CASES = [
    # ws, M, K_total, N, dtype, noncontiguous W, MMA tile, cluster, loops
    (2, 2048, 4096, 1024, "bfloat16", False, (256, 256), (2, 1), 4),
    (2, 2048, 4096, 1536, "float16", True, (256, 256), (2, 1), 1),
    (4, 2048, 4096, 3072, "bfloat16", False, (256, 128), (4, 1), 1),
    (4, 2048, 4096, 2048, "float16", False, (256, 256), (2, 1), 1),
    (8, 2048, 4096, 1024, "bfloat16", False, (256, 256), (2, 1), 2),
    (8, 2048, 4096, 1536, "float16", False, (256, 256), (2, 1), 2),
]


@pytest.mark.parametrize(
    "world_size,M,K,N,dtype_str,noncontiguous_weight,mma_tiler_mn,cluster_shape_mn,loops",
    BLACKWELL_CORRECTNESS_CASES,
)
def test_cutlass_blackwell_correctness(
    world_size,
    M,
    K,
    N,
    dtype_str,
    noncontiguous_weight,
    mma_tiler_mn,
    cluster_shape_mn,
    loops,
):
    _skip_if_insufficient_gpus(world_size)
    if _gpu_major_via_smi() < 10:
        pytest.skip("CUTLASS Blackwell backend requires SM >= 100")
    multi_process_parallel(
        world_size,
        _blackwell_correctness_worker,
        target_args=(
            dtype_str,
            M,
            K,
            N,
            noncontiguous_weight,
            mma_tiler_mn,
            cluster_shape_mn,
            loops,
        ),
    )
