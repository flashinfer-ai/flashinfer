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

"""Blackwell CUTLASS/CuTe DSL workspace and tensor adapters for GEMM+RS.

This module owns the FlashInfer ``backend="cutlass_blackwell"`` bridge for the
ported CUTLASS CuTe DSL Blackwell GEMM+reduce-scatter kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Literal

import torch
import torch.distributed as dist

_ALIGNMENT_BYTES = 16


try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as cutlass_utils
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack
except (ImportError, OSError) as exc:  # pragma: no cover - depends on Blackwell env.
    cutlass = None
    cute = None
    cutlass_torch = None
    cutlass_utils = None
    cuda = None
    from_dlpack = None
    _CUTLASS_IMPORT_ERROR = exc
else:
    _CUTLASS_IMPORT_ERROR = None

try:
    import nvshmem.core
except (ImportError, OSError) as exc:  # pragma: no cover - depends on Blackwell env.
    nvshmem = None
    _NVSHMEM_IMPORT_ERROR = exc
else:
    _NVSHMEM_IMPORT_ERROR = None


class BlackwellGemmRSUnavailableError(RuntimeError):
    """Required Blackwell GEMM+RS runtime dependencies are unavailable."""


def _require_blackwell_deps() -> None:
    missing = []
    if _CUTLASS_IMPORT_ERROR is not None:
        missing.append(f"nvidia-cutlass-dsl and cuda-python ({_CUTLASS_IMPORT_ERROR})")
    if _NVSHMEM_IMPORT_ERROR is not None:
        missing.append(f"nvshmem4py ({_NVSHMEM_IMPORT_ERROR})")
    if missing:
        raise BlackwellGemmRSUnavailableError(
            "backend='cutlass_blackwell' is unavailable; install and initialize "
            "the Blackwell dependencies: " + "; ".join(missing)
        )


def _require_blackwell_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise BlackwellGemmRSUnavailableError(
            "backend='cutlass_blackwell' requires a CUDA device."
        )
    major, minor = torch.cuda.get_device_capability(device)
    if major < 10:
        raise BlackwellGemmRSUnavailableError(
            "backend='cutlass_blackwell' requires SM100 or newer, "
            f"got SM{major}{minor}."
        )


def _require_16b_alignment(name: str, tensor: torch.Tensor) -> None:
    if tensor.data_ptr() % _ALIGNMENT_BYTES != 0:
        raise RuntimeError(
            f"Blackwell GEMM+RS {name} must be {_ALIGNMENT_BYTES}-byte aligned "
            "for packed 128-bit multimem and peer operations."
        )


def _require_world_group(group: dist.ProcessGroup) -> None:
    """Reject subgroups until the CuTe DSL kernel uses group-local rank state."""
    if group is not dist.group.WORLD:
        raise NotImplementedError(
            "backend='cutlass_blackwell' currently supports only dist.group.WORLD. "
            "The CUTLASS CuTe DSL kernel reads torch.distributed global rank/world "
            "state during construction, so process subgroups would use incorrect "
            "rank mapping."
        )


def _cutlass_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    raise ValueError(f"Blackwell GEMM+RS supports bf16/fp16, got {dtype}.")


def _check_contiguous_cuda(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _check_cuda(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor.")


def _mark_dynamic(cute_tensor, leading_dim: int | None = None):
    if leading_dim is None:
        return cute_tensor.mark_layout_dynamic()
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def _as_l1_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"expected a 2-D tensor, got {tuple(tensor.shape)}.")
    stride0, stride1 = tensor.stride()
    return tensor.as_strided(
        (tensor.shape[0], tensor.shape[1], 1), (stride0, stride1, 1)
    )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _debug_nvshmem_teams(prefix: str, *, rank: int, world_size: int) -> None:
    if not _env_flag("FLASHINFER_GRS_DEBUG"):
        return

    def _safe_call(fn):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - debug-only path.
            return f"{type(exc).__name__}: {exc}"

    teams = {}
    for name in ("TEAM_WORLD", "TEAM_NODE", "TEAM_SHARED"):
        if hasattr(nvshmem.core.Teams, name):
            team = getattr(nvshmem.core.Teams, name)
            teams[name] = {
                "handle": team,
                "my_pe": _safe_call(lambda team=team: nvshmem.core.team_my_pe(team)),
                "n_pes": _safe_call(lambda team=team: nvshmem.core.team_n_pes(team)),
            }
    print(
        f"[flashinfer_grs_debug:{prefix}] "
        f"rank={rank} torch_world_size={world_size} "
        f"nvshmem_my_pe={_safe_call(nvshmem.core.my_pe)} "
        f"nvshmem_n_pes={_safe_call(nvshmem.core.n_pes)} "
        f"teams={teams}",
        flush=True,
    )


def cute_tensor_from_torch(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    leading_dim: int | None = None,
    assumed_align: int = 16,
    require_contiguous: bool = True,
):
    """Create a CuTe tensor view over an existing torch tensor.

    This is a view/adapter. It does not allocate or copy.
    """
    _require_blackwell_deps()
    if require_contiguous:
        _check_contiguous_cuda(tensor, "tensor")
    else:
        _check_cuda(tensor, "tensor")
    if tensor.dtype != dtype:
        raise ValueError(
            f"tensor dtype mismatch: got {tensor.dtype}, expected {dtype}."
        )

    cutlass_dtype = _cutlass_dtype(dtype)
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = cutlass_dtype
    cute_tensor = _mark_dynamic(cute_tensor, leading_dim)
    return cutlass_torch.convert_cute_tensor(
        tensor,
        cute_tensor,
        cutlass_dtype,
        is_dynamic_layout=True,
    )


def make_a_tensor(X_local: torch.Tensor):
    """Adapt FlashInfer X_local [M, K_local] to CUTLASS A [M, K_local, 1]."""
    if X_local.ndim != 2:
        raise ValueError(f"X_local must be 2-D, got {tuple(X_local.shape)}.")
    # Contiguous X_local has K as the leading dynamic dimension in CUTLASS terms.
    return cute_tensor_from_torch(
        _as_l1_tensor(X_local), dtype=X_local.dtype, leading_dim=1
    )


def make_b_tensor_staged(
    W_local: torch.Tensor,
    *,
    staging: torch.Tensor | None = None,
) -> tuple[object, torch.Tensor]:
    """Create CUTLASS B [N, K_local, 1] from FlashInfer W_local [K_local, N].

    This currently uses a contiguous transpose staging tensor.  It is useful for
    an initial correctness port, but production performance must remove this
    copy by teaching the kernel to consume W_local's native [K, N] layout.
    """
    W_nk = stage_b_tensor(W_local, staging=staging)
    W_nk_l1 = _as_l1_tensor(W_nk)
    return cute_tensor_from_torch(W_nk_l1, dtype=W_local.dtype, leading_dim=1), W_nk


def make_b_tensor_nocopy(W_local: torch.Tensor):
    """Create logical CUTLASS B [N, K_local, 1] directly over W_local [K_local, N]."""
    if W_local.ndim != 2:
        raise ValueError(f"W_local must be 2-D, got {tuple(W_local.shape)}.")
    _check_contiguous_cuda(W_local, "W_local")
    k_local, n = W_local.shape
    b_view = W_local.as_strided((n, k_local, 1), (1, n, 1))
    return cute_tensor_from_torch(
        b_view,
        dtype=W_local.dtype,
        leading_dim=0,
        require_contiguous=False,
    )


def stage_b_tensor(
    W_local: torch.Tensor,
    *,
    staging: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stage FlashInfer W_local [K_local, N] as CUTLASS B [N, K_local]."""
    if W_local.ndim != 2:
        raise ValueError(f"W_local must be 2-D, got {tuple(W_local.shape)}.")
    _check_cuda(W_local, "W_local")
    if staging is None:
        W_nk = W_local.transpose(0, 1).contiguous()
    else:
        expected_shape = (W_local.shape[1], W_local.shape[0])
        if tuple(staging.shape) != expected_shape:
            raise ValueError(
                f"B staging shape mismatch: got {tuple(staging.shape)}, "
                f"expected {expected_shape}."
            )
        if staging.dtype != W_local.dtype:
            raise ValueError(
                f"B staging dtype mismatch: got {staging.dtype}, expected {W_local.dtype}."
            )
        if staging.device != W_local.device:
            raise ValueError(
                f"B staging device mismatch: got {staging.device}, expected {W_local.device}."
            )
        if not staging.is_contiguous():
            raise ValueError("B staging tensor must be contiguous.")
        staging.copy_(W_local.transpose(0, 1))
        W_nk = staging
    return W_nk


_SUPPORTED_BLACKWELL_WORLD_SIZES = (2, 4, 8)
_FP16_ELEMENTS_PER_16B = 8


@dataclass(frozen=True)
class BlackwellGemmRSConfig:
    mma_tiler_mn: tuple[int, int] = (256, 256)
    cluster_shape_mn: tuple[int, int] = (2, 1)
    use_2cta_instrs: bool = True
    use_tma_store: bool = True
    reduce_scatter: str = "two_shot"
    b_layout: Literal["staged", "nocopy"] = "nocopy"
    allow_staged_fallback: bool = True

    def __post_init__(self) -> None:
        if self.b_layout not in ("staged", "nocopy"):
            raise ValueError(
                f"Blackwell GEMM+RS b_layout must be 'staged' or 'nocopy', "
                f"got {self.b_layout!r}."
            )
        mma_m, mma_n = self.mma_tiler_mn
        valid_m = (128, 256) if self.use_2cta_instrs else (64, 128)
        if mma_m not in valid_m or mma_n not in range(32, 257, 32):
            raise ValueError(
                "Invalid Blackwell GEMM+RS MMA tile: "
                f"mma_tiler_mn={self.mma_tiler_mn}, "
                f"use_2cta_instrs={self.use_2cta_instrs}."
            )
        cluster_m, cluster_n = self.cluster_shape_mn
        if (
            cluster_m <= 0
            or cluster_n <= 0
            or cluster_m & (cluster_m - 1)
            or cluster_n & (cluster_n - 1)
            or cluster_m * cluster_n > 16
            or cluster_m % (2 if self.use_2cta_instrs else 1) != 0
        ):
            raise ValueError(
                "Invalid Blackwell GEMM+RS cluster shape: "
                f"cluster_shape_mn={self.cluster_shape_mn}."
            )
        if not self.use_tma_store:
            raise ValueError("Blackwell two-shot GEMM+RS requires use_tma_store=True.")
        if self.reduce_scatter != "two_shot":
            raise ValueError(
                "Blackwell GEMM+RS currently supports only reduce_scatter='two_shot'."
            )


def _validate_blackwell_problem_shape(
    *, M: int, N: int, K_local: int, world_size: int, config: BlackwellGemmRSConfig
) -> None:
    if M <= 0 or N <= 0 or K_local <= 0:
        raise ValueError(
            f"M, N, and K_local must be positive, got M={M}, N={N}, K_local={K_local}."
        )
    if world_size not in _SUPPORTED_BLACKWELL_WORLD_SIZES:
        raise NotImplementedError(
            "Blackwell two-shot GEMM+RS currently supports world sizes "
            f"{_SUPPORTED_BLACKWELL_WORLD_SIZES}, got {world_size}."
        )
    if M % world_size != 0:
        raise ValueError(f"M={M} must be divisible by world_size={world_size}.")

    mma_m, _ = config.mma_tiler_mn
    if M % mma_m != 0:
        raise ValueError(
            f"M={M} must be divisible by MMA tile M={mma_m}; M-tail tiles are "
            "not supported by the two-shot rank-ownership mapping."
        )
    m_tiles = M // mma_m
    if m_tiles < world_size or m_tiles % world_size != 0:
        raise ValueError(
            f"M={M} creates {m_tiles} MMA M tiles, which must be at least and "
            f"divisible by world_size={world_size}."
        )

    cta_m = mma_m // (2 if config.use_2cta_instrs else 1)
    if cta_m % world_size != 0:
        raise ValueError(
            f"CTA tile M={cta_m} must be divisible by world_size={world_size}."
        )
    if K_local % _FP16_ELEMENTS_PER_16B != 0 or N % _FP16_ELEMENTS_PER_16B != 0:
        raise ValueError(
            "Blackwell GEMM+RS requires K_local and N to be 16-byte aligned "
            f"({_FP16_ELEMENTS_PER_16B} fp16/bf16 elements), got K_local={K_local}, N={N}."
        )


def _nocopy_ineligibility_reason(W_local: torch.Tensor) -> str | None:
    if W_local.ndim != 2:
        return f"W_local must be 2-D, got shape={tuple(W_local.shape)}"
    if not W_local.is_cuda:
        return "W_local must be a CUDA tensor"
    if W_local.dtype not in (torch.bfloat16, torch.float16):
        return f"W_local dtype must be bf16/fp16, got {W_local.dtype}"
    k_local, n = W_local.shape
    if tuple(W_local.stride()) != (n, 1):
        return (
            "W_local must have native contiguous [K,N] strides "
            f"({n}, 1), got {tuple(W_local.stride())}"
        )
    if W_local.data_ptr() % _ALIGNMENT_BYTES != 0:
        return f"W_local data pointer must be {_ALIGNMENT_BYTES}-byte aligned"
    elements_per_16b = _ALIGNMENT_BYTES // W_local.element_size()
    if n % elements_per_16b != 0:
        return f"N={n} must be divisible by {elements_per_16b} elements"
    if k_local <= 0:
        return "K_local must be positive"
    return None


def _resolve_b_layout(
    W_local: torch.Tensor, config: BlackwellGemmRSConfig
) -> Literal["staged", "nocopy"]:
    if config.b_layout == "staged":
        return "staged"
    reason = _nocopy_ineligibility_reason(W_local)
    if reason is None:
        return "nocopy"
    if config.allow_staged_fallback:
        return "staged"
    raise ValueError(
        "Blackwell GEMM+RS no-copy B layout is not eligible and staged fallback "
        f"is disabled: {reason}."
    )


def _select_multicast_team(world_size: int) -> int:
    """Use the all-PE NVSHMEM team expected by the GEMM+RS kernel."""
    team = nvshmem.core.Teams.TEAM_WORLD
    try:
        team_size = nvshmem.core.team_n_pes(team)
    except Exception as exc:
        raise BlackwellGemmRSUnavailableError(
            "backend='cutlass_blackwell' requires NVSHMEM to be initialized "
            "collectively before constructing BlackwellGemmRSWorkspace."
        ) from exc
    if team_size != world_size:
        raise RuntimeError(
            "Blackwell GEMM+RS requires a multicast team matching the process "
            f"group world size. TEAM_WORLD has {team_size} PEs but "
            f"torch.distributed world_size is {world_size}."
        )
    return team


class BlackwellGemmRSWorkspace:
    """NVSHMEM workspace matching the CUTLASS Blackwell GEMM+RS example.

    The workspace owns the full symmetric C tensor, multicast aliases, peer
    tensors, and barrier flags required by the two-shot reduce-scatter epilogue.
    """

    def __init__(
        self,
        *,
        M: int,
        N: int,
        K_local: int,
        group: dist.ProcessGroup,
        dtype: torch.dtype,
        device: torch.device,
        config: BlackwellGemmRSConfig | None = None,
    ) -> None:
        _require_world_group(group)
        _require_blackwell_deps()
        _require_blackwell_device(device)
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"Blackwell GEMM+RS supports bf16/fp16, got {dtype}.")
        resolved_config = config or BlackwellGemmRSConfig()
        world_size = dist.get_world_size(group)
        _validate_blackwell_problem_shape(
            M=M,
            N=N,
            K_local=K_local,
            world_size=world_size,
            config=resolved_config,
        )

        self.M = M
        self.N = N
        self.K_local = K_local
        self.group = group
        self.world_size = world_size
        self.rank = dist.get_rank(group)
        self.dtype = dtype
        self.device = device
        self.config = resolved_config
        self.multicast_team = _select_multicast_team(self.world_size)
        self._destroyed = False
        self._compiled_gemm = None
        self._compiled_key = None
        self._compiled_a_tensor = None
        self._compiled_b_tensor = None

        self.c_full = nvshmem.core.tensor((M, N), dtype=dtype)
        self.c_full.zero_()
        _debug_nvshmem_teams("workspace", rank=self.rank, world_size=self.world_size)
        self.w_staging = (
            torch.empty((N, K_local), dtype=dtype, device=device)
            if self.config.b_layout == "staged"
            else None
        )
        self.last_b_layout: Literal["staged", "nocopy"] | None = None
        self.c_full_mc = nvshmem.core.get_multicast_tensor(
            self.multicast_team, self.c_full
        )
        self.c_peer_tensors = [
            nvshmem.core.get_peer_tensor(self.c_full, peer)
            for peer in range(self.world_size)
        ]
        _require_16b_alignment("local C tensor", self.c_full)
        _require_16b_alignment("multicast C tensor", self.c_full_mc)
        for peer, tensor in enumerate(self.c_peer_tensors):
            _require_16b_alignment(f"peer C tensor for rank {peer}", tensor)

        self.c_full_l1 = _as_l1_tensor(self.c_full)
        self.c_full_mc_l1 = _as_l1_tensor(self.c_full_mc)
        self.c_peer_l1_tensors = [_as_l1_tensor(t) for t in self.c_peer_tensors]

        self.c_tensor = cute_tensor_from_torch(
            self.c_full_l1, dtype=dtype, leading_dim=1
        )
        self.c_tensor_mc = from_dlpack(self.c_full_mc_l1, assumed_align=16)
        self.c_tensor_mc = _mark_dynamic(self.c_tensor_mc, leading_dim=1)
        self.c_peer_cute_tensors = [from_dlpack(t) for t in self.c_peer_l1_tensors]

        self.barrier_flag_torch = nvshmem.core.tensor(
            (self._num_barrier_flags(),), dtype=torch.int32
        )
        self.barrier_flag_torch.fill_(0)
        self.barrier_flag_torch_mc = nvshmem.core.get_multicast_tensor(
            self.multicast_team, self.barrier_flag_torch
        )
        self.barrier_flag = from_dlpack(self.barrier_flag_torch, assumed_align=16)
        self.barrier_flag = _mark_dynamic(self.barrier_flag)
        self.barrier_flag_mc = from_dlpack(self.barrier_flag_torch_mc, assumed_align=16)
        self.barrier_flag_mc = _mark_dynamic(self.barrier_flag_mc)

    def _num_barrier_flags(self) -> int:
        cta_tile_m = self.config.mma_tiler_mn[0] // (
            2 if self.config.use_2cta_instrs else 1
        )
        cta_tile_n = self.config.mma_tiler_mn[1]
        num_m_tiles = self.M // cta_tile_m
        num_n_tiles = (self.N + cta_tile_n - 1) // cta_tile_n
        num_tiles = num_m_tiles * num_n_tiles
        num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        return num_tiles + num_sms

    def get_w_staging(self) -> torch.Tensor:
        """Return lazily allocated local staging storage for B-layout fallback."""
        if self.w_staging is None:
            self.w_staging = torch.empty(
                (self.N, self.K_local), dtype=self.dtype, device=self.device
            )
        return self.w_staging

    @property
    def local_output(self) -> torch.Tensor:
        """Rank-local view, valid until the next call or workspace destruction."""
        m_local = self.M // self.world_size
        start = self.rank * m_local
        return self.c_full[start : start + m_local]

    def reset(self) -> None:
        # The CUTLASS two-shot kernel returns per-tile flags to zero before exit.
        # Host-side memset of NVSHMEM symmetric memory is extremely expensive
        # and would dominate measured latency.
        return

    def is_compatible(
        self,
        *,
        M: int,
        N: int,
        K_local: int,
        world_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        return (
            M == self.M
            and N == self.N
            and K_local == self.K_local
            and world_size == self.world_size
            and dtype == self.dtype
            and device == self.device
            and not self._destroyed
        )

    def destroy(self) -> None:
        """Synchronize outstanding work and collectively release NVSHMEM storage."""
        if self._destroyed:
            return
        torch.cuda.synchronize(self.device)
        nvshmem.core.free_tensor(self.c_full_mc)
        for peer, tensor in enumerate(self.c_peer_tensors):
            if peer != self.rank:
                nvshmem.core.free_tensor(tensor)
        nvshmem.core.free_tensor(self.c_full)
        nvshmem.core.free_tensor(self.barrier_flag_torch_mc)
        nvshmem.core.free_tensor(self.barrier_flag_torch)
        del self.w_staging
        self._destroyed = True

    def __enter__(self) -> "BlackwellGemmRSWorkspace":
        return self

    def __exit__(self, *args) -> None:
        self.destroy()


def _validate_run_inputs(
    X_local: torch.Tensor,
    W_local: torch.Tensor,
    group: dist.ProcessGroup,
    workspace: BlackwellGemmRSWorkspace,
) -> tuple[int, int, int, int]:
    _check_contiguous_cuda(X_local, "X_local")
    _check_cuda(W_local, "W_local")
    if X_local.ndim != 2:
        raise ValueError(f"X_local must be 2-D, got {tuple(X_local.shape)}.")
    if W_local.ndim != 2:
        raise ValueError(f"W_local must be 2-D, got {tuple(W_local.shape)}.")
    if X_local.dtype != W_local.dtype:
        raise ValueError(
            f"X_local and W_local dtype mismatch: {X_local.dtype} != {W_local.dtype}."
        )
    if X_local.device != W_local.device:
        raise ValueError(
            f"X_local and W_local device mismatch: {X_local.device} != {W_local.device}."
        )

    M, K_local = X_local.shape
    K_w, N = W_local.shape
    if K_w != K_local:
        raise ValueError(f"K mismatch: X_local K={K_local}, W_local K={K_w}.")

    world_size = dist.get_world_size(group)
    _validate_blackwell_problem_shape(
        M=M,
        N=N,
        K_local=K_local,
        world_size=world_size,
        config=workspace.config,
    )
    if not workspace.is_compatible(
        M=M,
        N=N,
        K_local=K_local,
        world_size=world_size,
        dtype=X_local.dtype,
        device=X_local.device,
    ):
        raise ValueError(
            "BlackwellGemmRSWorkspace is not compatible with this call: "
            f"call M={M} N={N} K_local={K_local} ws={world_size} "
            f"dtype={X_local.dtype} device={X_local.device}; "
            f"workspace M={workspace.M} N={workspace.N} "
            f"K_local={workspace.K_local} ws={workspace.world_size} "
            f"dtype={workspace.dtype} device={workspace.device}."
        )
    return M, K_local, N, world_size


def _current_cu_stream() -> cuda.CUstream:
    torch_stream = torch.cuda.current_stream()
    return cuda.CUstream(torch_stream.cuda_stream)


def _make_kernel(kernel_cls, config: BlackwellGemmRSConfig):
    return kernel_cls(
        cutlass.Float32,
        config.use_2cta_instrs,
        config.mma_tiler_mn,
        config.cluster_shape_mn,
        config.use_tma_store,
        reduce_scatter=config.reduce_scatter,
    )


def _compile_key(
    X_local: torch.Tensor,
    W_local: torch.Tensor,
    workspace: BlackwellGemmRSWorkspace,
    kernel_cls,
    effective_b_layout: Literal["staged", "nocopy"],
) -> tuple:
    config = workspace.config
    return (
        kernel_cls,
        tuple(X_local.shape),
        tuple(W_local.shape),
        tuple(workspace.c_full.shape),
        X_local.dtype,
        W_local.dtype,
        X_local.device.index,
        workspace.world_size,
        X_local.data_ptr(),
        W_local.data_ptr() if effective_b_layout == "nocopy" else 0,
        tuple(W_local.stride()) if effective_b_layout == "nocopy" else (),
        workspace.w_staging.data_ptr() if workspace.w_staging is not None else 0,
        workspace.c_full.data_ptr(),
        config.mma_tiler_mn,
        config.cluster_shape_mn,
        config.use_2cta_instrs,
        config.use_tma_store,
        config.reduce_scatter,
        effective_b_layout,
    )


def run_blackwell_gemm_rs_with_kernel(
    X_local: torch.Tensor,
    W_local: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    kernel_cls,
    workspace: BlackwellGemmRSWorkspace,
    stream: cuda.CUstream | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Run one Blackwell CUTLASS GEMM+RS call using a supplied kernel class.

    This is the bridge between FlashInfer caller tensors/workspace and the
    CUTLASS CuTeDSL `PersistentDenseGemmKernel`.  The kernel class is injected
    so the library does not import from benchmark files.  Once the kernel is
    extracted into FlashInfer, this function can call that class directly and
    grow a compile cache around it.
    """
    _require_world_group(group)
    _require_blackwell_deps()
    if torch.cuda.get_device_capability(X_local.device)[0] < 10:
        raise NotImplementedError("Blackwell GEMM+RS requires SM >= 100.")

    M, K_local, N, world_size = _validate_run_inputs(X_local, W_local, group, workspace)
    config = workspace.config
    effective_b_layout = _resolve_b_layout(W_local, config)
    workspace.last_b_layout = effective_b_layout
    if effective_b_layout == "staged":
        workspace.get_w_staging()
    profile = os.environ.get("FLASHINFER_GRS_PROFILE") == "1"
    workspace.reset()

    profile_copy_ms = None
    profile_kernel_ms = None
    profile_compile_ms = None
    key = _compile_key(X_local, W_local, workspace, kernel_cls, effective_b_layout)
    cache_hit = workspace._compiled_key == key and workspace._compiled_gemm is not None
    if profile:
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        copy_start.record()
    if cache_hit:
        if effective_b_layout == "staged":
            stage_b_tensor(W_local, staging=workspace.get_w_staging())
        a_tensor = workspace._compiled_a_tensor
        b_tensor = workspace._compiled_b_tensor
    else:
        a_tensor = make_a_tensor(X_local)
        if effective_b_layout == "staged":
            b_tensor, _ = make_b_tensor_staged(
                W_local, staging=workspace.get_w_staging()
            )
        else:
            b_tensor = make_b_tensor_nocopy(W_local)
    if profile:
        copy_end.record()
        torch.cuda.synchronize()
        profile_copy_ms = copy_start.elapsed_time(copy_end)
    if stream is None:
        stream = _current_cu_stream()

    if not cache_hit:
        gemm = _make_kernel(kernel_cls, config)
        if not gemm.can_implement(a_tensor, b_tensor, workspace.c_tensor):
            raise ValueError(
                "CUTLASS Blackwell GEMM+RS configuration is not implementable: "
                f"M={M} N={N} K_local={K_local} ws={world_size} "
                f"dtype={X_local.dtype} config={config}."
            )

        max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
            config.cluster_shape_mn[0] * config.cluster_shape_mn[1]
        )
        compile_t0 = time.perf_counter()
        workspace._compiled_gemm = cute.compile(
            gemm,
            a_tensor,
            b_tensor,
            workspace.c_tensor,
            max_active_clusters,
            stream,
            c_mc=workspace.c_tensor_mc,
            c_peer_tensors=workspace.c_peer_cute_tensors,
            barrier_flag=workspace.barrier_flag,
            barrier_flag_mc=workspace.barrier_flag_mc,
        )
        profile_compile_ms = (time.perf_counter() - compile_t0) * 1000.0
        workspace._compiled_key = key
        workspace._compiled_a_tensor = a_tensor
        workspace._compiled_b_tensor = b_tensor
    compiled_gemm = workspace._compiled_gemm
    if profile:
        kernel_start = torch.cuda.Event(enable_timing=True)
        kernel_end = torch.cuda.Event(enable_timing=True)
        kernel_start.record()
    compiled_gemm(
        a_tensor,
        b_tensor,
        workspace.c_tensor,
        stream,
        c_mc=workspace.c_tensor_mc,
        c_peer_tensors=workspace.c_peer_cute_tensors,
        barrier_flag=workspace.barrier_flag,
        barrier_flag_mc=workspace.barrier_flag_mc,
    )
    if profile:
        kernel_end.record()
        torch.cuda.synchronize()
        profile_kernel_ms = kernel_start.elapsed_time(kernel_end)
        print(
            "[run_blackwell_gemm_rs_with_kernel profile] "
            f"rank={dist.get_rank(group)} copy_ms={profile_copy_ms:.6f} "
            f"kernel_ms={profile_kernel_ms:.6f} "
            f"compiled={profile_compile_ms is not None} "
            f"compile_ms={profile_compile_ms}",
            flush=True,
        )
    if verbose and dist.get_rank(group) == 0:
        print(
            "[run_blackwell_gemm_rs_with_kernel] "
            f"M={M} N={N} K_local={K_local} ws={world_size} "
            f"dtype={X_local.dtype} config={config} "
            f"effective_b_layout={effective_b_layout}",
            flush=True,
        )
    return workspace.local_output


def gemm_reduce_scatter_blackwell_cutlass(
    X_local: torch.Tensor,
    W_local: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    verbose: bool = False,
    workspace: BlackwellGemmRSWorkspace | None = None,
) -> torch.Tensor:
    """Run FlashInfer GEMM+RS through the ported CUTLASS Blackwell kernel.

    This is the explicit SM100 performance path. Callers must pass a
    `BlackwellGemmRSWorkspace` because NVSHMEM allocation is collective and
    implicit caching would hide both collective ordering and memory ownership.
    The returned tensor aliases workspace storage and remains valid only until
    the next call using that workspace or until `workspace.destroy()`.
    """
    if workspace is None:
        raise ValueError(
            "backend='cutlass_blackwell' requires a BlackwellGemmRSWorkspace. "
            "The workspace owns the symmetric output storage returned by this call."
        )
    _require_world_group(group)

    try:
        from .cutlass_blackwell_gemm_rs import PersistentDenseGemmKernel
    except (ImportError, RuntimeError) as exc:
        raise BlackwellGemmRSUnavailableError(
            "backend='cutlass_blackwell' could not load the CuTe DSL kernel; "
            "verify nvidia-cutlass-dsl, cuda-python, nvshmem4py, and the "
            "NVSHMEM host library."
        ) from exc

    return run_blackwell_gemm_rs_with_kernel(
        X_local,
        W_local,
        group,
        kernel_cls=PersistentDenseGemmKernel,
        workspace=workspace,
        verbose=verbose,
    )
