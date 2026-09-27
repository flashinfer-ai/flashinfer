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

# Experimental Cake backend: the Kimi-K3 TP12 fused LatentMoE communication
# tail for SM100 / SM103 twelve-rank multi-node NVLink domains (GB200 /
# GB300 NVL72; flashinfer-ai/flashinfer#4542, tracker #4254).
#
# Per tensor-parallel rank ``r`` of twelve, ``M`` tokens, BF16::
#
#     latent = KimiRMSNorm(sum_r routed_partial_r)                  [M, 3584]
#     out    = BF16(latent @ up_weight.T + sum_r shared_partial_r)  [M, 7168]
#
# ``out`` is replicated and bitwise identical on every rank.  The fused tail
# is three launches per rank on the current stream (CUDA-Graph capturable):
#
# * ``K1``  Lamport all-reduce of ``routed_partial`` fused with KimiRMSNorm ->
#   the normalised latent ``y`` on every rank (one-shot for ``M <= 16``, the
#   token-sliced two-shot form above);
# * ``K2``  the up-projection of ``y`` with this rank's contiguous 640 / 512-row
#   slice of ``up_weight`` (``7168 = 8 x 640 + 4 x 512``: 7168 is not divisible
#   by 12): the generated K2-stream kernel (SIMT weight streaming, fp32 slice)
#   for ``M <= K2_STREAM_MAX_TOKENS``, cuBLAS (``torch.mm``, BF16) above;
# * ``K3``  column reduce-scatter of ``shared_partial`` to the owner rank, add of
#   the up-projection slice, one BF16 rounding, multicast all-gather into the
#   caller-owned ``out``; one CTA per token below ``K3_PERSIST_MIN_TOKENS``, the
#   persistent token pipeline (``min(M, SM count)`` CTAs per column half) above.
#
# The three Lamport workspaces are FlashInfer's own MNNVL all-reduce
# workspaces (``MNNVLAllReduceFusionWorkspace``: three rotating symmetric
# buffers initialised to negative zero, multicast mapping, nine-word flags),
# created once per process group by :class:`KimiK3Tp12TailWorkspace`.
# The host runtime here mirrors the Cake TP12 tail module
# (``kimi_k3_tp12_tail.py``); the generated sources live in ``csrc/``.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_jit import (
    KERNELS,
    MODULES,
    kernel_module_name,
    load_cake_kimi_k3_tp12_tail_module,
    route_available,
)

HIDDEN = 7168
LATENT = 3584
WORLD_SIZE = 12
RMS_EPS = 1.0e-5
ELEMENT_BYTES = 2  # BF16 payload elements
#: Output-column widths per rank (rank order), 128-column granular.
PARTITION = (640,) * 8 + (512,) * 4
#: One-shot K1 up to this many tokens, two-shot above.
ONESHOT_MAX_TOKENS = 16
#: Grouped owner poll-reduce loads up to this many tokens, pinned above.
GROUPED_MAX_TOKENS = 256
#: One 16-byte packet per thread covers one 3584-wide row (K1) or one 3584-wide half row (K3).
THREADS = 448
K3_GRID_Y = 2
#: K3 runs the persistent token pipeline (``k3_persist:*``, grid ``(min(M, SM count), 2, 1)``: CTA ``b`` walks tokens
#: ``b, b + P, ...`` scattering token ``t_k`` while reducing ``t_{k-1}`` and gathering ``t_{k-2}``) from this many tokens
#: on; below it the one-CTA-per-token form (``k3:*``, grid ``(M, 2, 1)``) is faster.  Pinned by the round-4 A/B on both
#: NVL72 racks (identical kernels, buffers, flags and numerics; only the CTA -> token mapping differs).
K3_PERSIST_MIN_TOKENS = 256
#: K2 runs the Cake SIMT weight-streaming slice GEMM (``k2_stream:n<cols>``: 128 CTAs x 448 threads stream the rank's
#: 4.6 MB BF16 weight slice once from HBM, fp32 output) and K3 its fp32-add form (``k3_f32:grouped``) for
#: ``M <= K2_STREAM_MAX_TOKENS``; above, K2 is cuBLAS (``torch.mm`` on the contiguous weight-row slice, BF16) and K3 the
#: BF16 form.  Pinned by the round-4 paired A/B on both racks (-6 .. -11 % at M = 1, 2, 4; tie at 8; slower at 16).
K2_STREAM_MAX_TOKENS = 4
K2_STREAM_GRID_X = 128
DEFAULT_MAX_TOKENS = 4096
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}


def col_begin(partition: tuple[int, ...] = PARTITION) -> tuple[int, ...]:
    """Prefix sums of the column partition: 13 boundaries, the last one 7168."""
    if len(partition) != WORLD_SIZE or sum(partition) != HIDDEN:
        raise ValueError(
            f"partition must have {WORLD_SIZE} widths summing to {HIDDEN}, got {partition}"
        )
    bounds = [0]
    for width in partition:
        bounds.append(bounds[-1] + width)
    return tuple(bounds)


COL_BEGIN = col_begin(PARTITION)


def poll_schedule_for(num_tokens: int) -> str:
    """Owner poll-reduce schedule of the two-shot K1 and of K3 for ``num_tokens``."""
    return "grouped" if num_tokens <= GROUPED_MAX_TOKENS else "pinned"


def k1_kernel_key(num_tokens: int, rank: int) -> str:
    if num_tokens <= ONESHOT_MAX_TOKENS:
        return f"k1_oneshot:r{rank}"
    return f"k1_twoshot:{poll_schedule_for(num_tokens)}"


def k2_form_for(num_tokens: int) -> str:
    """``"stream"`` (Cake K2-stream kernel, fp32 slice) for ``num_tokens <= K2_STREAM_MAX_TOKENS``, else ``"cublas"``."""
    return "stream" if num_tokens <= K2_STREAM_MAX_TOKENS else "cublas"


def k2_kernel_key(num_tokens: int, rank: int) -> Optional[str]:
    """The K2 kernel key of ``rank`` (its column width selects the module), or ``None`` when K2 is cuBLAS."""
    if k2_form_for(num_tokens) == "stream":
        return f"k2_stream:n{PARTITION[rank]}"
    return None


def k3_form_for(num_tokens: int) -> str:
    """``"persist"`` (persistent token pipeline) for ``num_tokens >= K3_PERSIST_MIN_TOKENS``, else ``"lamport"``."""
    return "persist" if num_tokens >= K3_PERSIST_MIN_TOKENS else "lamport"


def k3_kernel_key(num_tokens: int) -> str:
    if k2_form_for(num_tokens) == "stream":
        # fp32 GEMM slice from K2-stream: the fp32-add K3 (one-CTA-per-token, grouped poll schedule; M <= 4 < 256)
        return f"k3_f32:{poll_schedule_for(num_tokens)}"
    prefix = "k3_persist" if k3_form_for(num_tokens) == "persist" else "k3"
    return f"{prefix}:{poll_schedule_for(num_tokens)}"


def k3_grid(num_tokens: int, sm_count: int) -> tuple[int, int, int]:
    """Launch grid of K3 for ``num_tokens`` on a device with ``sm_count`` SMs (both forms, two column halves)."""
    if not isinstance(sm_count, int) or sm_count <= 0:
        raise ValueError(f"sm_count must be a positive integer, got {sm_count!r}")
    if k3_form_for(num_tokens) == "persist":
        return (min(num_tokens, sm_count), K3_GRID_Y, 1)
    return (num_tokens, K3_GRID_Y, 1)


def route_kernel_keys(num_tokens: int, rank: int) -> tuple[str, ...]:
    """The kernel keys the runtime launches for ``num_tokens`` on ``rank``, in launch order: ``(K1, K3)`` when K2 is
    cuBLAS, ``(K1, K2, K3)`` when K2 is the Cake K2-stream kernel (``num_tokens <= K2_STREAM_MAX_TOKENS``)."""
    k2 = k2_kernel_key(num_tokens, rank)
    if k2 is None:
        return k1_kernel_key(num_tokens, rank), k3_kernel_key(num_tokens)
    return k1_kernel_key(num_tokens, rank), k2, k3_kernel_key(num_tokens)


def workspace_buffer_bytes(max_tokens: int) -> dict[str, int]:
    """Bytes of one Lamport buffer (of three) per workspace for up to ``max_tokens`` tokens."""
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"max_tokens must be a positive integer, got {max_tokens!r}")
    oneshot_tokens = min(ONESHOT_MAX_TOKENS, max_tokens)
    twoshot_tokens = math.ceil(max_tokens / WORLD_SIZE) * WORLD_SIZE
    return {
        # every rank receives the other ranks' packets of every token
        "k1_oneshot": oneshot_tokens * LATENT * WORLD_SIZE * ELEMENT_BYTES,
        # scatter stage (ceil(M / 12) x 12 x 3584) + broadcast stage (M x 3584)
        "k1_twoshot": 2 * twoshot_tokens * LATENT * ELEMENT_BYTES,
        # scatter stage (M x 12 x n_max) + broadcast stage (M x 7168 <= M x 12 x n_max)
        "k3": 2 * max_tokens * WORLD_SIZE * max(PARTITION) * ELEMENT_BYTES,
    }


def _device_arch(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "the Kimi-K3 TP12 tail programs require compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    return arch


def generated_program_available(device: torch.device) -> bool:
    """True when every kernel of the route is registered for ``device``'s architecture."""
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        return False
    from .cake_jit import required_kernel_keys

    return route_available(arch, required_kernel_keys())


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class KimiK3Tp12TailWorkspace:
    """Caller-owned state of one twelve-rank process group.

    Three FlashInfer MNNVL Lamport workspaces (one-shot K1, two-shot K1, K3),
    the per-workspace unicast pointer tables, the normalised-latent and the
    up-projection slice buffers for up to ``max_tokens`` tokens.  Create it
    once (collective: every rank of ``group`` must call it), reuse it for every
    token count up to ``max_tokens``, and :meth:`destroy` it before the
    process group is destroyed.
    """

    def __init__(
        self,
        *,
        rank: int,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        group: Any = None,
        device: Optional[torch.device] = None,
        comm_backend: Any = None,
    ) -> None:
        import torch.distributed as dist

        from ...comm.comm_backend import TorchDistBackend
        from ...comm.mapping import Mapping
        from ...comm.trtllm_mnnvl_ar import MNNVLAllReduceFusionWorkspace

        if not isinstance(rank, int) or not 0 <= rank < WORLD_SIZE:
            raise ValueError(
                f"rank must be an integer in [0, {WORLD_SIZE}), got {rank!r}"
            )
        if comm_backend is None:
            if not dist.is_initialized():
                raise RuntimeError(
                    "KimiK3Tp12TailWorkspace needs an initialised torch.distributed "
                    "process group (or an explicit comm_backend)"
                )
            comm_backend = TorchDistBackend(group)
        world = int(comm_backend.Get_size())
        if world != WORLD_SIZE:
            raise ValueError(
                f"the Kimi-K3 TP12 tail needs a {WORLD_SIZE}-rank group, got {world}"
            )
        if int(comm_backend.Get_rank()) != rank:
            raise ValueError(
                f"rank {rank} does not match the process group rank {comm_backend.Get_rank()}"
            )
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.index is not None:
            torch.cuda.set_device(device.index)
        self.device = torch.device("cuda", torch.cuda.current_device())
        # persistent K3 grid: one CTA per SM and column half (the Cake runtime's default k3p_ctas)
        self.sm_count = int(torch.cuda.get_device_properties(self.device).multi_processor_count)
        self.rank = rank
        self.world_size = WORLD_SIZE
        self.max_tokens = int(max_tokens)
        self.arch = _device_arch(self.device)
        self.my_col_begin = COL_BEGIN[rank]
        self.my_cols = PARTITION[rank]
        mapping = Mapping(world_size=WORLD_SIZE, rank=rank, tp_size=WORLD_SIZE)
        self.buffers: dict[str, MNNVLAllReduceFusionWorkspace] = {}
        self.peer_ptrs: dict[str, torch.Tensor] = {}
        self._destroyed = False
        try:
            for name, size in workspace_buffer_bytes(self.max_tokens).items():
                ws = MNNVLAllReduceFusionWorkspace(
                    mapping, buffer_size_in_bytes=size, comm_backend=comm_backend
                )
                self.buffers[name] = ws
                self.peer_ptrs[name] = torch.tensor(
                    [int(p) for p in ws.ptrs], dtype=torch.int64, device=self.device
                )
            self.y = torch.empty(
                (self.max_tokens, LATENT), dtype=torch.bfloat16, device=self.device
            )
            self.gemm = torch.empty(
                (self.max_tokens, self.my_cols),
                dtype=torch.bfloat16,
                device=self.device,
            )
            # fp32 up-projection slice of the K2-stream route (M <= K2_STREAM_MAX_TOKENS)
            self.gemm_f32 = torch.empty(
                (min(self.max_tokens, K2_STREAM_MAX_TOKENS), self.my_cols),
                dtype=torch.float32,
                device=self.device,
            )
        except Exception:
            self.destroy()
            raise
        torch.cuda.synchronize(self.device)
        comm_backend.barrier()

    def destroy(self) -> None:
        """Release the symmetric buffers (collective teardown order is the caller's)."""
        if self._destroyed:
            return
        self._destroyed = True
        for ws in list(self.buffers.values()):
            ws.destroy()
        self.buffers.clear()
        self.peer_ptrs.clear()

    def flags(self, name: str) -> torch.Tensor:
        return self.buffers[name].buffer_flags

    def multicast_ptr(self, name: str) -> int:
        return int(self.buffers[name].mc_ptr)

    def local_unicast_ptr(self, name: str) -> int:
        return int(self.buffers[name].uc_ptr_local)


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------


def _bind(module_name: str, kwargs: dict[str, Any]) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the generated argument plan of ``module_name`` and load its entry."""
    record = MODULES[module_name]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments = []
    for kind, name in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid[name])
        elif name in kwargs:
            arguments.append(kwargs[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} ({kind}); "
                f"host binding provides {sorted(kwargs)}"
            )
    module = load_cake_kimi_k3_tp12_tail_module(module_name)
    return getattr(module, record["ffi_entry"]), tuple(arguments)


@dataclass(frozen=True)
class _Launch:
    stage: str
    key: str
    module: str
    kwargs: dict[str, Any] = field(repr=False)
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)

    def __call__(self) -> None:
        self.entry(*self.arguments)


@dataclass(frozen=True)
class KimiK3Tp12TailRunner:
    """The prepared launch sequence of one fused tail call on one rank.

    ``launch()`` submits K1, the up-projection slice (the K2-stream kernel or
    cuBLAS) and K3 in order on
    the current torch stream with no CUDA allocation and no host
    synchronisation; the kernels read every operand on device at launch, so
    the runner (or a CUDA Graph capturing it) replays for new values written
    into the same buffers.  Every rank of the group must launch the same
    token count; prepare a new runner when a shape or a tensor binding
    changes.
    """

    num_tokens: int
    rank: int
    arch: str
    k1: _Launch
    k3: _Launch
    y: torch.Tensor
    gemm: torch.Tensor
    up_weight_slice: torch.Tensor
    out: torch.Tensor
    k2: Optional[_Launch] = None  # Cake K2-stream kernel (M <= K2_STREAM_MAX_TOKENS); None = cuBLAS torch.mm

    @property
    def kernel_keys(self) -> tuple[str, ...]:
        """Generated-kernel keys in launch order (``(K1, K3)`` or ``(K1, K2, K3)``)."""
        if self.k2 is None:
            return (self.k1.key, self.k3.key)
        return (self.k1.key, self.k2.key, self.k3.key)

    @property
    def module_names(self) -> tuple[str, ...]:
        if self.k2 is None:
            return (self.k1.module, self.k3.module)
        return (self.k1.module, self.k2.module, self.k3.module)

    @property
    def launch_count(self) -> int:
        return 3

    def launch(self) -> torch.Tensor:
        with tvm_ffi.use_torch_stream():
            self.k1()
            if self.k2 is None:
                torch.mm(self.y, self.up_weight_slice.t(), out=self.gemm)
            else:
                self.k2()
            self.k3()
        return self.out

    __call__ = launch


def _check(
    t: torch.Tensor,
    shape: tuple[int, ...],
    name: str,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous {dtype} tensor of shape {tuple(shape)}, "
            f"got {tuple(t.shape)} {t.dtype}"
        )


def prepare_kimi_k3_tp12_tail(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    workspace: KimiK3Tp12TailWorkspace,
) -> KimiK3Tp12TailRunner:
    """Validate the operands, select the route for ``M`` and bind the launches.

    The JIT modules are built and loaded here; prepare outside CUDA Graph
    capture.  ``up_weight`` is the replicated ``[7168, 3584]`` BF16 weight;
    the rank's slice is a contiguous row-slice view (no copy).
    """
    if not isinstance(workspace, KimiK3Tp12TailWorkspace):
        raise TypeError("workspace must be a KimiK3Tp12TailWorkspace")
    if workspace._destroyed:
        raise RuntimeError("the workspace has been destroyed")
    M = int(routed_partial.shape[0]) if routed_partial.dim() == 2 else -1
    if M <= 0:
        raise ValueError("routed_partial must be a [M, 3584] tensor with M >= 1")
    if workspace.max_tokens < M:
        raise ValueError(
            f"M={M} exceeds the workspace capacity max_tokens={workspace.max_tokens}"
        )
    _check(routed_partial, (M, LATENT), "routed_partial")
    _check(shared_partial, (M, HIDDEN), "shared_partial")
    _check(norm_weight, (LATENT,), "norm_weight")
    _check(up_weight, (HIDDEN, LATENT), "up_weight")
    _check(out, (M, HIDDEN), "out")
    tensors = dict(
        routed_partial=routed_partial,
        shared_partial=shared_partial,
        norm_weight=norm_weight,
        up_weight=up_weight,
        out=out,
    )
    devices = {t.device for t in tensors.values()}
    if len(devices) != 1 or next(iter(devices)) != workspace.device:
        raise ValueError(
            f"every operand must live on the workspace device {workspace.device}, "
            f"got {sorted(map(str, devices))}"
        )
    arch = workspace.arch
    rank = workspace.rank
    keys = route_kernel_keys(M, rank)
    k1_key, k3_key = keys[0], keys[-1]
    k2_key = keys[1] if len(keys) == 3 else None
    k1_module = kernel_module_name(arch, k1_key)
    k3_module = kernel_module_name(arch, k3_key)
    y = workspace.y[:M]
    # K3 reads the fp32 slice of the K2-stream kernel below K2_STREAM_MAX_TOKENS, cuBLAS's BF16 slice above
    gemm = workspace.gemm_f32[:M] if k2_key is not None else workspace.gemm[:M]
    up_weight_slice = up_weight[
        workspace.my_col_begin : workspace.my_col_begin + workspace.my_cols
    ]
    if k1_key.startswith("k1_oneshot:"):
        name = "k1_oneshot"
        k1_kwargs: dict[str, Any] = dict(
            routed=routed_partial,
            y_out=y,
            gamma=norm_weight,
            mcast_ptr=workspace.multicast_ptr(name),
            local_unicast_ptr=workspace.local_unicast_ptr(name),
            buffer_flags=workspace.flags(name),
            num_tokens=M,
            epsilon=float(RMS_EPS),
            grid=(M, 1, 1),
        )
    else:
        name = "k1_twoshot"
        k1_kwargs = dict(
            routed=routed_partial,
            y_out=y,
            gamma=norm_weight,
            peer_ptrs=workspace.peer_ptrs[name],
            mcast_ptr=workspace.multicast_ptr(name),
            buffer_flags=workspace.flags(name),
            num_tokens=M,
            rank=rank,
            epsilon=float(RMS_EPS),
            grid=(M, 1, 1),
        )
    k3_kwargs: dict[str, Any] = dict(
        shared=shared_partial,
        gemm_slice=gemm,
        out=out,
        peer_ptrs=workspace.peer_ptrs["k3"],
        mcast_ptr=workspace.multicast_ptr("k3"),
        buffer_flags=workspace.flags("k3"),
        num_tokens=M,
        rank=rank,
        my_col_begin=workspace.my_col_begin,
        my_cols=workspace.my_cols,
        gemm_plane_stride=M * workspace.my_cols,
        num_gemm_splits=1,
        grid=k3_grid(M, workspace.sm_count),
    )
    k1_entry, k1_args = _bind(k1_module, k1_kwargs)
    k3_entry, k3_args = _bind(k3_module, k3_kwargs)
    k2_launch = None
    if k2_key is not None:
        k2_module = kernel_module_name(arch, k2_key)
        k2_kwargs: dict[str, Any] = dict(
            y=y,
            w_slice=up_weight_slice,
            out=gemm,
            num_tokens=M,
            grid=(K2_STREAM_GRID_X, 1, 1),
        )
        k2_entry, k2_args = _bind(k2_module, k2_kwargs)
        k2_launch = _Launch("k2", k2_key, k2_module, k2_kwargs, k2_entry, k2_args)
    return KimiK3Tp12TailRunner(
        num_tokens=M,
        rank=rank,
        arch=arch,
        k1=_Launch("k1", k1_key, k1_module, k1_kwargs, k1_entry, k1_args),
        k3=_Launch("k3", k3_key, k3_module, k3_kwargs, k3_entry, k3_args),
        y=y,
        gemm=gemm,
        up_weight_slice=up_weight_slice,
        out=out,
        k2=k2_launch,
    )


__all__ = [
    "COL_BEGIN",
    "DEFAULT_MAX_TOKENS",
    "GROUPED_MAX_TOKENS",
    "K2_STREAM_MAX_TOKENS",
    "K3_PERSIST_MIN_TOKENS",
    "HIDDEN",
    "KERNELS",
    "KimiK3Tp12TailRunner",
    "KimiK3Tp12TailWorkspace",
    "LATENT",
    "MODULES",
    "ONESHOT_MAX_TOKENS",
    "PARTITION",
    "RMS_EPS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "THREADS",
    "WORLD_SIZE",
    "col_begin",
    "generated_program_available",
    "k1_kernel_key",
    "k2_form_for",
    "k2_kernel_key",
    "k3_form_for",
    "k3_grid",
    "k3_kernel_key",
    "poll_schedule_for",
    "prepare_kimi_k3_tp12_tail",
    "route_kernel_keys",
    "workspace_buffer_bytes",
]
