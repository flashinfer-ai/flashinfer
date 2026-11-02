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

# CuTe-DSL launch adapter for the fixed-shape EP16 session. Imported only for
# the explicit backend="cute_dsl" choice. Compilation and workspace bindings
# are prepared during session creation; the session owns validation,
# collective setup, and the current-stream context. The generated programs
# live beside the CUDA closure under csrc/, one per architecture and return
# protocol, and are compiled for the device's exact architecture through
# FlashInfer's CuTe JIT cache.

from __future__ import annotations

import functools
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from ...jit import cute_dsl_core
from ...moe_ep.cake_mxfp8_megamoe_ep16 import CakeMxfp8MegaMoeEp16Weights
from . import backend as _backend
from . import jit as _jit

# Host-tensor ranks the generated launch is compiled for, in its argument order:
# pointer-only carriers flat (1), the TMA sources [expert, row, column] (3) and
# the packed scale planes [n, 128] (2), then the published rows, ids, weights and
# route terms (1 each).
_FUSED_RANKS: tuple[int, ...] = (1, 1, 1, 3, 2, 3, 3, 1, 3, 2, 3, 3, 1, 1)
_FUSED_RANKS += (1,) * 14
_REDUCE_RANKS = (1, 1)
_FUSED_GRID = (144, 1, 1)  # 72 two-CTA clusters, all co-resident
_REDUCE_THREADS = 128  # eight hidden columns per thread
_PEER_CAPACITY = 32


_ROLES = ("fused", "fused_return_all_cta", "topk_reduce")


def _source_files(arch: str) -> tuple[str, ...]:
    """Use one complete source key for every kernel of this architecture."""

    package = Path(__file__).resolve().parent
    flashinfer = package.parent.parent
    paths = [
        *(_jit.cute_dsl_source(role, arch) for role in _ROLES),
        package / "cute_dsl.py",
        package / "backend.py",
        package / "jit.py",
        package / "__init__.py",
        flashinfer / "moe_ep" / "cake_mxfp8_megamoe_ep16.py",
        Path(cute_dsl_core.__file__).resolve(),
    ]
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"CuTe-DSL kernel source is unavailable: {path.name}")
    return tuple(str(path) for path in paths)


@functools.cache
def _import_source(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{__name__}._generated_{path.stem}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def _load_kernels(
    device_index: int, return_all_cta: bool
) -> tuple[Any, Any, ModuleType, ModuleType]:
    with torch.cuda.device(device_index):
        arch = _jit.device_arch(
            _jit.require_cute_dsl_capability(torch.cuda.get_device_capability())
        )
        # Each generated program fixes its target architecture. Do not let the
        # JIT cache label the same object with an incompatible target.
        requested = os.environ.get("CUTE_DSL_ARCH")
        if requested and requested.replace("_", "") != arch.replace("_", ""):
            raise ValueError(
                f"CUTE_DSL_ARCH={requested} does not match the device architecture {arch}"
            )
        sources = _source_files(arch)
        fused_role = "fused_return_all_cta" if return_all_cta else "fused"
        reducer_role = "topk_reduce"
        fused_module = _import_source(_jit.cute_dsl_source(fused_role, arch))
        reducer_module = _import_source(_jit.cute_dsl_source(reducer_role, arch))
        fused = cute_dsl_core.build_and_load_cute_dsl_kernel(
            "cake_mxfp8_megamoe_ep16",
            f"{fused_role}_{arch}",
            functools.partial(
                fused_module.compile_program,
                ranks=_FUSED_RANKS,
                aligned=(True,) * len(_FUSED_RANKS),
            ),
            extra_key_files=sources,
        )
        reducer = cute_dsl_core.build_and_load_cute_dsl_kernel(
            "cake_mxfp8_megamoe_ep16",
            f"{reducer_role}_{arch}",
            functools.partial(
                reducer_module.compile_program,
                ranks=_REDUCE_RANKS,
                aligned=(True,) * len(_REDUCE_RANKS),
            ),
            extra_key_files=sources,
        )
    return fused, reducer, fused_module, reducer_module


def _host_check(module: ModuleType) -> Any:
    names = [name for name in vars(module) if name.startswith("check_launch_")]
    if len(names) != 1:
        raise RuntimeError("generated CuTe-DSL program must define one host check")
    return getattr(module, names[0])


def _flat(tensor: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype != dtype or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"CuTe-DSL requires a contiguous CUDA {dtype} carrier")
    if tensor.data_ptr() % 16:
        raise ValueError("CuTe-DSL buffer addresses must be 16-byte aligned")
    return tensor.view(-1)


def _tma_source(tensor: torch.Tensor) -> torch.Tensor:
    """A contiguous [expert, row, column] TMA source, 16-byte aligned."""

    if tensor.ndim != 3 or min(int(size) for size in tensor.shape) <= 0:
        raise ValueError("invalid three-dimensional TMA source geometry")
    _flat(tensor, dtype=tensor.dtype)
    return tensor


def _scale_plane(tensor: torch.Tensor) -> torch.Tensor:
    # Packed scales use uint8 storage with a 128-byte innermost dimension.
    return _flat(tensor, dtype=torch.uint8).view(-1, 128)


def _peer_table(symm: Any) -> torch.Tensor:
    """The by-value peer table the generated launch reads on the host."""

    pointers = symm.peer_pointers
    if not isinstance(pointers, tuple) or len(pointers) != _PEER_CAPACITY:
        raise ValueError("CuTe-DSL peer tables must contain 32 host addresses")
    if any(
        type(pointer) is not int or not 0 < pointer < 2**64 for pointer in pointers[:16]
    ):
        raise ValueError("EP16 requires 16 nonzero unsigned peer addresses")
    if any(type(pointer) is not int or pointer != 0 for pointer in pointers[16:]):
        raise ValueError("unused CuTe-DSL peer slots must be zero")
    # Preserve all address bits through the signed int64 host table.
    return torch.tensor(
        [pointer if pointer < 2**63 else pointer - 2**64 for pointer in pointers],
        dtype=torch.int64,
    )


class _Runner:
    def __init__(
        self,
        *,
        device: torch.device,
        tokens_per_rank: int,
        rank: int,
        weights: CakeMxfp8MegaMoeEp16Weights,
        workspace: Any,
    ) -> None:
        if type(tokens_per_rank) is not int or tokens_per_rank not in (16, 32, 64):
            raise ValueError("CuTe-DSL supports 16, 32, or 64 tokens per rank")
        if type(rank) is not int or not 0 <= rank < 16:
            raise ValueError("CuTe-DSL requires an EP16 rank")
        if device.type != "cuda" or device.index is None:
            raise ValueError("CuTe-DSL requires an explicit CUDA device index")
        if tuple(workspace.flags.tensor.shape) != (66,):
            raise ValueError("CuTe-DSL requires the 66-word rank-barrier workspace")
        self._device = device
        self._tokens = tokens_per_rank
        # Keep the symmetric handles, original tensors, and all views alive.
        self._weights = weights
        self._workspace = workspace
        w = workspace
        activation = _tma_source(w.activation_bf16)
        fc1_workspace = _tma_source(w.fc1_workspace_bf16)
        # Launch arguments between the three per-call inputs and the epoch.
        self._before_epoch = (
            _tma_source(weights.w13.view(torch.uint8)),
            _scale_plane(weights.w13_scale),
            activation,
            activation,  # The N16 map reads the same carrier.
            _flat(activation, dtype=torch.bfloat16),
            _tma_source(weights.w2.view(torch.uint8)),
            _scale_plane(weights.w2_scale),
            fc1_workspace,
            fc1_workspace,
            _flat(fc1_workspace, dtype=torch.bfloat16),
            _flat(w.fc2_output_bf16, dtype=torch.bfloat16),
            _flat(w.route_map_i32, dtype=torch.int32),
            _flat(w.route_scale_f32, dtype=torch.float32),
            _flat(w.route_counts_u32, dtype=torch.uint32),
            _flat(w.fc1_done, dtype=torch.uint32),
            _flat(w.publication_done, dtype=torch.uint32),
            _flat(w.publication_visible, dtype=torch.uint32),
            _flat(w.dispatch_done, dtype=torch.uint32),
            _flat(w.compute_done, dtype=torch.uint32),
            _flat(w.return_done, dtype=torch.uint32),
            _flat(w.return_visible, dtype=torch.uint32),
        )
        route_terms = _flat(w.route_terms.tensor, dtype=torch.bfloat16)
        self._after_epoch = (
            tokens_per_rank,
            16,
            rank,
            _peer_table(w.flags),
            _flat(w.published_hidden.tensor, dtype=torch.bfloat16),
            _peer_table(w.published_hidden),
            _flat(w.published_topk_ids.tensor, dtype=torch.int32),
            _peer_table(w.published_topk_ids),
            _flat(w.published_topk_weights.tensor, dtype=torch.float32),
            _peer_table(w.published_topk_weights),
            route_terms,
            _peer_table(w.route_terms),
            *_FUSED_GRID,
        )
        self._output = w.output_bf16
        reduce_grid = tokens_per_rank * 3072 // 8 // _REDUCE_THREADS
        self._reduce_args = (
            route_terms,
            _flat(self._output, dtype=torch.bfloat16),
            tokens_per_rank,
            reduce_grid,
            1,
            1,
        )
        # 32 tokens per rank keep the all-CTA return protocol; 16 and 64 use the
        # CTA-0 coordinator. Each protocol is its own generated program.
        self._fused, self._reduce, fused_module, reducer_module = _load_kernels(
            device.index, tokens_per_rank == 32
        )
        # The generated host checks validate the bound geometry once and return
        # the signature (ranks, 16-byte alignment) the launch was compiled for.
        sample = (
            torch.empty((tokens_per_rank * 3072,), dtype=torch.bfloat16, device=device),
            torch.empty((tokens_per_rank * 8,), dtype=torch.int64, device=device),
            torch.empty((tokens_per_rank * 8,), dtype=torch.float32, device=device),
        )
        signature = _host_check(fused_module)(
            *sample, *self._before_epoch, 0, *self._after_epoch[:-3]
        )
        if signature != (_FUSED_RANKS, (True,) * len(_FUSED_RANKS)):
            raise RuntimeError("CuTe-DSL fused launch signature is inconsistent")
        signature = _host_check(reducer_module)(*self._reduce_args[:3])
        if signature != (_REDUCE_RANKS, (True,) * len(_REDUCE_RANKS)):
            raise RuntimeError("CuTe-DSL reduce launch signature is inconsistent")
        self._input_tensors: tuple[torch.Tensor, ...] = ()
        self._input_views: tuple[torch.Tensor, ...] = ()
        self._input_signature: tuple[Any, ...] = ()

    def run(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        launch_epoch: int,
        out: torch.Tensor,
    ) -> None:
        _backend._validate_launch_epoch(launch_epoch)
        if out.data_ptr() != self._output.data_ptr():
            raise ValueError("CuTe-DSL output must alias the session workspace")
        tensors = (hidden_states, topk_ids, topk_weights)
        signature = tuple(
            (tensor.data_ptr(), tensor.device, tensor.dtype) for tensor in tensors
        )
        if signature != self._input_signature:
            views = (
                _flat(hidden_states, dtype=torch.bfloat16),
                _flat(topk_ids, dtype=torch.int64),
                _flat(topk_weights, dtype=torch.float32),
            )
            self._input_tensors = tensors
            self._input_views = views
            self._input_signature = signature
        # The enclosing session supplies the TVM-FFI current Torch stream.
        # These calls allocate no device storage and perform no host-to-device
        # copies; the peer tables are read on the host.
        self._fused(
            *self._input_views,
            *self._before_epoch,
            launch_epoch,
            *self._after_epoch,
        )
        self._reduce(*self._reduce_args)


def create_runner(
    *,
    device: torch.device,
    tokens_per_rank: int,
    rank: int,
    weights: CakeMxfp8MegaMoeEp16Weights,
    workspace: Any,
) -> _Runner:
    """Prepare the generated kernels and allocation-free session bindings."""

    return _Runner(
        device=device,
        tokens_per_rank=tokens_per_rank,
        rank=rank,
        weights=weights,
        workspace=workspace,
    )
