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
# collective setup, and the current-stream context.

from __future__ import annotations

import functools
import importlib
import os
from pathlib import Path
from typing import Any

import torch

from ...jit import cute_dsl_core
from ...moe_ep.cake_mxfp8_megamoe_ep16 import CakeMxfp8MegaMoeEp16Weights
from . import backend as _backend


def _source_files() -> tuple[str, ...]:
    """Use one complete source key for every kernel in this op family."""

    package = Path(__file__).resolve().parent
    flashinfer = package.parent.parent
    paths = [
        package / "kernels" / "fused_cta0.py",
        package / "kernels" / "fused_all_ctas.py",
        package / "kernels" / "topk_reduce.py",
        package / "cute_dsl.py",
        package / "backend.py",
        package / "__init__.py",
        flashinfer / "moe_ep" / "cake_mxfp8_megamoe_ep16.py",
        Path(cute_dsl_core.__file__).resolve(),
    ]
    # Namespace packages need no initializer; include one if the distribution
    # supplies it, since importing a kernel then executes that source too.
    kernels_init = package / "kernels" / "__init__.py"
    if kernels_init.is_file():
        paths.append(kernels_init)
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"CuTe-DSL kernel source is unavailable: {path.name}")
    return tuple(str(path) for path in paths)


@functools.cache
def _load_kernels(device_index: int, fused_variant: str) -> tuple[Any, Any]:
    # The generated host functions select sm_103a explicitly. Do not let the
    # JIT cache label the same object with an incompatible environment target.
    arch = os.environ.get("CUTE_DSL_ARCH") or "sm_103a"
    if arch.replace("_", "") != "sm103a":
        raise ValueError("Cake CuTe-DSL requires CUTE_DSL_ARCH=sm_103a when set")
    sources = _source_files()
    with torch.cuda.device(device_index):
        if torch.cuda.get_device_capability() != (10, 3):
            raise RuntimeError("Cake CuTe-DSL requires compute capability 10.3")
        kernels = []
        for name in (fused_variant, "topk_reduce"):
            module = importlib.import_module(f"{__package__}.kernels.{name}")
            kernels.append(
                cute_dsl_core.build_and_load_cute_dsl_kernel(
                    "cake_mxfp8_megamoe_ep16",
                    name,
                    module.compile_program,
                    extra_key_files=sources,
                )
            )
    return tuple(kernels)


def _flat(tensor: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype != dtype or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"CuTe-DSL requires a contiguous CUDA {dtype} carrier")
    if tensor.data_ptr() % 16:
        raise ValueError("CuTe-DSL buffer addresses must be 16-byte aligned")
    return tensor.view(-1)


def _tma_4d(tensor: torch.Tensor, *, inner: int) -> tuple[Any, ...]:
    """Flatten a contiguous [expert, row, column] tensor and its TMA ABI.

    Descriptor dimensions are [inner, row, column / inner, expert]. The
    following three strides are in 16-byte units, not elements or bytes.
    """

    if tensor.ndim != 3 or tensor.shape[-1] % inner:
        raise ValueError("invalid four-dimensional TMA source geometry")
    experts, rows, columns = (int(size) for size in tensor.shape)
    if min(experts, rows, columns) <= 0:
        raise ValueError("TMA source dimensions must be positive")
    byte_strides = tuple(
        stride * tensor.element_size() for stride in (columns, inner, rows * columns)
    )
    if any(stride % 16 for stride in byte_strides):
        raise ValueError("TMA strides must be multiples of 16 bytes")
    return (
        _flat(tensor, dtype=tensor.dtype),
        inner,
        rows,
        columns // inner,
        experts,
        *(stride // 16 for stride in byte_strides),
    )


def _tma_scale(tensor: torch.Tensor) -> tuple[Any, ...]:
    # Packed scales use uint8 storage with a 128-byte innermost dimension.
    packed = tensor.view(-1, 128)
    return (_flat(packed, dtype=torch.uint8), 128, int(packed.shape[0]), 8)


def _peer_arguments(symm: Any) -> tuple[int, ...]:
    pointers = symm.peer_pointers
    if not isinstance(pointers, tuple) or len(pointers) != 32:
        raise ValueError("CuTe-DSL peer tables must contain 32 host addresses")
    if any(
        type(pointer) is not int or not 0 < pointer < 2**64 for pointer in pointers[:16]
    ):
        raise ValueError("EP16 requires 16 nonzero unsigned peer addresses")
    if any(type(pointer) is not int or pointer != 0 for pointer in pointers[16:]):
        raise ValueError("unused CuTe-DSL peer slots must be zero")
    # Preserve all address bits through the generated signed Int64 host ABI.
    return tuple(
        pointer if pointer < 2**63 else pointer - 2**64 for pointer in pointers
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
            raise ValueError("CuTe-DSL requires the 66-word owner-warp flag workspace")
        self._device = device
        self._tokens = tokens_per_rank
        # Keep the symmetric handles, original tensors, and all flat views alive.
        self._weights = weights
        self._workspace = workspace
        w = workspace
        fc1_weight = _tma_4d(weights.w13.view(torch.uint8), inner=128)
        fc2_weight = _tma_4d(weights.w2.view(torch.uint8), inner=128)
        activation = _tma_4d(w.activation_bf16, inner=64)
        fc1_workspace = _tma_4d(w.fc1_workspace_bf16, inner=64)
        # Physical host parameters between the three input carriers and epoch.
        # Each TMA parameter expands to its carrier, dimensions, then strides.
        self._before_epoch = (
            *fc1_weight,
            *_tma_scale(weights.w13_scale),
            *activation,
            *activation,  # N16 map has the same global dimensions and strides.
            activation[0],
            *fc2_weight,
            *_tma_scale(weights.w2_scale),
            *fc1_workspace,
            *fc1_workspace,
            fc1_workspace[0],
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
            *_peer_arguments(w.flags),
            _flat(w.published_hidden.tensor, dtype=torch.bfloat16),
            *_peer_arguments(w.published_hidden),
            _flat(w.published_topk_ids.tensor, dtype=torch.int32),
            *_peer_arguments(w.published_topk_ids),
            _flat(w.published_topk_weights.tensor, dtype=torch.float32),
            *_peer_arguments(w.published_topk_weights),
            route_terms,
            *_peer_arguments(w.route_terms),
            144,
            1,
            1,
        )
        if len(self._before_epoch) != 69 or len(self._after_epoch) != 170:
            raise RuntimeError(
                "CuTe-DSL fused physical argument layout is inconsistent"
            )
        self._output = w.output_bf16
        self._reduce_args = (
            route_terms,
            _flat(self._output, dtype=torch.bfloat16),
            tokens_per_rank,
            3 * tokens_per_rank,  # 384 eight-element tiles per token / 128 threads.
            1,
            1,
        )
        variant = "fused_all_ctas" if tokens_per_rank == 32 else "fused_cta0"
        self._fused, self._reduce = _load_kernels(device.index, variant)
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
        # peer-table copies or descriptor uploads.
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
