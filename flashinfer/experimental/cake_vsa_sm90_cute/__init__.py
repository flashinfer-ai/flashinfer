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
"""CuTe DSL build of the Cake SM90 variable block-sparse attention kernels.

``backend="cake_cute"`` runs the same fifteen Hopper kernels as
``backend="cake"`` (persistent pair/split kernel plus the small-selection,
split-KV and cluster variants), rendered as CuTe DSL Python modules instead of
CUDA C++.  The planner, plan rows and launch geometry are shared with the CUDA
route (:mod:`flashinfer.cake_vsa_sm90`); only the kernel build differs:

* one module per stage under this package (``<stage>.py``), each exposing
  ``compile_program()`` which JIT-compiles the kernel with ``cute.compile``
  under the TVM-FFI host ABI;
* ``cake_vsa_sm90_cute_manifest.json`` records, per stage, the ordered argument
  plan, the TMA descriptor dimension/stride expressions of every TMA-loaded
  tensor, the by-value plan parameters bound as device buffers and the kernel's
  launch geometry.

The compiled entry takes flattened 1-D tensor views; for each TMA source the
descriptor's global dimensions and 16-byte strides follow as ``Int64``
arguments (evaluated here from the manifest expressions against the actual
tensor), then the scalar parameters, then ``grid_x/grid_y/grid_z``.  The launch
stream is the TVM-FFI environment stream (``tvm_ffi.use_torch_stream``).

One divergence from the CUDA build: the ``plan`` / ``hdr`` by-value kernel
parameters (constant-bank ``LDC`` reads in CUDA) are read-only device buffers
here (``LDG``), because the CuTe DSL exposes no addressable by-value array
parameter.  The planner uploads them once per plan; ``run`` waits on that
upload's event before the launch.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

_PACKAGE_DIR = Path(__file__).resolve().parent
_MANIFEST_NAME = "cake_vsa_sm90_cute_manifest.json"
_SCHEMA = "cake.cutedsl_export.v1"
_NAME = "cake_vsa_sm90_cute"
_ARCH = "sm_90a"

STAGES = (
    "attention",
    "attention_queue",
    "small_k1",
    "small_k3",
    "small_k4",
    "small_k6",
    "small_k1s",
    "small_k3s",
    "small_k4s",
    "small_k6s",
    "small_k2c4",
    "small_k3c3",
    "small_k3c6",
    "small_k4c2",
    "small_k4c4",
    "small_k6c2",
)

_TORCH_DTYPES = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
    "i16": torch.int16,
    "i32": torch.int32,
    "u32": torch.uint32,
    "u64": torch.uint64,
    "i64": torch.int64,
}


def is_available() -> bool:
    """``True`` when the CuTe DSL stack is importable (kernels are JIT-only)."""
    return (
        importlib.util.find_spec("cutlass") is not None
        and importlib.util.find_spec("cutlass.cute") is not None
    )


@functools.cache
def manifest() -> Dict[str, Any]:
    path = _PACKAGE_DIR / _MANIFEST_NAME
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != _SCHEMA
        or value.get("name") != _NAME
        or value.get("library") != "flashinfer"
        or value.get("arch") != _ARCH
    ):
        raise RuntimeError("invalid Cake SM90 VSA CuTe DSL manifest")
    modules = value.get("modules")
    if not isinstance(modules, list) or len(modules) != len(STAGES):
        raise RuntimeError("Cake SM90 VSA CuTe DSL manifest must list every stage")
    return value


def record(stage: str) -> Dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(
            f"unknown Cake SM90 VSA stage {stage!r}; expected one of {STAGES}"
        )
    matches = [item for item in manifest()["modules"] if item.get("stage") == stage]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one CuTe DSL module for stage {stage!r}, got {len(matches)}"
        )
    return matches[0]


def _module_path(item: Mapping[str, Any]) -> Path:
    relative = Path(str(item["path"]))
    path = _PACKAGE_DIR / relative.name
    if not path.is_file():
        raise FileNotFoundError(f"generated CuTe DSL source is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item.get("source_sha256"):
        raise RuntimeError(f"generated CuTe DSL source hash mismatch: {path.name}")
    return path


def _eval_dim(tree: Any, tensor: torch.Tensor) -> int:
    """Evaluate one manifest TMA expression tree against ``tensor``."""
    if isinstance(tree, bool):
        raise TypeError("TMA expression leaves are integers, not booleans")
    if isinstance(tree, int):
        return tree
    leaf = tree.get("leaf")
    if leaf is not None:
        index = int(tree["index"])
        if leaf == "axis":
            return int(tensor.size(index))
        if leaf == "stride":
            return int(tensor.stride(index))
        if leaf == "outer":
            trailing = 1
            for axis in range(1, index + 1):
                trailing *= int(tensor.size(-axis))
            return int(tensor.numel()) // trailing
        raise ValueError(f"unknown TMA expression leaf {leaf!r}")
    op = tree["op"]
    a = _eval_dim(tree["a"], tensor)
    b = _eval_dim(tree["b"], tensor)
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "floordiv":
        return a // b
    if op == "floormod":
        return a % b
    if op == "eq":
        return int(a == b)
    raise ValueError(f"unknown TMA expression operator {op!r}")


def tma_metadata(
    name: str, tma: Mapping[str, Any], tensor: torch.Tensor
) -> Tuple[int, ...]:
    """``(dims..., strides/16B...)`` of one TMA source, as the compiled entry expects."""
    if tensor.ndim < int(tma["min_source_rank"]):
        raise ValueError(
            f"{name} needs at least {tma['min_source_rank']} dimensions, got {tensor.ndim}"
        )
    if int(tensor.stride(-1)) != 1:
        raise ValueError(f"{name} must have unit innermost stride")
    for check in tma["checks"]:
        if not _eval_dim(check, tensor):
            raise ValueError(f"{name} failed a TMA descriptor check ({check})")
    dims = tuple(_eval_dim(expr, tensor) for expr in tma["global_dim"])
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"{name} resolved non-positive TMA dimensions {dims}")
    allow_oob = set(int(axis) for axis in tma["allow_oob_box"])
    for index, box in enumerate(tma["box_shape"]):
        if index not in allow_oob and int(box) > dims[index]:
            raise ValueError(
                f"{name}: TMA box {tuple(tma['box_shape'])} exceeds dimensions {dims}"
            )
    bits = int(tma["element_bits"])
    strides16 = []
    for axis, (expr, extent) in enumerate(
        zip(tma["global_strides"], dims[1:], strict=True), start=1
    ):
        stride = _eval_dim(expr, tensor)
        if stride < 0:
            raise ValueError(f"{name} resolved a negative TMA stride on axis {axis}")
        if stride == 0:
            if extent != 1:
                raise ValueError(
                    f"{name} resolved a zero TMA stride on axis {axis} with extent {extent}"
                )
            strides16.append(0)
            continue
        stride_bits = stride * bits
        if stride_bits % 128:
            raise ValueError(
                f"{name}: TMA stride on axis {axis} is not a multiple of 16 bytes"
            )
        strides16.append(stride_bits // 128)
    return (*dims, *strides16)


class CuteStage:
    """One compiled stage: argument marshalling per the manifest's plan."""

    def __init__(self, stage: str):
        self.stage = stage
        self.record = record(stage)
        self.arg_plan: Sequence[Tuple[str, str]] = [
            (str(kind), str(name)) for kind, name in self.record["arg_plan"]
        ]
        self.tma: Mapping[str, Any] = self.record["tma"]
        self.threads = int(self.record["threads"])
        self.cluster_dims = tuple(int(dim) for dim in self.record["cluster_dims"])
        self.dynamic_smem_bytes = int(self.record["dynamic_smem_bytes"])
        self.param_arrays: Mapping[str, Any] = self.record["param_array_device_buffers"]
        # By-value arrays kept in the kernel parameter space as Uint64 scalars.
        self.param_slots: Mapping[str, Any] = self.record["param_array_scalar_slots"]
        self._entry = None

    def _compile(self):
        if not is_available():
            raise RuntimeError(
                "backend='cake_cute' needs the CuTe DSL (pip install nvidia-cutlass-dsl)"
            )
        path = _module_path(self.record)
        spec = importlib.util.spec_from_file_location(
            f"{__name__}._generated_{self.stage}", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        factory = getattr(module, str(self.record["compile_factory"]))
        return factory()

    @property
    def entry(self):
        if self._entry is None:
            self._entry = self._compile()
        return self._entry

    def arguments(
        self, bindings: Mapping[str, Any], grid: Tuple[int, int, int]
    ) -> list:
        """Flatten ``bindings`` (kernel parameter name -> tensor / scalar) per the plan."""
        args: list = []
        grid_values = iter(grid)
        for kind, name in self.arg_plan:
            if kind == "grid":
                args.append(int(next(grid_values)))
                continue
            try:
                value = bindings[name]
            except KeyError as exc:
                raise KeyError(
                    f"stage {self.stage!r} needs kernel argument {name!r}"
                ) from exc
            if kind == "buffer":
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{name} must be a torch.Tensor")
                if not value.is_cuda or not value.is_contiguous():
                    raise ValueError(f"{name} must be a contiguous CUDA tensor")
                if name in self.param_arrays:
                    spec = self.param_arrays[name]
                    if value.dtype != _TORCH_DTYPES[
                        str(spec["dtype"])
                    ] or value.numel() != int(spec["length"]):
                        raise ValueError(
                            f"{name} must be a {spec['dtype']} device buffer of {spec['length']} elements"
                        )
                args.append(value.view(-1))
                if name in self.tma:
                    args.extend(tma_metadata(name, self.tma[name], value))
            elif kind == "scalar":
                args.append(value)
            elif kind == "slots":
                args.extend(self.slot_words(name, value))
            else:
                raise NotImplementedError(
                    f"argument kind {kind!r} is not used by this route"
                )
        return args

    def slot_words(self, name: str, value: Any) -> list:
        """A by-value array as its little-endian Uint64 scalar arguments (payload zero padded)."""
        spec = self.param_slots[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a CPU torch.Tensor")
        if value.is_cuda or value.dtype != _TORCH_DTYPES[str(spec["dtype"])]:
            raise TypeError(
                f"{name} must be a CPU {spec['dtype']} tensor of {spec['length']} elements"
            )
        if value.numel() != int(spec["length"]):
            raise ValueError(
                f"{name} must be a CPU {spec['dtype']} tensor of {spec['length']} elements"
            )
        payload = value.contiguous().view(-1).numpy().tobytes()
        slots = int(spec["slots"])
        payload += b"\0" * (slots * 8 - len(payload))
        return [
            int.from_bytes(payload[8 * i : 8 * i + 8], "little", signed=True)
            for i in range(slots)
        ]

    def run(self, bindings: Mapping[str, Any], grid: Tuple[int, int, int]) -> None:
        """Launch on the current torch stream (graph-capturable)."""
        import tvm_ffi

        entry = self.entry
        args = self.arguments(bindings, grid)
        with tvm_ffi.use_torch_stream():
            entry(*args)


@functools.cache
def load_stage(stage: str) -> CuteStage:
    return CuteStage(stage)


def compile_all() -> Dict[str, int]:
    """Compile every stage (warm-up / CI); returns dynamic SMEM per stage."""
    result = {}
    for stage in STAGES:
        loaded = load_stage(stage)
        loaded.entry  # noqa: B018 - compiles
        result[stage] = loaded.dynamic_smem_bytes
    return result


__all__ = [
    "STAGES",
    "CuteStage",
    "compile_all",
    "is_available",
    "load_stage",
    "manifest",
    "record",
    "tma_metadata",
]
