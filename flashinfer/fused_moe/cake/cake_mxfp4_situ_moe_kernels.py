#
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Kernel loader of the ``backend="cake"`` (CUDA C++) package of the Cake MXFP4 SiTU routed MoE.

The generated CUDA translation units and their ``tvm_ffi_utils`` bindings live under
``csrc/fused_moe/cake_mxfp4_situ_moe/cuda`` and are inventoried by the content-addressed manifest
``csrc/fused_moe/cake_mxfp4_situ_moe/cake_mxfp4_situ_moe_manifest.json`` (schema ``cake.library_export.v5``).  This module authenticates
every source against the manifest before it is compiled, JIT-builds one module per
launched form through FlashInfer's TVM-FFI JIT, and marshals named launch bindings
into the module's positional argument plan.  Kernels are selected by the trace-time
constants recorded in ``route.form`` (never by file name).

The CUDA modules keep the pointer TMA ABI of the validated production build: every
launch receives a caller-owned, 128-byte aligned descriptor workspace
(``tma_workspace_bytes`` per module) that the binding fills before the kernel runs.
:class:`Kernel.bind` allocates that workspace once per bound kernel (outside CUDA
graph capture) so ``launch`` itself performs no allocation.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm103a_nvcc_flags

BACKEND = "cake"
PACKAGE_NAME = "cake_mxfp4_situ_moe"
GENERATED_ROOT = "csrc/fused_moe/cake_mxfp4_situ_moe/cuda"
MANIFEST_RELPATH = (
    "csrc/fused_moe/cake_mxfp4_situ_moe/cake_mxfp4_situ_moe_manifest.json"
)
ARCH = "sm_103a"
_SCHEMA = "cake.library_export.v5"
_TENSOR_KINDS = frozenset({"buffer", "tma_buffer"})


def is_available() -> bool:
    """``True`` when the manifest is installed (the kernels are JIT-built on first use)."""
    try:
        _manifest_path()
    except FileNotFoundError:
        return False
    return True


def _csrc_dir() -> Path:
    """Installed ``flashinfer/data/csrc`` or the source checkout's ``csrc``."""
    relative = MANIFEST_RELPATH.removeprefix("csrc/")
    installed = jit_env.FLASHINFER_CSRC_DIR
    if (installed / relative).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3] / "csrc"
    if (checkout / relative).is_file():
        return checkout
    raise FileNotFoundError(
        f"{PACKAGE_NAME}: manifest {MANIFEST_RELPATH} is not installed"
    )


def _manifest_path() -> Path:
    return _csrc_dir() / MANIFEST_RELPATH.removeprefix("csrc/")


@functools.cache
def manifest() -> dict[str, Any]:
    """The validated manifest of this package."""
    value = json.loads(_manifest_path().read_text(encoding="utf-8"))
    contract = value.get("contract", {})
    if (
        value.get("schema") != _SCHEMA
        or value.get("producer") != "cake"
        or value.get("library") != "flashinfer"
        or value.get("name") != PACKAGE_NAME
        or contract.get("backend") != BACKEND
        or contract.get("arch") != ARCH
    ):
        raise RuntimeError(f"invalid {PACKAGE_NAME} manifest")
    modules = value.get("modules")
    if not isinstance(modules, list) or not modules:
        raise RuntimeError(f"empty {PACKAGE_NAME} module inventory")
    return value


def cake_revision() -> str:
    """The Cake revision both packages were rendered from."""
    return str(manifest()["producer_revision"])


def modules() -> list[dict[str, Any]]:
    return [dict(item) for item in manifest()["modules"] if item.get("arch") == ARCH]


def unsupported_forms() -> dict[str, str]:
    """Forms of the family this backend does not build, with the recorded reason."""
    return dict(manifest()["contract"].get("unsupported_forms", {}))


def source_path(relative_path: str) -> Path:
    """Authenticated absolute path of one generated source (SHA-256 checked against the manifest)."""
    inventory = {item["path"]: item for item in manifest()["files"]}
    receipt = inventory.get(relative_path)
    if not isinstance(receipt, dict):
        raise FileNotFoundError(
            f"generated source is absent from the manifest: {relative_path}"
        )
    path = _csrc_dir() / relative_path.removeprefix("csrc/")
    if not path.is_file():
        raise FileNotFoundError(f"generated source is not installed: {relative_path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != receipt.get("sha256"):
        raise RuntimeError(f"generated source hash mismatch: {relative_path}")
    return path


def record(stage: str) -> dict[str, Any]:
    """The manifest module whose ``route.stage`` is ``stage``."""
    matches = [
        item for item in modules() if dict(item.get("route", {})).get("stage") == stage
    ]
    if len(matches) != 1:
        raise KeyError(
            f"{PACKAGE_NAME}: expected one module for stage {stage!r}, found {len(matches)}"
        )
    return matches[0]


def find_form(**form: Any) -> dict[str, Any]:
    """The unique module whose ``route.form`` carries every given key with the given value."""
    matches = [
        item
        for item in modules()
        if all(
            dict(item["route"].get("form", {})).get(key) == value
            for key, value in form.items()
        )
    ]
    if len(matches) != 1:
        raise KeyError(
            f"{PACKAGE_NAME}: {len(matches)} modules match form {form}; "
            + (
                "the form is not in this package"
                if not matches
                else "the selection is ambiguous"
            )
        )
    return matches[0]


class BoundKernel:
    """One kernel bound to a device: owns the descriptor workspace and launches with named bindings."""

    def __init__(self, kernel: "Kernel", device: torch.device):
        self.kernel = kernel
        self.device = device
        self.tma_workspace = (
            torch.empty(kernel.tma_workspace_bytes, dtype=torch.uint8, device=device)
            if kernel.tma_workspace_bytes
            else None
        )

    def launch(self, grid: tuple[int, int, int], **bindings: Any) -> None:
        """Launch on the current torch stream (graph-capturable; no allocation)."""
        self.kernel.run(grid, tma_workspace=self.tma_workspace, **bindings)


class Kernel:
    """One generated module: JIT-built on first launch, named bindings marshalled per the argument plan."""

    def __init__(self, item: Mapping[str, Any]):
        self.record = dict(item)
        self.stage: str = str(self.record["route"]["stage"])
        self.form: dict[str, Any] = dict(self.record["route"].get("form", {}))
        self.arg_plan: list[tuple[str, str]] = [
            (str(kind), str(name)) for kind, name in self.record["arg_plan"]
        ]
        self.launch_record: dict[str, Any] = dict(self.record["launch"])
        self.tma_workspace_bytes = int(self.record.get("tma_workspace_bytes", 0))
        self._entry: Any = None

    @property
    def block(self) -> tuple[int, int, int]:
        return tuple(int(dim) for dim in self.launch_record["block"])  # type: ignore[return-value]

    @property
    def cluster(self) -> tuple[int, int, int]:
        return tuple(int(dim) for dim in self.launch_record["cluster"])  # type: ignore[return-value]

    @property
    def dynamic_smem_bytes(self) -> int:
        return int(self.launch_record["dynamic_smem_bytes"])

    @property
    def cooperative(self) -> bool:
        return bool(self.launch_record["cooperative"])

    @property
    def use_pdl(self) -> bool:
        return bool(self.launch_record["use_pdl"])

    @property
    def names(self) -> tuple[str, ...]:
        """Binding names of the launch (tensors and scalars; workspace and grid are supplied by the loader)."""
        return tuple(
            name for kind, name in self.arg_plan if kind not in {"workspace", "grid"}
        )

    def jit_spec(self):
        units = self.record["translation_units"]
        csrc = _csrc_dir()
        include_dir = jit_env.FLASHINFER_INCLUDE_DIR
        if not include_dir.is_dir():
            include_dir = Path(__file__).resolve().parents[3] / "include"
        return gen_jit_spec(
            name=f"{self.record['name']}_{ARCH}",
            sources=[source_path(units["device"]), source_path(units["binding"])],
            extra_cuda_cflags=[*sm103a_nvcc_flags, *self.record["compile_flags"]],
            extra_ldflags=["-lcuda"],
            extra_include_paths=[csrc, include_dir],
        )

    @property
    def entry(self):
        if self._entry is None:
            module = self.jit_spec().build_and_load()
            self._entry = getattr(module, str(self.record["ffi_entry"]))
        return self._entry

    def arguments(
        self,
        grid: tuple[int, int, int],
        *,
        tma_workspace: torch.Tensor | None,
        bindings: Mapping[str, Any],
    ) -> list[Any]:
        """Flatten ``bindings`` (kernel parameter name -> tensor / scalar) per the argument plan."""
        args: list[Any] = []
        grid_values = iter(grid)
        unused = set(bindings)
        for kind, name in self.arg_plan:
            if kind == "grid":
                value = int(next(grid_values))
                if value <= 0:
                    raise ValueError(f"{self.stage}: grid dimensions must be positive")
                args.append(value)
                continue
            if kind == "workspace":
                if tma_workspace is None:
                    raise ValueError(
                        f"{self.stage}: this module needs a TMA descriptor workspace"
                    )
                if (
                    not tma_workspace.is_cuda
                    or tma_workspace.dtype != torch.uint8
                    or tma_workspace.numel() < self.tma_workspace_bytes
                    or tma_workspace.data_ptr() % 128
                ):
                    raise ValueError(
                        f"{self.stage}: TMA workspace must be {self.tma_workspace_bytes} aligned uint8 bytes"
                    )
                args.append(tma_workspace)
                continue
            if name not in bindings:
                raise KeyError(f"{self.stage}: missing kernel argument {name!r}")
            unused.discard(name)
            value = bindings[name]
            if kind in _TENSOR_KINDS:
                if not isinstance(value, torch.Tensor) or not value.is_cuda:
                    raise TypeError(f"{self.stage}: {name} must be a CUDA torch.Tensor")
                args.append(value)
            elif kind == "parameter":
                if isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"{self.stage}: {name} is a scalar parameter, got a tensor"
                    )
                args.append(value)
            else:
                raise NotImplementedError(
                    f"{self.stage}: argument kind {kind!r} is not used by this package"
                )
        if unused:
            raise KeyError(
                f"{self.stage}: unexpected kernel arguments {sorted(unused)}"
            )
        return args

    def run(
        self,
        grid: tuple[int, int, int],
        *,
        tma_workspace: torch.Tensor | None = None,
        **bindings: Any,
    ) -> None:
        """Launch on the current torch stream (graph-capturable)."""
        import tvm_ffi

        entry = self.entry
        args = self.arguments(grid, tma_workspace=tma_workspace, bindings=bindings)
        with tvm_ffi.use_torch_stream():
            entry(*args)

    def bind(self, device: torch.device) -> BoundKernel:
        """Allocate this kernel's descriptor workspace on ``device`` (call outside graph capture)."""
        return BoundKernel(self, torch.device(device))


@functools.cache
def load(stage: str) -> Kernel:
    """The kernel of one ``route.stage``."""
    return Kernel(record(stage))


def _load_form(**form: Any) -> Kernel:
    return load(str(find_form(**form)["route"]["stage"]))


def select_gemm1(n_tile: int, kbps: int) -> Kernel:
    """The swap-AB GEMM1 SiTU form the plan selected."""
    return _load_form(kind="gemm1_swapab", n_tile=int(n_tile), kbps=int(kbps))


def select_gemm2(n_tile: int, kbps: int, m_group: int) -> Kernel:
    """The swap-AB GEMM2 finalize form the plan selected."""
    return _load_form(
        kind="gemm2_swapab", n_tile=int(n_tile), kbps=int(kbps), m_group=int(m_group)
    )


def select_routing(config: Any) -> Kernel:
    """The fused routing kernel of one ``RoutingConfig``.

    Every trace-time field must match; ``max_rows`` reaches the kernel only through
    ``staged_rows``, so configurations differing in ``max_rows`` alone share one module.
    """
    return _load_form(
        kind="routing",
        mode=config.mode,
        threads=int(config.threads),
        single_tile_per_expert=bool(config.single_tile_per_expert),
        max_routes=int(config.max_routes),
        clear=bool(config.clear),
        dispatch_lists=bool(config.dispatch_lists),
        staged_rows=int(config.staged_rows),
        cluster=int(config.cluster),
        split_layout=bool(config.split_layout),
    )


def compile_all() -> dict[str, int]:
    """JIT-build every module of the package (warm-up / CI); returns dynamic SMEM per stage."""
    built = {}
    for item in modules():
        kernel = load(str(item["route"]["stage"]))
        kernel.entry  # noqa: B018 - builds the module
        built[kernel.stage] = kernel.dynamic_smem_bytes
    return built


__all__ = [
    "ARCH",
    "BACKEND",
    "BoundKernel",
    "Kernel",
    "PACKAGE_NAME",
    "cake_revision",
    "compile_all",
    "find_form",
    "is_available",
    "load",
    "manifest",
    "modules",
    "record",
    "select_gemm1",
    "select_gemm2",
    "select_routing",
    "source_path",
    "unsupported_forms",
]
