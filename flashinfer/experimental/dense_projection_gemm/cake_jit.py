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
import importlib
import json
from types import ModuleType
from typing import Any, Optional

from . import cake_build

# Registry of the generated programs of this package (schema v2, the layout of the deterministic package's
# ``cake_jit``).  Populated verbatim by the generated-program export (``export.families.dense_projection_gemm``); do
# not edit by hand.  Three literals, each wrapped in ``# fmt: off`` / ``# fmt: on`` and rendered one record per line,
# so the export replaces a literal from its head line up to the next ``# fmt: on`` line:
#
# * ``ARG_PLANS``: the distinct argument plans, keyed by the first 12 hex digits of the SHA-256 of the plan's compact
#   JSON (``arg_plan_key``): the ordered ``[kind, name]`` pairs the generated binding's entry takes.
# * ``MODULES``: one record per delivered program (``template``: the symbol the host planner derives for the launch;
#   the kernel symbol, the architecture-free ``library`` stem of the ``build.json`` entries ``<library>_<arch>``, the
#   FFI entry, the ``arg_plan`` key, the launch geometry baked into the module and the architectures it serves).
# * ``KERNELS``: ``arch -> template -> module name``, the table the host planner resolves a launch through.
#
# Only the programs of the non-deterministic schedule live here (program names ``cake_dense_projection_gemm_<hex>``,
# templates ``dense_proj_gemm_atomic_*`` / ``router_fp32_gemm_atomic_*``).  Every program the deterministic package
# registers (``cake_dense_projection_det_gemm_<hex>``, templates ``dense_proj_det_gemm_*``,
# ``router_fp32_det_gemm_*``, ``split_fp32_det_bf16x3``) is served by that package: :func:`select_module` and
# :func:`load_cake_dense_projection_gemm_module` forward its templates and names to it.

# fmt: off
ARG_PLANS: dict[str, list[list[str]]] = {
}
# fmt: on

# fmt: off
MODULES: dict[str, dict[str, Any]] = {
    "cake_dense_projection_gemm_1d46f84995b7baec3e04": {"template": "dense_proj_gemm_atomic_nn_n128_t", "kernel": "kernel_cake_dense_projection_gemm_1d46f84995b7baec3e04", "library": "cake_dense_projection_gemm_1d46f84995b7baec3e04", "ffi_entry": "run", "arg_plan": [["tma_buffer", "A"], ["tma_buffer", "B"], ["buffer", "out"], ["buffer", "ws"], ["buffer", "counters"], ["parameter", "M"], ["parameter", "N"], ["parameter", "K"], ["parameter", "num_l"], ["parameter", "num_splits"], ["parameter", "ldo"], ["parameter", "out_l"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]], "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_103a", "sm_107a"], "build": {"module": "tart.bmm_bf16.sm100_sm103_sm107.dense_projection_gemm", "kwargs": {"block_n": 128, "out_f32": False, "out_t": True, "split_order": "arrival"}}},
    "cake_dense_projection_gemm_243446e7407f2bed9b45": {"template": "dense_proj_gemm_atomic_nn_n64_t", "kernel": "kernel_cake_dense_projection_gemm_243446e7407f2bed9b45", "library": "cake_dense_projection_gemm_243446e7407f2bed9b45", "ffi_entry": "run", "arg_plan": [["tma_buffer", "A"], ["tma_buffer", "B"], ["buffer", "out"], ["buffer", "ws"], ["buffer", "counters"], ["parameter", "M"], ["parameter", "N"], ["parameter", "K"], ["parameter", "num_l"], ["parameter", "num_splits"], ["parameter", "ldo"], ["parameter", "out_l"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]], "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_103a", "sm_107a"], "build": {"module": "tart.bmm_bf16.sm100_sm103_sm107.dense_projection_gemm", "kwargs": {"block_n": 64, "out_f32": False, "out_t": True, "split_order": "arrival"}}},
    "cake_dense_projection_gemm_546922f892bf12518fd7": {"template": "dense_proj_gemm_atomic_nn_n128_f32_t", "kernel": "kernel_cake_dense_projection_gemm_546922f892bf12518fd7", "library": "cake_dense_projection_gemm_546922f892bf12518fd7", "ffi_entry": "run", "arg_plan": [["tma_buffer", "A"], ["tma_buffer", "B"], ["buffer", "out"], ["buffer", "ws"], ["buffer", "counters"], ["parameter", "M"], ["parameter", "N"], ["parameter", "K"], ["parameter", "num_l"], ["parameter", "num_splits"], ["parameter", "ldo"], ["parameter", "out_l"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]], "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_103a", "sm_107a"], "build": {"module": "tart.bmm_bf16.sm100_sm103_sm107.dense_projection_gemm", "kwargs": {"block_n": 128, "out_f32": True, "out_t": True, "split_order": "arrival"}}},
    "cake_dense_projection_gemm_5c912a20d4917fc98c98": {"template": "router_fp32_gemm_atomic_nn_t", "kernel": "kernel_cake_dense_projection_gemm_5c912a20d4917fc98c98", "library": "cake_dense_projection_gemm_5c912a20d4917fc98c98", "ffi_entry": "run", "arg_plan": [["tma_buffer", "A"], ["tma_buffer", "B"], ["buffer", "out32"], ["tma_buffer", "D"], ["parameter", "M"], ["parameter", "N"], ["parameter", "m_tiles"], ["parameter", "n_tiles"], ["parameter", "k_iters"], ["parameter", "ldo"], ["parameter", "units"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]], "launch": {"block": [448, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_103a", "sm_107a"], "build": {"module": "tart.gemm_fp32_bf16x3.sm100_sm103_sm107.router_fp32_gemm", "kwargs": {"a_mn": True, "b_mn": True, "out_t": True}}},
    "cake_dense_projection_gemm_7555c5ce07ba13117bb3": {"template": "dense_proj_gemm_atomic_nn_n64_f32_t", "kernel": "kernel_cake_dense_projection_gemm_7555c5ce07ba13117bb3", "library": "cake_dense_projection_gemm_7555c5ce07ba13117bb3", "ffi_entry": "run", "arg_plan": [["tma_buffer", "A"], ["tma_buffer", "B"], ["buffer", "out"], ["buffer", "ws"], ["buffer", "counters"], ["parameter", "M"], ["parameter", "N"], ["parameter", "K"], ["parameter", "num_l"], ["parameter", "num_splits"], ["parameter", "ldo"], ["parameter", "out_l"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]], "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_103a", "sm_107a"], "build": {"module": "tart.bmm_bf16.sm100_sm103_sm107.dense_projection_gemm", "kwargs": {"block_n": 64, "out_f32": True, "out_t": True, "split_order": "arrival"}}},
    "cake_dense_projection_gemm_bbd8e1061de96c7ebc7b": {"template": "dense_proj_gemm_atomic_nn_n256_f32_t", "kernel": "kernel_cake_dense_projection_gemm_bbd8e1061de96c7ebc7b", "library": "cake_dense_projection_gemm_bbd8e1061de96c7ebc7b", "ffi_entry": "run", "arg_plan": [["tma_buffer", "A"], ["tma_buffer", "B"], ["buffer", "out"], ["buffer", "ws"], ["buffer", "counters"], ["parameter", "M"], ["parameter", "N"], ["parameter", "K"], ["parameter", "num_l"], ["parameter", "num_splits"], ["parameter", "ldo"], ["parameter", "out_l"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]], "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_103a", "sm_107a"], "build": {"module": "tart.bmm_bf16.sm100_sm103_sm107.dense_projection_gemm", "kwargs": {"block_n": 256, "out_f32": True, "out_t": True, "split_order": "arrival"}}},
    "cake_dense_projection_gemm_e717e449f20f5a8edfc3": {"template": "dense_proj_gemm_atomic_nn_n256_t", "kernel": "kernel_cake_dense_projection_gemm_e717e449f20f5a8edfc3", "library": "cake_dense_projection_gemm_e717e449f20f5a8edfc3", "ffi_entry": "run", "arg_plan": [["tma_buffer", "A"], ["tma_buffer", "B"], ["buffer", "out"], ["buffer", "ws"], ["buffer", "counters"], ["parameter", "M"], ["parameter", "N"], ["parameter", "K"], ["parameter", "num_l"], ["parameter", "num_splits"], ["parameter", "ldo"], ["parameter", "out_l"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]], "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_103a", "sm_107a"], "build": {"module": "tart.bmm_bf16.sm100_sm103_sm107.dense_projection_gemm", "kwargs": {"block_n": 256, "out_f32": False, "out_t": True, "split_order": "arrival"}}},
    "cake_dense_projection_gemm_e827d953a13c9a594460": {"template": "router_fp32_gemm_atomic_kk", "kernel": "kernel_cake_dense_projection_gemm_e827d953a13c9a594460", "library": "cake_dense_projection_gemm_e827d953a13c9a594460", "ffi_entry": "run", "arg_plan": [["tma_buffer", "A"], ["tma_buffer", "B"], ["buffer", "out32"], ["tma_buffer", "D"], ["parameter", "M"], ["parameter", "N"], ["parameter", "m_tiles"], ["parameter", "n_tiles"], ["parameter", "k_iters"], ["parameter", "ldo"], ["parameter", "units"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]], "launch": {"block": [448, 1, 1], "cluster": [2, 1, 1]}, "arches": ["sm_100a", "sm_103a", "sm_107a"], "build": {"module": "tart.gemm_fp32_bf16x3.sm100_sm103_sm107.router_fp32_gemm", "kwargs": {}}},
}
# fmt: on

# fmt: off
KERNELS: dict[str, dict[str, str]] = {
    "sm_100a": {"dense_proj_gemm_atomic_nn_n128_f32_t": "cake_dense_projection_gemm_546922f892bf12518fd7", "dense_proj_gemm_atomic_nn_n128_t": "cake_dense_projection_gemm_1d46f84995b7baec3e04", "dense_proj_gemm_atomic_nn_n256_f32_t": "cake_dense_projection_gemm_bbd8e1061de96c7ebc7b", "dense_proj_gemm_atomic_nn_n256_t": "cake_dense_projection_gemm_e717e449f20f5a8edfc3", "dense_proj_gemm_atomic_nn_n64_f32_t": "cake_dense_projection_gemm_7555c5ce07ba13117bb3", "dense_proj_gemm_atomic_nn_n64_t": "cake_dense_projection_gemm_243446e7407f2bed9b45", "router_fp32_gemm_atomic_kk": "cake_dense_projection_gemm_e827d953a13c9a594460", "router_fp32_gemm_atomic_nn_t": "cake_dense_projection_gemm_5c912a20d4917fc98c98"},
    "sm_103a": {"dense_proj_gemm_atomic_nn_n128_f32_t": "cake_dense_projection_gemm_546922f892bf12518fd7", "dense_proj_gemm_atomic_nn_n128_t": "cake_dense_projection_gemm_1d46f84995b7baec3e04", "dense_proj_gemm_atomic_nn_n256_f32_t": "cake_dense_projection_gemm_bbd8e1061de96c7ebc7b", "dense_proj_gemm_atomic_nn_n256_t": "cake_dense_projection_gemm_e717e449f20f5a8edfc3", "dense_proj_gemm_atomic_nn_n64_f32_t": "cake_dense_projection_gemm_7555c5ce07ba13117bb3", "dense_proj_gemm_atomic_nn_n64_t": "cake_dense_projection_gemm_243446e7407f2bed9b45", "router_fp32_gemm_atomic_kk": "cake_dense_projection_gemm_e827d953a13c9a594460", "router_fp32_gemm_atomic_nn_t": "cake_dense_projection_gemm_5c912a20d4917fc98c98"},
    "sm_107a": {"dense_proj_gemm_atomic_nn_n128_f32_t": "cake_dense_projection_gemm_546922f892bf12518fd7", "dense_proj_gemm_atomic_nn_n128_t": "cake_dense_projection_gemm_1d46f84995b7baec3e04", "dense_proj_gemm_atomic_nn_n256_f32_t": "cake_dense_projection_gemm_bbd8e1061de96c7ebc7b", "dense_proj_gemm_atomic_nn_n256_t": "cake_dense_projection_gemm_e717e449f20f5a8edfc3", "dense_proj_gemm_atomic_nn_n64_f32_t": "cake_dense_projection_gemm_7555c5ce07ba13117bb3", "dense_proj_gemm_atomic_nn_n64_t": "cake_dense_projection_gemm_243446e7407f2bed9b45", "router_fp32_gemm_atomic_kk": "cake_dense_projection_gemm_e827d953a13c9a594460", "router_fp32_gemm_atomic_nn_t": "cake_dense_projection_gemm_5c912a20d4917fc98c98"},
}
# fmt: on

# Generated file list: the device / binding translation units of every registered module live flat under
# ``csrc/cake_dense_projection_gemm/`` next to this file (one architecture-neutral pair per program, built for every
# architecture in the record's ``arches``); ``build.json`` lists them per library entry.  They are written by the
# Cake exporter together with the three registries above and the manifest.

# The architectures this package registers programs for (``KERNELS`` keys and every record's ``arches`` are drawn
# from it).
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
TRACKING_ISSUE = "flashinfer-ai/flashinfer#5677"  # the dense projection GEMM family (both numerical modes)

#: The program-name prefix of this package and of the deterministic package.
PROGRAM_PREFIX = "cake_dense_projection_gemm_"
DET_PROGRAM_PREFIX = "cake_dense_projection_det_gemm_"
#: The instance templates of this package's programs, and of the deterministic package's.  The token ``atomic``
#: keeps every template of this package different from every template the deterministic package registers and
#: from every template it registered before its rename (``dense_proj_gemm_<layout>...``, ``router_fp32_gemm_<ab>``).
TEMPLATE_PREFIXES = ("dense_proj_gemm_atomic_", "router_fp32_gemm_atomic_")
DET_TEMPLATE_PREFIXES = (
    "dense_proj_det_gemm_",
    "router_fp32_det_gemm_",
    "split_fp32_det_bf16x3",
)

#: The deterministic package: a sibling of this package (``<parent>.dense_projection_det_gemm``), resolved on first
#: use and never through an absolute ``flashinfer`` import; a vendored copy binds it explicitly
#: (``cake_backend.use_deterministic_backend``).
DETERMINISTIC_PACKAGE = "dense_projection_det_gemm"
_det_package: Optional[str] = None


def bind_deterministic_package(package: str) -> None:
    """Resolve the deterministic package as the dotted package ``package`` from now on (the hook behind
    ``cake_backend.use_deterministic_backend``)."""
    global _det_package
    _det_package = str(package)


def deterministic_module(name: str) -> ModuleType:
    """The submodule ``name`` (``cake_backend``, ``cake_jit``, ``cake_build``) of the deterministic package."""
    package = _det_package
    if package is None:
        parent = __name__.rpartition(".")[0].rpartition(".")[0]
        if not parent:
            raise RuntimeError(
                f"{__name__.rpartition('.')[0]} delegates its deterministic path to the package "
                f"{DETERMINISTIC_PACKAGE!r}, which a vendored copy has no sibling of; vendor that package too and "
                "bind it with cake_backend.use_deterministic_backend(<its cake_backend module>)"
            )
        package = f"{parent}.{DETERMINISTIC_PACKAGE}"
    return importlib.import_module(f"{package}.{name}")


def is_det_template(template: str) -> bool:
    """Whether ``template`` names an instance of the deterministic package."""
    return template.startswith(DET_TEMPLATE_PREFIXES)


def is_det_program(name: str) -> bool:
    """Whether ``name`` names a program of the deterministic package."""
    return name.startswith(DET_PROGRAM_PREFIX)


def arg_plan_key(plan) -> str:
    """Registry key of an argument plan: the first 12 hex digits of the SHA-256 of its compact JSON
    (``[[kind, name], ...]`` with ``,`` / ``:`` separators) -- the export computes the same key."""
    normalised = [[str(kind), str(name)] for kind, name in plan]
    return hashlib.sha256(
        json.dumps(normalised, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def arg_plan_of(record: dict[str, Any]) -> list[list[str]]:
    """The argument plan of a registry record: a ``str`` is a key into ``ARG_PLANS``; an inline list of
    ``[kind, name]`` pairs is taken as is.  Fails closed on a key ``ARG_PLANS`` does not hold."""
    plan = record["arg_plan"]
    if isinstance(plan, str):
        try:
            return ARG_PLANS[plan]
        except KeyError:
            raise KeyError(
                f"registry record {record.get('library', record.get('kernel'))!r} names the "
                f"argument plan {plan!r}, which ARG_PLANS does not hold (keys: {sorted(ARG_PLANS)})"
            ) from None
    return [list(item) for item in plan]


def registered_templates(arch: str) -> tuple[str, ...]:
    """Instance templates this package registers for ``arch``, sorted (the deterministic package's are
    ``deterministic_module("cake_jit").registered_templates(arch)``)."""
    return tuple(sorted(KERNELS.get(arch, {})))


def select_module(arch: str, template: str) -> str:
    """Return the registered module name serving ``template`` on ``arch``: a template of the deterministic package
    resolves through that package's registry, any other through this package's."""
    if is_det_template(template):
        return deterministic_module("cake_jit").select_module(arch, template)
    name = KERNELS.get(arch, {}).get(template)
    if name is None:
        registered = registered_templates(arch)
        raise NotImplementedError(
            f"The generated non-deterministic dense projection GEMM instance {template!r} for {arch} is "
            f"not registered in this checkout (registered: {list(registered) or 'none'}; "
            f"see {TRACKING_ISSUE})"
        )
    record = MODULES.get(name)
    if (
        record is None
        or arch not in record.get("arches", ())
        or record.get("template") != template
    ):
        raise ValueError(
            f"registry record {name!r} does not serve template {template!r} on {arch}"
        )
    return name


def library_name(name: str, arch: str) -> str:
    """``build.json`` entry of this package's program ``name`` built for ``arch``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise ValueError(
            f"generated dense projection GEMM module {name!r} is not registered for {arch} "
            f"(registered: {list(record['arches'])})"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def _load_module(name: str, arch: str):
    return cake_build.load_library(library_name(name, arch))


def load_cake_dense_projection_gemm_module(name: str, arch: str):
    """Load (once per program and architecture) the generated program ``name`` for ``arch``: this package's
    prebuilt library under ``aot/`` when it ships one, else FlashInfer's JIT build of its ``build.json`` entry; a
    program of the deterministic package is loaded by that package's loader."""
    if is_det_program(name):
        return deterministic_module(
            "cake_jit"
        ).load_cake_dense_projection_det_gemm_module(name, arch)
    return _load_module(name, arch)


def specs_for_arch(arch: str):
    """JIT specs of every ``build.json`` entry of ``arch`` without a prebuilt library (the
    ``flashinfer.jit.build_jit_specs`` warm-up); none while the package registers no program."""
    return cake_build.jit_specs(arch) if MODULES else []
