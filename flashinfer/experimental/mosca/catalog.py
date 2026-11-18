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

# Catalog of the MOSCA-optimized kernels in FlashInfer (experimental).
#
# MOSCA is an NVIDIA kernel-optimization project. It starts from a kernel that
# already serves a model, measures it on that model's cells (operation, shape,
# dtype, parallel layout, GPU), and changes the kernel or its launch
# configuration where a change is proven faster with unchanged outputs.
#
# Every kernel this package ships has exactly one entry below. The entry is the
# description that the README, the tests and the runtime coverage check read:
#
# * the FlashInfer API it plugs into and the config that selects it;
# * the upstream kernel it is derived from (a vendored tree, see _vendor.py);
# * each MOSCA change as a separate switch, and the files that implement it;
# * the configurations it has been validated on.
#
# Data only: importing this module loads no kernel, JIT, tvm-ffi or torch code.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Feature:
    """One MOSCA change, exposed as a switch of the kernel's config."""

    name: str
    #: How to turn it on, as a config keyword (``'small_t=True'``).
    switch: str
    summary: str
    #: Preprocessor macro that gates the change in the kernel's device source;
    #: ``None`` for a change that lives in host code only.
    macro: str | None
    #: Files (relative to ``flashinfer/experimental/mosca``) that implement it.
    where: tuple[str, ...]
    #: Whether outputs are bitwise identical with the switch on and off.
    result_preserving: bool = True


@dataclass(frozen=True)
class ValidatedCell:
    """One configuration a kernel has been validated on."""

    model: str
    num_experts: int
    hidden: int
    intermediate: int
    top_k: int
    gpu: str
    compute_capability: tuple[int, int]
    ep_sizes: tuple[int, ...]
    #: Parallel layout of the serving deployment this cell stands for.
    deployment: str
    token_layout: str
    #: Feature names exercised in this cell.
    features: tuple[str, ...]
    correctness: str
    performance: str | None = None


@dataclass(frozen=True)
class MoscaKernel:
    """One MOSCA kernel shipped by FlashInfer."""

    #: Registered name (``kernel_name`` of the config).
    name: str
    #: Subpackage of ``flashinfer.experimental.mosca`` that holds it.
    family: str
    #: FlashInfer API it plugs into.
    api: str
    #: Config class that selects it (the only way to select it).
    config: str
    #: Upstream kernel it is derived from.
    base_kernel: str
    #: Vendored upstream tree it builds on (``vendor/<tree>.json``).
    vendored_tree: str
    features: tuple[Feature, ...]
    validated: tuple[ValidatedCell, ...]

    def feature(self, name: str) -> Feature:
        for feature in self.features:
            if feature.name == name:
                return feature
        raise KeyError(f"{self.name} has no feature {name!r}")

    def find_cell(
        self,
        *,
        num_experts: int,
        hidden: int,
        intermediate: int,
        top_k: int,
        ep_size: int,
        token_layout: str,
        compute_capability: tuple[int, int],
        features: tuple[str, ...] = (),
    ) -> ValidatedCell | None:
        """The validated cell that covers a configuration, or ``None``.

        A cell covers the configuration when the shape, EP size, token layout
        and GPU match and every enabled feature was exercised in that cell.
        """
        for cell in self.validated:
            if (
                (cell.num_experts, cell.hidden, cell.intermediate, cell.top_k)
                == (num_experts, hidden, intermediate, top_k)
                and ep_size in cell.ep_sizes
                and cell.token_layout == token_layout
                and tuple(cell.compute_capability) == tuple(compute_capability)
                and set(features) <= set(cell.features)
            ):
                return cell
        return None


# ----------------------------------------------------------------------------
# sm100_fp8_fp4_bf16_mosca: DeepGEMM's SM100 FP8 x FP4 MegaMoE + MOSCA changes
# ----------------------------------------------------------------------------

_MEGAMOE_HEADER = "csrc/megamoe/sm100_fp8_fp4_mega_moe.cuh"
_MEGAMOE_PLANNER = "csrc/megamoe/sm100_fp8_fp4_planner.cu"
_MEGAMOE_LAUNCHER = "csrc/megamoe/sm100_fp8_fp4_launcher.cu"


def _dsv4_pro_cell(**kwargs: Any) -> ValidatedCell:
    """A cell on the DeepSeek-V4-Pro MoE layer: 384 routed experts, hidden 7168,
    expert intermediate 3072, top-6 routing, SwiGLU with input clamp 10."""
    return ValidatedCell(
        model="DeepSeek-V4-Pro MoE layer",
        num_experts=384,
        hidden=7168,
        intermediate=3072,
        top_k=6,
        **kwargs,
    )


_MEGAMOE_TESTS = "tests/experimental/mosca/test_mosca_megamoe_multirank.py"
_MEGAMOE_BENCH = "benchmarks/moe_ep/bench_mosca_megamoe.py"

SM100_FP8_FP4_MEGAMOE = MoscaKernel(
    name="sm100_fp8_fp4_bf16_mosca",
    family="megamoe",
    api="flashinfer.moe_ep.MoEEpMegaLayer (MegaConfig)",
    config="flashinfer.experimental.mosca.megamoe.Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig",
    base_kernel="DeepGEMM sm100_fp8_fp4_mega_moe (sgl-project/DeepGEMM 1ff3932)",
    vendored_tree="deep_gemm",
    features=(
        Feature(
            name="small_t",
            switch="small_t=True",
            summary=(
                "When every token fits SM 0's first dispatch round, SM 0 "
                "publishes the per-expert counts: the cross-SM count round "
                "(global atomics) and one grid sync are skipped."
            ),
            macro="DGX_SMALL_T",
            where=(_MEGAMOE_HEADER,),
        ),
        Feature(
            name="sparse_count",
            switch="sparse_count=True",
            summary=(
                "With small_t, only the SMs that own tokens add to the global "
                "send counters."
            ),
            macro="DGX_SPARSE_COUNT",
            where=(_MEGAMOE_HEADER,),
        ),
        Feature(
            name="in_kernel_all_gather",
            switch='token_layout="replicated"',
            summary=(
                "Every rank passes the same batch (attention TP); the kernel "
                "writes every rank's output rows into a symmetric gathered "
                "region during combine, so the post-MoE all-gather disappears."
            ),
            macro="DGX_BCAST",
            where=(
                _MEGAMOE_HEADER,
                _MEGAMOE_PLANNER,
                _MEGAMOE_LAUNCHER,
                "megamoe/backend.py",
            ),
        ),
        Feature(
            name="tile_rule",
            switch='tile_rule="t>=41&t<=56:block_m=32;..."',
            summary=(
                "Per-call tile override by the per-rank token count, validated "
                "against the kernel's static constraints. Outputs are unchanged; "
                "a rule's speed is specific to the deployment it was tuned in "
                "(DSV4_PRO_DEP8_TILE_RULE: SGLang DEP8 with the shared expert in "
                "kernel), so no rule is part of a validated cell here."
            ),
            macro=None,
            where=(_MEGAMOE_PLANNER,),
        ),
    ),
    validated=(
        _dsv4_pro_cell(
            gpu="B200",
            compute_capability=(10, 0),
            ep_sizes=(1, 4, 8),
            deployment="TP attention + EP MoE (every rank holds the same batch)",
            token_layout="replicated",
            features=("small_t", "in_kernel_all_gather"),
            correctness=(
                "bitwise == the partitioned kernel on each rank's tensor_split "
                "slice + NCCL all-gather for batches of 1, 3, 8, 13, 64, 200 and "
                "512 tokens; torch oracle (rel L2 < 5 %); CUDA-graph replay == "
                f"eager ({_MEGAMOE_TESTS})"
            ),
            performance=(
                "kernel time vs sm100_fp8_fp4_bf16_deepgemm + NCCL all-gather, "
                f"EP8 ({_MEGAMOE_BENCH} --layout replicated)"
            ),
        ),
        _dsv4_pro_cell(
            gpu="B200",
            compute_capability=(10, 0),
            ep_sizes=(1, 4, 8),
            deployment="DP attention + EP MoE (every rank holds its own tokens)",
            token_layout="partitioned",
            features=("small_t", "sparse_count"),
            correctness=(
                "bitwise == the switches-off kernel and == "
                "sm100_fp8_fp4_bf16_deepgemm for small_t, sparse_count, num_sms and "
                "DSV4_PRO_DEP8_TILE_RULE at per-rank token counts "
                "0, 1, 7, 16, 20, 21, 48, 64, 128, 300 (every window of the rule) "
                "and at uneven per-rank counts; torch oracle (rel L2 < 5 %) for "
                "every switch; CUDA-graph replay == eager, including counts never "
                f"run eagerly ({_MEGAMOE_TESTS}); planner == deep_gemm for every "
                "per-rank count 0..1920 (tests/experimental/mosca/test_mosca_megamoe.py)"
            ),
            performance=(
                "kernel time vs sm100_fp8_fp4_bf16_deepgemm, EP8 "
                f"({_MEGAMOE_BENCH} --layout partitioned)"
            ),
        ),
    ),
)

#: Every MOSCA kernel shipped by FlashInfer.
KERNELS: tuple[MoscaKernel, ...] = (SM100_FP8_FP4_MEGAMOE,)


def get_kernel(name: str) -> MoscaKernel:
    for kernel in KERNELS:
        if kernel.name == name:
            return kernel
    raise KeyError(
        f"no MOSCA kernel named {name!r}; known: {[k.name for k in KERNELS]}"
    )


__all__ = [
    "Feature",
    "KERNELS",
    "MoscaKernel",
    "SM100_FP8_FP4_MEGAMOE",
    "ValidatedCell",
    "get_kernel",
]
