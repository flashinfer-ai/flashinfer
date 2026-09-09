"""JIT spec for the SM120 SVDQuant fused NVFP4 GEMM module.

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

import os

import jinja2

from .. import env as jit_env
from ..core import JitSpec, current_compilation_context, gen_jit_spec
from ..utils import write_if_different
from .svdquant_sm120_configs import SVDQUANT_SM120_CONFIGS
from .svdquant_sm120_variants import LORA_LADDER_RUNGS, PRODUCTION_LADDER_RUNGS


class SvdquantSm120VariantError(ValueError):
    """Raised when incompatible SM120 diagnostic variants are requested."""


def _selected_items(items: tuple[tuple[bool, str], ...]) -> list[str]:
    return [item for selected, item in items if selected]


def gen_gemm_sm120_module_cutlass_nvfp4_svdquant(
    lora_every_split: bool = False,
    lora_dedicated: bool = False,
    lora_ladder: str = "",
    lora_epi_overlap: bool = True,
    lora_neutral_barrier: bool = False,
    static_sched: bool = False,
    bias_coalesced: bool = False,
    lora_true_rank_tail: bool = True,
    lora_smem_swizzle: bool = False,
    row80_lora_byte_exact: bool = False,
    row80_stages: int = 0,
    stock_parity_kstep: int = 0,
    diagnostic_probe: bool = False,
    lora_rank: int = 32,
) -> JitSpec:
    """SM120 SVDQuant fused NVFP4 GEMM + smooth quantization JIT module.

    ``lora_every_split`` builds a deliberately incorrect kernel variant that folds
    the LoRA product into every Stream-K split's partial accumulator instead of
    only the epilogue owner. It exists solely so the duplicate-application
    detector in the test suite can be shown to detect; never use it for real work.

    ``lora_dedicated`` builds every config with its own one-slot D/L1 TMA pipeline.

    ``lora_ladder`` selects a measurement-only rung of the byte-exact LoRA path
    whose OUTPUT LACKS THE LoRA TERM: ``"residual"`` compiles the LoRA stage out
    entirely (no producer step, no consumer), ``"transfer"`` keeps the full
    transfer protocol but compiles out the serialized smem->rmem copies and bf16
    tail MMAs. Fixed-tactic deltas across rungs decompose the LoRA fixed cost
    (transfer minus residual = protocol; full minus transfer = tail). Ladder
    rungs are numerically wrong by construction, exist only for the fixed-cost
    ladder tool, and cannot combine with the other variant flags. The
    ``"production_residual"`` and ``"production_transfer"`` rungs remove the
    same work while retaining row80's production stage count, shared-memory
    swizzle, true-rank layout, bias prefetch, and overlap-enabled epilogue type.

    ``lora_epi_overlap`` (the production DEFAULT) stages D/L1 into registers,
    releases the overlay stage before the epilogue, and folds the rank-32 bf16
    LoRA product into each epilogue subtile's accumulator block inside the
    store loop instead of serially before it. Bit-identical to the serial
    path; measured to recover part of the serialized tail on many-wave
    shapes (the balance is genuine issue-bound compute) with no regression
    beyond noise anywhere. ``lora_epi_overlap=False`` builds the serial-tail
    fallback variant. The other variant flags keep the serial path (the
    overlap cannot combine with them), except ``lora_neutral_barrier``, which
    retains the fold via its own register-staging overload.

    ``lora_neutral_barrier`` builds every config with the smem-neutral
    independent LoRA barrier (Sm120LoRaPath::kSmemNeutralBarrier): D/L1 stage
    through the dedicated policy's true-rank TMA descriptors and one-slot
    pipeline, but land at the borrow stage base inside the residual A/B
    buffers - no smem carveout beyond the 128-byte barrier allowance, no
    dummy SF reloads, and the mainloop runs exactly its K residual steps.
    Cannot combine with ``lora_dedicated``, ``lora_every_split``, or
    ``lora_ladder``; composes with ``lora_epi_overlap``.

    ``static_sched`` selects the stock static scheduler; ``bias_coalesced`` stages
    eligible swapped-tile bias vectors through shared memory.

    ``lora_true_rank_tail`` (the production DEFAULT) stages and folds only the
    rank-32 payload columns on the consumer side. On K256 tiles the smem->rmem
    copies, D/L1 register fragments, and bf16 tail MMAs halve relative to the
    storage-width path; the byte-exact fallback still preserves its TMA boxes,
    dummy SF reloads, and pipeline byte accounting, while the row80 side slot
    keeps a padded K64 physical layout. K128 tiles are unchanged because their
    payload width already equals their storage width.
    ``lora_true_rank_tail=False`` builds the storage-width fallback
    (``_lora_storage_tail``). Composes with ``lora_epi_overlap``,
    ``static_sched``, and ``bias_coalesced``; the dedicated, every-split,
    neutral-barrier, and ladder variants replace or compile out the
    byte-exact consumer and keep the storage-width layout.

    Row80 keeps matched K256 producer/consumer swizzles in both production
    paths: the default independent side slot and the byte-exact overlay
    fallback. ``lora_smem_swizzle`` remains a build-wide diagnostic variant:
    it changes the producer TMA layout and consumer tensor view together to a
    128-byte swizzle while retaining the baseline copy atom. K128 layouts
    remain unchanged. It composes with the overlapped or serial byte-exact
    consumer, but not with variants that replace or compile out that path.

    Row80 uses an independent, padded K64, swizzled D/L1 slot by default. Set
    ``row80_lora_byte_exact=True`` only for the cache-distinct production A/B
    fallback that overlays D/L1 on the residual pipeline's next stage.

    ``diagnostic_probe`` additionally exports
    ``nvfp4_svdquant_producer_probe_sm120``, the producer half of the fused
    linear, for the isolated producer/K3 decomposition tool. It adds a symbol
    and changes nothing else: the fused route, its kernels, and its tactics are
    the production ones, which is the point -- the probe must measure what
    production runs. It therefore requires the production tuning defaults and
    cannot combine with any variant that replaces part of the measured path.
    The build is cached under its own module name, so the production module
    stays byte-identical to a build without this flag.

    """
    production_ladder = lora_ladder in PRODUCTION_LADDER_RUNGS
    if lora_ladder and lora_ladder not in LORA_LADDER_RUNGS:
        raise ValueError(f"unknown lora_ladder rung: {lora_ladder!r}")
    if stock_parity_kstep not in (0, 1, 2, 3):
        raise ValueError(f"unknown stock_parity_kstep arm: {stock_parity_kstep!r}")
    if row80_stages not in (0, 4, 5):
        raise SvdquantSm120VariantError(
            f"row80_stages must be 0, 4, or 5; got {row80_stages!r}"
        )
    if lora_rank <= 0 or lora_rank % 32:
        raise ValueError(
            f"lora_rank must be a positive multiple of 32; got {lora_rank!r}"
        )
    production_defaults = (
        lora_epi_overlap
        and lora_true_rank_tail
        and row80_stages == 0
        and not static_sched
        and not bias_coalesced
    )
    if production_ladder and not production_defaults:
        raise SvdquantSm120VariantError(
            "production ladder rungs require the production row80 tuning defaults"
        )
    if row80_lora_byte_exact and (
        not production_defaults
        or lora_every_split
        or lora_dedicated
        or lora_ladder
        or lora_neutral_barrier
        or lora_smem_swizzle
        or stock_parity_kstep
    ):
        raise SvdquantSm120VariantError(
            "the row80 byte-exact fallback requires the production tuning defaults"
        )
    if stock_parity_kstep and lora_ladder != "residual":
        # The parity arms perturb only the ptxas issue-slot schedule of the
        # steady-state K loop; they exist solely for the residual-build
        # matched-pair comparison and are unreachable from production.
        raise ValueError("stock_parity_kstep arms measure the residual build only")
    if lora_ladder and (lora_every_split or lora_dedicated):
        raise ValueError("lora_ladder rungs measure the byte-exact path only")
    if lora_neutral_barrier and (lora_every_split or lora_dedicated or lora_ladder):
        raise ValueError(
            "lora_neutral_barrier cannot combine with the dedicated, every-split, "
            "or ladder variants"
        )
    if lora_every_split or lora_dedicated or (lora_ladder and not production_ladder):
        # These variants study or replace pieces of the serial byte-exact path.
        lora_epi_overlap = False
    uses_historical_storage_layout = (
        lora_every_split
        or lora_dedicated
        or (lora_ladder and not production_ladder)
        or lora_neutral_barrier
    )
    if lora_smem_swizzle and (
        lora_every_split or lora_dedicated or lora_ladder or lora_neutral_barrier
    ):
        raise SvdquantSm120VariantError(
            "lora_smem_swizzle measures the byte-exact path only and cannot "
            "combine with variants that replace it"
        )
    if uses_historical_storage_layout:
        # These variants replace or compile out the byte-exact consumer; keep
        # their historical storage-width layout and cache names.
        lora_true_rank_tail = False
    if diagnostic_probe and (
        not production_defaults
        or lora_every_split
        or lora_dedicated
        or lora_ladder
        or lora_neutral_barrier
        or lora_smem_swizzle
        or row80_lora_byte_exact
        or stock_parity_kstep
    ):
        raise SvdquantSm120VariantError(
            "the diagnostic probe seam exports the production producer half and "
            "requires the production tuning defaults"
        )
    row80_lora_side_slot = production_defaults and not any(
        (
            row80_lora_byte_exact,
            lora_every_split,
            lora_dedicated,
            lora_neutral_barrier,
            lora_smem_swizzle,
            stock_parity_kstep,
            lora_ladder and not production_ladder,
        )
    )
    variant = (
        ("_lora_every_split" if lora_every_split else "")
        + ("_lora_dedicated" if lora_dedicated else "")
        + (f"_lora_ladder_{lora_ladder}" if lora_ladder else "")
        + ("_lora_neutral_barrier" if lora_neutral_barrier else "")
        + (f"_stock_parity_kstep{stock_parity_kstep}" if stock_parity_kstep else "")
        + ("_row80_lora_byte_exact" if row80_lora_byte_exact else "")
    )
    if not uses_historical_storage_layout and not lora_true_rank_tail:
        # Storage-width fallback of the byte-exact consumer (opt-out of the
        # default rank-32 payload slice).
        variant += "_lora_storage_tail"
    if variant == "_lora_storage_tail" and not lora_epi_overlap:
        variant += "_serial_tail"
    if variant == "_lora_neutral_barrier" and not lora_epi_overlap:
        # Keep the serial-tail neutral build cache-distinct from the default
        # (epi-overlap) neutral build.
        variant += "_serial_tail"
    if not variant and not lora_epi_overlap:
        variant = "_lora_serial_tail"
    variant += "".join(
        _selected_items(
            (
                (lora_smem_swizzle, "_lora_smem_swizzle"),
                # Rank 32 stays un-suffixed so the production module keeps its identity
                # and its tuned cache; another rank is a separate build, the way the
                # upstream cute-dsl path compiles one specialization per rank.
                (lora_rank != 32, f"_rank{lora_rank}"),
                (bool(row80_stages), f"_row80_stages{row80_stages}"),
                (static_sched, "_static_sched"),
                (bias_coalesced, "_bias_coalesced"),
                # Keep the probe build cache-distinct so the production module is
                # byte-identical to a build that never saw this flag.
                (diagnostic_probe, "_diagnostic_probe"),
            )
        )
    )
    gen_directory = (
        jit_env.FLASHINFER_GEN_SRC_DIR
        / f"gen_gemm_sm120_cutlass_nvfp4_svdquant{variant}"
    )
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = [
        jit_env.FLASHINFER_CSRC_DIR / "nvfp4_svdquant_gemm_cutlass_sm120.cu",
        jit_env.FLASHINFER_CSRC_DIR / "nvfp4_smooth_quantize_sm100.cu",
    ]

    with open(
        jit_env.FLASHINFER_CSRC_DIR / "nvfp4_svdquant_gemm_cutlass_sm120.jinja"
    ) as f:
        kernel_inst_templ = jinja2.Template(f.read())
        for config in SVDQUANT_SM120_CONFIGS:
            dest_path = gen_directory / f"nvfp4_svdquant_gemm_cutlass_sm120_{config}.cu"
            source_paths.append(dest_path)
            source = kernel_inst_templ.render(config=config)
            write_if_different(dest_path, source)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )
    ladder_flags = {
        "residual": "-DSVDQ_SM120_LORA_LADDER_RESIDUAL_ONLY",
        "transfer": "-DSVDQ_SM120_LORA_LADDER_TRANSFER_ONLY",
        "production_residual": "-DSVDQ_SM120_LORA_LADDER_RESIDUAL_ONLY",
        "production_transfer": "-DSVDQ_SM120_LORA_LADDER_TRANSFER_ONLY",
    }
    ladder_flag = ladder_flags.get(lora_ladder)
    variant_flags = _selected_items(
        (
            (lora_every_split, "-DSVDQ_SM120_LORA_EVERY_SPLIT"),
            (lora_dedicated, "-DSVDQ_SM120_LORA_DEDICATED"),
            (bool(ladder_flag), ladder_flag or ""),
            (production_ladder, "-DSVDQ_SM120_ROW80_PRODUCTION_PARITY"),
            (lora_neutral_barrier, "-DSVDQ_SM120_LORA_NEUTRAL_BARRIER"),
            (lora_epi_overlap, "-DSVDQ_SM120_LORA_EPI_OVERLAP"),
            (static_sched, "-DSVDQ_SM120_STATIC_SCHED"),
            (bias_coalesced, "-DSVDQ_SM120_BIAS_COALESCED"),
            (lora_true_rank_tail, "-DSVDQ_SM120_LORA_TRUE_RANK_TAIL"),
            (lora_smem_swizzle, "-DSVDQ_SM120_LORA_SMEM_SWIZZLE"),
            (row80_lora_side_slot, "-DSVDQ_SM120_ROW80_LORA_SIDE_SLOT"),
            (diagnostic_probe, "-DFLASHINFER_ENABLE_SVDQ_SM120_DIAGNOSTIC_PROBE"),
        )
    )
    if row80_stages:
        variant_flags.append(f"-DSVDQ_SM120_ROW80_STAGES={row80_stages}")
    if stock_parity_kstep:
        variant_flags.append(f"-DSVDQ_SM120_STOCK_PARITY_KSTEP={stock_parity_kstep}")
    if lora_rank != 32:
        variant_flags.append(f"-DSVDQ_SM120_LORA_RANK={lora_rank}")
    return gen_jit_spec(
        f"nvfp4_svdquant_gemm_cutlass_sm120{variant}",
        source_paths,
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP4",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
            "-DFLASHINFER_ENABLE_SVDQ_SM120_K1_K2_FUSION",
        ]
        + variant_flags,
        extra_cflags=[
            "-DFAST_BUILD",
        ],
        extra_ldflags=["-lcublasLt"],
    )
