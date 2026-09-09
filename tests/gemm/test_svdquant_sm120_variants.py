"""Host-only tests for SM120 SVDQuant JIT diagnostic variants."""

from flashinfer.jit.gemm.svdquant_sm120 import (
    SvdquantSm120VariantError,
    gen_gemm_sm120_module_cutlass_nvfp4_svdquant as gen,
)


def test_production_ladder_rungs_keep_row80_production_traits() -> None:
    """Production ladder rungs retain row80 tuning around the removed work."""
    for rung, ladder_macro in (
        ("production_residual", "SVDQ_SM120_LORA_LADDER_RESIDUAL_ONLY"),
        ("production_transfer", "SVDQ_SM120_LORA_LADDER_TRANSFER_ONLY"),
    ):
        spec = gen(lora_ladder=rung)
        flags = set(spec.extra_cuda_cflags)

        assert spec.name.endswith(f"_lora_ladder_{rung}")
        assert f"-D{ladder_macro}" in flags
        assert "-DSVDQ_SM120_ROW80_PRODUCTION_PARITY" in flags
        assert "-DSVDQ_SM120_ROW80_LORA_SIDE_SLOT" in flags
        assert "-DSVDQ_SM120_LORA_EPI_OVERLAP" in flags
        assert "-DSVDQ_SM120_LORA_TRUE_RANK_TAIL" in flags


def test_legacy_ladder_rungs_keep_original_diagnostic_codegen() -> None:
    """The existing ladder remains cache- and codegen-compatible."""
    for rung in ("residual", "transfer"):
        spec = gen(lora_ladder=rung)
        flags = set(spec.extra_cuda_cflags)

        assert "-DSVDQ_SM120_ROW80_PRODUCTION_PARITY" not in flags
        assert "-DSVDQ_SM120_LORA_EPI_OVERLAP" not in flags
        assert "-DSVDQ_SM120_LORA_TRUE_RANK_TAIL" not in flags


def test_production_ladder_rungs_reject_nonproduction_overrides() -> None:
    """A production-labelled rung fails closed on every row80 tuning override."""
    overrides = (
        {"lora_epi_overlap": False},
        {"lora_true_rank_tail": False},
        {"row80_stages": 4},
        {"row80_stages": 5},
        {"static_sched": True},
        {"bias_coalesced": True},
    )

    for rung in ("production_residual", "production_transfer"):
        for override in overrides:
            try:
                gen(lora_ladder=rung, **override)
            except SvdquantSm120VariantError:
                continue
            raise AssertionError((rung, override))


def test_row80_lora_side_slot_is_the_production_default() -> None:
    """Production enables the independent row80 slot without renaming the module."""
    spec = gen()
    flags = set(spec.extra_cuda_cflags)

    assert spec.name.endswith("nvfp4_svdquant_gemm_cutlass_sm120")
    assert "-DSVDQ_SM120_ROW80_LORA_SIDE_SLOT" in flags
    assert "-DSVDQ_SM120_LORA_EPI_OVERLAP" in flags
    assert "-DSVDQ_SM120_LORA_TRUE_RANK_TAIL" in flags


def test_sm120_svdquant_module_links_cublaslt() -> None:
    """The exact fused LoRA-down helper has an explicit runtime dependency."""
    assert gen().extra_ldflags == ["-lcublasLt"]


def test_row80_byte_exact_fallback_is_production_locked() -> None:
    """The old row80 overlay remains an explicit, cache-distinct A/B fallback."""
    spec = gen(row80_lora_byte_exact=True)
    flags = set(spec.extra_cuda_cflags)

    assert spec.name.endswith("_row80_lora_byte_exact")
    assert "-DSVDQ_SM120_ROW80_LORA_SIDE_SLOT" not in flags
    assert "-DSVDQ_SM120_LORA_EPI_OVERLAP" in flags
    assert "-DSVDQ_SM120_LORA_TRUE_RANK_TAIL" in flags

    overrides = (
        {"lora_every_split": True},
        {"lora_dedicated": True},
        {"lora_ladder": "residual"},
        {"lora_neutral_barrier": True},
        {"lora_epi_overlap": False},
        {"lora_true_rank_tail": False},
        {"lora_smem_swizzle": True},
        {"row80_stages": 4},
        {"static_sched": True},
        {"bias_coalesced": True},
    )
    for override in overrides:
        try:
            gen(row80_lora_byte_exact=True, **override)
        except SvdquantSm120VariantError:
            continue
        raise AssertionError(override)


def test_nonproduction_variants_do_not_inherit_the_row80_side_slot() -> None:
    """Whole-module diagnostics keep their historical path selection."""
    specs = (
        gen(lora_every_split=True),
        gen(lora_dedicated=True),
        gen(lora_ladder="residual"),
        gen(lora_neutral_barrier=True),
        gen(lora_smem_swizzle=True),
        gen(row80_stages=4),
        gen(static_sched=True),
        gen(bias_coalesced=True),
        gen(stock_parity_kstep=1, lora_ladder="residual"),
    )

    for spec in specs:
        assert "-DSVDQ_SM120_ROW80_LORA_SIDE_SLOT" not in set(spec.extra_cuda_cflags)
