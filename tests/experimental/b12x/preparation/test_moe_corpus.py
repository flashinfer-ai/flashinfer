"""Reviewed MoE geometry contracts independent of offline tuning."""

from __future__ import annotations

from b12x.testing.moe_corpus import (
    COMMON_DECODE_TOKENS,
    COMMON_MOE_MODELS,
    COMMON_PLAN_TOKEN_COUNTS,
    COMMON_PREFILL_TOKEN_CAPACITIES,
    COMMON_TP_SIZES,
    MOE_RECIPES,
    MoeModelGeometry,
    MoeRecipe,
    expand_physical_geometries,
)


def test_common_models_expand_across_supported_parallelism() -> None:
    geometries = expand_physical_geometries()
    covered = {
        (alias.model_id, geometry.recipe.recipe_id, alias.tp_size)
        for geometry in geometries
        for alias in geometry.aliases
    }
    recipes_by_family = {
        family: tuple(recipe for recipe in MOE_RECIPES if recipe.family_id == family)
        for family in {recipe.family_id for recipe in MOE_RECIPES}
    }
    for model in COMMON_MOE_MODELS:
        for family in model.recipe_families:
            for recipe in recipes_by_family[family]:
                if model.activation in recipe.compatible_activations:
                    for tp_size in model.tp_sizes:
                        assert (model.model_id, recipe.recipe_id, tp_size) in covered


def test_unaligned_three_wide_shard_is_padded_instead_of_rejected() -> None:
    recipe = MoeRecipe(
        recipe_id="nvfp4-test", family_id="test", quant_mode="nvfp4",
        source_format="modelopt_nvfp4", intermediate_alignment=16,
        minimum_intermediate_size=16, compatible_activations=("silu",),
    )
    model = MoeModelGeometry(
        model_id="small-test", hidden_size=256, intermediate_size=47,
        num_experts=16, native_top_k=2, activation="silu",
        recipe_families=(recipe.family_id,), source="test", tp_sizes=(16,),
    )
    (geometry,) = expand_physical_geometries(models=(model,), recipes=(recipe,))
    (alias,) = geometry.aliases

    assert alias.logical_intermediate_sizes == (2, 3)
    assert alias.physical_intermediate_size == 16
    assert alias.padding_per_tp_group == 209


def test_recipe_families_only_expand_compatible_activations() -> None:
    recipes = (
        MoeRecipe(recipe_id="silu-recipe", family_id="shared", quant_mode="nvfp4",
                  source_format="modelopt_nvfp4", intermediate_alignment=16,
                  minimum_intermediate_size=16, compatible_activations=("silu",)),
        MoeRecipe(recipe_id="relu-recipe", family_id="shared", quant_mode="w4a16",
                  source_format="modelopt_nvfp4", intermediate_alignment=64,
                  minimum_intermediate_size=64, compatible_activations=("relu2",)),
    )
    model = MoeModelGeometry(
        model_id="silu-model", hidden_size=256, intermediate_size=96,
        num_experts=16, native_top_k=2, activation="silu",
        recipe_families=("shared",), source="test", tp_sizes=(1,),
    )

    geometries = expand_physical_geometries(models=(model,), recipes=recipes)
    assert [geometry.recipe.recipe_id for geometry in geometries] == ["silu-recipe"]


def test_moe_token_axes_keep_decode_and_prefill_capacity_distinct() -> None:
    assert COMMON_DECODE_TOKENS == (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128)
    assert COMMON_PREFILL_TOKEN_CAPACITIES == (512, 1_024, 2_048, 4_096, 8_192)
    assert COMMON_PLAN_TOKEN_COUNTS == (*COMMON_DECODE_TOKENS, *COMMON_PREFILL_TOKEN_CAPACITIES)


def test_qwen_nondivisible_parallel_shards_share_a_padded_geometry() -> None:
    aliases = [
        (geometry, alias)
        for geometry in expand_physical_geometries()
        for alias in geometry.aliases
        if alias.model_id == "qwen3.8-flash-next-180b"
        and alias.tp_size == 3
        and geometry.recipe.recipe_id == "modelopt-nvfp4"
    ]
    assert len(aliases) == 1
    geometry, alias = aliases[0]
    assert alias.logical_intermediate_sizes == (213, 214)
    assert geometry.intermediate_size == 224
