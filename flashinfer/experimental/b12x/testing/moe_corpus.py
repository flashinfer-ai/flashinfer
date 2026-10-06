"""Reviewed model corpus for offline MoE decode profile generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

COMMON_TP_SIZES = tuple(range(1, 17))
COMMON_TOP_K = (1, 2, 4, 6, 8, 10, 12, 16)
COMMON_DECODE_TOKENS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128)
COMMON_PREFILL_TOKEN_CAPACITIES = (512, 1_024, 2_048, 4_096, 8_192)
COMMON_PLAN_TOKEN_COUNTS = (
    *COMMON_DECODE_TOKENS,
    *COMMON_PREFILL_TOKEN_CAPACITIES,
)
COMMON_ROUTE_PATTERNS = ("balanced", "hot", "zipf", "disjoint")


def align_up(value: int, alignment: int) -> int:
    if value <= 0 or alignment <= 0:
        raise ValueError("value and alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True, kw_only=True)
class MoeRecipe:
    recipe_id: str
    family_id: str
    quant_mode: str
    source_format: str
    intermediate_alignment: int
    minimum_intermediate_size: int
    compatible_activations: tuple[str, ...]
    trellis_variant: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.recipe_id
            or not self.family_id
            or not self.quant_mode
            or not self.source_format
        ):
            raise ValueError("MoE recipe identifiers must be non-empty")
        if self.intermediate_alignment <= 0:
            raise ValueError("intermediate_alignment must be positive")
        if self.minimum_intermediate_size <= 0:
            raise ValueError("minimum_intermediate_size must be positive")
        if self.minimum_intermediate_size % self.intermediate_alignment:
            raise ValueError(
                "minimum_intermediate_size must satisfy the recipe alignment"
            )
        if not self.compatible_activations or len(self.compatible_activations) != len(
            set(self.compatible_activations)
        ):
            raise ValueError("compatible_activations must be non-empty and unique")
        if any(not activation for activation in self.compatible_activations):
            raise ValueError("compatible_activations must contain non-empty strings")
        is_trellis = self.source_format in {"exl3", "b12x_trellis"}
        if is_trellis and not self.trellis_variant:
            raise ValueError("Trellis recipes require a trellis_variant")
        if not is_trellis and self.trellis_variant is not None:
            raise ValueError("trellis_variant is valid only for Trellis recipes")

    def physical_intermediate_size(self, logical_size: int) -> int:
        return align_up(
            max(int(logical_size), self.minimum_intermediate_size),
            self.intermediate_alignment,
        )


@dataclass(frozen=True, kw_only=True)
class MoeModelGeometry:
    model_id: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    native_top_k: int
    activation: str
    recipe_families: tuple[str, ...]
    source: str
    tp_sizes: tuple[int, ...] = COMMON_TP_SIZES

    physical_intermediate_alignment: int | None = None
    def __post_init__(self) -> None:
        if not self.model_id or not self.activation or not self.source:
            raise ValueError("model geometry labels must be non-empty")
        for name in ("hidden_size", "intermediate_size", "num_experts"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.native_top_k <= self.num_experts:
            raise ValueError("native_top_k must be in the expert range")
        if not self.recipe_families or len(self.recipe_families) != len(
            set(self.recipe_families)
        ):
            raise ValueError("recipe_families must be non-empty and unique")
        if any(not family for family in self.recipe_families):
            raise ValueError("recipe_families must contain non-empty strings")
        if not self.tp_sizes or any(tp <= 0 for tp in self.tp_sizes):
            raise ValueError("tp_sizes must contain positive values")
        if len(self.tp_sizes) != len(set(self.tp_sizes)):
            raise ValueError("tp_sizes must be unique")

        if self.physical_intermediate_alignment is not None and (
            self.physical_intermediate_alignment <= 0
        ):
            raise ValueError("physical_intermediate_alignment must be positive when set")

@dataclass(frozen=True, kw_only=True)
class MoeBenchmarkPreset:
    preset_id: str
    model_id: str
    recipe_id: str
    tp_size: int

    def __post_init__(self) -> None:
        if not self.preset_id or not self.model_id or not self.recipe_id:
            raise ValueError("MoE benchmark preset fields must be non-empty")
        if self.tp_size <= 0:
            raise ValueError("MoE benchmark preset TP size must be positive")


@dataclass(frozen=True, kw_only=True)
class MoeGeometryAlias:
    model_id: str
    tp_size: int
    global_intermediate_size: int
    logical_intermediate_sizes: tuple[int, ...]
    physical_intermediate_size: int
    padding_per_tp_group: int
    native_top_k: int
    source: str


@dataclass(frozen=True, kw_only=True)
class MoePhysicalGeometry:
    recipe: MoeRecipe
    activation: str
    num_experts: int
    hidden_size: int
    intermediate_size: int
    aliases: tuple[MoeGeometryAlias, ...]

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.recipe.recipe_id,
            self.activation,
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
        )

    @property
    def native_top_ks(self) -> frozenset[int]:
        return frozenset(alias.native_top_k for alias in self.aliases)


@dataclass(frozen=True, kw_only=True)
class MoeSweepCase:
    case_id: str
    geometry: MoePhysicalGeometry
    top_k: int
    num_tokens: int
    route_pattern: str

    @property
    def routed_rows(self) -> int:
        return self.num_tokens * self.top_k

    @property
    def is_model_native_top_k(self) -> bool:
        return self.top_k in self.geometry.native_top_ks

    def query(self) -> dict[str, object]:
        return {
            "activation": self.geometry.activation,
            "hidden_size": self.geometry.hidden_size,
            "intermediate_size": self.geometry.intermediate_size,
            "num_experts": self.geometry.num_experts,
            "num_tokens": self.num_tokens,
            "quant_mode": self.geometry.recipe.quant_mode,
            "routed_rows": self.routed_rows,
            "source_format": self.geometry.recipe.source_format,
            "top_k": self.top_k,
        }


MOE_RECIPES = (
    MoeRecipe(
        recipe_id="modelopt-nvfp4-auto",
        family_id="modelopt-nvfp4",
        quant_mode="nvfp4_auto",
        source_format="modelopt_nvfp4",
        intermediate_alignment=16,
        minimum_intermediate_size=16,
        compatible_activations=("silu",),
    ),
    MoeRecipe(
        recipe_id="modelopt-nvfp4",
        family_id="modelopt-nvfp4",
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        intermediate_alignment=16,
        minimum_intermediate_size=16,
        compatible_activations=("silu", "situ", "swigluoai_uninterleave", "relu2"),
    ),
    MoeRecipe(
        recipe_id="modelopt-w4a16",
        family_id="modelopt-nvfp4",
        quant_mode="w4a16",
        source_format="modelopt_nvfp4",
        intermediate_alignment=64,
        minimum_intermediate_size=64,
        compatible_activations=("silu", "situ", "swigluoai_uninterleave", "relu2"),
    ),
    MoeRecipe(
        recipe_id="modelopt-w4a8-nvfp4",
        family_id="modelopt-nvfp4",
        quant_mode="w4a8_nvfp4",
        source_format="modelopt_nvfp4",
        intermediate_alignment=32,
        minimum_intermediate_size=32,
        compatible_activations=("silu", "situ"),
    ),
    MoeRecipe(
        recipe_id="e8m0-w4a16",
        family_id="fp4-e8m0-k32",
        quant_mode="w4a16",
        source_format="fp4_e8m0_k32",
        intermediate_alignment=16,
        minimum_intermediate_size=16,
        compatible_activations=("silu", "situ", "swigluoai_uninterleave", "relu2"),
    ),
    MoeRecipe(
        recipe_id="e8m0-w4a8",
        family_id="fp4-e8m0-k32",
        quant_mode="w4a8_mx",
        source_format="fp4_e8m0_k32",
        intermediate_alignment=32,
        minimum_intermediate_size=32,
        compatible_activations=("silu", "situ", "relu2"),
    ),
    MoeRecipe(
        recipe_id="compressed-tensors-w4a16",
        family_id="compressed-tensors",
        quant_mode="w4a16",
        source_format="compressed_tensors",
        intermediate_alignment=128,
        minimum_intermediate_size=128,
        compatible_activations=("silu", "situ", "swigluoai_uninterleave", "relu2"),
    ),
    MoeRecipe(
        recipe_id="trellis-glm-w4a16",
        family_id="trellis-glm-mcg",
        quant_mode="w4a16",
        source_format="b12x_trellis",
        intermediate_alignment=128,
        minimum_intermediate_size=128,
        compatible_activations=("silu",),
        trellis_variant="glm-mcg-projection-tiered",
    ),
    MoeRecipe(
        recipe_id="trellis-k3-w4a16",
        family_id="trellis-k3-lut",
        quant_mode="w4a16",
        source_format="b12x_trellis",
        intermediate_alignment=256,
        minimum_intermediate_size=256,
        compatible_activations=("situ",),
        trellis_variant="k3-lut-uniform-hadamard",
    ),
)


COMMON_MOE_MODELS = (
    MoeModelGeometry(
        model_id="deepseek-v4.1-flash",
        hidden_size=5120,
        intermediate_size=2304,
        num_experts=384,
        native_top_k=6,
        activation="silu",
        recipe_families=("fp4-e8m0-k32",),
        physical_intermediate_alignment=32,
        source="DeepSeek-V4.1-Flash inference/model.py:830-904",
        tp_sizes=(1, 4, 8),
    ),
    MoeModelGeometry(
        model_id="deepseek-v4.1-dspark",
        hidden_size=5120,
        intermediate_size=2304,
        num_experts=128,
        native_top_k=3,
        activation="silu",
        recipe_families=("fp4-e8m0-k32",),
        physical_intermediate_alignment=32,
        source="DeepSeek-V4.1-Flash inference/model.py:830-904 DSpark expert geometry",
        tp_sizes=(1, 4, 8),
    ),
    MoeModelGeometry(
        model_id="qwen3.5-35b-a3b",
        hidden_size=2048,
        intermediate_size=512,
        num_experts=256,
        native_top_k=8,
        activation="silu",
        recipe_families=("modelopt-nvfp4",),
        source="Qwen3.5-35B-A3B-NVFP4 config.json",
    ),
    MoeModelGeometry(
        model_id="qwen3.5-122b-a10b",
        hidden_size=3072,
        intermediate_size=1024,
        num_experts=256,
        native_top_k=8,
        activation="silu",
        recipe_families=("modelopt-nvfp4",),
        source="Qwen3.5-122B-A10B-NVFP4 config.json",
    ),
    MoeModelGeometry(
        model_id="qwen3.5-397b-a17b",
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        native_top_k=10,
        activation="silu",
        recipe_families=("modelopt-nvfp4",),
        source="Qwen3.5-397B-A17B-NVFP4 config.json",
    ),
    MoeModelGeometry(
        model_id="qwen3.8-flash-next-180b",
        hidden_size=2560,
        intermediate_size=640,
        num_experts=512,
        native_top_k=10,
        activation="silu",
        recipe_families=("modelopt-nvfp4",),
        source="Qwen3.8 Flash Next checkpoint config.json",
    ),
    MoeModelGeometry(
        model_id="nvidia-nano3.5",
        hidden_size=2688,
        intermediate_size=1856,
        num_experts=128,
        native_top_k=6,
        activation="relu2",
        recipe_families=("modelopt-nvfp4", "compressed-tensors"),
        source="Nano3.5 W4A16 benchmark profile",
    ),
    MoeModelGeometry(
        model_id="nvidia-nemotron-3-super-120b",
        hidden_size=1024,
        intermediate_size=2688,
        num_experts=512,
        native_top_k=22,
        activation="relu2",
        recipe_families=("modelopt-nvfp4",),
        source="benchmark_moe.MODEL_PROFILES['nemotron-backbone']",
    ),
    MoeModelGeometry(
        model_id="dsv4f-shape",
        hidden_size=6144,
        intermediate_size=2048,
        num_experts=256,
        native_top_k=8,
        activation="silu",
        recipe_families=("modelopt-nvfp4", "fp4-e8m0-k32"),
        source="benchmark_moe dsv4f and dsv4f-nvfp4 shape profiles",
    ),
    MoeModelGeometry(
        model_id="deepseek-v4-flash",
        hidden_size=4096,
        intermediate_size=2048,
        num_experts=256,
        native_top_k=6,
        activation="silu",
        recipe_families=("fp4-e8m0-k32",),
        source="benchmark_moe.MODEL_PROFILES['deepseek-v4-flash']",
    ),
    MoeModelGeometry(
        model_id="minimax-m3",
        hidden_size=6144,
        intermediate_size=3072,
        num_experts=128,
        native_top_k=4,
        activation="swigluoai_uninterleave",
        recipe_families=("modelopt-nvfp4",),
        source="MiniMax-M3 benchmark profile",
    ),
    MoeModelGeometry(
        model_id="laguna-s2.1",
        hidden_size=3072,
        intermediate_size=1024,
        num_experts=256,
        native_top_k=10,
        activation="silu",
        recipe_families=("modelopt-nvfp4",),
        source="Laguna S-2.1 benchmark profile",
    ),
    MoeModelGeometry(
        model_id="minimax-m2.7",
        hidden_size=3072,
        intermediate_size=1536,
        num_experts=256,
        native_top_k=8,
        activation="silu",
        recipe_families=("modelopt-nvfp4",),
        source="benchmark_moe.MODEL_PROFILES['minimax-m27']",
    ),
    MoeModelGeometry(
        model_id="glm-5.1",
        hidden_size=6144,
        intermediate_size=2048,
        num_experts=256,
        native_top_k=8,
        activation="silu",
        recipe_families=("modelopt-nvfp4",),
        source="benchmark_moe.MODEL_PROFILES['glm51']",
    ),
    MoeModelGeometry(
        model_id="glm-5.2",
        hidden_size=6144,
        intermediate_size=2048,
        num_experts=256,
        native_top_k=8,
        activation="silu",
        recipe_families=("modelopt-nvfp4", "trellis-glm-mcg"),
        source="benchmark_moe.MODEL_PROFILES['glm52']",
    ),
    MoeModelGeometry(
        model_id="glm-5.3",
        hidden_size=6144,
        intermediate_size=2048,
        num_experts=256,
        native_top_k=8,
        activation="silu",
        recipe_families=("modelopt-nvfp4",),
        source="GLM-5.3-NVFP4 config.json",
    ),
    MoeModelGeometry(
        model_id="glm-5.3-flash",
        hidden_size=4096,
        intermediate_size=2048,
        num_experts=288,
        native_top_k=8,
        activation="silu",
        recipe_families=("modelopt-nvfp4",),
        source="benchmark_moe.MODEL_PROFILES['glm53-flash-shape']",
    ),
    MoeModelGeometry(
        model_id="kimi-k3",
        hidden_size=3584,
        intermediate_size=3072,
        num_experts=896,
        native_top_k=16,
        activation="situ",
        recipe_families=("trellis-k3-lut",),
        source="Kimi-K3 TP12 production geometry",
    ),
)


MOE_BENCHMARK_PRESETS = (
    MoeBenchmarkPreset(
        preset_id="qwen38-flash-next",
        model_id="qwen3.8-flash-next-180b",
        recipe_id="modelopt-nvfp4",
        tp_size=1,
    ),
    MoeBenchmarkPreset(
        preset_id="qwen38-flash-next-shape",
        model_id="qwen3.8-flash-next-180b",
        recipe_id="modelopt-nvfp4",
        tp_size=1,
    ),
    MoeBenchmarkPreset(
        preset_id="qwen397b",
        model_id="qwen3.5-397b-a17b",
        recipe_id="modelopt-nvfp4",
        tp_size=4,
    ),
    MoeBenchmarkPreset(
        preset_id="nemotron-backbone",
        model_id="nvidia-nemotron-3-super-120b",
        recipe_id="modelopt-nvfp4",
        tp_size=1,
    ),
    MoeBenchmarkPreset(
        preset_id="nano35-w4a16",
        model_id="nvidia-nano3.5",
        recipe_id="modelopt-w4a16",
        tp_size=1,
    ),
    MoeBenchmarkPreset(
        preset_id="nano35-w4a16-shape",
        model_id="nvidia-nano3.5",
        recipe_id="modelopt-w4a16",
        tp_size=1,
    ),
    MoeBenchmarkPreset(
        preset_id="dsv4f",
        model_id="dsv4f-shape",
        recipe_id="e8m0-w4a16",
        tp_size=2,
    ),
    MoeBenchmarkPreset(
        preset_id="dsv4f-nvfp4",
        model_id="dsv4f-shape",
        recipe_id="modelopt-nvfp4",
        tp_size=2,
    ),
    MoeBenchmarkPreset(
        preset_id="minimax-m3-shape",
        model_id="minimax-m3",
        recipe_id="modelopt-nvfp4",
        tp_size=4,
    ),
    MoeBenchmarkPreset(
        preset_id="laguna-s21-shape",
        model_id="laguna-s2.1",
        recipe_id="modelopt-nvfp4",
        tp_size=1,
    ),
    MoeBenchmarkPreset(
        preset_id="deepseek-v4-flash",
        model_id="deepseek-v4-flash",
        recipe_id="e8m0-w4a16",
        tp_size=4,
    ),
    MoeBenchmarkPreset(
        preset_id="deepseek-v4.1-flash",
        model_id="deepseek-v4.1-flash",
        recipe_id="e8m0-w4a8",
        tp_size=4,
    ),
    MoeBenchmarkPreset(
        preset_id="glm51",
        model_id="glm-5.1",
        recipe_id="modelopt-nvfp4",
        tp_size=8,
    ),
    MoeBenchmarkPreset(
        preset_id="glm52",
        model_id="glm-5.2",
        recipe_id="modelopt-w4a8-nvfp4",
        tp_size=8,
    ),
    MoeBenchmarkPreset(
        preset_id="glm53-flash",
        model_id="glm-5.3-flash",
        recipe_id="modelopt-nvfp4",
        tp_size=1,
    ),
    MoeBenchmarkPreset(
        preset_id="glm53-flash-shape",
        model_id="glm-5.3-flash",
        recipe_id="modelopt-w4a16",
        tp_size=1,
    ),
    MoeBenchmarkPreset(
        preset_id="minimax-m27",
        model_id="minimax-m2.7",
        recipe_id="modelopt-nvfp4",
        tp_size=2,
    ),
    MoeBenchmarkPreset(
        preset_id="minimax-m3",
        model_id="minimax-m3",
        recipe_id="modelopt-nvfp4",
        tp_size=2,
    ),
)


def _logical_shard_sizes(total: int, tp_size: int) -> tuple[int, ...]:
    if tp_size > total:
        return ()
    widths = {
        ((rank + 1) * total) // tp_size - (rank * total) // tp_size
        for rank in range(tp_size)
    }
    return tuple(sorted(widths))


def expand_physical_geometries(
    *,
    models: tuple[MoeModelGeometry, ...] = COMMON_MOE_MODELS,
    recipes: tuple[MoeRecipe, ...] = MOE_RECIPES,
) -> tuple[MoePhysicalGeometry, ...]:
    """Expand models over TP and merge identical rank-local kernel geometry."""

    recipes_by_id = {recipe.recipe_id: recipe for recipe in recipes}
    if len(recipes_by_id) != len(recipes):
        raise ValueError("MoE recipe IDs must be unique")
    recipes_by_family: dict[str, list[MoeRecipe]] = {}
    for recipe in recipes:
        recipes_by_family.setdefault(recipe.family_id, []).append(recipe)
    aliases_by_key: dict[tuple[object, ...], list[MoeGeometryAlias]] = {}
    recipe_by_key: dict[tuple[object, ...], MoeRecipe] = {}
    for model in models:
        for family_id in model.recipe_families:
            try:
                family_recipes = recipes_by_family[family_id]
            except KeyError as exc:
                raise ValueError(
                    f"model {model.model_id!r} references unknown recipe family "
                    f"{family_id!r}"
                ) from exc
            compatible_recipes = tuple(
                recipe
                for recipe in family_recipes
                if model.activation in recipe.compatible_activations
            )
            if not compatible_recipes:
                raise ValueError(
                    f"recipe family {family_id!r} has no recipe compatible with "
                    f"activation {model.activation!r}"
                )
            for recipe in compatible_recipes:
                for tp_size in model.tp_sizes:
                    logical_sizes = _logical_shard_sizes(
                        model.intermediate_size, tp_size,
                    )
                    local_experts = model.num_experts
                    if not logical_sizes:
                        continue
                    logical_max = max(logical_sizes)
                    physical_alignment = (
                        model.physical_intermediate_alignment
                        or recipe.intermediate_alignment
                    )
                    physical_size = align_up(
                        max(logical_max, recipe.minimum_intermediate_size),
                        physical_alignment,
                    )
                    key = (
                        recipe.recipe_id,
                        model.activation,
                        local_experts,
                        model.hidden_size,
                        physical_size,
                    )
                    recipe_by_key[key] = recipe
                    aliases_by_key.setdefault(key, []).append(
                        MoeGeometryAlias(
                            model_id=model.model_id,
                            tp_size=tp_size,
                            global_intermediate_size=model.intermediate_size,
                            logical_intermediate_sizes=logical_sizes,
                            physical_intermediate_size=physical_size,
                            padding_per_tp_group=physical_size * tp_size - model.intermediate_size,
                            native_top_k=model.native_top_k,
                            source=model.source,
                        )
                    )
    geometries = []
    for key in sorted(aliases_by_key):
        recipe_id, activation, num_experts, hidden_size, intermediate_size = key
        geometries.append(
            MoePhysicalGeometry(
                recipe=recipe_by_key[key],
                activation=str(activation),
                num_experts=int(num_experts),
                hidden_size=int(hidden_size),
                intermediate_size=int(intermediate_size),
                aliases=tuple(
                    sorted(
                        aliases_by_key[key],
                        key=lambda alias: (alias.model_id, alias.tp_size),
                    )
                ),
            )
        )
    return tuple(geometries)


def _case_id(
    geometry: MoePhysicalGeometry,
    *,
    top_k: int,
    num_tokens: int,
    route_pattern: str,
) -> str:
    key = (
        *geometry.key,
        int(top_k),
        int(num_tokens),
        route_pattern,
    )
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:12]
    geometry_id = (
        f"e{geometry.num_experts}-k{geometry.hidden_size}-n{geometry.intermediate_size}"
    )
    return f"{geometry_id}-tk{top_k}-m{num_tokens}-{route_pattern}-{digest}"


def expand_sweep_cases(
    *,
    geometries: tuple[MoePhysicalGeometry, ...] | None = None,
    top_ks: tuple[int, ...] = COMMON_TOP_K,
    token_counts: tuple[int, ...] = COMMON_PLAN_TOKEN_COUNTS,
    route_patterns: tuple[str, ...] = COMMON_ROUTE_PATTERNS,
) -> tuple[MoeSweepCase, ...]:
    """Cross deduplicated physical shapes with common runtime axes."""

    geometries = geometries or expand_physical_geometries()
    common_top_ks = set(top_ks)
    cases = []
    for geometry in geometries:
        geometry_top_ks = sorted(common_top_ks | geometry.native_top_ks)
        for top_k in geometry_top_ks:
            if not 1 <= top_k <= geometry.num_experts:
                continue
            for num_tokens in token_counts:
                if num_tokens <= 0:
                    raise ValueError("token counts must be positive")
                for route_pattern in route_patterns:
                    if not route_pattern:
                        raise ValueError("route patterns must be non-empty")
                    cases.append(
                        MoeSweepCase(
                            case_id=_case_id(
                                geometry,
                                top_k=top_k,
                                num_tokens=num_tokens,
                                route_pattern=route_pattern,
                            ),
                            geometry=geometry,
                            top_k=top_k,
                            num_tokens=num_tokens,
                            route_pattern=route_pattern,
                        )
                    )
    return tuple(cases)


def corpus_manifest() -> dict[str, object]:
    payload = {
        "schema_version": 3,
        "tp_sizes": list(COMMON_TP_SIZES),
        "top_k": list(COMMON_TOP_K),
        "decode_tokens": list(COMMON_DECODE_TOKENS),
        "prefill_token_capacities": list(COMMON_PREFILL_TOKEN_CAPACITIES),
        "route_patterns": list(COMMON_ROUTE_PATTERNS),
        "recipes": [asdict(recipe) for recipe in MOE_RECIPES],
        "models": [asdict(model) for model in COMMON_MOE_MODELS],
        "benchmark_presets": [asdict(preset) for preset in MOE_BENCHMARK_PRESETS],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["corpus_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return payload


__all__ = [
    "COMMON_DECODE_TOKENS",
    "COMMON_MOE_MODELS",
    "COMMON_PLAN_TOKEN_COUNTS",
    "COMMON_PREFILL_TOKEN_CAPACITIES",
    "COMMON_ROUTE_PATTERNS",
    "COMMON_TOP_K",
    "COMMON_TP_SIZES",
    "MOE_BENCHMARK_PRESETS",
    "MOE_RECIPES",
    "MoeBenchmarkPreset",
    "MoeGeometryAlias",
    "MoeModelGeometry",
    "MoePhysicalGeometry",
    "MoeRecipe",
    "MoeSweepCase",
    "align_up",
    "corpus_manifest",
    "expand_physical_geometries",
    "expand_sweep_cases",
]
