from dataclasses import asdict, dataclass
from functools import cache

from .catalog import (
    FUSED_MOE_VARIANTS,
    get_fused_moe_configs,
    kernel_config_permutations,
    supports_shape,
)


@dataclass(frozen=True)
class ColonelsTactic:
    """One fused MoE launch configuration, independent of model dimensions."""

    variant: str = "bf16_sm100"
    block_m: int = 64
    block_n: int = 64
    warp_n: int = 4
    stages: int = 4
    clusters: int = 1
    red_store: bool = False
    cluster_fallback: bool = False
    weight_eviction: bool = False
    epilogue_warps: int = 4
    producer_warps: int = 4
    z_split: int = 1

    def __post_init__(self):
        if self.variant not in FUSED_MOE_VARIANTS:
            raise ValueError(f"Unknown Colonels variant: {self.variant}")
        for name in (
            "block_m",
            "warp_n",
            "stages",
            "clusters",
            "epilogue_warps",
            "producer_warps",
            "z_split",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.block_n) is not int or self.block_n < (
            0 if "gmem" in self.variant else 1
        ):
            raise ValueError("Invalid block_n")
        for name in ("red_store", "cluster_fallback", "weight_eviction"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")

    def kernel_config(self):
        result = asdict(self)
        result.pop("variant")
        result.update(
            sm100=self.variant.startswith("bf16_sm100"),
            gmem="gmem" in self.variant,
            tmem="tmem" in self.variant,
            gemv=self.variant == "bf16_gemv",
            persistent=self.variant.endswith("persistent"),
        )
        return result

    def supports_shape(self, hidden_size, intermediate_size, gated):
        n = intermediate_size * (2 if gated else 1)
        alignment = (
            256
            if self.variant.startswith("w8a8") or self.variant == "bf16_gemv"
            else 128
        )
        n_alignment = (
            16
            if self.variant == "bf16_gemv"
            else 128
            if self.variant.startswith("bf16_sm100")
            else 256
        )
        return (
            hidden_size > 0
            and intermediate_size > 0
            and hidden_size % alignment == 0
            and n % n_alignment == 0
            and supports_shape(self.kernel_config(), hidden_size, n)
        )

    def compilation_parameters(self):
        values = dict(
            BM=self.block_m,
            BN=self.block_n,
            WN=self.warp_n,
            CLUSTERS=self.clusters,
            RED_STORE=int(self.red_store),
            WEIGHT_EVICTION=int(self.weight_eviction),
            EPILOGUE_WARPS=self.epilogue_warps,
            PRODUCER_WARPS=self.producer_warps,
            Z_SPLIT=self.z_split,
        )
        if not self.variant.startswith("bf16_sm100"):
            values["STAGES"] = self.stages
        return tuple(sorted(values.items()))


def _variant(config, arch, fp8):
    if fp8:
        name = "w8a8"
    elif arch == 90:
        name = "bf16"
    else:
        name = "bf16_sm100"
        if config.get("gmem"):
            name += "_gmem"
        elif config.get("tmem"):
            name += "_tmem"
    return name + ("_persistent" if config.get("persistent") else "")


@cache
def get_colonels_tactics(arch, *, fp8=False, exhaustive=False):
    """Enumerate the complete catalogue, or a bounded default search across families."""
    from ...api import SwiGLU

    configs = get_fused_moe_configs(arch, "fp8" if fp8 else "bf16", SwiGLU())
    fields = ColonelsTactic.__dataclass_fields__
    tactics = [
        ColonelsTactic(
            variant=_variant(c, arch, fp8),
            **{k: v for k, v in c.items() if k in fields},
        )
        for c in configs
    ]
    if not fp8:
        tactics += [
            ColonelsTactic(variant="bf16_gemv", **c)
            for c in kernel_config_permutations("bf16_gemv")
        ]
    if exhaustive:
        return tuple(tactics)
    return tuple(
        t
        for t in tactics
        if (
            (
                t.variant == "bf16_gemv"
                and t.block_n == 128
                and t.warp_n == 4
                and t.z_split == 1
            )
            or (
                t.variant != "bf16_gemv"
                and t.block_m in (32, 64, 128)
                and t.block_n in ((0, 64) if arch == 103 else (32, 64))
                and (arch == 103 or t.stages in (2, 3))
                and t.warp_n == (8 if arch == 90 else 4)
                and t.clusters == 1
                and not t.red_store
                and not t.weight_eviction
                and t.epilogue_warps == 4
                and t.producer_warps == 4
            )
            # Large prefill batches also need the clustered GMEM configuration.
            or (
                t.variant == "bf16_sm100_gmem_persistent"
                and t.block_m == 256
                and t.stages == 6
                and t.clusters == 2
                and not t.red_store
                and t.epilogue_warps == 8
                and t.producer_warps == 8
            )
        )
    )
