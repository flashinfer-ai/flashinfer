from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ...api import ActivationConfig


FUSED_MOE_VARIANTS = (
    "bf16",
    "bf16_persistent",
    "w8a8",
    "w8a8_persistent",
    "bf16_gemv",
    "bf16_sm100",
    "bf16_sm100_tmem",
    "bf16_sm100_tmem_persistent",
    "bf16_sm100_persistent",
    "bf16_sm100_gmem",
    "bf16_sm100_gmem_persistent",
)

KernelArgs = dict[str, int | bool]


def get_fused_moe_configs(
    arch: Literal[90, 103],
    dtype: Literal["bf16", "fp8"],
    activation: ActivationConfig,
) -> list[KernelArgs]:
    """Return configs for ``fused_moe_bf16`` or ``fused_moe_w8a8``.

    ``arch`` is the SM version (90 or 103); ``fp8`` denotes W8A8
    e4m3 with 128x128 block scaling. Activations use the shared ``SwiGLU()``,
    ``ReLU2()``, or ``ClampedReLU2(limit=...)`` configs. Unsupported activation
    semantics raise NotImplementedError. Supported activations with no kernel
    for the architecture/dtype return an empty list; unknown architectures or
    dtypes raise ValueError.

    Pass each dictionary directly as the wrapper's ``config`` argument. It
    includes the flags selecting persistent, SM100, SMEM/TMEM/GMEM variants.
    GEMV uses a separate wrapper and routing inputs, so it is not included.
    Enumeration does not compile kernels or inspect the current GPU. K/N
    divisibility, shared memory and compiled-kernel availability still need
    checking before launch. GMEM also needs a scratchpad supplied to the wrapper.
    No heuristic pruning is applied.
    """
    # The shared API imports ColonelsTactic, which imports this catalogue.
    from .activations import activation_kind

    if arch not in (90, 103):
        raise ValueError(f"Unsupported architecture {arch!r}; expected 90 or 103")
    if dtype not in ("bf16", "fp8"):
        raise ValueError(f"Unknown dtype {dtype!r}; expected 'bf16' or 'fp8'")
    kind = activation_kind(activation)
    gated = activation.is_gated
    kernels: tuple[str, ...]
    if dtype == "fp8":
        kernels = ("w8a8", "w8a8_persistent") if arch == 90 and gated else ()
    elif arch == 90:
        kernels = ("bf16", "bf16_persistent")
    else:
        kernels = tuple(
            kernel for kernel in FUSED_MOE_VARIANTS if kernel.startswith("bf16_sm100")
        )
    configs: list[KernelArgs] = []
    for kernel in kernels:
        flags: KernelArgs = {"persistent": kernel.endswith("_persistent")}
        if dtype == "bf16":
            flags.update(
                sm100=arch == 103,
                gmem="gmem" in kernel,
                tmem="tmem" in kernel,
                gated=gated,
            )
            flags["activation"] = kind
        configs.extend(
            {**args, **flags} for args in _kernel_arg_permutations(kernel, gated=gated)
        )
    return configs


def kernel_config_permutations(
    kernel: str, *, include_stages: bool = True
) -> list[KernelArgs]:
    """Enumerate tuning parameters; SM100 source generation omits runtime stages."""
    if kernel not in FUSED_MOE_VARIANTS:
        raise ValueError(f"Unknown fused MoE kernel: {kernel}")
    if kernel == "bf16_gemv":
        return [
            {"block_n": bn, "warp_n": wn, "z_split": z}
            for bn, wn in product((16, 32, 64, 128, 256, 512), (1, 2, 4, 8, 16))
            if bn // wn >= 16 and not (bn == 512 and wn == 16)
            for z in (1, 2, 4)
        ]

    sm100 = kernel.startswith("bf16_sm100")
    gmem = "gmem" in kernel
    smem = sm100 and not gmem and "tmem" not in kernel
    block_ms: tuple[int, ...]
    if sm100:
        tiles = [(64, 4), (128, 4), (256, 4)]
        if "tmem" in kernel:
            tiles = [(64, 4), (128, 4)]
        if gmem:
            tiles = [(0, 4)]
        block_ms = (64, 128, 256) if gmem or smem else (64, 128)
    else:
        persistent = kernel.endswith("_persistent")
        tiles = [(32, 8), (64, 8)] if persistent else [(32, 8), (64, 8), (64, 4)]
        block_ms = tuple(range(16, 129, 16) if persistent else range(8, 129, 8))

    configs: list[KernelArgs] = [
        {
            "block_m": bm,
            "block_n": bn,
            "warp_n": wn,
            **({"stages": stages} if include_stages else {}),
        }
        for (bn, wn), bm in product(tiles, block_ms)
        for stages in (
            range(1, 8 if sm100 and bm == 256 else 6) if include_stages else (1,)
        )
    ]
    if kernel == "w8a8_persistent" or gmem or smem:
        configs = [
            {**config, "red_store": red_store}
            for config, red_store in product(configs, (False, True))
            if not gmem or config.get("stages", 1) <= 6 or red_store
        ]
    if smem:
        configs = [
            {**config, "clusters": 2 if config["block_m"] == 256 else 1}
            for config in configs
        ]
    if gmem:
        configs = [
            {**config, "clusters": clusters}
            for config, clusters in product(configs, (1, 2, 4, 8, 16))
            if config["block_m"] != 256 or clusters >= 2
        ]
        configs = [
            {
                **config,
                "weight_eviction": weight_eviction,
                "epilogue_warps": epilogue_warps,
            }
            for config, weight_eviction, epilogue_warps in product(
                configs, (False, True), (4, 8)
            )
        ]
        configs = [
            {**config, "producer_warps": warps}
            for config, warps in product(configs, (4, 8))
        ]
    return configs


def _kernel_arg_permutations(
    kernel: str, *, gated: bool | None = None
) -> list[KernelArgs]:
    configs = kernel_config_permutations(kernel)
    if kernel.startswith("bf16") and kernel != "bf16_gemv":
        gated_modes = (False, True) if gated is None else (gated,)
        configs = [
            {**config, "gated": mode} for config, mode in product(configs, gated_modes)
        ]
    if "gmem" in kernel:
        # Preferred cluster fallback is a launch option, not a separate instantiation.
        configs = [
            {**config, "cluster_fallback": fallback}
            for config in configs
            for fallback in (
                (False, True)
                if config["block_m"] == 256 and config["clusters"] >= 4
                else (False,)
            )
        ]
    return configs


def supports_shape(config: KernelArgs, k: int, n: int) -> bool:
    if config.get("gemv", False):
        # GEMV's block_n is the whole UP tile; warp_n partitions it into warps.
        return k % 256 == 0 and n >= config["block_n"] and n % config["block_n"] == 0
    if config.get("gmem", False):
        groups = config["clusters"] // (2 if config["block_m"] == 256 else 1)
        barrier_groups = (
            groups // 2 if config.get("cluster_fallback", False) else groups
        )
        return (
            (n + 255) // 256 >= groups
            and k % (256 * groups) == 0
            and n <= 64 * 256 * barrier_groups
        )
    tile_n = config["block_n"] * config["warp_n"]
    if config.get("sm100", False):
        if not config.get("tmem", False):
            return tile_n <= (n + 255) // 256 * 256
        if n % 256:
            return config["block_n"] == 64
    return n >= tile_n and n % tile_n == 0
