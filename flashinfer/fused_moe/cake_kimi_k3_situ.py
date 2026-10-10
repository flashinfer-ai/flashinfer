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

import math
from functools import cache

import torch
import tvm_ffi

from ..jit.cake_kimi_k3_situ import (
    MODULES,
    cake_situ_program,
    cake_situ_stage_entries,
    get_cake_situ_module,
    get_cake_situ_submit,
)
from ..tllm_enums import ActivationType
from ..utils import get_compute_capability

_H, _I, _E, _TOP_K = 3584, 384, 896, 16
_LAYOUT = "trtllm_shuffled_nvfp4_group16"
_STATE_ATTR = "_flashinfer_cake_situ_workspace"
_MAX_TOKENS = 16384
_FORCED_TILE_N16_TOKENS = (32, 64, 128, 256)
_N8_W2A_M16_TOKENS = (16,)
_N32_CLAIM8_TOKENS = (512, 1024)
_MID_WORK5FD_TOKENS = (2048, 4096)
_LARGE_C7_TOKENS = (8192, 16384)
# Architectures whose single-token route runs the fused quantization + routing
# FC1 (three kernels: FC1, FC2, finalize); the others launch the separate
# single-token quantization + router kernel before the FC1 (four kernels).
_M1_FUSED_ARCHES = ("sm_103a",)
# Architectures whose 8192- and 16384-token route runs the FC2 that stores an
# 8-bit per-tile partial with per-tile FP32 scales and the finalize that
# consumes it (eight kernels); the others run the pre-shuffled scale-factor
# route (nine kernels).
_LARGE_Q8I_ARCHES = ("sm_103a",)
# The 512- and 1024-token routes launch the routing kernel as one thread-block
# cluster of this many CTAs (the kernel declares the matching cluster size).
_ROUTE_MC_CLUSTER = 8
# FC2 device-workfeed pools. The 16-token route runs an FC2 program built for
# two resident CTAs per SM (__launch_bounds__(512, 2)), so its pool holds
# 2 * SM // (_H // 128) rows (280 CTAs on a 148-SM part); the 32..256-token
# rows run twelve FC2 CTAs per N-tile (336 CTAs, two per SM) and the 512- and
# 1024-token rows a seven-row pool (196 CTAs). The pool only sets how many
# CTAs share the work: the router seeds the workfeed counter with the pool
# size and every CTA claims its next tile through an atomic increment.
_N8_W2A_M16_FC2_GRID_N_SM_FACTOR = 2
_M64_CLAIM8_FC2_GRID_N = 12
_N32_CLAIM8_FC2_GRID_N = 7
# Pre-shuffled FC1 scale factors (the 16384-token route). The tile-N128 FC1
# consumes K in steps of 512 elements; for every (N-tile, K-step) the writer
# emits the 4096-byte shared-memory image FC1 expects (128 rows x 32 bytes).
_FC1_K_STEP = 512
_FC1_K_TILES = _H // _FC1_K_STEP
_SFB_IMAGE_BYTES = 128 * (_FC1_K_STEP // 16)
_SFB_IMAGE_BLOCKS = _SFB_IMAGE_BYTES // 512

# --------------------------------------------------------------------------
# Route table: selector -> (ordered stages, {stage: logical kernel key},
# FC2 device-workfeed pool rows: None = whole tile range, "sm" = one CTA per
# SM, "sm2" = two CTAs per SM, int = fixed row count).
# --------------------------------------------------------------------------
_GENERIC = (
    "route_reset",
    "quant",
    "route_histogram",
    "route_prefix",
    "route_scatter",
    "fc1",
    "fc2",
    "finalize",
)
_FUSED = ("quant", "fused_router", "fc1", "fc2", "finalize")
_QUANT_ROUTE = ("quant_route", "fc1", "fc2", "finalize")
_M1_FUSED = ("fc1", "fc2", "finalize")
_LARGE = _GENERIC[:5] + ("sfb_shuffle",) + _GENERIC[5:]


def _generic(tile_n, fc1=None, fc2=None, finalize=None, extra=None, stages=_GENERIC):
    keys = {
        "fc1": fc1 or f"fc1:n{tile_n}",
        "fc2": fc2 or f"fc2:n{tile_n}",
        "finalize": finalize or f"finalize:n{tile_n}",
        **(extra or {}),
    }
    # Stage order: the prepared plan and the tests read the route in this order.
    return {stage: keys.get(stage, stage) for stage in stages}


_SELECTORS = {
    "m1": (
        _QUANT_ROUTE,
        {
            "quant_route": "quant_route:m1",
            "fc1": "fc1:n8_small",
            "fc2": "fc2:n8_m1",
            "finalize": "finalize:feature",
        },
        None,
    ),
    "m1_s2a": (
        _M1_FUSED,
        {
            "fc1": "fc1:n8_s2a",
            "fc2": "fc2:n8_m1",
            "finalize": "finalize:feature",
        },
        None,
    ),
    "n8_feature": (
        _QUANT_ROUTE,
        {
            "quant_route": "quant_route:s2b",
            "fc1": "fc1:n8_s3b",
            "fc2": "fc2:n8_workfeed",
            "finalize": "finalize:feature",
        },
        "sm",
    ),
    "n8_w2a_m16": (
        _QUANT_ROUTE,
        {
            "quant_route": "quant_route:s2b",
            "fc1": "fc1:n8_s3b",
            "fc2": "fc2:n8_w2a_s4",
            "finalize": "finalize:feature",
        },
        "sm2",
    ),
    "m64_claim8": (
        _FUSED,
        {
            "quant": "quant",
            "fused_router": "fused_router:single_cta_v16",
            "fc1": "fc1:n16_claim8",
            "fc2": "fc2:n16_claim8",
            "finalize": "finalize:feature",
        },
        _M64_CLAIM8_FC2_GRID_N,
    ),
    "n32_claim8": (
        _FUSED,
        {
            "quant": "quant:n32_claim8",
            "fused_router": "fused_router:cluster_v16_mc8",
            "fc1": "fc1:n32_claim8",
            "fc2": "fc2:n32_claim8",
            "finalize": "finalize:n32_claim8",
        },
        _N32_CLAIM8_FC2_GRID_N,
    ),
    "mid_work5fd": (_GENERIC, _generic(128, fc1="fc1:n128_work5fd"), None),
    "large_c7": (
        _LARGE,
        _generic(
            128,
            fc1="fc1:n128_sfbs",
            extra={"sfb_shuffle": "sfb_shuffle:n128"},
            stages=_LARGE,
        ),
        None,
    ),
    "large_q8i": (
        _GENERIC,
        _generic(128, fc2="fc2:n128_q8i", finalize="finalize:n128_q8i"),
        None,
    ),
    8: (_GENERIC, _generic(8), None),
    16: (_GENERIC, _generic(16), None),
    32: (_GENERIC, _generic(32), None),
    128: (_GENERIC, _generic(128), None),
}


def _tile_n(num_tokens):
    if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
        raise TypeError("num_tokens must be an integer")
    if not 1 <= num_tokens <= _MAX_TOKENS:
        raise ValueError(f"Cake SiTU supports 1 through {_MAX_TOKENS} tokens")
    if num_tokens <= 128:
        return 8
    if num_tokens <= 256:
        return 16
    if num_tokens <= 1024:
        return 32
    return 128


def _selector(arch, num_tokens):
    tile_n = _tile_n(num_tokens)
    if num_tokens == 1:
        return "m1_s2a" if arch in _M1_FUSED_ARCHES else "m1"
    if num_tokens in _N8_W2A_M16_TOKENS:
        return "n8_w2a_m16"
    if num_tokens == 8:
        return "n8_feature"
    if num_tokens in _FORCED_TILE_N16_TOKENS:
        return "m64_claim8"
    if num_tokens in _N32_CLAIM8_TOKENS:
        return "n32_claim8"
    if num_tokens in _MID_WORK5FD_TOKENS:
        return "mid_work5fd"
    if num_tokens in _LARGE_C7_TOKENS:
        return "large_q8i" if arch in _LARGE_Q8I_ARCHES else "large_c7"
    return tile_n


def _route(arch, num_tokens):
    """``(stages, {stage: kernel key})`` of one token count on one architecture."""
    stages, kernels, _pool = _SELECTORS[_selector(arch, num_tokens)]
    return list(stages), dict(kernels)


def _geometry(num_tokens, arch=None):
    tile_n = 16 if num_tokens in _FORCED_TILE_N16_TOKENS else _tile_n(num_tokens)
    total_pairs = num_tokens * _TOP_K
    occupied = min(_E, total_pairs)
    max_tiles = occupied + (total_pairs - occupied) // tile_n
    return tile_n, total_pairs, max_tiles


def _fc2_grid_n(selector, max_tiles, sm_count):
    pool = _SELECTORS[selector][2]
    if pool is None:
        return max_tiles
    if pool == "sm":
        return min(max_tiles, max(1, sm_count // (_H // 128)))
    if pool == "sm2":
        return min(
            max_tiles,
            max(1, _N8_W2A_M16_FC2_GRID_N_SM_FACTOR * sm_count // (_H // 128)),
        )
    return min(max_tiles, pool)


def _sfb_shuffled_bytes(max_tiles):
    return max_tiles * _FC1_K_TILES * _SFB_IMAGE_BYTES


def _workspace_layout(num_tokens, arch=None):
    tile_n, total_pairs, max_tiles = _geometry(num_tokens, arch)
    rows = max_tiles * tile_n
    fields = (
        ("situ_beta", torch.float32, (_E,), 4),
        ("situ_linear_beta", torch.float32, (_E,), 4),
        ("x_packed", torch.uint8, (num_tokens, _H // 2), 1),
        ("x_scales", torch.uint8, (num_tokens, _H // 16), 1),
        ("expert_counts", torch.int32, (_E,), 4),
        ("expert_tile_offsets", torch.int32, (_E,), 4),
        ("expert_scatter_offsets", torch.int32, (_E,), 4),
        ("tile_expert", torch.int32, (max_tiles,), 4),
        ("tile_mn_limit", torch.int32, (max_tiles,), 4),
        ("total_tiles", torch.int32, (1,), 4),
        ("route_map", torch.int32, (rows + 1,), 4),
        ("token_to_permuted", torch.int32, (total_pairs,), 4),
        ("intermediate_packed", torch.uint8, (rows, _I // 2), 1),
        ("intermediate_scales", torch.uint8, (rows, _I // 16), 1),
        ("expert_output", torch.bfloat16, (rows, _H), 2),
    )
    # The layout is architecture-independent: a field that any architecture's
    # selector of this token count needs (the FC2 work counter, the
    # pre-shuffled scale images) is present on both.
    selectors = {_selector(name, num_tokens) for name in ("sm_100a", "sm_103a")}
    if any(_SELECTORS[selector][2] is not None for selector in selectors):
        fields += (("fc2_work_counter", torch.int32, (1,), 4),)
    if "large_c7" in selectors:
        fields += (("sfb_shuffled", torch.uint8, (_sfb_shuffled_bytes(max_tiles),), 1),)
    layout, offset = {}, 0
    for name, dtype, shape, element_bytes in fields:
        offset = (offset + 127) // 128 * 128
        nbytes = math.prod(shape) * element_bytes
        layout[name] = (offset, nbytes, dtype, shape)
        offset += nbytes
    return layout, (offset + 127) // 128 * 128


def _validate_contract(options, *, size_query=False):
    if options["activation_type"] != ActivationType.Situ:
        raise ValueError("backend='cake' requires activation_type=ActivationType.Situ")
    if options["output_dtype"] != torch.bfloat16:
        raise ValueError("Cake SiTU output_dtype must be torch.bfloat16")
    if options["tp_size"] != 8 or not 0 <= options["tp_rank"] < 8:
        raise ValueError("Cake SiTU expects TP8-local weights and tp_size=8")
    if options["ep_size"] != 1 or options["ep_rank"] != 0:
        raise ValueError("Cake SiTU requires all 896 experts on the local TP rank")
    unsupported = (
        "min_latency_mode",
        "use_deepseek_fp8_block_scale",
        "use_w4_group_scaling",
        "use_mxfp8_act_scaling",
        "use_packed_weights",
        "use_wfp4afp8_humming",
    )
    if any(options[key] for key in unsupported):
        raise ValueError("Cake SiTU requires the documented group-16 NVFP4 layout")
    if not options["use_fused_finalize"]:
        raise ValueError("Cake SiTU always includes routed finalization")
    if size_query:
        if (
            options["hidden_size"],
            options["intermediate_size"],
            options["num_experts_total"],
            options["top_k"],
        ) != (_H, _I, _E, _TOP_K):
            raise ValueError("Cake SiTU requires H=3584, I=384, E=896, top_k=16")
        if (
            options["x_dtype"] != torch.bfloat16
            or options["weight_dtype"] != torch.uint8
        ):
            raise ValueError("Cake SiTU requires BF16 input and packed uint8 weights")
        return
    if options["cluster_size"] != 1 or options["cluster_rank"] != 0:
        raise ValueError("Cake SiTU does not perform cross-rank collective operations")
    if options["enable_alltoall"]:
        raise ValueError("Cake SiTU does not perform all-to-all")
    if options["enable_pdl"] is False:
        raise ValueError("Cake SiTU preserves its generated per-stage PDL policy")
    if options["profile_ids"] is not None:
        raise ValueError("Cake SiTU uses its prepared metadata route")
    if any(
        options[key] is not None
        for key in (
            "fc1_expert_biases",
            "fc2_expert_biases",
            "input_sf",
            "swiglu_alpha",
            "swiglu_beta",
            "swiglu_limit",
        )
    ):
        raise ValueError("Cake SiTU requires unbiased BF16 input and SiTU parameters")


def _cake_situ_workspace_size(options):
    # No device query, module load, CUDA allocation or initialization.
    _validate_contract(options, size_query=True)
    max_tokens = options["max_num_tokens"]
    # ``_workspace_layout`` is not monotonic in ``num_tokens``: the token counts
    # that force tile-N16 need more scratch rows than the tile-N8 counts that
    # follow them, and the pre-shuffled scale-factor route carries
    # ``sfb_shuffled`` (the 8192-token layout is larger than the ordinary
    # layout up to 8647 tokens), so a maximum-size buffer covers every forced
    # and pre-shuffled count at or below ``max_num_tokens`` as well as
    # ``max_num_tokens`` itself.
    return max(
        _workspace_layout(num_tokens)[1]
        for num_tokens in (max_tokens, *_FORCED_TILE_N16_TOKENS, *_LARGE_C7_TOKENS)
        if num_tokens <= max_tokens
    )


def _tensor(tensor, name, *, shape, dtype, device):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    allowed = dtype if isinstance(dtype, tuple) else (dtype,)
    if (
        tuple(tensor.shape) != shape
        or tensor.dtype not in allowed
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous, shape {shape}, dtype {allowed}, device {device}"
        )
    return tensor


def _workspace_key(workspace):
    return (workspace.data_ptr(), workspace.numel(), workspace.device)


@cache
def _device_facts(device_index):
    """Architecture name and SM count of one CUDA device, queried once."""
    major, minor = get_compute_capability(torch.device("cuda", device_index))
    arch = {(10, 0): "sm_100a", (10, 3): "sm_103a"}.get((major, minor))
    if arch is None:
        raise ValueError("Cake SiTU requires SM100a or SM103a")
    return arch, torch.cuda.get_device_properties(device_index).multi_processor_count


class _Slot:
    """A call-time argument of a stage (input, routing, weights, scales, output)."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


# Per-call tensors by the stage argument names that bind them. Everything
# else in a stage's argument plan is fixed at prepare time.
_CALL_ARGS = {
    "quant": {"x": "x", "qx": "qx"},
    "quant_route": {
        "x": "x",
        "qx": "qx",
        "topk_ids": "ids",
        "s2b_x": "x",
        "s2b_qx": "qx",
    },
    "fused_router": {"topk_ids": "ids"},
    "route_histogram": {"topk_ids": "ids"},
    "route_scatter": {"topk_ids": "ids"},
    "fc1": {
        "A": "w1",
        "SFA": "sf1",
        "scale_c": "qa",
        "scale_gate": "decode1",
        "clamp_limit": "qa",
        "act_alpha": "alpha",
        "act_beta": "beta",
        # The single-token fused program quantizes and routes inside FC1: its
        # plan also names the quantizer and router inputs.
        "x": "x",
        "qx": "qx",
        "topk_ids": "ids",
    },
    "fc2": {"A": "w2", "SFA": "sf2", "scale_c": "decode2"},
    "finalize": {"route_weights": "weights", "out": "out"},
}


def _plan_names(prepared, stage, name):
    """Whether the program bound to ``stage`` takes an argument called ``name``."""
    plan = MODULES[prepared["modules"][stage]]["arg_plan"]
    return any(item == name for _kind, item in plan)


def _q8_partial_views(views, tile_n, max_tiles):
    """The 8-bit FC2 partial ``[rows, H]`` and its per-tile FP32 scales.

    The programs that store the FC2 result as an 8-bit per-tile partial write
    ``max_tiles * (H // 128)`` FP32 scales next to it; both are carved out of
    the BF16 ``expert_output`` scratch, whose byte size (``2 * rows * H``)
    exceeds ``rows * H + 4 * max_tiles * (H // 128)`` for every token count,
    so the workspace layout and size are those of every other route.
    """
    rows = max_tiles * tile_n
    scratch = views["expert_output"].view(torch.uint8).view(-1)
    partial = scratch[: rows * _H].view(rows, _H)
    scales = scratch[rows * _H : rows * _H + max_tiles * (_H // 128) * 4]
    return partial, scales.view(torch.float32)


def _stage_constants(stage, views, prepared, num_tokens):
    """Prepare-time values of one stage's arguments (grids, scratch views, scalars)."""
    tile_n = prepared["tile_n"]
    total_pairs = prepared["total_pairs"]
    max_tiles = prepared["max_tiles"]
    selector = prepared["selector"]
    workfeed = _SELECTORS[selector][2] is not None
    route_grid = ((total_pairs + 255) // 256, 1, 1)
    route_views = {
        name: views[name]
        for name in (
            "expert_counts",
            "expert_tile_offsets",
            "expert_scatter_offsets",
            "route_map",
            "token_to_permuted",
            "tile_expert",
            "tile_mn_limit",
            "total_tiles",
        )
    }
    if stage == "quant":
        return dict(
            grid=(num_tokens, 1, 1),
            packed=views["x_packed"],
            scales=views["x_scales"],
            M=num_tokens,
        )
    if stage == "quant_route":
        values = dict(
            # The single-token program runs one block; the s2b program's block 0
            # routes and blocks 1..num_tokens quantize.
            grid=(1, 1, 1) if selector == "m1" else (num_tokens + 1, 1, 1),
            packed=views["x_packed"],
            scales=views["x_scales"],
            s2b_packed=views["x_packed"],
            s2b_scales=views["x_scales"],
            s2b_num_tokens=num_tokens,
            M=num_tokens,
            total_pairs=total_pairs,
            num_experts=_E,
            max_tiles=max_tiles,
            top_k=_TOP_K,
            tile_n=tile_n,
            **route_views,
        )
        if workfeed:
            values.update(
                fc2_work_counter=views["fc2_work_counter"],
                fc2_pool_ctas=prepared["fc2_pool_ctas"],
            )
        return values
    if stage == "fused_router":
        return dict(
            grid=(_ROUTE_MC_CLUSTER, 1, 1) if selector == "n32_claim8" else (1, 1, 1),
            total_pairs=total_pairs,
            num_experts=_E,
            max_tiles=max_tiles,
            top_k=_TOP_K,
            tile_n=tile_n,
            fc2_work_counter=views["fc2_work_counter"],
            fc2_pool_ctas=prepared["fc2_pool_ctas"],
            **route_views,
        )
    if stage == "route_reset":
        reset_items = max(_E, max_tiles * tile_n + 1)
        values = dict(
            grid=((reset_items + 255) // 256, 1, 1),
            num_experts=_E,
            max_tiles=max_tiles,
            tile_n=tile_n,
            **route_views,
        )
        if workfeed:
            values.update(
                fc2_work_counter=views["fc2_work_counter"],
                fc2_pool_ctas=prepared["fc2_pool_ctas"],
            )
        return values
    if stage == "route_histogram":
        return dict(grid=route_grid, total_pairs=total_pairs, **route_views)
    if stage == "route_prefix":
        return dict(
            grid=(1, 1, 1),
            num_experts=_E,
            tile_n=tile_n,
            tile_n_shift=tile_n.bit_length() - 1,
            **route_views,
        )
    if stage == "route_scatter":
        return dict(
            grid=route_grid,
            total_pairs=total_pairs,
            top_k=_TOP_K,
            tile_n=tile_n,
            tile_n_shift=tile_n.bit_length() - 1,
            **route_views,
        )
    if stage == "sfb_shuffle":
        return dict(
            grid=(max_tiles, 1, 1),
            SFB=views["x_scales"],
            SFBS=views["sfb_shuffled"],
            K=_H,
            K_tiles=_FC1_K_TILES,
            grid_n=max_tiles,
            **route_views,
        )
    if stage == "fc1":
        values = dict(
            grid=(_I // 64, max_tiles, 1),
            B=views["x_packed"],
            SFB=views["x_scales"],
            C=views["intermediate_packed"],
            SFC=views["intermediate_scales"],
            M_out=_I,
            K=_H,
            grid_m=_I // 64,
            grid_n=max_tiles,
            K_tiles=_FC1_K_TILES,
            **route_views,
        )
        if selector == "large_c7":
            values["SFBS"] = views["sfb_shuffled"].view(
                max_tiles, _FC1_K_TILES * _SFB_IMAGE_BLOCKS, 2, 256
            )
        return values
    if stage == "fc2":
        values = dict(
            grid=(_H // 128, prepared["fc2_grid_n"], 1),
            B=views["intermediate_packed"].view(max_tiles, tile_n, _I // 2),
            SFB=(
                views["intermediate_scales"].view(max_tiles, _I // 64, 2, 256)
                if tile_n == 128
                else views["intermediate_scales"].view(
                    max_tiles * (tile_n // 8), _I // 64, 32
                )
            ),
            C_tma=views["expert_output"],
            C=views["expert_output"],
            M=_H,
            K=_I,
            grid_m=_H // 128,
            grid_n=max_tiles,
            K_tiles=1 if selector in ("m64_claim8", "n32_claim8") else 2,
            **route_views,
        )
        if workfeed:
            values.update(
                num_non_exiting_ctas=views["total_tiles"],
                work_counter=views["fc2_work_counter"],
            )
        if _plan_names(prepared, stage, "partial_scale"):
            # The program stores an 8-bit per-tile partial and its FP32 scales
            # instead of the BF16 expert output.
            partial, partial_scale = _q8_partial_views(views, tile_n, max_tiles)
            values.update(C_tma=partial, C=partial, partial_scale=partial_scale)
        return values
    if stage == "finalize":
        feature = selector in ("m1", "m1_s2a", "n8_feature", "n8_w2a_m16", "m64_claim8")
        vec = 2 if tile_n == 8 else 8
        values = dict(
            grid=(
                num_tokens,
                1
                if selector == "n32_claim8"
                else 4
                if feature
                else (_H + 128 * vec - 1) // (128 * vec),
                1,
            ),
            expert_output=views["expert_output"],
            token_to_permuted=views["token_to_permuted"],
            M=num_tokens,
        )
        if _plan_names(prepared, stage, "partial_scale"):
            # The program reads the FC2 stage's 8-bit partial and its scales.
            partial, partial_scale = _q8_partial_views(views, tile_n, max_tiles)
            values.update(expert_output=partial, partial_scale=partial_scale)
        return values
    raise ValueError(f"unknown SiTU stage {stage!r}")


def _flatten(stage, arg_plan, constants):
    """Positional argument vector of one stage: prepare-time values and call slots."""
    slots = _CALL_ARGS.get(stage, {})
    flat = []
    for kind, name in arg_plan:
        if kind == "grid":
            flat.append(constants["grid"][("grid_x", "grid_y", "grid_z").index(name)])
        elif name in slots:
            flat.append(_Slot(slots[name]))
        elif name in constants:
            flat.append(constants[name])
        else:
            raise ValueError(f"stage {stage} argument {name!r} has no prepared value")
    return flat


def cake_fused_moe_prepare_workspace(
    workspace_buffer,
    num_tokens,
    *,
    backend="cake",
    weight_layout=_LAYOUT,
):
    """Prepare a caller-owned SiTU workspace outside graph capture.

    This helper initializes SiTU's default per-expert 4/25 tensors, loads the
    stage programs of the token count's route and pre-flattens every launch
    argument that does not change between calls. It allocates no tensor
    storage. The caller allocates ``workspace_buffer`` using
    ``cutlass_fused_moe_workspace_size``. Call once for every token count that
    will use this buffer, on the stream that will submit work (or establish
    ordinary caller-side stream ordering).

    ``weight_layout='trtllm_shuffled_nvfp4_group16'`` declares the physical
    packed weight/block-scale layout described in the SiTU backend guide.
    The buffer and its prepared views must not be resized or used concurrently.
    The helper owns no graph. A capture/replay owner may use the prepared call.
    """
    if backend != "cake" or weight_layout != _LAYOUT:
        raise ValueError(
            "this preparation helper requires the documented Cake SiTU layout"
        )
    if (
        not isinstance(workspace_buffer, torch.Tensor)
        or not workspace_buffer.is_cuda
        or workspace_buffer.dtype not in (torch.uint8, torch.int8)
        or workspace_buffer.ndim != 1
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.data_ptr() % 128
    ):
        raise ValueError(
            "workspace_buffer must be contiguous CUDA uint8/int8 and 128-byte aligned"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare the SiTU workspace before graph capture")
    with torch.cuda.device(workspace_buffer.device):
        arch, sm_count = _device_facts(workspace_buffer.device.index)
        layout, nbytes = _workspace_layout(num_tokens, arch)
        if workspace_buffer.numel() < nbytes:
            raise ValueError(f"workspace_buffer needs at least {nbytes} bytes")
        key = _workspace_key(workspace_buffer)
        state = getattr(workspace_buffer, _STATE_ATTR, None)
        if state is None or state["buffer_key"] != key:
            state = {
                "buffer_key": key,
                "arch": arch,
                "weight_layout": weight_layout,
                "shapes": {},
            }
            setattr(workspace_buffer, _STATE_ATTR, state)
        views = {
            name: workspace_buffer.narrow(0, offset, size).view(dtype).view(shape)
            for name, (offset, size, dtype, shape) in layout.items()
        }
        # These two fixed-prefix regions are shared by all prepared shapes.
        views["situ_beta"].fill_(4.0)
        views["situ_linear_beta"].fill_(25.0)
        selector = _selector(arch, num_tokens)
        stages, kernels = _route(arch, num_tokens)
        tile_n, total_pairs, max_tiles = _geometry(num_tokens, arch)
        fc2_grid_n = _fc2_grid_n(selector, max_tiles, sm_count)
        prepared = {
            "views": views,
            "arch": arch,
            "selector": selector,
            "tile_n": tile_n,
            "total_pairs": total_pairs,
            "max_tiles": max_tiles,
            "fc2_grid_n": fc2_grid_n,
            "fc2_pool_ctas": (_H // 128) * fc2_grid_n,
            "stages": stages,
            "modules": {
                stage: cake_situ_program(arch, kernels[stage]) for stage in stages
            },
        }
        # One flat submission plan: [prepare, submit, argc, args...] per stage
        # (the stage shim's two-phase entry points and its argument vector), with
        # the call-dependent slots recorded by position and filled at call time.
        launches, plan, slots = [], [], []
        for stage in stages:
            name = prepared["modules"][stage]
            module = get_cake_situ_module(name, arch)
            record = MODULES[name]
            constants = _stage_constants(stage, views, prepared, num_tokens)
            flat = _flatten(stage, record["arg_plan"], constants)
            launches.append((getattr(module, record["ffi_entry"]), flat))
            plan += [*cake_situ_stage_entries(name, arch), len(flat)]
            for value in flat:
                if type(value) is _Slot:
                    slots.append((len(plan), value.name))
                    plan.append(None)
                else:
                    plan.append(value)
        prepared["launches"] = launches
        prepared["submit"] = (get_cake_situ_submit(arch).run, plan, tuple(slots))
        state["shapes"][num_tokens] = prepared
    return workspace_buffer


def _prepared(workspace, num_tokens):
    if workspace is None:
        raise ValueError(
            "backend='cake' requires an explicitly prepared workspace_buffer"
        )
    state = getattr(workspace, _STATE_ATTR, None)
    if state is None or state["buffer_key"] != _workspace_key(workspace):
        raise ValueError("call cake_fused_moe_prepare_workspace before submitting SiTU")
    if num_tokens not in state["shapes"]:
        raise ValueError(f"workspace is not prepared for {num_tokens} tokens")
    return state["shapes"][num_tokens]


def _cake_situ_workspace_views(workspace, num_tokens):
    """Actual live scratch objects, also used for direct-launch parity checks."""
    return {
        name: tensor
        for name, tensor in _prepared(workspace, num_tokens)["views"].items()
        if name not in {"situ_beta", "situ_linear_beta"}
    }


def _cake_situ_call_args(options, prepared):
    """Validate the per-call tensors once and name them for the launch slots."""
    x = options["input"]
    num_tokens = x.shape[0]
    device = x.device
    views = prepared["views"]
    _tensor(x, "input", shape=(num_tokens, _H), dtype=torch.bfloat16, device=device)
    if not x.is_cuda:
        raise ValueError("Cake SiTU input must be on CUDA")
    if options["workspace_buffer"].device != device:
        raise ValueError("workspace_buffer and input must be on the same device")
    ids = _tensor(
        options["token_selected_experts"],
        "token_selected_experts",
        shape=(num_tokens, _TOP_K),
        dtype=torch.int32,
        device=device,
    )
    weights = _tensor(
        options["token_final_scales"],
        "token_final_scales",
        shape=(num_tokens, _TOP_K),
        dtype=torch.bfloat16,
        device=device,
    )
    w1 = _tensor(
        options["fc1_expert_weights"],
        "fc1_expert_weights",
        shape=(_E, 2 * _I, _H // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2 = _tensor(
        options["fc2_expert_weights"],
        "fc2_expert_weights",
        shape=(_E, _H, _I // 2),
        dtype=torch.uint8,
        device=device,
    )
    out = _tensor(
        options["output"],
        "output",
        shape=(num_tokens, _H),
        dtype=torch.bfloat16,
        device=device,
    )
    scales = options["quant_scales"]
    if not isinstance(scales, (list, tuple)) or len(scales) != 6:
        raise ValueError("Cake SiTU requires the six NVFP4 scale tensors")
    qx, sf1, decode1, qa, sf2, decode2 = scales
    if qx.numel() != 1:
        raise ValueError("qX must contain one float32 value")
    _tensor(qx, "qX", shape=tuple(qx.shape), dtype=torch.float32, device=device)
    sf_dtype = (torch.uint8, torch.float8_e4m3fn)
    _tensor(
        sf1,
        "FC1 block scales",
        shape=(_E, 2 * _I, _H // 16),
        dtype=sf_dtype,
        device=device,
    )
    _tensor(
        sf2, "FC2 block scales", shape=(_E, _H, _I // 16), dtype=sf_dtype, device=device
    )
    for name, tensor in (
        ("FC1 dequant scale", decode1),
        ("qA", qa),
        ("FC2 dequant scale", decode2),
    ):
        _tensor(tensor, name, shape=(_E,), dtype=torch.float32, device=device)
    alpha = options["situ_beta"]
    beta = options["situ_linear_beta"]
    if alpha is None:
        alpha = views["situ_beta"]
    if beta is None:
        beta = views["situ_linear_beta"]
    _tensor(alpha, "situ_beta", shape=(_E,), dtype=torch.float32, device=device)
    _tensor(beta, "situ_linear_beta", shape=(_E,), dtype=torch.float32, device=device)
    return dict(
        x=x,
        qx=qx,
        ids=ids.view(-1),
        weights=weights.view(-1),
        w1=w1,
        sf1=sf1.view(torch.uint8).view(_E * (_I // 64), _H // 64, 2, 256),
        w2=w2,
        sf2=sf2.view(torch.uint8).view(_E * (_H // 128), _I // 64, 2, 256),
        qa=qa,
        decode1=decode1,
        decode2=decode2,
        alpha=alpha,
        beta=beta,
        out=out,
    )


def _cake_situ_stage_bindings(options, prepared):
    """``stage -> argument name -> value`` of one call, for parity checks and tests."""
    call = _cake_situ_call_args(options, prepared)
    bindings = {}
    for stage in prepared["stages"]:
        record = MODULES[prepared["modules"][stage]]
        constants = _stage_constants(
            stage, prepared["views"], prepared, options["input"].shape[0]
        )
        values = {}
        for value, (_kind, name) in zip(
            _flatten(stage, record["arg_plan"], constants),
            record["arg_plan"],
            strict=True,
        ):
            values[name] = call[value.name] if isinstance(value, _Slot) else value
        values["grid"] = constants["grid"]
        bindings[stage] = values
    return bindings


def _cake_situ_fused_moe(options):
    _validate_contract(options)
    prepared = _prepared(options["workspace_buffer"], options["input"].shape[0])
    call = _cake_situ_call_args(options, prepared)
    # Tensor views and argument packing allocate no CUDA storage. The prepared
    # plan is patched in place at its call slots and submitted through one FFI
    # call; the submit entry prepares every stage (argument checks, tensor maps,
    # kernel parameters) before it launches the first, so the launches reach the
    # torch stream back to back as from the shipped sequence bindings.
    submit, plan, slots = prepared["submit"]
    for index, name in slots:
        plan[index] = call[name]
    with tvm_ffi.use_torch_stream():
        submit(*plan)
    return options["output"]
