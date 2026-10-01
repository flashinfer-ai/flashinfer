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

import torch
import tvm_ffi

from ..jit.cake_kimi_k3_situ import (
    PROGRAMS,
    cake_situ_sequence,
    get_cake_situ_module,
)
from ..utils import get_compute_capability
from ..tllm_enums import ActivationType


_H, _I, _E, _TOP_K = 3584, 384, 896, 16
_LAYOUT = "trtllm_shuffled_nvfp4_group16"
_STATE_ATTR = "_flashinfer_cake_situ_workspace"
_N32_CLAIM8_ARCHES = ("sm_100a", "sm_103a")
# The 512- and 1024-token routes launch the routing kernel as one thread-block
# cluster of this many CTAs (the kernel declares the matching cluster size).
_ROUTE_MC_CLUSTER = 8
_M256_C12_ARCHES = ("sm_100a",)
_LARGE_C7_ARCHES = ("sm_103a", "sm_100a")
_LARGE_C7_TOKENS = (16384,)
# Pre-shuffled FC1 scale factors (the ``large_c7`` route). The tile-N128 FC1
# consumes K in steps of 512 elements. For every (N-tile, K-step) the
# scale-factor writer emits the 4096-byte shared-memory image FC1 expects
# (128 rows x 32 bytes; one byte per 16-element group) into ``sfb_shuffled``,
# and FC1 loads each image with a single TMA copy. The FC1 TMA view addresses
# an image as eight 512-byte blocks (one per 64 elements of K) of 2 x 256 bytes.
_FC1_K_STEP = 512
_FC1_K_TILES = _H // _FC1_K_STEP
_SFB_IMAGE_BYTES = 128 * (_FC1_K_STEP // 16)
_SFB_IMAGE_BLOCKS = _SFB_IMAGE_BYTES // 512
# The 16-token SM103 and SM100 routes run an FC2 program built for two resident CTAs per
# SM (four pipeline stages, __launch_bounds__(512, 2)), so its device-workfeed
# pool holds 2 * SM // (_H // 128) rows: 280 CTAs on a 148-SM part instead of
# the 140 of the single-CTA FC2 program. The pool size only sets how many CTAs
# share the work: the router seeds the workfeed counter with the pool size and
# every CTA claims its next tile through an atomic increment, exiting once the
# counter passes the tile count, so a CTA that is not co-resident starts later
# and takes whatever remains; no CTA waits for another.
_N8_W2A_M16_ARCHES = ("sm_103a", "sm_100a")
_N8_W2A_M16_TOKENS = (16,)
_N8_W2A_M16_FC2_GRID_N_SM_FACTOR = 2


def _tile_n(num_tokens):
    if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
        raise TypeError("num_tokens must be an integer")
    if not 1 <= num_tokens <= 16384:
        raise ValueError("Cake SiTU supports 1 through 16384 tokens")
    if num_tokens <= 128:
        return 8
    if num_tokens <= 256:
        return 16
    if num_tokens <= 1024:
        return 32
    return 128


def _geometry(num_tokens, arch=None):
    tile_n = (
        16
        if (num_tokens in (32, 64, 128) and arch in (None, "sm_100a", "sm_103a"))
        or (num_tokens == 256 and arch in (None, "sm_100a", "sm_103a"))
        else _tile_n(num_tokens)
    )
    total_pairs = num_tokens * _TOP_K
    occupied = min(_E, total_pairs)
    max_tiles = occupied + (total_pairs - occupied) // tile_n
    return tile_n, total_pairs, max_tiles


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
    if (
        num_tokens in (8, 16)
        or (num_tokens in (32, 64, 128) and arch in (None, "sm_100a", "sm_103a"))
        or (num_tokens == 256 and arch in (None, "sm_100a", "sm_103a"))
        or (num_tokens in (512, 1024) and arch in (None, *_N32_CLAIM8_ARCHES))
    ):
        fields += (("fc2_work_counter", torch.int32, (1,), 4),)
    if num_tokens in _LARGE_C7_TOKENS and arch in (None, *_LARGE_C7_ARCHES):
        # One image per (N-tile, K-step). For 16384 tokens (2937 tiles) this
        # adds 2937 * 7 * 4096 = 84,209,664 bytes (80.3 MiB) to the layout.
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


_FORCED_TILE_N16_TOKENS = (32, 64, 128, 256)


def _cake_situ_workspace_size(options):
    # No device query, module load, CUDA allocation or initialization.
    _validate_contract(options, size_query=True)
    max_tokens = options["max_num_tokens"]
    # ``_workspace_layout`` is not monotonic in ``num_tokens``: the token counts
    # that force tile-N16 (32 and 64) need more scratch rows than the tile-N8
    # counts that follow them, so a maximum-size buffer must cover every forced
    # count at or below ``max_num_tokens`` as well as ``max_num_tokens`` itself.
    return max(
        _workspace_layout(num_tokens)[1]
        for num_tokens in (max_tokens, *_FORCED_TILE_N16_TOKENS)
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


def cake_fused_moe_prepare_workspace(
    workspace_buffer,
    num_tokens,
    *,
    backend="cake",
    weight_layout=_LAYOUT,
):
    """Prepare a caller-owned SiTU workspace outside graph capture.

    This helper initializes SiTU's default per-expert 4/25 tensors and loads
    the exact generated route. It allocates no tensor storage. The caller
    allocates ``workspace_buffer`` using ``cutlass_fused_moe_workspace_size``.
    Call once for every token count that will use this buffer, on the stream
    that will submit work (or establish ordinary caller-side stream ordering).

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
        major, minor = get_compute_capability(workspace_buffer.device)
        arch = {(10, 0): "sm_100a", (10, 3): "sm_103a"}.get((major, minor))
        if arch is None:
            raise ValueError("Cake SiTU requires SM100a or SM103a")
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
        m64_claim8 = arch in ("sm_100a", "sm_103a") and num_tokens in (32, 64, 128, 256)
        mid_work5fd = arch in ("sm_100a", "sm_103a") and num_tokens in (2048, 4096)
        n32_claim8 = arch in _N32_CLAIM8_ARCHES and num_tokens in (512, 1024)
        m256_c12 = arch in _M256_C12_ARCHES and num_tokens == 256
        large_c7 = arch in _LARGE_C7_ARCHES and num_tokens in _LARGE_C7_TOKENS
        n8_w2a_m16 = arch in _N8_W2A_M16_ARCHES and num_tokens in _N8_W2A_M16_TOKENS
        tile_n, total_pairs, max_tiles = _geometry(num_tokens, arch)
        fc2_device_workfeed = num_tokens in (8, 16) or m64_claim8 or n32_claim8
        fc2_grid_n = max_tiles
        if fc2_device_workfeed:
            sm_count = torch.cuda.get_device_properties(
                workspace_buffer.device
            ).multi_processor_count
            fc2_grid_n = min(max_tiles, max(1, sm_count // (_H // 128)))
            if m64_claim8:
                # The 32- to 256-token routes measured best with a six-row
                # pool: 6 * 28 = 168 FC2 CTAs on the 148-SM B200 and B300.
                fc2_grid_n = min(max_tiles, 6)
                if arch in ("sm_100a", "sm_103a"):
                    fc2_grid_n = min(
                        max_tiles, 12
                    )  # inc23 (N16 claim8 FC2 2 CTAs/SM, s2b2 v39 program): 336 FC2 CTAs on the 148-SM B300 for the sm_103a claim8 rows M32/M64/M128/M256; inc24: the same twelve rows = 336 FC2 CTAs on the 148-SM B200 for the sm_100a claim8 rows M32/M64/M128/M256 (2 CTAs/SM x 148 + 40)
            if n8_w2a_m16:
                # Two resident CTAs per SM: the pool has
                # _N8_W2A_M16_FC2_GRID_N_SM_FACTOR * SM // (_H // 128) rows
                # (10 rows, 280 CTAs, on 148 SMs). It sizes the FC2 grid and is
                # passed to the fused router as fc2_pool_ctas.
                fc2_grid_n = min(
                    max_tiles,
                    max(1, _N8_W2A_M16_FC2_GRID_N_SM_FACTOR * sm_count // (_H // 128)),
                )
            if n32_claim8:
                # The 512- and 1024-token routes measured best with a seven-row
                # pool: 7 * 28 = 196 FC2 CTAs on the 148-SM B200 and B300.
                fc2_grid_n = min(max_tiles, 7)
        feature_finalize = num_tokens in (1, 8, 16) or m64_claim8
        program_key = cake_situ_sequence(
            arch,
            tile_n,
            single_token=num_tokens == 1,
            feature_finalize=feature_finalize,
            m64_claim8=m64_claim8,
            mid_work5fd=mid_work5fd,
            n32_claim8=n32_claim8,
            m256_c12=m256_c12,
            large_c7=large_c7,
            n8_w2a_m16=n8_w2a_m16,
        )
        module = get_cake_situ_module(program_key)
        # The fused quantization + router programs declare the stage in their
        # argument plan; resolve that once here rather than on every call.
        fused_quant_route = any(
            name == "quant_route.s2b_num_tokens"
            for _, name in PROGRAMS[program_key]["arg_plan"]
        )
        state["shapes"][num_tokens] = {
            "views": views,
            "tile_n": tile_n,
            "total_pairs": total_pairs,
            "max_tiles": max_tiles,
            "program_key": program_key,
            "module": module,
            "feature_finalize": feature_finalize,
            "m64_claim8": m64_claim8,
            "mid_work5fd": mid_work5fd,
            "n32_claim8": n32_claim8,
            "m256_c12": m256_c12,
            "large_c7": large_c7,
            "n8_w2a_m16": n8_w2a_m16,
            "fused_quant_route": fused_quant_route,
            "fc2_device_workfeed": fc2_device_workfeed,
            "fc2_grid_n": fc2_grid_n,
            "fc2_pool_ctas": (_H // 128) * fc2_grid_n,
            "entry": getattr(module, PROGRAMS[program_key]["ffi_entry"]),
        }
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


def _cake_situ_stage_bindings(options, prepared):
    x = options["input"]
    num_tokens = x.shape[0]
    device = x.device
    views = prepared["views"]
    tile_n = prepared["tile_n"]
    total_pairs = prepared["total_pairs"]
    max_tiles = prepared["max_tiles"]
    fc2_device_workfeed = prepared["fc2_device_workfeed"]
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
    for name, tensor in (("qX", qx),):
        if tensor.numel() != 1:
            raise ValueError(f"{name} must contain one float32 value")
        _tensor(
            tensor, name, shape=tuple(tensor.shape), dtype=torch.float32, device=device
        )
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
    flat_ids, flat_weights = ids.view(-1), weights.view(-1)
    if num_tokens == 1:
        stages = {
            "quant_route": dict(
                grid=(1, 1, 1),
                x=x,
                qx=qx,
                packed=views["x_packed"],
                scales=views["x_scales"],
                topk_ids=flat_ids,
                **{
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
                },
                M=num_tokens,
            ),
        }
    elif fc2_device_workfeed:
        stages = {
            "quant": dict(
                grid=(num_tokens, 1, 1),
                x=x,
                qx=qx,
                packed=views["x_packed"],
                scales=views["x_scales"],
                M=num_tokens,
            ),
            "fused_router": dict(
                # One cluster of _ROUTE_MC_CLUSTER CTAs for the 512- and
                # 1024-token routes; every other route keeps the single CTA.
                grid=(
                    (_ROUTE_MC_CLUSTER, 1, 1) if prepared["n32_claim8"] else (1, 1, 1)
                ),
                topk_ids=flat_ids,
                **{
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
                },
                total_pairs=total_pairs,
                num_experts=_E,
                max_tiles=max_tiles,
                top_k=_TOP_K,
                tile_n=tile_n,
                fc2_work_counter=views["fc2_work_counter"],
                fc2_pool_ctas=prepared["fc2_pool_ctas"],
            ),
        }
    else:
        route_grid = ((total_pairs + 255) // 256, 1, 1)
        reset_items = max(_E, max_tiles * tile_n + 1)
        stages = {
            "route_reset": dict(
                grid=((reset_items + 255) // 256, 1, 1),
                **{
                    name: views[name]
                    for name in (
                        "expert_counts",
                        "expert_scatter_offsets",
                        "route_map",
                        "tile_expert",
                        "tile_mn_limit",
                        "total_tiles",
                    )
                },
                num_experts=_E,
                max_tiles=max_tiles,
                tile_n=tile_n,
                **(
                    {
                        "fc2_work_counter": views["fc2_work_counter"],
                        "fc2_pool_ctas": prepared["fc2_pool_ctas"],
                    }
                    if fc2_device_workfeed
                    else {}
                ),
            ),
            "quant": dict(
                grid=(num_tokens, 1, 1),
                x=x,
                qx=qx,
                packed=views["x_packed"],
                scales=views["x_scales"],
                M=num_tokens,
            ),
            "route_histogram": dict(
                grid=route_grid,
                topk_ids=flat_ids,
                expert_counts=views["expert_counts"],
                total_pairs=total_pairs,
            ),
            "route_prefix": dict(
                grid=(1, 1, 1),
                **{
                    name: views[name]
                    for name in (
                        "expert_counts",
                        "expert_tile_offsets",
                        "expert_scatter_offsets",
                        "total_tiles",
                    )
                },
                num_experts=_E,
                tile_n=tile_n,
                tile_n_shift=tile_n.bit_length() - 1,
            ),
            "route_scatter": dict(
                grid=route_grid,
                topk_ids=flat_ids,
                **{
                    name: views[name]
                    for name in (
                        "expert_counts",
                        "expert_tile_offsets",
                        "expert_scatter_offsets",
                        "route_map",
                        "token_to_permuted",
                        "tile_expert",
                        "tile_mn_limit",
                    )
                },
                total_pairs=total_pairs,
                top_k=_TOP_K,
                tile_n=tile_n,
                tile_n_shift=tile_n.bit_length() - 1,
            ),
        }
    if prepared["large_c7"]:
        stages["sfb_shuffle"] = dict(
            grid=(max_tiles, 1, 1),
            SFB=views["x_scales"],
            **{
                name: views[name]
                for name in ("route_map", "tile_mn_limit", "total_tiles")
            },
            SFBS=views["sfb_shuffled"],
            K=_H,
            K_tiles=_FC1_K_TILES,
            grid_n=max_tiles,
        )
    stages.update(
        {
            "fc1": dict(
                grid=(_I // 64, max_tiles, 1),
                A=w1,
                B=views["x_packed"],
                SFA=sf1.view(torch.uint8).view(_E * (_I // 64), _H // 64, 2, 256),
                SFB=views["x_scales"],
                C=views["intermediate_packed"],
                SFC=views["intermediate_scales"],
                **{
                    name: views[name]
                    for name in (
                        "route_map",
                        "tile_expert",
                        "tile_mn_limit",
                        "total_tiles",
                    )
                },
                scale_c=qa,
                scale_gate=decode1,
                clamp_limit=qa,
                act_alpha=alpha,
                act_beta=beta,
                M_out=_I,
                K=_H,
                grid_m=_I // 64,
                grid_n=max_tiles,
                K_tiles=_FC1_K_TILES,
                **(
                    {
                        "SFBS": views["sfb_shuffled"].view(
                            max_tiles, _FC1_K_TILES * _SFB_IMAGE_BLOCKS, 2, 256
                        )
                    }
                    if prepared["large_c7"]
                    else {}
                ),
            ),
            "fc2": dict(
                grid=(_H // 128, prepared["fc2_grid_n"], 1),
                A=w2,
                B=views["intermediate_packed"].view(max_tiles, tile_n, _I // 2),
                SFA=sf2.view(torch.uint8).view(_E * (_H // 128), _I // 64, 2, 256),
                SFB=(
                    views["intermediate_scales"].view(max_tiles, _I // 64, 2, 256)
                    if tile_n == 128
                    else views["intermediate_scales"].view(
                        max_tiles * (tile_n // 8), _I // 64, 32
                    )
                ),
                C_tma=views["expert_output"],
                C=views["expert_output"],
                scale_c=decode2,
                **{name: views[name] for name in ("tile_expert", "tile_mn_limit")},
                **(
                    {
                        "num_non_exiting_ctas": views["total_tiles"],
                        "work_counter": views["fc2_work_counter"],
                    }
                    if fc2_device_workfeed
                    else {"total_tiles": views["total_tiles"]}
                ),
                M=_H,
                K=_I,
                grid_m=_H // 128,
                grid_n=max_tiles,
                K_tiles=1 if prepared["m64_claim8"] or prepared["n32_claim8"] else 2,
            ),
            "finalize": dict(
                grid=(
                    num_tokens,
                    1
                    if prepared["n32_claim8"]
                    else 4
                    if prepared["feature_finalize"]
                    else (_H + 128 * (2 if tile_n == 8 else 8) - 1)
                    // (128 * (2 if tile_n == 8 else 8)),
                    1,
                ),
                expert_output=views["expert_output"],
                route_weights=flat_weights,
                token_to_permuted=views["token_to_permuted"],
                out=out,
                M=num_tokens,
            ),
        }
    )
    # The fused quantization + router programs run one launch of grid
    # (num_tokens + 1, 1, 1): block 0 routes and blocks 1..num_tokens quantize.
    # The separate "quant" and "fused_router" entries stay in `stages` but are
    # not referenced by their argument plan.
    if prepared["fused_quant_route"]:
        stages["quant_route"] = dict(
            stages["fused_router"],
            grid=(num_tokens + 1, 1, 1),
            s2b_x=x,
            s2b_qx=qx,
            s2b_packed=views["x_packed"],
            s2b_scales=views["x_scales"],
            s2b_num_tokens=num_tokens,
        )
    return stages


def _cake_situ_flat_args(stages, program_key):
    args = []
    for kind, qualified_name in PROGRAMS[program_key]["arg_plan"]:
        stage, name = qualified_name.split(".", 1)
        values = stages[stage]
        if kind == "grid":
            args.append(values["grid"][("grid_x", "grid_y", "grid_z").index(name)])
        else:
            args.append(values[name])
    return args


def _cake_situ_fused_moe(options):
    _validate_contract(options)
    num_tokens = options["input"].shape[0]
    prepared = _prepared(options["workspace_buffer"], num_tokens)
    stages = _cake_situ_stage_bindings(options, prepared)
    args = _cake_situ_flat_args(stages, prepared["program_key"])
    # Tensor views and Python argument packing allocate no CUDA storage. The
    # generated packed FFI entry submits the complete authored stage sequence.
    with tvm_ffi.use_torch_stream():
        prepared["entry"](*args)
    return options["output"]
