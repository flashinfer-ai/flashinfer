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
    PROGRAMS, cake_situ_sequence, get_cake_situ_module,
)
from ..utils import get_compute_capability
from ..tllm_enums import ActivationType


_H, _I, _E, _TOP_K = 3584, 384, 896, 16
_LAYOUT = "trtllm_shuffled_nvfp4_group16"
_STATE_ATTR = "_flashinfer_cake_situ_workspace"


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


def _geometry(num_tokens):
    tile_n = _tile_n(num_tokens)
    total_pairs = num_tokens * _TOP_K
    occupied = min(_E, total_pairs)
    max_tiles = occupied + (total_pairs - occupied) // tile_n
    return tile_n, total_pairs, max_tiles


def _workspace_layout(num_tokens):
    tile_n, total_pairs, max_tiles = _geometry(num_tokens)
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
        "min_latency_mode", "use_deepseek_fp8_block_scale", "use_w4_group_scaling",
        "use_mxfp8_act_scaling", "use_packed_weights", "use_wfp4afp8_humming",
    )
    if any(options[key] for key in unsupported):
        raise ValueError("Cake SiTU requires the documented group-16 NVFP4 layout")
    if not options["use_fused_finalize"]:
        raise ValueError("Cake SiTU always includes routed finalization")
    if size_query:
        if (options["hidden_size"], options["intermediate_size"],
            options["num_experts_total"], options["top_k"]) != (_H, _I, _E, _TOP_K):
            raise ValueError("Cake SiTU requires H=3584, I=384, E=896, top_k=16")
        if options["x_dtype"] != torch.bfloat16 or options["weight_dtype"] != torch.uint8:
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
    if any(options[key] is not None for key in (
        "fc1_expert_biases", "fc2_expert_biases", "input_sf", "swiglu_alpha",
        "swiglu_beta", "swiglu_limit",
    )):
        raise ValueError("Cake SiTU requires unbiased BF16 input and SiTU parameters")


def _cake_situ_workspace_size(options):
    # No device query, module load, CUDA allocation or initialization.
    _validate_contract(options, size_query=True)
    return _workspace_layout(options["max_num_tokens"])[1]


def _tensor(tensor, name, *, shape, dtype, device):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    allowed = dtype if isinstance(dtype, tuple) else (dtype,)
    if (tuple(tensor.shape) != shape or tensor.dtype not in allowed
            or tensor.device != device or not tensor.is_contiguous()):
        raise ValueError(f"{name} must be contiguous, shape {shape}, dtype {allowed}, device {device}")
    return tensor


def _workspace_key(workspace):
    return (workspace.data_ptr(), workspace.numel(), workspace.device)


def cutlass_fused_moe_prepare_workspace(
    workspace_buffer, num_tokens, *, backend="cake", weight_layout=_LAYOUT,
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
        raise ValueError("this preparation helper requires the documented Cake SiTU layout")
    if (not isinstance(workspace_buffer, torch.Tensor) or not workspace_buffer.is_cuda
            or workspace_buffer.dtype not in (torch.uint8, torch.int8)
            or workspace_buffer.ndim != 1 or not workspace_buffer.is_contiguous()
            or workspace_buffer.data_ptr() % 128):
        raise ValueError("workspace_buffer must be contiguous CUDA uint8/int8 and 128-byte aligned")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare the SiTU workspace before graph capture")
    layout, nbytes = _workspace_layout(num_tokens)
    if workspace_buffer.numel() < nbytes:
        raise ValueError(f"workspace_buffer needs at least {nbytes} bytes")
    with torch.cuda.device(workspace_buffer.device):
        major, minor = get_compute_capability(workspace_buffer.device)
        arch = {(10, 0): "sm_100a", (10, 3): "sm_103a"}.get((major, minor))
        if arch is None:
            raise ValueError("Cake SiTU requires SM100a or SM103a")
        key = _workspace_key(workspace_buffer)
        state = getattr(workspace_buffer, _STATE_ATTR, None)
        if state is None or state["buffer_key"] != key:
            state = {"buffer_key": key, "arch": arch, "weight_layout": weight_layout, "shapes": {}}
            setattr(workspace_buffer, _STATE_ATTR, state)
        views = {
            name: workspace_buffer.narrow(0, offset, size).view(dtype).view(shape)
            for name, (offset, size, dtype, shape) in layout.items()
        }
        # These two fixed-prefix regions are shared by all prepared shapes.
        views["situ_beta"].fill_(4.0)
        views["situ_linear_beta"].fill_(25.0)
        tile_n, total_pairs, max_tiles = _geometry(num_tokens)
        program_key = cake_situ_sequence(arch, tile_n)
        module = get_cake_situ_module(program_key)
        state["shapes"][num_tokens] = {
            "views": views, "tile_n": tile_n, "total_pairs": total_pairs,
            "max_tiles": max_tiles, "program_key": program_key, "module": module,
            "entry": getattr(module, PROGRAMS[program_key]["ffi_entry"]),
        }
    return workspace_buffer


def _prepared(workspace, num_tokens):
    if workspace is None:
        raise ValueError("backend='cake' requires an explicitly prepared workspace_buffer")
    state = getattr(workspace, _STATE_ATTR, None)
    if state is None or state["buffer_key"] != _workspace_key(workspace):
        raise ValueError("call cutlass_fused_moe_prepare_workspace before submitting SiTU")
    if num_tokens not in state["shapes"]:
        raise ValueError(f"workspace is not prepared for {num_tokens} tokens")
    return state["shapes"][num_tokens]


def _cake_situ_workspace_views(workspace, num_tokens):
    """Actual live scratch objects, also used for direct-launch parity checks."""
    return {name: tensor for name, tensor in _prepared(workspace, num_tokens)["views"].items()
            if name not in {"situ_beta", "situ_linear_beta"}}


def _cake_situ_stage_bindings(options, prepared):
    x = options["input"]
    num_tokens = x.shape[0]
    device = x.device
    views = prepared["views"]
    tile_n = prepared["tile_n"]
    total_pairs = prepared["total_pairs"]
    max_tiles = prepared["max_tiles"]
    _tensor(x, "input", shape=(num_tokens, _H), dtype=torch.bfloat16, device=device)
    if not x.is_cuda:
        raise ValueError("Cake SiTU input must be on CUDA")
    if options["workspace_buffer"].device != device:
        raise ValueError("workspace_buffer and input must be on the same device")
    ids = _tensor(options["token_selected_experts"], "token_selected_experts",
                  shape=(num_tokens, _TOP_K), dtype=torch.int32, device=device)
    weights = _tensor(options["token_final_scales"], "token_final_scales",
                      shape=(num_tokens, _TOP_K), dtype=torch.bfloat16, device=device)
    w1 = _tensor(options["fc1_expert_weights"], "fc1_expert_weights",
                 shape=(_E, 2 * _I, _H // 2), dtype=torch.uint8, device=device)
    w2 = _tensor(options["fc2_expert_weights"], "fc2_expert_weights",
                 shape=(_E, _H, _I // 2), dtype=torch.uint8, device=device)
    out = _tensor(options["output"], "output", shape=(num_tokens, _H),
                  dtype=torch.bfloat16, device=device)
    scales = options["quant_scales"]
    if not isinstance(scales, (list, tuple)) or len(scales) != 6:
        raise ValueError("Cake SiTU requires the six NVFP4 scale tensors")
    qx, sf1, decode1, qa, sf2, decode2 = scales
    for name, tensor in (("qX", qx),):
        if tensor.numel() != 1:
            raise ValueError(f"{name} must contain one float32 value")
        _tensor(tensor, name, shape=tuple(tensor.shape), dtype=torch.float32, device=device)
    sf_dtype = (torch.uint8, torch.float8_e4m3fn)
    _tensor(sf1, "FC1 block scales", shape=(_E, 2 * _I, _H // 16), dtype=sf_dtype, device=device)
    _tensor(sf2, "FC2 block scales", shape=(_E, _H, _I // 16), dtype=sf_dtype, device=device)
    for name, tensor in (("FC1 dequant scale", decode1), ("qA", qa), ("FC2 dequant scale", decode2)):
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
    route_grid = ((total_pairs + 255) // 256, 1, 1)
    reset_items = max(_E, max_tiles * tile_n + 1)
    stages = {
        "route_reset": dict(
            grid=((reset_items + 255) // 256, 1, 1),
            **{name: views[name] for name in ("expert_counts", "expert_scatter_offsets",
                "route_map", "tile_expert", "tile_mn_limit", "total_tiles")},
            num_experts=_E, max_tiles=max_tiles, tile_n=tile_n),
        "quant": dict(grid=(num_tokens, 1, 1), x=x, qx=qx,
                      packed=views["x_packed"], scales=views["x_scales"], M=num_tokens),
        "route_histogram": dict(grid=route_grid, topk_ids=flat_ids,
                                expert_counts=views["expert_counts"], total_pairs=total_pairs),
        "route_prefix": dict(grid=(1, 1, 1),
            **{name: views[name] for name in ("expert_counts", "expert_tile_offsets",
                "expert_scatter_offsets", "total_tiles")}, num_experts=_E, tile_n=tile_n),
        "route_scatter": dict(grid=route_grid, topk_ids=flat_ids,
            **{name: views[name] for name in ("expert_counts", "expert_tile_offsets",
                "expert_scatter_offsets", "route_map", "token_to_permuted", "tile_expert", "tile_mn_limit")},
            total_pairs=total_pairs, top_k=_TOP_K, tile_n=tile_n),
        "fc1": dict(grid=(_I // 64, max_tiles, 1), A=w1, B=views["x_packed"],
            SFA=sf1.view(torch.uint8).view(_E * (_I // 64), _H // 64, 2, 256),
            SFB=views["x_scales"], C=views["intermediate_packed"], SFC=views["intermediate_scales"],
            **{name: views[name] for name in ("route_map", "tile_expert", "tile_mn_limit", "total_tiles")},
            scale_c=qa, scale_gate=decode1, clamp_limit=qa, act_alpha=alpha, act_beta=beta,
            M_out=_I, K=_H, grid_m=_I // 64, grid_n=max_tiles, K_tiles=_H // 512),
        "fc2": dict(grid=(_H // 128, max_tiles, 1), A=w2,
            B=views["intermediate_packed"].view(max_tiles, tile_n, _I // 2),
            SFA=sf2.view(torch.uint8).view(_E * (_H // 128), _I // 64, 2, 256),
            SFB=(views["intermediate_scales"].view(max_tiles, _I // 64, 2, 256)
                 if tile_n == 128 else views["intermediate_scales"].view(max_tiles * (tile_n // 8), _I // 64, 32)),
            C_tma=views["expert_output"], C=views["expert_output"], scale_c=decode2,
            **{name: views[name] for name in ("tile_expert", "tile_mn_limit", "total_tiles")},
            M=_H, K=_I, grid_m=_H // 128, grid_n=max_tiles, K_tiles=2),
        "finalize": dict(grid=(num_tokens, 1, 1), expert_output=views["expert_output"],
            route_weights=flat_weights, token_to_permuted=views["token_to_permuted"], out=out, M=num_tokens),
    }
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
    # generated packed FFI entry submits all eight stages on the caller stream.
    with tvm_ffi.use_torch_stream():
        prepared["entry"](*args)
    return options["output"]
