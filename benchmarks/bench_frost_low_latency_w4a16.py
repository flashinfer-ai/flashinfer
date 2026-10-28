# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4/MXFP4/MXINT4/E4M3 weight-only MoE: shared FMA versus architecture-specific natives.

Synthetic model geometry, not serving throughput. Search native stage pairs,
freeze winners, validate with ABBA and independent warm CUDA-event chains.
"""

import argparse
import hashlib
import inspect
import importlib.metadata as metadata
import json
from pathlib import Path
from statistics import median
import subprocess
import torch
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from flashinfer.fused_moe import (
    FrostLowLatencyConfig,
    ExecutionConfig,
    BackendOptions,
    CuTileNvfp4Bf16Config,
    CuteDslConfig,
    CutlassBf16Config,
    SM12xNvfp4Bf16Config,
    CuTileMxfp4Bf16Config,
    CutlassW4A16Config,
    PrimsTsConfig,
    TrtllmMxInt4Config,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
)
from dataclasses import replace
from flashinfer.autotuner import autotune
from flashinfer.testing import bench_gpu_time_with_cupti


# Same deterministic checkpoint generator as the cuTile adapter tests.
def _quantize_weights(
    weight: torch.Tensor,
    *,
    scale_block_size: int = 16,
):
    if scale_block_size not in (16, 32):
        raise ValueError(f"unsupported FP4 scale block size {scale_block_size}")
    shape = weight.shape
    columns = shape[-1]
    rows = weight.numel() // columns
    flat_weight = weight.reshape(rows, columns)
    packed = torch.empty(rows, columns // 2, dtype=torch.uint8, device=weight.device)
    scale_dtype = (
        torch.float8_e4m3fn if scale_block_size == 16 else torch.float8_e8m0fnu
    )
    scales = torch.empty(
        rows,
        columns // scale_block_size,
        dtype=scale_dtype,
        device=weight.device,
    )
    dequantized = torch.empty_like(flat_weight)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=weight.device,
    )
    code_points = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=weight.device,
    )
    # Bound the FP32 and int64 quantization temporaries for model-sized expert
    # banks; the unchunked reference path peaks at tens of GiB.
    max_chunk_elements = 8 * 1024 * 1024
    chunk_rows = max(1, max_chunk_elements // columns)
    for begin in range(0, rows, chunk_rows):
        end = min(begin + chunk_rows, rows)
        groups = (
            flat_weight[begin:end]
            .float()
            .reshape(-1, columns // scale_block_size, scale_block_size)
        )
        if scale_block_size == 16:
            scale = (groups.abs().amax(dim=-1) / 6.0).to(scale_dtype)
            divisor = scale.float().clamp_min(2.0**-9)
        else:
            scale = torch.pow(
                2.0,
                torch.ceil(
                    torch.log2(groups.abs().amax(dim=-1).div(6.0).clamp_min(2.0**-127))
                ),
            ).to(scale_dtype)
            divisor = scale.float()
        scales[begin:end].copy_(scale)
        values = groups / divisor.unsqueeze(-1)
        codes = torch.bucketize(values.abs(), boundaries, right=False)
        codes |= (values < 0).to(torch.int64) << 3
        codes = codes.reshape(end - begin, columns)
        packed[begin:end].copy_(
            (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        )
        decoded = code_points[codes & 7]
        decoded = torch.where((codes & 8).bool(), -decoded, decoded)
        dequantized[begin:end].copy_(
            (
                decoded.reshape(
                    end - begin, columns // scale_block_size, scale_block_size
                )
                * scale.float().unsqueeze(-1)
            ).reshape(end - begin, columns)
        )
    return (
        packed.reshape(*shape[:-1], columns // 2),
        scales.reshape(*shape[:-1], columns // scale_block_size),
        dequantized.reshape(shape),
    )


def _sm120_tc_native(x, w1, s1, a1, w2, s2, a2, ids, scores):
    """SM120 BF16 decode / FP16 Tensor Core paths with supported native tiles."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
        select_route_block_size_m,
        unswizzle_expert_scales,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        prepare_w4a16_packed_weights,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
        allocate_sm120_moe_workspace,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel import (
        compile_w4a16_fused_moe,
        run_w4a16_moe,
        _SMALL_BATCH_TILE_CONFIGS,
        _LARGE_BATCH_TILE_CONFIGS,
        _direct_topk_route_m_cap,
        _tc_decode_m_cap,
    )
    from flashinfer.quantization.fp4_quantization import block_scale_interleave

    t, h = x.shape
    e, i, k = w1.shape[0], w2.shape[2] * 2, ids.shape[1]

    def swizzle(sf, rows, cols):
        physical = torch.stack([block_scale_interleave(row) for row in sf]).reshape_as(
            sf
        )
        recovered = unswizzle_expert_scales(physical, rows=rows, cols=cols)
        torch.testing.assert_close(recovered.view(torch.uint8), sf, atol=0, rtol=0)
        return physical

    sw1, sw2 = swizzle(s1, 2 * i, h), swizzle(s2, h, i)
    prepared = prepare_w4a16_packed_weights(
        w1,
        sw1,
        a1,
        w2,
        sw2,
        a2,
        activation="silu",
        params_dtype=x.dtype,
        source_format="modelopt",
    )
    workspace = allocate_sm120_moe_workspace(
        state_E=e,
        weight_E=e,
        k=h,
        n=i,
        num_topk=k,
        device=x.device,
        routed_rows=t * k,
        quant_mode="w4a16",
        activation="silu",
    )
    output = torch.empty_like(x)
    props = torch.cuda.get_device_properties(x.device)
    block = select_route_block_size_m(t, k, e)
    compiled = {}
    fp16 = x.dtype is torch.float16
    tc_decode = not fp16 and t <= _tc_decode_m_cap(
        int(props.major) * 10 + int(props.minor), "bf16", "packed", "e4m3_k16", h, i, k
    )
    direct = tc_decode or t <= _direct_topk_route_m_cap(
        "fp16" if fp16 else "bf16", "packed", "e4m3_k16", h, i, k
    )
    tiles = tuple(
        dict.fromkeys((*_SMALL_BATCH_TILE_CONFIGS, *_LARGE_BATCH_TILE_CONFIGS))
    )

    def call(tactic):
        tactic = tuple(tactic)
        fused = None
        if tactic != (-1,):
            if tactic not in compiled:
                compiled[tactic] = compile_w4a16_fused_moe(
                    size_m=t,
                    hidden_size=h,
                    intermediate_size=i,
                    num_experts=e,
                    top_k=k,
                    activation="silu",
                    apply_router_weight_on_input=False,
                    zero_fc2_output=False,
                    moe_block_size=block,
                    max_m_blocks=t * k
                    if direct
                    else workspace.block_expert_ids.numel(),
                    element_dtype="fp16" if fp16 else "bf16",
                    fast_math=True,
                    sms=props.multi_processor_count,
                    max_shared_mem=props.shared_memory_per_block_optin,
                    weight_layout="packed",
                    scale_format="e4m3_k16",
                    direct_topk_routes=direct,
                    tc_decode_fused_sum=tc_decode,
                    force_tile_config=tactic,
                )
            fused = compiled[tactic]
            assert fused.tc_decode_fused_sum == tc_decode
            assert fused.direct_topk_routes == direct
        return run_w4a16_moe(
            x,
            prepared,
            scores,
            ids,
            activation="silu",
            intermediate_cache13=workspace.intermediate_cache13.view(x.dtype),
            intermediate_cache2=workspace.intermediate_cache2.view(x.dtype),
            output=output,
            fc1_c_tmp=workspace.fc1_c_tmp,
            fc2_c_tmp=workspace.fc2_c_tmp,
            packed_route_indices=workspace.packed_route_indices,
            block_expert_ids=workspace.block_expert_ids,
            packed_route_count=workspace.packed_route_count,
            expert_offsets=workspace.expert_offsets,
            expert_map=None,
            fast_math=True,
            fused_launch=fused,
        )

    tactics = [(-1,)] + [
        (ak, an, bk, bn) for ak, an, ath in tiles for bk, bn, bth in tiles if ath == bth
    ]
    return call, tactics


def _cute_dsl_nvfp4_view(w1, s1, a1, w2, s2, a2):
    from flashinfer.fused_moe.prepare import _interleave_linear_and_gate

    def mma_scales(sf):
        e, rows, cols = sf.shape
        physical = (
            sf.reshape(e, rows // 128, 4, 32, cols // 4, 4)
            .permute(0, 1, 4, 3, 2, 5)
            .contiguous()
        )
        logical = physical.permute(3, 4, 1, 5, 2, 0)
        inverse = logical.permute(5, 2, 1, 0, 4, 3).contiguous().reshape_as(sf)
        torch.testing.assert_close(inverse, sf, atol=0, rtol=0)
        return logical.view(torch.float8_e4m3fn)

    return dict(
        w1_weight=_interleave_linear_and_gate(w1, 64, 1),
        w1_weight_sf=mma_scales(_interleave_linear_and_gate(s1, 64, 1)),
        w1_alpha=a1,
        w2_weight=w2,
        w2_weight_sf=mma_scales(s2),
        w2_alpha=a2,
    )


def _expand_nvfp4_bf16(q, scales):
    """Exact offline control: four times packed-weight storage, no quantization."""
    result = torch.empty(
        q.shape[0], q.shape[1], q.shape[2] * 2, device=q.device, dtype=torch.bfloat16
    )
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=q.device)
    for begin in range(0, q.shape[0], 8):
        chunk = q[begin : begin + 8]
        codes = torch.stack((chunk & 15, chunk >> 4), dim=-1).flatten(-2).long()
        decoded = (
            lut[codes & 7]
            * (1 - 2 * ((codes & 8) != 0).float())
            * scales[begin : begin + 8]
            .view(torch.float8_e4m3fn)
            .float()
            .repeat_interleave(16, -1)
        )
        converted = decoded.bfloat16()
        torch.testing.assert_close(converted.float(), decoded, atol=0, rtol=0)
        result[begin : begin + 8].copy_(converted)
    return result


def _expand_weight_only_fp16(q, sf, global_scale, weight_format):
    """Exact checkpoint expansion; fail rather than round an unsupported value."""
    packed = weight_format != "w8a16"
    shape = (*q.shape[:-1], q.shape[-1] * (2 if packed else 1))
    result = torch.empty(shape, device=q.device, dtype=torch.float16)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=q.device)
    for begin in range(0, q.shape[0], 8):
        chunk, scale = q[begin : begin + 8], sf[begin : begin + 8]
        if not packed:
            decoded = chunk.float() * scale[:, None, None]
        else:
            codes = torch.stack((chunk & 15, chunk >> 4), dim=-1).flatten(-2).long()
            if weight_format == "mxint4":
                decoded = torch.where(codes < 8, codes, codes - 16).float()
                scale, block = scale.float(), 32
            else:
                decoded = lut[codes & 7] * (1 - 2 * ((codes & 8) != 0).float())
                block = 32 if weight_format == "mxfp4" else 16
                scale = scale.view(
                    torch.float8_e8m0fnu if block == 32 else torch.float8_e4m3fn
                ).float()
            decoded *= scale.repeat_interleave(block, -1)
        decoded *= global_scale[begin : begin + 8, None, None]
        converted = decoded.half()
        torch.testing.assert_close(converted.float(), decoded, atol=0, rtol=0)
        result[begin : begin + 8].copy_(converted)
    return result


def _expanded_fp16_native(x, w1, w2, ids, scores):
    """Dense CUTLASS control; expansion and its larger storage are explicit."""
    from flashinfer.fused_moe.core import (
        get_cutlass_fused_moe_module,
        cutlass_fused_moe,
        cutlass_fused_moe_workspace_size,
    )
    from flashinfer.fused_moe import SwiGLU
    from flashinfer.autotuner import AutoTuner, TuningConfig

    t, h, i, e, k = x.shape[0], x.shape[1], w2.shape[2], w1.shape[0], ids.shape[1]
    cc = torch.cuda.get_device_capability(x.device)
    module = get_cutlass_fused_moe_module(str(cc[0] * 10 + cc[1]))
    inner = module.MoERunner(
        x_dtype=torch.float16,
        weight_dtype=torch.float16,
        output_dtype=torch.float16,
        top_k=k,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        cluster_size=1,
        cluster_rank=0,
        enable_alltoall=False,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=False,
        use_mxfp8_act_scaling=False,
        min_latency_mode=False,
        enable_pdl=False,
        activation_type=SwiGLU().type,
        use_packed_weights=False,
        use_fused_finalize=True,
        use_wfp4afp8_humming=False,
    )
    stage_choices = []
    with autotune(True):
        for stage in (1, 2):
            inner.gemm_idx_for_tuning = stage
            stage_choices.append(
                AutoTuner.get().rank_tactics(
                    f"frost_control_cutlass_fp16_expanded_stage{stage}",
                    [inner],
                    TuningConfig(),
                    [x, w1, None, w2, None],
                    k=2,
                    gemm_idx=stage,
                )
            )
    inner.gemm_idx_for_tuning = None
    tactics = [
        -1,
        *[(int(one), int(two)) for one in stage_choices[0] for two in stage_choices[1]],
    ]
    workspace = torch.empty(
        cutlass_fused_moe_workspace_size(
            t,
            h,
            i,
            e,
            k,
            x_dtype=torch.float16,
            weight_dtype=torch.float16,
            output_dtype=torch.float16,
            use_w4_group_scaling=False,
            device=x.device,
        ),
        device=x.device,
        dtype=torch.uint8,
    )
    output = torch.empty_like(x)

    def call(tactic):
        with autotune(False):
            value = cutlass_fused_moe(
                x,
                ids,
                scores,
                w1,
                w2,
                torch.float16,
                [],
                output=output,
                use_w4_group_scaling=False,
                tune_max_num_tokens=t,
                enable_pdl=False,
                profile_ids=None if tactic == -1 else list(tactic),
                workspace_buffer=workspace,
            )
        return value[0] if isinstance(value, (list, tuple)) else value

    return call, tactics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weight-format",
        choices=("nvfp4", "mxfp4", "mxint4", "w8a16"),
        default="nvfp4",
    )
    p.add_argument("--activation-dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument(
        "--native-backend",
        choices=(
            "cutile",
            "cutlass",
            "prims-ts",
            "trtllm-mxint4",
            "cute-dsl",
            "sm12x",
            "sm120-tc",
            "expanded-bf16",
            "expanded-fp16",
            "frost-ungrouped",
        ),
        default="cutile",
    )
    p.add_argument("--timer", choices=("cupti", "globaltimer"), default="cupti")
    p.add_argument("--tokens", type=int, default=1)
    p.add_argument(
        "--native-tactic-json",
        help="Freeze an advertised native tactic from an earlier result for validation",
    )
    p.add_argument(
        "--native-full-blocks", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=512)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--cache-mode", choices=("warm", "cold"), default="warm")
    p.add_argument("--expert-reuse", choices=("shared", "disjoint"), default="disjoint")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    is_fp16 = a.activation_dtype == "fp16"
    is_w8 = a.weight_format == "w8a16"
    dtype = torch.float16 if is_fp16 else torch.bfloat16
    quant_format = QuantFormat.FP16 if is_fp16 else QuantFormat.BF16
    expanded_fp16 = a.native_backend == "expanded-fp16"
    reuse_control = a.native_backend == "frost-ungrouped"
    preact_boundary = a.activation_dtype + "_preact"
    if a.timer == "cupti":
        import cupti.cupti  # noqa: F401

        if int(metadata.version("cupti-python").split(".")[0]) < 13:
            raise RuntimeError("Install cupti-python >= 13 for this benchmark")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cc = torch.cuda.get_device_capability()
    nv_native_archs = {
        "expanded-bf16": ((9, 0),),
        "cutile": ((9, 0), (12, 0), (12, 1)),
        "cute-dsl": ((10, 0), (10, 3)),
        "sm12x": ((12, 0), (12, 1)),
        "sm120-tc": ((12, 0), (12, 1)),
    }
    if reuse_control:
        if not is_w8 or cc != (12, 0):
            raise ValueError("frost-ungrouped is the SM120 W8A16 reuse ablation")
    elif expanded_fp16:
        if not is_fp16 or cc not in ((9, 0), (10, 0), (10, 3)):
            raise ValueError(
                "expanded-fp16 requires FP16 activations on SM90/SM100/SM103"
            )
    elif is_fp16:
        if (
            cc != (12, 0)
            or a.weight_format != "nvfp4"
            or a.native_backend != "sm120-tc"
        ):
            raise ValueError(
                "FP16 comparison requires SM120 NVFP4 sm120-tc or expanded-fp16 on SM90/SM100/SM103"
            )
    elif is_w8:
        raise ValueError(
            "The W8A16 control requires --activation-dtype fp16 --native-backend expanded-fp16"
        )
    if not expanded_fp16 and a.weight_format == "nvfp4":
        if cc not in nv_native_archs.get(a.native_backend, ()):
            raise RuntimeError(
                "Select an architecture-appropriate NVFP4 native baseline"
            )
    elif a.weight_format != "nvfp4" and a.native_backend in (
        "cute-dsl",
        "sm12x",
        "sm120-tc",
        "expanded-bf16",
    ):
        raise ValueError("This native adapter supports NVFP4/BF16 only")
    mx_native_archs = {
        "cutlass": ((9, 0),),
        "prims-ts": ((10, 0), (10, 3)),
        "cutile": ((9, 0), (12, 0), (12, 1)),
    }
    if (
        not expanded_fp16
        and a.weight_format == "mxfp4"
        and cc not in mx_native_archs.get(a.native_backend, ())
    ):
        raise RuntimeError("Select an architecture-appropriate MXFP4 native baseline")
    is_int = a.weight_format == "mxint4"
    if not expanded_fp16 and is_int != (a.native_backend == "trtllm-mxint4"):
        raise ValueError("MXINT4 comparison requires --native-backend trtllm-mxint4")
    is_mx = a.weight_format == "mxfp4"
    scale_block = 32 if is_mx or is_int else 16
    scale_dtype = (
        torch.bfloat16
        if is_int
        else (torch.float8_e8m0fnu if is_mx else torch.float8_e4m3fn)
    )
    t, h, i, e, k = a.tokens, a.hidden, a.intermediate, a.experts, a.top_k
    assert h % 128 == i % 128 == 0 and t * k <= e and 1 <= t <= 8 and k <= 32
    torch.manual_seed(20261024)
    x = torch.randn(t, h, device="cuda", dtype=dtype)

    def weights(rows, cols):
        chunks = []
        for begin in range(0, e, 16):
            value = (
                torch.randn(min(16, e - begin), rows, cols, device="cuda") / cols**0.5
            ).bfloat16()
            if is_w8:
                q = (value.float() * 256.0).to(torch.float8_e4m3fn)
                sf = torch.full((value.shape[0],), 1.0 / 256.0, device=value.device)
                chunks.append((q, sf))
            else:
                q, sf, _ = _quantize_weights(value, scale_block_size=scale_block)
                chunks.append((q, sf.view(torch.uint8)))
        return torch.cat([q for q, sf in chunks]), torch.cat([sf for q, sf in chunks])

    if is_int:
        from flashinfer.fused_moe.prepare import _mxint4_quantize

        assert h % 256 == i % 256 == 0
        w1_bf16 = (torch.randn(e, 2 * i, h, device=x.device) / h**0.5).bfloat16()
        w2_bf16 = (torch.randn(e, h, i, device=x.device) / i**0.5).bfloat16()
        w1, s1 = _mxint4_quantize(w1_bf16)
        w2, s2 = _mxint4_quantize(w2_bf16)
        s1 = s1.reshape(e, 2 * i, h // 32).contiguous()
        s2 = s2.reshape(e, h, i // 32).contiguous()
    else:
        w1, s1 = weights(2 * i, h)
        w2, s2 = weights(h, i)
    a1 = torch.ones(e, device="cuda")
    a2 = torch.ones(e, device="cuda")
    if is_fp16 and a.weight_format == "nvfp4" and not expanded_fp16:
        # Preserve the checkpoint exactly while avoiding native FP16 preparation's
        # tiny-block-scale clamp. Both paths receive these equivalent scales.
        for sf in (s1, s2):
            values = sf.view(torch.float8_e4m3fn).float() * 128.0
            converted = values.to(torch.float8_e4m3fn)
            torch.testing.assert_close(converted.float(), values, atol=0, rtol=0)
            sf.copy_(converted.view(torch.uint8))
        a1.fill_(1.0 / 128.0)
        a2.fill_(1.0 / 128.0)
    ids = torch.arange(t * k, device="cuda", dtype=torch.int32).reshape(t, k)
    if a.expert_reuse == "shared":
        ids.remainder_(k)
    scores = torch.randn(t, k, device="cuda").softmax(-1).bfloat16().float()
    original_ids, original_scores = ids.clone(), scores.clone()
    if reuse_control:
        option, native_key, view = (
            FrostLowLatencyConfig(),
            "frost_ungrouped_16_tactics",
            None,
        )
    elif expanded_fp16:
        option, native_key, view = (
            FrostLowLatencyConfig(),
            "cutlass_fp16_expanded",
            None,
        )
    elif is_int:
        option = TrtllmMxInt4Config()
        native_key = "trtllm_mxint4_routed"
        # Official preparation calls the same deterministic canonical quantizer
        # and then permutes payloads/scales. Both paths consume identical values.
        view = option.prepare_weights(
            w1_bf16, w2_bf16, num_local_experts=e, hidden_size=h, intermediate_size=i
        )
        del w1_bf16, w2_bf16
    elif a.native_backend == "expanded-bf16":
        assert torch.equal(a1, torch.ones_like(a1))
        assert torch.equal(a2, torch.ones_like(a2))
        option, native_key = CutlassBf16Config(), "cutlass_bf16"
        view = option.prepare_weights(
            _expand_nvfp4_bf16(w1, s1),
            _expand_nvfp4_bf16(w2, s2),
            num_local_experts=e,
            hidden_size=h,
            intermediate_size=i,
        )
    elif a.native_backend == "cute-dsl":
        option, native_key = CuteDslConfig(), "cute_dsl"
        view = _cute_dsl_nvfp4_view(w1, s1, a1, w2, s2, a2)
    elif a.native_backend == "sm120-tc":
        # The direct native adapter below has no unified-runner tactic wrapper.
        option, native_key, view = (
            FrostLowLatencyConfig(),
            ("sm120_fp16_tc" if is_fp16 else "sm120_fused_tc"),
            None,
        )
    elif a.native_backend in ("cutile", "sm12x"):
        option = (
            SM12xNvfp4Bf16Config()
            if a.native_backend == "sm12x"
            else (CuTileMxfp4Bf16Config() if is_mx else CuTileNvfp4Bf16Config())
        )
        args = (
            (w1, s1.view(scale_dtype), w2, s2.view(scale_dtype))
            if is_mx
            else (w1, s1.view(scale_dtype), a1, w2, s2.view(scale_dtype), a2)
        )
        view = option.prepare_weights(
            *args, num_local_experts=e, hidden_size=h, intermediate_size=i
        )
        native_key = (
            "sm12x_nvfp4_bf16"
            if a.native_backend == "sm12x"
            else ("cutile_mxfp4" if is_mx else "cutile_nvfp4")
        )
    elif a.native_backend == "cutlass":
        from flashinfer.fused_moe.prepare import (
            interleave_moe_weights_for_sm90_mixed_gemm,
            interleave_moe_scales_for_sm90_mixed_gemm,
        )

        option = CutlassW4A16Config()
        native_key = "cutlass_w4a16"
        view = dict(
            fc1_expert_weights=interleave_moe_weights_for_sm90_mixed_gemm(w1, "fp4"),
            fc1_expert_scales=interleave_moe_scales_for_sm90_mixed_gemm(s1),
            fc2_expert_weights=interleave_moe_weights_for_sm90_mixed_gemm(w2, "fp4"),
            fc2_expert_scales=interleave_moe_scales_for_sm90_mixed_gemm(s2),
        )
    else:
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )
        from flashinfer.quantization.fp4_quantization import block_scale_interleave

        option = PrimsTsConfig()
        native_key = "prims_ts"
        cache = {}

        def native_layout(q, sf, gated):
            permute = (
                _maybe_get_cached_w3_w1_permute_indices
                if gated
                else get_w2_permute_indices_with_cache
            )
            order = permute(cache, q[0], 128, is_gated_act_gemm=True).to(x.device)
            scale_order = permute(
                cache, sf[0], 128, num_elts_per_sf=16, is_gated_act_gemm=True
            ).to(x.device)
            weights = q[:, order].contiguous()
            torch.testing.assert_close(
                weights[:, torch.argsort(order)], q, atol=0, rtol=0
            )
            scales = torch.stack(
                [block_scale_interleave(row[scale_order].contiguous()) for row in sf]
            ).reshape_as(sf)
            # Native launcher metadata is E4M3; MX mode interprets the E8M0 bytes.
            return weights, scales.view(torch.float8_e4m3fn)

        nw1, ns1 = native_layout(w1, s1, True)
        nw2, ns2 = native_layout(w2, s2, False)
        view = dict(
            gemm1_weights=nw1,
            gemm1_weights_scale=ns1,
            gemm2_weights=nw2,
            gemm2_weights_scale=ns2,
            output1_scale_scalar=a1,
            output1_scale_gate_scalar=a1,
            output2_scale_scalar=a2,
        )
    cfg = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(
            weight=QuantFormat.FP8PerTensor
            if is_w8
            else QuantFormat.MXINT4
            if is_int
            else (QuantFormat.MXFP4 if is_mx else QuantFormat.NVFP4),
            activation=quant_format,
            output=quant_format,
        ),
        ExpertConfig(i),
        backend=BackendOptions((option,)),
        execution=ExecutionConfig(tune_max_num_tokens=t),
    )
    act = MoEActivationPack(x, None, ids, scores)
    if reuse_control:
        native_tactics = [(a, b) for a in (1, 2, 4, 8) for b in (1, 2, 4, 8)]
    elif expanded_fp16:
        native, native_tactics = _expanded_fp16_native(
            x,
            _expand_weight_only_fp16(w1, s1, a1, a.weight_format),
            _expand_weight_only_fp16(w2, s2, a2, a.weight_format),
            ids,
            scores,
        )
    elif a.native_backend == "sm120-tc":
        native, native_tactics = _sm120_tc_native(
            x, w1, s1, a1, w2, s2, a2, ids, scores
        )
    else:
        pack = MoEWeightPack()
        pack.prepare_for(native_key, view)
        native_cfg = (
            replace(cfg, quant=QuantConfig(QuantFormat.BF16, QuantFormat.BF16))
            if a.native_backend == "expanded-bf16"
            else cfg
        )
        layer = MoELayer(native_cfg)
        with autotune(False):
            layer(act, pack)
        bucket, (runner, initial) = next(iter(layer._winners.items()))
        # Stage-ranking hooks mutate routing tensors. Give them private routing storage.
        tuning_inputs = runner.pack_inputs(
            MoEActivationPack(x, None, ids.clone(), scores.clone()), pack
        )
        with autotune(True):
            advertised = runner.get_valid_tactics(tuning_inputs, None)
        torch.testing.assert_close(ids, original_ids, atol=0, rtol=0)
        torch.testing.assert_close(scores, original_scores, atol=0, rtol=0)
        full_pairs = []
        if a.native_backend == "cutile":
            actual_inputs = runner.pack_inputs(act, pack)
            block = runner._fp4_block_size(ids.numel())
            for native_block in (
                runner._block_sizes if a.native_full_blocks else (block,)
            ):
                choices = [
                    runner._stage_tactics(
                        actual_inputs, stage=stage, block_size=native_block
                    )
                    for stage in (1, 2)
                ]
                full_pairs.extend(
                    (native_block, *one, *two)
                    for one in choices[0]
                    for two in choices[1]
                )
        advertised = [*advertised, *full_pairs]

        def canonical_tactic(value):
            if isinstance(value, (tuple, list)) or type(value).__name__ == "Array":
                return tuple(canonical_tactic(v) for v in value)
            return value

        native_tactics = []
        defaults = [initial, -1] if a.native_backend == "cute-dsl" else [initial]
        for tactic in map(canonical_tactic, [*defaults, *advertised]):
            if tactic not in native_tactics:
                native_tactics.append(tactic)

        def native(tactic):
            layer._winners[bucket] = (
                runner,
                tuple(tactic) if isinstance(tactic, (tuple, list)) else tactic,
            )
            return layer(act, pack)

    frost_config = replace(
        cfg, backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),))
    )
    frost_pack = MoEWeightPack()
    frost_view = (
        FrostLowLatencyConfig.prepare_fp8_weight_only_weights(w1, s1, w2, s2)
        if is_w8
        else FrostLowLatencyConfig.prepare_mxint4_weight_only_weights(w1, s1, w2, s2)
        if is_int
        else FrostLowLatencyConfig.prepare_mxfp4_weight_only_weights(
            w1, s1.view(scale_dtype), w2, s2.view(scale_dtype)
        )
        if is_mx
        else FrostLowLatencyConfig.prepare_nvfp4_weight_only_weights(
            w1, s1.view(scale_dtype), a1, w2, s2.view(scale_dtype), a2
        )
    )
    frost_pack.prepare_for("frost_low_latency", frost_view)
    frost_layer = MoELayer(frost_config)
    with autotune(False):
        frost_layer(act, frost_pack)
    frost_bucket, (frost_runner, _) = next(iter(frost_layer._winners.items()))
    assert isinstance(frost_runner, FrostLowLatencyRunner)

    def frost(tactic):
        frost_layer._winners[frost_bucket] = (frost_runner, ("ffma", *tactic))
        return frost_layer(act, frost_pack)

    if reuse_control:
        native = frost

    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")

    def decode(q, sf):
        if is_w8:
            return q.double() * sf.double()
        code = torch.stack((q & 15, q >> 4), dim=-1).flatten(-2).long()
        if is_int:
            values = torch.where(code < 8, code, code - 16).float()
            return (values * sf.float().repeat_interleave(32, -1)).double()
        return (
            lut[code & 7]
            * (1 - 2 * ((code & 8) != 0).float())
            * sf.view(scale_dtype).float().repeat_interleave(scale_block, -1)
        ).double()

    references = {
        boundary: torch.zeros(t, h, device="cuda")
        for boundary in (preact_boundary, "fp32_preact", "fp32_preact_expert")
    }
    for token in range(t):
        for slot, expert in enumerate(ids[token].tolist()):
            pre = (decode(w1[expert], s1[expert]) @ x[token].double()).float() * a1[
                expert
            ]
            for boundary, reference in references.items():
                up, gate = (
                    pre.to(dtype).float() if boundary == preact_boundary else pre
                ).chunk(2)
                hidden = (up * torch.nn.functional.silu(gate)).to(dtype)
                value = (decode(w2[expert], s2[expert]) @ hidden.double()).float() * a2[
                    expert
                ]
                if boundary != "fp32_preact_expert":
                    value = value.to(dtype).float()
                reference[token] += value * scores[token, slot]
    references = {name: value.to(dtype) for name, value in references.items()}

    def err(v, r):
        delta = v.double() - r.double()
        return dict(
            relative_l2=float(delta.norm() / r.double().norm().clamp_min(1e-30)),
            normalized_max=float(
                delta.abs().max() / r.double().abs().max().clamp_min(1e-30)
            ),
        )

    moe_root = Path(inspect.getfile(MoELayer)).parent
    repo_root = moe_root.parents[1]
    source_files = {
        Path(__file__),
        *moe_root.glob("*.py"),
        *(moe_root / "cutile").rglob("*.py"),
        *(moe_root / "cute_dsl").rglob("*.py"),
        *(moe_root.parent / "cute_dsl").rglob("*.py"),
        *(moe_root.parent / "cutile").rglob("*.py"),
        *(moe_root.parent / "prims_ts").rglob("*.py"),
        *Path(inspect.getfile(FrostLowLatencyRunner)).parent.rglob("*.py"),
        *(repo_root / "csrc/fused_moe/cutlass_backend").rglob("*.cuh"),
        *(
            repo_root / "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm"
        ).rglob("*.inl"),
    }
    native_header = (
        repo_root
        / "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
    )
    if native_header.exists():
        source_files.add(native_header)
    result = dict(
        workload=vars(a) | {"output": str(a.output)},
        gpu=torch.cuda.get_device_name(),
        cc=torch.cuda.get_device_capability(),
        scope=f"{a.activation_dtype.upper()} input/output + routed FC1/activation + FC2/combine; identical checkpoint payloads, scales and routes. Host/preparation/routing network/shared experts/communication/serving excluded",
        numerical_contract=(
            "Both paths use identical E4M3 checkpoints, input/preactivation/intermediate/expert/output rounding and ordered combine. Compare the best original 16 tactics with the expanded candidate set."
            if reuse_control
            else "FP16 input/intermediate/output; Frost FP4/E4M3 rounds preactivation to FP16, MXINT4 retains FP32. Expanded CUTLASS rounds preactivation to FP16; SM120 native retains FP32 preactivation/partials. Each has a separate reference."
            if is_fp16
            else "BF16 input and intermediate. Unless noted otherwise, BF16 expert outputs with FP32 accumulation/combine. FP4 Frost/cuTile/SM90 CUTLASS round FC1 preactivation to BF16; Prims-TS and both MXINT4 paths retain FP32 through activation. CuTe DSL also retains FP32 preactivation; SM12x and SM120 fused TC retain FP32 expert partials. SM12x uses FP16 partial products/accumulation internally. SM120 fused TC combines atomically into BF16 output. Separate references are recorded; different boundaries are not bitwise-equivalent math."
        ),
        native_backend=native_key,
        native_weight_storage=(
            "Identical E4M3 checkpoint and per-expert FP32 scales on both paths"
            if reuse_control
            else f"Exact offline {a.weight_format} expansion to FP16; {2 if is_w8 else 4}x payload storage; expansion excluded; not a native low-bit implementation"
            if expanded_fp16
            else "Exact offline NVFP4 expansion to BF16; unit global scales; 4x packed-weight bytes; expansion excluded from timing; not a native four-bit implementation"
            if a.native_backend == "expanded-bf16"
            else "Native packed four-bit checkpoint"
        ),
        timer="CUPTI single-call first-to-last GPU activity span"
        if a.timer == "cupti"
        else "Explicit globaltimer CUDA-graph interval including stamp boundary overhead",
        rows=[],
        validation=[],
        rejected=[],
        selection_refresh=[],
        completed=False,
        versions={
            z: metadata.version(z)
            for z in (
                "torch",
                "nvidia-cutlass-dsl",
                "apache-tvm-ffi",
            )
        },
        source_hashes={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_files)
        },
        actual_route_ids=ids.cpu().tolist(),
        actual_route_scores=scores.cpu().tolist(),
    )

    def hardware():
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem,power.draw,temperature.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()

    if a.timer == "cupti":
        result["versions"]["cupti-python"] = metadata.version("cupti-python")
    if a.native_backend == "cutile":
        result["versions"]["cuda-tile"] = metadata.version("cuda-tile")
    result["hardware_start"] = hardware()

    def save():
        a.output.write_text(json.dumps(result, indent=2) + "\n")

    def check(v, name):
        errors = {key: err(v, ref) for key, ref in references.items()}
        own = errors[
            "fp32_preact_expert"
            if name == "native" and a.native_backend in ("sm12x", "sm120-tc")
            else preact_boundary
            if (name == "native" and (expanded_fp16 or reuse_control))
            or (
                not is_int
                and (
                    name == "frost"
                    or a.native_backend in ("cutile", "cutlass", "expanded-bf16")
                )
            )
            else "fp32_preact"
        ]
        assert (
            torch.isfinite(v).all()
            and own["relative_l2"] < (0.003 if is_fp16 else 0.015)
            and own["normalized_max"] < (0.01 if is_fp16 else 0.06)
        ), (name, errors)
        return errors

    if a.timer == "globaltimer":
        from flashinfer.utils import get_globaltimer_kernel

        stamp = get_globaltimer_kernel()
        if stamp is None:
            raise RuntimeError("globaltimer is unavailable")
        flush = (
            torch.empty(512 * 1024**2, device=x.device, dtype=torch.uint8)
            if a.cache_mode == "cold"
            else None
        )

        def globaltimer_samples(call, repeats):
            call()
            timestamps = torch.empty((repeats, 2), device=x.device, dtype=torch.int64)
            starts = [timestamps[j, :1] for j in range(repeats)]
            ends = [timestamps[j, 1:] for j in range(repeats)]
            stamp(starts[0])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for start, end in zip(starts, ends, strict=True):
                    if flush is not None:
                        flush.zero_()
                    stamp(start)
                    call()
                    stamp(end)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            samples = (
                (timestamps[:, 1] - timestamps[:, 0]).cpu().double() / 1000
            ).tolist()
            assert min(samples) > 0
            return samples

        result["empty_interval_us"] = globaltimer_samples(lambda: None, 50)

    def measure(name, tactic, phase="selection", group="rows"):
        if phase == "selection":
            saved = next(
                (
                    row
                    for row in result["rows"]
                    if row["backend"] == name
                    and row["tactic"]
                    == (list(tactic) if isinstance(tactic, (tuple, list)) else tactic)
                ),
                None,
            )
            if saved is not None:
                return saved
        forward = (
            (lambda: frost(tactic)) if name == "frost" else (lambda: native(tactic))
        )
        # MoELayer may allocate a new result during graph capture. Retain that
        # result and validate it after replay, rather than the eager warmup's
        # distinct allocation.
        replay_output = [None]

        def call():
            replay_output[0] = forward()
            return replay_output[0]

        check(call(), name)
        if a.timer == "globaltimer":
            samples = globaltimer_samples(call, 20 if phase == "selection" else 50)
        else:
            samples = [
                1000 * v
                for v in bench_gpu_time_with_cupti(
                    call,
                    dry_run_iters=8,
                    repeat_iters=20 if phase == "selection" else 50,
                    use_cuda_graph=True,
                    cold_l2_cache=a.cache_mode == "cold",
                )
            ]
        row = dict(
            backend=name,
            tactic=list(tactic) if isinstance(tactic, (tuple, list)) else tactic,
            phase=phase,
            median_us=median(samples),
            samples_us=samples,
            errors=check(replay_output[0], name),
        )
        result[group].append(row)
        save()
        print(phase, name, tactic, row["median_us"], row["errors"], flush=True)
        return row

    frost_inputs = frost_runner.pack_inputs(act, frost_pack)
    for tactic in frost_runner.get_valid_tactics(frost_inputs, None):
        assert tactic[0] == "ffma"
        measure("frost", tuple(tactic[1:]))
    for tactic in native_tactics:
        measure("native", tactic)
    result["native_search"] = (
        "All 16 original Frost FC1/FC2 column pairs; identical checkpoint and numerical contract"
        if reuse_control
        else "All advertised native choices plus default; cuTile additionally enumerates the selected/all supported block-size stage pairs; SM120 native uses all same-thread small/large-tile pairs; expanded FP16 ranks top-two per stage plus default"
    )
    result["native_tactic_count"] = len(native_tactics)
    selected = {
        name: min(
            (r for r in result["rows"] if r["backend"] == name),
            key=lambda r: r["median_us"],
        )["tactic"]
        for name in ("frost", "native")
    }
    if a.native_tactic_json is not None:

        def immutable(value):
            return (
                tuple(immutable(v) for v in value)
                if isinstance(value, (list, tuple))
                else value
            )

        forced = immutable(json.loads(a.native_tactic_json))
        if forced not in [immutable(tactic) for tactic in native_tactics]:
            raise ValueError(
                "The requested native tactic was not in this run's candidate set"
            )
        selected["native"] = forced
        result["native_selection"] = (
            "Previously measured finalist, explicitly frozen for ABBA validation"
        )
    result["selected"] = selected
    save()
    for rnd in range(4):
        for label, name in (
            ("A1", "native"),
            ("B1", "frost"),
            ("B2", "frost"),
            ("A2", "native"),
        ):
            measure(name, selected[name], f"ABBA-{rnd}-{label}", "validation")
    result["validation_median_us"] = {
        name: median(
            r["median_us"] for r in result["validation"] if r["backend"] == name
        )
        for name in selected
    }
    result["speedup"] = (
        result["validation_median_us"]["native"]
        / result["validation_median_us"]["frost"]
    )
    result["aa_ratio"] = {
        name: median(
            r["median_us"]
            for j, r in enumerate(result["validation"])
            if r["backend"] == name and j % 4 in (0, 1)
        )
        / median(
            r["median_us"]
            for j, r in enumerate(result["validation"])
            if r["backend"] == name and j % 4 in (2, 3)
        )
        for name in selected
    }

    def event_chain(call):
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(128):
                call()
        for _ in range(3):
            graph.replay()
        times = []
        for _ in range(20):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / 128)
        return dict(median_us=median(times), samples_us=times)

    if a.timer == "cupti":
        result["warm_event_chain_control"] = {
            name: event_chain(
                (lambda: frost(selected[name]))
                if name == "frost"
                else (lambda: native(selected[name]))
            )
            for name in selected
        }
    torch.testing.assert_close(ids, original_ids, atol=0, rtol=0)
    torch.testing.assert_close(scores, original_scores, atol=0, rtol=0)
    result["hardware_end"] = hardware()
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
