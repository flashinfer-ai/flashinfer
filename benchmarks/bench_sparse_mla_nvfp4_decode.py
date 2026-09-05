# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paired FP8/NVFP4 DeepSeek-V4 sparse-MLA decode benchmark for SM120.

The NVFP4 path includes online Q/P quantization and constructs each selected-V
tile inside the attention CTA. Both formats start from the same BF16 Q/KV
tensors and use the same indices.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from flashinfer.mla import nvfp4_quantize_pack_sparse_mla_cache
from flashinfer.mla._sparse_mla_nvfp4_sm120 import (
    get_sparse_mla_nvfp4_sm120_module,
)
from flashinfer.mla._sparse_mla_sm120 import sparse_mla_sm120_decode_dsv4
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import is_sm120a_supported


_D_LATENT = 512
_PAGE_SIZE = 64


def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor) -> torch.Tensor:
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil())


def _fp32_to_ue8m0_bytes(scale_fp32: torch.Tensor) -> torch.Tensor:
    bits = scale_fp32.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def _quantize_fp8_cache(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack BF16 KV into the upstream DSV4 FP8 FOOTER cache ABI."""
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2
    scale_bytes = num_tiles + 1
    bytes_per_token = data_stride + scale_bytes
    num_pages, page_size, num_kv_heads, dim = kv_bf16.shape
    assert dim == _D_LATENT and num_kv_heads == 1
    kv = kv_bf16.squeeze(2)
    result = torch.zeros(
        num_pages,
        page_size * bytes_per_token,
        dtype=torch.uint8,
        device=kv.device,
    )

    for tile_idx in range(num_tiles):
        tile = kv[..., tile_idx * tile_size : (tile_idx + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        ue8m0 = _fp32_to_ue8m0_bytes(scale)
        for token_idx in range(page_size):
            data_offset = token_idx * data_stride + tile_idx * tile_size
            result[:, data_offset : data_offset + tile_size] = fp8[:, token_idx].view(
                torch.uint8
            )
            scale_offset = page_size * data_stride + token_idx * scale_bytes + tile_idx
            result[:, scale_offset] = ue8m0[:, token_idx]

    rope = kv[..., d_nope:].contiguous().view(torch.uint8)
    rope = rope.reshape(num_pages, page_size, d_rope * 2)
    for token_idx in range(page_size):
        rope_offset = token_idx * data_stride + d_nope
        result[:, rope_offset : rope_offset + d_rope * 2] = rope[:, token_idx]
    return result.view(num_pages, page_size, 1, bytes_per_token)


def _median_us(fn, warmup_ms: int, measure_ms: int) -> float:
    fn()
    torch.cuda.synchronize()
    measurements = bench_gpu_time(
        fn, dry_run_time_ms=warmup_ms, repeat_time_ms=measure_ms
    )
    return float(np.median(measurements)) * 1e3


def _allocate_decode_scratch(
    num_tokens: int, num_heads: int, num_splits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    mid_out = torch.empty(
        num_tokens,
        num_heads,
        num_splits,
        _D_LATENT,
        dtype=torch.bfloat16,
        device=device,
    )
    mid_lse = torch.empty(
        num_tokens, num_heads, num_splits, dtype=torch.float32, device=device
    )
    output = torch.empty(
        num_tokens, num_heads, _D_LATENT, dtype=torch.bfloat16, device=device
    )
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)
    return mid_out, mid_lse, output, out_lse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--topk", type=int, nargs="+", default=(128, 512))
    parser.add_argument(
        "--cpb", type=int, nargs="+", default=None, help="optional tactic subset"
    )
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="benchmark each backend's automatic chunks-per-block tactic",
    )
    parser.add_argument("--num-pages", type=int, default=128)
    parser.add_argument("--extra-topk", type=int, default=0)
    parser.add_argument("--extra-page-size", type=int, choices=(2, 64), default=64)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measure-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--with-lengths-sink",
        action="store_true",
        help=(
            "include full top-k length tensors and an attention sink, matching "
            "the DeepSeek-V4 serving/autotune call contract"
        ),
    )
    parser.add_argument(
        "--profile-once",
        action="store_true",
        help=(
            "warm up, bracket exactly one inclusive NVFP4 decode call with "
            "the CUDA Profiler API, and exit"
        ),
    )
    parser.add_argument(
        "--probe-prefill-kernel",
        action="store_true",
        help=(
            "time the 64-head single-pass streaming prefill CTA at the "
            "decode token count as a head-grouping diagnostic"
        ),
    )
    parser.add_argument(
        "--diagnose-splits",
        action="store_true",
        help="print packed partial-output validity for one explicit CPB and exit",
    )
    args = parser.parse_args()

    if not is_sm120a_supported(torch.device("cuda")):
        raise SystemExit("NVFP4 sparse MLA requires SM120/SM121")
    if args.num_heads not in (16, 32, 64, 128):
        raise SystemExit("NVFP4 decode supports 16/32/64/128 heads")
    if args.profile_once and len(args.topk) != 1:
        raise SystemExit("--profile-once requires exactly one --topk value")
    if args.profile_once and args.cpb is not None and len(args.cpb) != 1:
        raise SystemExit("--profile-once accepts at most one --cpb value")
    if args.auto_only and args.cpb is not None:
        raise SystemExit("--auto-only and --cpb are mutually exclusive")

    torch.manual_seed(args.seed)
    kv_bf16 = (
        torch.randn(
            args.num_pages,
            _PAGE_SIZE,
            1,
            _D_LATENT,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10.0
    ).clamp(-1, 1)
    q = (
        torch.randn(
            args.num_tokens,
            args.num_heads,
            _D_LATENT,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10.0
    ).clamp(-1, 1)
    fp8_cache = _quantize_fp8_cache(kv_bf16)
    nvfp4_cache = nvfp4_quantize_pack_sparse_mla_cache(kv_bf16.squeeze(2))
    fp8_extra_cache = None
    nvfp4_extra_cache = None
    extra_indices = None
    if args.extra_topk > 0:
        extra_num_pages = max(
            args.num_pages,
            (args.extra_topk + args.extra_page_size - 1) // args.extra_page_size,
        )
        extra_bf16 = (
            torch.randn(
                extra_num_pages,
                args.extra_page_size,
                1,
                _D_LATENT,
                dtype=torch.bfloat16,
                device="cuda",
            )
            / 10.0
        ).clamp(-1, 1)
        fp8_extra_cache = _quantize_fp8_cache(extra_bf16)
        nvfp4_extra_cache = nvfp4_quantize_pack_sparse_mla_cache(extra_bf16.squeeze(2))
        extra_indices = torch.randint(
            0,
            extra_num_pages * args.extra_page_size,
            (args.num_tokens, args.extra_topk),
            dtype=torch.int32,
            device="cuda",
        )
    nvfp4_module = get_sparse_mla_nvfp4_sm120_module()
    sm_scale = _D_LATENT**-0.5
    attn_sink = (
        torch.zeros(args.num_heads, dtype=torch.float32, device="cuda")
        if args.with_lengths_sink
        else None
    )

    print(
        "topk,extra_topk,extra_page_size,cpb,fp8_full_us,"
        "nvfp4_full_us,nvfp4_stage1_us,nvfp4_merge_us,speedup_pct"
    )
    for topk in args.topk:
        if topk not in (128, 512):
            raise ValueError("the initial NVFP4 decode prototype supports topk 128/512")
        num_splits = (topk + 63) // 64 + (args.extra_topk + 63) // 64
        indices = torch.randint(
            0,
            args.num_pages * _PAGE_SIZE,
            (args.num_tokens, topk),
            dtype=torch.int32,
            device="cuda",
        )
        topk_length = (
            torch.full((args.num_tokens,), topk, dtype=torch.int32, device="cuda")
            if args.with_lengths_sink
            else None
        )
        extra_topk_length = (
            torch.full(
                (args.num_tokens,),
                args.extra_topk,
                dtype=torch.int32,
                device="cuda",
            )
            if args.with_lengths_sink and args.extra_topk > 0
            else None
        )
        fp8_mid, fp8_mid_lse, fp8_out, fp8_lse = _allocate_decode_scratch(
            args.num_tokens, args.num_heads, num_splits
        )
        nv_mid, nv_mid_lse, nv_out, nv_lse = _allocate_decode_scratch(
            args.num_tokens, args.num_heads, num_splits
        )

        def run_nv(cpb: int, *, stage1_only: bool) -> None:
            nvfp4_module.sparse_mla_sm120_nvfp4_decode(
                q,
                nvfp4_cache,
                indices,
                nv_mid,
                nv_mid_lse,
                nv_out,
                nv_lse,
                num_splits,
                sm_scale,
                topk_length,
                attn_sink,
                nvfp4_extra_cache,
                extra_indices,
                extra_topk_length,
                cpb,
                stage1_only,
            )

        if args.probe_prefill_kernel:
            if args.num_heads != 64:
                raise ValueError("--probe-prefill-kernel currently requires H=64")

            def run_nv_prefill_cta() -> None:
                nvfp4_module.sparse_mla_sm120_nvfp4_prefill(
                    q,
                    nvfp4_cache,
                    indices,
                    nv_out,
                    nv_lse,
                    sm_scale,
                    None,
                    None,
                    nvfp4_extra_cache,
                    extra_indices,
                    None,
                )

            prefill_cta_us = _median_us(
                run_nv_prefill_cta, args.warmup_ms, args.measure_ms
            )
            print(
                "HEAD64_CTA_PROBE,"
                f"tokens={args.num_tokens},topk={topk},"
                f"extra_topk={args.extra_topk},us={prefill_cta_us:.3f}"
            )
            return

        if args.diagnose_splits:
            if args.cpb is None or len(args.cpb) != 1:
                raise ValueError("--diagnose-splits requires one explicit --cpb")
            diagnose_cpb = args.cpb[0]
            active_splits = (num_splits + diagnose_cpb - 1) // diagnose_cpb
            nv_mid.fill_(float("nan"))
            nv_mid_lse.fill_(float("nan"))
            run_nv(diagnose_cpb, stage1_only=True)
            torch.cuda.synchronize()
            packed_mid = nv_mid.view(-1)[
                : (args.num_tokens * args.num_heads * active_splits * _D_LATENT)
            ].view(args.num_tokens, args.num_heads, active_splits, _D_LATENT)
            packed_lse = nv_mid_lse.view(-1)[
                : (args.num_tokens * args.num_heads * active_splits)
            ].view(args.num_tokens, args.num_heads, active_splits)
            for head_lo in range(0, args.num_heads, 16):
                head_hi = min(head_lo + 16, args.num_heads)
                mid_finite = packed_mid[:, head_lo:head_hi].isfinite().float().mean()
                lse_finite = packed_lse[:, head_lo:head_hi].isfinite().float().mean()
                print(
                    f"SPLIT_DIAG,heads={head_lo}:{head_hi},"
                    f"mid_finite={mid_finite.item():.6f},"
                    f"lse_finite={lse_finite.item():.6f},"
                    f"lse_min={packed_lse[:, head_lo:head_hi].nan_to_num().min().item():.6g},"
                    f"lse_max={packed_lse[:, head_lo:head_hi].nan_to_num().max().item():.6g}"
                )
            if nvfp4_extra_cache is not None and args.extra_page_size == 64:

                def direct_section(
                    section_cache: torch.Tensor, section_indices: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    section_splits = (section_indices.shape[1] + 63) // 64
                    section_mid, section_mid_lse, section_out, section_lse = (
                        _allocate_decode_scratch(
                            args.num_tokens, args.num_heads, section_splits
                        )
                    )
                    nvfp4_module.sparse_mla_sm120_nvfp4_decode(
                        q,
                        section_cache,
                        section_indices,
                        section_mid,
                        section_mid_lse,
                        section_out,
                        section_lse,
                        section_splits,
                        sm_scale,
                        None,
                        None,
                        None,
                        None,
                        None,
                        section_splits,
                        False,
                    )
                    return section_out, section_lse

                if active_splits == 2:
                    main_direct, main_direct_lse = direct_section(nvfp4_cache, indices)
                    extra_direct, extra_direct_lse = direct_section(
                        nvfp4_extra_cache, extra_indices
                    )
                    torch.cuda.synchronize()
                    for split, direct, direct_lse in (
                        (0, main_direct, main_direct_lse),
                        (1, extra_direct, extra_direct_lse),
                    ):
                        cosine = torch.nn.functional.cosine_similarity(
                            packed_mid[:, :, split].float(), direct.float(), dim=-1
                        )
                        print(
                            f"PARTIAL_DIAG,split={split},"
                            f"cosine={cosine.mean().item():.6g},"
                            f"max_abs={(packed_mid[:, :, split].float() - direct.float()).abs().max().item():.6g},"
                            f"lse_max_abs={(packed_lse[:, :, split] - direct_lse).abs().max().item():.6g}"
                        )
            return

        cpb_values = (
            (0,)
            if args.auto_only
            else args.cpb
            if args.cpb is not None
            else range(1, num_splits + 1)
        )
        if any(cpb < 0 or cpb > num_splits for cpb in cpb_values):
            raise ValueError(f"cpb must be 0 (auto) or in [1, {num_splits}]")

        if args.profile_once:
            profile_cpb = args.cpb[0] if args.cpb is not None else 0
            run_nv(profile_cpb, stage1_only=False)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            run_nv(profile_cpb, stage1_only=False)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(
                "profiled_nvfp4_calls=1,"
                f"tokens={args.num_tokens},heads={args.num_heads},"
                f"topk={topk},extra_topk={args.extra_topk},"
                f"cpb={profile_cpb if profile_cpb else 'auto'}"
            )
            return

        best_fp8 = float("inf")
        best_nv = float("inf")
        best_fp8_cpb = None
        best_nv_cpb = 0
        for cpb in cpb_values:
            fp8_cpb = None if cpb == 0 else cpb

            def run_fp8() -> None:
                sparse_mla_sm120_decode_dsv4(
                    q,
                    fp8_cache,
                    indices,
                    fp8_mid,
                    fp8_mid_lse,
                    fp8_out,
                    fp8_lse,
                    sm_scale,
                    topk_length=topk_length,
                    attn_sink=attn_sink,
                    chunks_per_block=fp8_cpb,
                    extra_kv_cache=fp8_extra_cache,
                    extra_indices=extra_indices,
                    extra_topk_length=extra_topk_length,
                )

            def run_nv_inclusive() -> None:
                run_nv(cpb, stage1_only=False)

            def run_nv_stage1() -> None:
                run_nv(cpb, stage1_only=True)

            fp8_us = _median_us(run_fp8, args.warmup_ms, args.measure_ms)
            inclusive_us = _median_us(run_nv_inclusive, args.warmup_ms, args.measure_ms)
            stage1_us = _median_us(run_nv_stage1, args.warmup_ms, args.measure_ms)
            merge_us = max(0.0, inclusive_us - stage1_us)
            speedup_pct = (fp8_us / inclusive_us - 1.0) * 100.0
            print(
                f"{topk},{args.extra_topk},{args.extra_page_size},{cpb},"
                f"{fp8_us:.3f},{inclusive_us:.3f},"
                f"{stage1_us:.3f},{merge_us:.3f},"
                f"{speedup_pct:.2f}"
            )
            if fp8_us < best_fp8:
                best_fp8, best_fp8_cpb = fp8_us, fp8_cpb
            if inclusive_us < best_nv:
                best_nv, best_nv_cpb = inclusive_us, cpb

        # Produce outputs once after timing and quantify the expected format
        # delta. This is not used to select a tactic.
        sparse_mla_sm120_decode_dsv4(
            q,
            fp8_cache,
            indices,
            fp8_mid,
            fp8_mid_lse,
            fp8_out,
            fp8_lse,
            sm_scale,
            topk_length=topk_length,
            attn_sink=attn_sink,
            chunks_per_block=best_fp8_cpb,
            extra_kv_cache=fp8_extra_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_length,
        )
        run_nv(best_nv_cpb, stage1_only=False)
        torch.cuda.synchronize()
        delta = (nv_out.float() - fp8_out.float()).abs()
        cosine = torch.nn.functional.cosine_similarity(
            nv_out.float(), fp8_out.float(), dim=-1
        )
        print(
            f"BEST,topk={topk},extra_topk={args.extra_topk},"
            f"extra_page_size={args.extra_page_size},"
            f"fp8_cpb={best_fp8_cpb if best_fp8_cpb is not None else 'auto'},"
            f"nvfp4_cpb={best_nv_cpb},"
            f"fp8_us={best_fp8:.3f},nvfp4_us={best_nv:.3f},"
            f"speedup_pct={(best_fp8 / best_nv - 1.0) * 100.0:.2f},"
            f"mae={delta.mean().item():.6g},max_abs={delta.max().item():.6g},"
            f"cosine_mean={cosine.mean().item():.6g},"
            f"lse_max_abs={(nv_lse - fp8_lse).abs().max().item():.6g}"
        )


if __name__ == "__main__":
    main()
