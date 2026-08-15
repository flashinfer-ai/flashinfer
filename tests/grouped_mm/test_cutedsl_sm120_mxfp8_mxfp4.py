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
from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from flashinfer.grouped_mm import moe_gemm_mxfp8_mxfp4_nt_groupwise
from flashinfer.utils import is_sm120a_supported

COS_SIM_THRESHOLD = 0.99
UE8M0_PACK_NUM = 4
SF_M_ALIGN = 4  # a scale word is 4B, so MN must be a multiple of 4 to keep the TMA stride 16B aligned
FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0


def skip_if_not_sm120():
    if not is_sm120a_supported(torch.device("cuda")):
        pytest.skip(
            "MXFP8 x MXFP4 cutedsl moe GEMM requires SM120 (RTX PRO 6000 Blackwell)."
        )


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def pack_ue8m0_to_int32(sf: torch.Tensor) -> torch.Tensor:
    """Pack UE8M0 fp32 scales into int32, 4 per word, little-endian along the last dim."""
    return (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)


def compute_padded_offset(offset: int, problem_idx: int, alignment: int) -> int:
    return (offset + problem_idx * (alignment - 1)) // alignment * alignment


def per_token_cast_to_mxfp8(
    x: torch.Tensor, gran_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token MXFP8 with UE8M0 scales; returns the E4M3 codes and the fp32 UE8M0 scales."""
    m, k = x.shape
    padded_k = align(k, gran_k)
    xp = torch.zeros((m, padded_k), dtype=torch.float32, device=x.device)
    xp[:, :k] = x.float()
    blocks = xp.view(m, padded_k // gran_k, gran_k)
    amax = blocks.abs().amax(dim=2).clamp_min(1e-4)
    sf = ceil_to_ue8m0(amax / FP8_E4M3_MAX)
    codes = (blocks / sf.unsqueeze(2)).to(torch.float8_e4m3fn)
    return codes.view(m, padded_k)[:, :k].contiguous(), sf


def pack_mxfp8_sfa(a_sf: torch.Tensor, m_indptr: torch.Tensor) -> torch.Tensor:
    """The A-scale ABI: MN-major int32 words, each expert starting on a 4-row aligned column."""
    m = a_sf.shape[0]
    sf = a_sf
    n_sf_padded = align(sf.shape[1], UE8M0_PACK_NUM)
    if n_sf_padded != sf.shape[1]:
        sf = F.pad(sf, (0, n_sf_padded - sf.shape[1]))
    packed = pack_ue8m0_to_int32(sf)
    offsets = m_indptr.tolist()
    num_experts = len(offsets) - 1
    m_padded = compute_padded_offset(m, num_experts, SF_M_ALIGN)
    out = torch.zeros(
        packed.shape[1], m_padded, dtype=torch.int32, device=a_sf.device
    )
    mn_major = packed.t().contiguous()
    for e in range(num_experts):
        start, end = offsets[e], offsets[e + 1]
        if start < end:
            padded = compute_padded_offset(start, e, SF_M_ALIGN)
            out[:, padded : padded + (end - start)] = mn_major[:, start:end]
    return out.view(torch.uint8)


def _e2m1_code(x: torch.Tensor) -> torch.Tensor:
    """E2M1 code: bit 3 is the sign, bits [2:0] index the magnitude grid."""
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=x.dtype
    )
    idx = torch.bucketize(x.abs().clamp_max(FP4_MAX), boundaries)
    code = idx.to(torch.uint8) | (((x < 0) & (idx != 0)).to(torch.uint8) << 3)
    return code


def dequant_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=codes.device,
        dtype=torch.float32,
    )
    idx = (codes & 0x07).to(torch.long)
    val = values[idx]
    return torch.where(((codes & 0x08) != 0) & (idx != 0), -val, val)


def per_block_cast_to_mxfp4(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weight MXFP4: two E2M1 nibbles per byte along K, plus one UE8M0 scale per 32 elements."""
    gran_k = 32
    n, k = w.shape
    padded_k = align(k, gran_k)
    wp = torch.zeros((n, padded_k), dtype=torch.float32, device=w.device)
    wp[:, :k] = w.float()
    blocks = wp.view(n, padded_k // gran_k, gran_k)
    amax = blocks.abs().amax(dim=2).clamp_min(1e-4)
    sf = ceil_to_ue8m0(amax / FP4_MAX)
    codes = _e2m1_code(blocks / sf.unsqueeze(2)).view(n, padded_k)
    pairs = codes.view(n, padded_k // 2, 2)
    packed = (pairs[:, :, 0] & 0x0F) | ((pairs[:, :, 1] & 0x0F) << 4)
    return packed[:, : k // 2].contiguous(), sf


def pack_mxfp4_sfb(sf_list) -> torch.Tensor:
    """The B-scale ABI: MN-major int32 words per expert, MN padded so the TMA stride is 16B aligned."""
    packed = []
    for sf in sf_list:
        p = pack_ue8m0_to_int32(sf).t().contiguous()
        mn = align(p.shape[1], SF_M_ALIGN)
        packed.append(F.pad(p, (0, mn - p.shape[1])).contiguous())
    return torch.stack(packed).view(torch.uint8)


def dequant_mxfp4(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    n, half_k = packed.shape
    k = half_k * 2
    codes = torch.zeros(n, k, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = (packed >> 4) & 0x0F
    return (dequant_e2m1_codes(codes).view(n, k // 32, 32) * sf[:, :, None]).view(n, k)


def build_case(expert_rows, n, k, k_gran, seed=0):
    """Quantize both operands and return the kernel inputs alongside the dequantized reference."""
    torch.random.manual_seed(seed)
    num_groups = len(expert_rows)
    offsets = [0]
    for rows in expert_rows:
        offsets.append(offsets[-1] + rows)
    token_num = offsets[-1]
    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    a = torch.randn((token_num, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((num_groups, n, k), dtype=torch.bfloat16, device="cuda") / math.sqrt(
        k
    )

    a_fp8, a_sf = per_token_cast_to_mxfp8(a, gran_k=k_gran)
    a_scale = pack_mxfp8_sfa(a_sf, m_indptr)
    a_deq = (
        a_fp8.float().view(token_num, k // k_gran, k_gran) * a_sf[:, :, None]
    ).view(token_num, k)

    quantized = [per_block_cast_to_mxfp4(b[i]) for i in range(num_groups)]
    b_fp4 = torch.stack([q.view(torch.uint8) for q, _ in quantized])
    b_scale = pack_mxfp4_sfb([sf for _, sf in quantized])

    ref = torch.zeros(token_num, n, dtype=torch.bfloat16, device="cuda")
    for i in range(num_groups):
        start, end = offsets[i], offsets[i + 1]
        if start < end:
            b_deq = dequant_mxfp4(quantized[i][0].view(torch.uint8), quantized[i][1])
            ref[start:end] = (a_deq[start:end] @ b_deq.t()).to(torch.bfloat16)
    return a_fp8, b_fp4, a_scale, b_scale, m_indptr, ref


def assert_close(out: torch.Tensor, ref: torch.Tensor, what: str):
    cos_sim = F.cosine_similarity(
        out.reshape(-1).float(), ref.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > COS_SIM_THRESHOLD, f"{what} cos_sim={cos_sim:.4f} < {COS_SIM_THRESHOLD}"


@pytest.mark.parametrize("num_groups", [2, 4])
@pytest.mark.parametrize("rows_per_group", [1, 8, 64, 128])
@pytest.mark.parametrize("n,k", [(2048, 3584), (3584, 2048)])
@pytest.mark.parametrize("k_gran", [32, 128])
def test_moe_gemm_mxfp8_mxfp4_nt_groupwise(num_groups, rows_per_group, n, k, k_gran):
    skip_if_not_sm120()
    a, b, a_scale, b_scale, m_indptr, ref = build_case(
        (rows_per_group,) * num_groups, n, k, k_gran
    )
    out = moe_gemm_mxfp8_mxfp4_nt_groupwise(
        a,
        b,
        a_scale,
        b_scale,
        m_indptr,
        scale_granularity_mnk=(1, 1, k_gran),
        out_dtype=torch.bfloat16,
    )
    assert_close(out, ref, "moe_gemm_mxfp8_mxfp4")
