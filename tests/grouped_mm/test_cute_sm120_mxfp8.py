"""
Copyright (c) 2025 by FlashInfer team.

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
from typing import Optional, Tuple, Union

import pytest
import torch
import torch.nn.functional as F

from flashinfer.deep_gemm import get_col_major_tma_aligned_packed_tensor
from flashinfer.grouped_mm import moe_gemm_mxfp8_nt_groupwise
from flashinfer.utils import is_sm120a_supported

COS_SIM_THRESHOLD = 0.99


def skip_if_not_sm120():
    if not is_sm120a_supported(torch.device("cuda")):
        pytest.skip("MXFP8 cute moe GEMM requires SM120 (RTX PRO 6000 Blackwell).")


# ===== Helpers COPIED FROM DeepGEMM (deep_gemm/utils/math.py)
# COPIED FROM DeepGEMM
def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


# COPIED FROM DeepGEMM
def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


# COPIED FROM DeepGEMM
def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


# COPIED FROM DeepGEMM
def pack_ue8m0_to_int(x: torch.Tensor):
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    assert (x.view(torch.int) & ((1 << 23) - 1) == 0).all()
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


# COPIED FROM DeepGEMM
def per_token_cast_to_fp8(
    x: torch.Tensor,
    use_ue8m0: bool,
    gran_k: int = 128,
    use_packed_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, gran_k)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (
        (x_view * (1.0 / sf.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    return x_fp8, pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf


# COPIED FROM DeepGEMM
def per_block_cast_to_fp8(
    x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (align(m, gran_k), align(n, gran_k)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, gran_k, x_padded.size(1) // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return (
        x_scaled.view_as(x_padded)[:m, :n].contiguous(),
        sf.view(x_view.size(0), x_view.size(2)),
    )


# COPIED FROM DeepGEMM
def transform_sf_into_required_layout(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: Union[Tuple[int, int, int], Tuple[int, int]],
    num_groups: Optional[int] = None,
    is_sfa: Optional[bool] = None,
) -> torch.Tensor:
    if len(recipe) == 3:
        assert is_sfa is not None
        gran_mn = recipe[0] if is_sfa else recipe[1]
        gran_k = recipe[2]
    elif len(recipe) == 2:
        gran_mn, gran_k = recipe
    else:
        raise ValueError(f"recipe must be 2-tuple or 3-tuple, got length {len(recipe)}")
    assert gran_k in (32, 128)
    if sf.dtype == torch.int32 and gran_mn == 1:
        return sf
    if sf.dtype == torch.float:
        if gran_mn != 1:
            sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // gran_mn)
        return get_col_major_tma_aligned_packed_tensor(sf)
    raise AssertionError(
        f"Unsupported sf transformation: dtype={sf.dtype}, gran=({gran_mn}, {gran_k})"
    )


def compute_padded_offset(offset: int, problem_idx: int, alignment: int) -> int:
    """Worst-case padded cumulative offset along M axis — each of the `problem_idx`
    prior groups assumed to need maximum `(alignment - 1)` padding rows. For this
    kernel `alignment = PACK_NSF` (sf is col-major + 4 UE8M0 pack into 1 int32 along M).
    """
    return (offset + problem_idx * (alignment - 1)) // alignment * alignment


def per_token_cast_to_mxfp8_for_moe_gemm(
    x: torch.Tensor,
    token_offset: torch.Tensor,
    gran_k: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Python-eager reference for the cute SM120 moe_gemm zero-padding quantize layout.

    Adapted from 6KD_fp8_block_scale/test/utils/layout.py — temporary stand-in for the
    dedicated CUDA quantize kernel removed in this PR. Uses DG helpers
    (per_token_cast_to_fp8, pack_ue8m0_to_int, align). Will be revisited in phase 2.
    """
    assert x.dim() == 2
    assert token_offset.dtype == torch.int32
    assert token_offset[0].item() == 0

    token_num, k = x.shape
    E = token_offset.numel() - 1
    PACK_NSF = 4
    PACK_NK = gran_k * PACK_NSF
    m_padded = compute_padded_offset(token_num, E, alignment=PACK_NSF)
    k_align = (k + PACK_NK - 1) // PACK_NK

    fp8_output = torch.empty((token_num, k), dtype=torch.float8_e4m3fn, device=x.device)
    sf_int = torch.zeros((k_align, m_padded), dtype=torch.int32, device=x.device)

    for i in range(E):
        start = token_offset[i].item()
        end = token_offset[i + 1].item()
        if start == end:
            continue
        actual_m = end - start
        expert_fp8, expert_sf_ue8m0 = per_token_cast_to_fp8(
            x[start:end], use_ue8m0=True, gran_k=gran_k
        )

        n_sf = ceil_div(k, gran_k)
        n_sf_padded = align(n_sf, PACK_NSF)
        if n_sf_padded != n_sf:
            pad = torch.zeros(
                (actual_m, n_sf_padded - n_sf), dtype=torch.float32, device=x.device
            )
            expert_sf_ue8m0 = torch.cat([expert_sf_ue8m0, pad], dim=1)

        packed_int = pack_ue8m0_to_int(expert_sf_ue8m0)

        fp8_output[start:end] = expert_fp8
        padded_offset = compute_padded_offset(start, i, alignment=PACK_NSF)
        sf_int[:, padded_offset : padded_offset + actual_m] = packed_int.t()

    return fp8_output, sf_int.transpose(0, 1)


def per_token_dequant_from_fp8(
    fp8: torch.Tensor,
    sf_ue8m0: torch.Tensor,
    gran_k: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequant per-token MXFP8: fp8 (m, k) + sf_ue8m0 (m, k_blocks) → bf16 (m, k)."""
    assert fp8.dim() == 2
    m, k = fp8.shape
    return (
        (fp8.view(m, -1, gran_k).to(torch.float32) * sf_ue8m0.unsqueeze(-1))
        .view(m, k)
        .to(dtype)
    )


def per_block_dequant_from_fp8(
    fp8: torch.Tensor,
    sf_ue8m0: torch.Tensor,
    gran_k: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequant per-block MXFP8: fp8 (g, n, k) + sf_ue8m0 (g, n_blocks, k_blocks) → bf16 (g, n, k)."""
    assert fp8.dim() == 3
    g, n, k = fp8.shape
    return (
        (
            fp8.view(g, n // gran_k, gran_k, k // gran_k, gran_k).to(torch.float32)
            * sf_ue8m0.view(g, n // gran_k, 1, k // gran_k, 1)
        )
        .view(g, n, k)
        .to(dtype)
    )


def per_block_resmooth_to_ue8m0(
    fp8: torch.Tensor, sf_fp32: torch.Tensor, gran_k: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resmooth per-block: dequant (fp8 + fp32 sf) and re-quantize to (fp8 + ue8m0 sf).

    Customer checkpoint use case — customer stores weights with FP32 (任意值) scale; cute
    SM120 GEMM kernel requires UE8M0 scale, so resmooth converts FP32 → UE8M0 scale.
    """
    assert fp8.dim() == 2
    m, k = fp8.shape
    m_blocks, k_blocks = sf_fp32.shape
    deq = (
        fp8.view(m_blocks, gran_k, k_blocks, gran_k).to(torch.float32)
        * sf_fp32.view(m_blocks, 1, k_blocks, 1)
    ).view(m, k)
    return per_block_cast_to_fp8(deq, use_ue8m0=True, gran_k=gran_k)


@pytest.mark.parametrize("num_groups", [2, 4])
@pytest.mark.parametrize("rows_per_group", [1, 8, 64, 128])
@pytest.mark.parametrize("n,k", [(4096, 7168), (7168, 4096)])
@pytest.mark.parametrize("k_gran", [32, 128])
@pytest.mark.parametrize("is_weight_scale_float", [True, False])
def test_moe_gemm_mxfp8_nt_groupwise(
    num_groups, rows_per_group, n, k, k_gran, is_weight_scale_float
):
    skip_if_not_sm120()
    torch.random.manual_seed(0)
    out_dtype = torch.bfloat16
    token_num = rows_per_group * num_groups

    a = torch.randn((token_num, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        (num_groups, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    m_indptr = torch.tensor(
        [i * rows_per_group for i in range(num_groups + 1)],
        dtype=torch.int32,
        device="cuda",
    )

    a_fp8, a_sf = per_token_cast_to_mxfp8_for_moe_gemm(a, m_indptr, gran_k=k_gran)

    a_deq = torch.zeros_like(a)
    for j in range(num_groups):
        start = int(m_indptr[j].item())
        end = int(m_indptr[j + 1].item())
        a_j_fp8, a_j_sf_ue8m0 = per_token_cast_to_fp8(
            a[start:end], use_ue8m0=True, gran_k=k_gran
        )
        a_deq[start:end] = per_token_dequant_from_fp8(
            a_j_fp8, a_j_sf_ue8m0, gran_k=k_gran, dtype=a.dtype
        )

    b_fp8_list, b_sf_ue8m0_list = [], []
    for i in range(num_groups):
        if is_weight_scale_float:
            b_i_fp8, b_i_sf = per_block_cast_to_fp8(b[i], use_ue8m0=True, gran_k=k_gran)
        else:
            # weight_scale stored with FP32; resmooth to UE8M0 once model weight loaded (not per inference call)
            b_i_fp8_raw, b_i_sf_fp32 = per_block_cast_to_fp8(
                b[i], use_ue8m0=False, gran_k=k_gran
            )
            b_i_fp8, b_i_sf = per_block_resmooth_to_ue8m0(
                b_i_fp8_raw, b_i_sf_fp32, gran_k=k_gran
            )
        b_fp8_list.append(b_i_fp8)
        b_sf_ue8m0_list.append(b_i_sf)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_sf_ue8m0 = torch.stack(b_sf_ue8m0_list, dim=0)
    b_sf = transform_sf_into_required_layout(
        b_sf_ue8m0,
        mn=n,
        k=k,
        recipe=(k_gran, k_gran),
        num_groups=num_groups,
        is_sfa=False,
    )
    b_deq = per_block_dequant_from_fp8(b_fp8, b_sf_ue8m0, gran_k=k_gran, dtype=b.dtype)

    ref = torch.zeros(token_num, n, dtype=out_dtype, device="cuda")
    for j in range(num_groups):
        start = int(m_indptr[j].item())
        end = int(m_indptr[j + 1].item())
        ref[start:end] = (a_deq[start:end] @ b_deq[j].t()).to(out_dtype)

    out = moe_gemm_mxfp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_sf,
        b_sf,
        m_indptr,
        scale_granularity_mnk=(1, 1, k_gran),
        out_dtype=out_dtype,
    )
    cos_sim = F.cosine_similarity(
        out.reshape(-1).float(), ref.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > COS_SIM_THRESHOLD, f"cos_sim={cos_sim:.4f} < {COS_SIM_THRESHOLD}"


@pytest.mark.parametrize("bad_scale", ["a_m", "a_k", "b_expert", "b_n", "b_k"])
def test_moe_gemm_mxfp8_nt_groupwise_rejects_bad_scale_shape(bad_scale):
    skip_if_not_sm120()
    torch.random.manual_seed(0)

    num_groups = 2
    rows_per_group = 1
    n = 4096
    k = 4096
    k_gran = 128
    token_num = rows_per_group * num_groups

    a = torch.randn((token_num, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        (num_groups, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    m_indptr = torch.tensor(
        [i * rows_per_group for i in range(num_groups + 1)],
        dtype=torch.int32,
        device="cuda",
    )

    a_fp8, a_sf = per_token_cast_to_mxfp8_for_moe_gemm(a, m_indptr, gran_k=k_gran)
    b_fp8_list, b_sf_ue8m0_list = [], []
    for i in range(num_groups):
        b_i_fp8, b_i_sf = per_block_cast_to_fp8(b[i], use_ue8m0=True, gran_k=k_gran)
        b_fp8_list.append(b_i_fp8)
        b_sf_ue8m0_list.append(b_i_sf)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_sf_ue8m0 = torch.stack(b_sf_ue8m0_list, dim=0)
    b_sf = transform_sf_into_required_layout(
        b_sf_ue8m0,
        mn=n,
        k=k,
        recipe=(k_gran, k_gran),
        num_groups=num_groups,
        is_sfa=False,
    )

    expected_message = "a_scale must have shape"
    if bad_scale == "a_m":
        a_sf = a_sf[:-1, :]
    elif bad_scale == "a_k":
        a_sf = a_sf[:, :-1]
    elif bad_scale == "b_expert":
        b_sf = b_sf[:-1, :, :]
        expected_message = "b_scale must have shape"
    elif bad_scale == "b_n":
        b_sf = b_sf[:, :-1, :]
        expected_message = "b_scale must have shape"
    elif bad_scale == "b_k":
        b_sf = b_sf[:, :, :-1]
        expected_message = "b_scale must have shape"

    with pytest.raises(Exception, match=expected_message):
        moe_gemm_mxfp8_nt_groupwise(
            a_fp8,
            b_fp8,
            a_sf,
            b_sf,
            m_indptr,
            scale_granularity_mnk=(1, 1, k_gran),
            out_dtype=torch.bfloat16,
        )


if __name__ == "__main__":
    pytest.main([__file__])
