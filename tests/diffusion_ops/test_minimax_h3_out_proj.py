# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the MiniMax-H3 direct-layout attention output projection + indexed gate + residual
operator (BF16 / MXFP8 / NVFP4) over the sequence-parallel receive layout ``[P, M, 56 // P, 128]``.

Acceptance rule (shared by the three variants): the operator rounds to BF16 three times after the
FP32 accumulation (o, gate * o, residual + p), so two correct implementations with different FP32
accumulation orders legitimately disagree by one BF16 ulp on a ~1e-6 fraction of the elements.
The check therefore bounds the number of elements outside ``atol + rtol * |ref|`` (1e-2 / 1e-2)
by ``max(4, 2e-7 * numel)``; a wrong tile, row or head segment produces thousands of violations.

For the quantized variants the quantized activation the kernel wrote is compared bit-exactly with
FlashInfer's own ``mxfp8_quantize`` / ``nvfp4_quantize`` of the unpacked logical activation, and
the output is compared with the reference math applied to that quantized activation and the
prepared (FlashInfer-quantized) weight.
"""

import math
from typing import Dict, Tuple

import pytest
import torch

from flashinfer.diffusion_ops import (
    minimax_h3_out_proj,
    minimax_h3_out_proj_mxfp8,
    minimax_h3_out_proj_nvfp4,
    minimax_h3_out_proj_reference,
    prepare_minimax_h3_o_weight_mxfp8,
    prepare_minimax_h3_o_weight_nvfp4,
)
from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import _unswizzle_sf_128x4
from flashinfer.diffusion_ops.minimax_h3_out_proj import (
    MINIMAX_H3_ATTN_DIM,
    MINIMAX_H3_GATE_ROWS,
    MINIMAX_H3_HEAD_DIM,
    MINIMAX_H3_HIDDEN,
    MINIMAX_H3_NUM_HEADS,
    MINIMAX_H3_SEQUENCE_PARALLEL_DEGREES,
    MXFP8_BLOCK,
    MXFP8_SF_COLS,
    MXFP8_SF_K_TILES,
    NVFP4_BLOCK,
    NVFP4_PACKED_COLS,
    NVFP4_SF_COLS,
    NVFP4_SF_K_TILES,
    _reference_from_operands,
    _weight_scale_tiles,
    minimax_h3_nvfp4_alpha,
    minimax_h3_nvfp4_global_scale,
    minimax_h3_unpack_attn_out,
    mxfp8_activation_scale_workspace_bytes,
    nvfp4_activation_scale_workspace_bytes,
)
from flashinfer.utils import get_compute_capability

ATOL = 1e-2
RTOL = 1e-2
MAX_VIOLATION_FRACTION = 2.0e-7
MAX_VIOLATIONS_FLOOR = 4
# One row, one partial pair of 128-row tiles (129 -> 2 tiles, 257 -> 3 tiles padded to 4) and the
# production token count of one rank at sequence-parallel degree 8.
M_VALUES = [1, 129, 257, 4824]
P_VALUES = list(MINIMAX_H3_SEQUENCE_PARALLEL_DEGREES)
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    return tuple(get_compute_capability(torch.device("cuda:0"))) in {(10, 0), (10, 3)}


requires_blackwell = pytest.mark.skipif(
    not _supported(), reason="requires a CUDA GPU with compute capability 10.0 or 10.3"
)


# --------------------------------------------------------------------------------------------
# Deterministic inputs
# --------------------------------------------------------------------------------------------


def make_model(device: torch.device, seed: int = 4616) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    o_weight = torch.empty(
        (MINIMAX_H3_HIDDEN, MINIMAX_H3_ATTN_DIM), dtype=torch.bfloat16, device=device
    ).normal_(0.0, 0.01, generator=g)
    gate = torch.empty(
        (MINIMAX_H3_GATE_ROWS, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    ).uniform_(-1.0, 1.0, generator=g)
    return {"o_weight": o_weight, "gate": gate}


def make_gate_index(rows: int, device: torch.device) -> torch.Tensor:
    """Nine contiguous segments over the rows, with out-of-range indices planted at a few rows
    (``-1`` where ``row % 101 == 50``, ``9`` where ``row % 103 == 60``) so the device-side guard
    (``gate = 0``, i.e. ``out = residual``) is exercised."""
    r = torch.arange(rows, dtype=torch.int64, device=device)
    idx = torch.div(r * MINIMAX_H3_GATE_ROWS, rows, rounding_mode="floor").clamp_max(
        MINIMAX_H3_GATE_ROWS - 1
    )
    idx = torch.where(r % 101 == 50, torch.full_like(idx, -1), idx)
    idx = torch.where(r % 103 == 60, torch.full_like(idx, MINIMAX_H3_GATE_ROWS), idx)
    return idx.to(torch.int32)


def make_inputs(
    rows: int, degree: int, device: torch.device, seed: int = 4616
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(attn_out [P, M, 56 // P, 128], gate_index [M], residual [M, 5376])``."""
    g = torch.Generator(device=device)
    g.manual_seed(seed + 7919 * rows + 31 * degree)
    attn_out = torch.empty(
        (degree, rows, MINIMAX_H3_NUM_HEADS // degree, MINIMAX_H3_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).normal_(0.0, 0.5, generator=g)
    residual = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    ).normal_(0.0, 1.0, generator=g)
    return attn_out, make_gate_index(rows, device), residual


# --------------------------------------------------------------------------------------------
# Reference math for the quantized operands
# --------------------------------------------------------------------------------------------


def mxfp8_dequantize(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """E4M3 ``[R, K]`` + UE8M0 bytes ``[R, K/32]`` -> FP32 ``[R, K]``."""
    rows, cols = q.shape
    scale = torch.exp2(sf.float() - 127.0)
    return (
        q.float().reshape(rows, cols // MXFP8_BLOCK, MXFP8_BLOCK) * scale[:, :, None]
    ).reshape(rows, cols)


def nvfp4_dequantize_scaled(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 ``[R, K/2]`` + UE4M3 bytes ``[R, K/16]`` -> FP32 ``q * sf`` (the operand the MMA
    consumes; the global scales enter through alpha)."""
    rows = packed.shape[0]
    grid = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    codes = torch.empty(
        (rows, 2 * packed.shape[1]), dtype=torch.uint8, device=packed.device
    )
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    mag = grid[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    sf_f = sf.view(torch.float8_e4m3fn).float()
    return (vals.reshape(rows, -1, NVFP4_BLOCK) * sf_f[..., None]).reshape(rows, -1)


def violation_stats(y: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    diff = (y.float() - ref.float()).abs()
    bad = diff > (ATOL + RTOL * ref.float().abs())
    numel = diff.numel()
    return {
        "numel": numel,
        "violations": int(bad.sum().item()),
        "budget": max(
            MAX_VIOLATIONS_FLOOR, int(math.ceil(MAX_VIOLATION_FRACTION * numel))
        ),
        "max_abs_err": float(diff.max().item()),
        "mean_abs_err": float(diff.mean().item()),
    }


def assert_within_budget(
    y: torch.Tensor, ref: torch.Tensor, what: str
) -> Dict[str, float]:
    assert y.shape == ref.shape and y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all(), f"{what}: non-finite output"
    stats = violation_stats(y, ref)
    assert stats["violations"] <= stats["budget"], f"{what}: {stats}"
    return stats


def flashinfer_mxfp8_activation(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer's own MXFP8 quantization of ``a`` -> (E4M3 ``[M, K]``, linear UE8M0 ``[M, K/32]``)."""
    from flashinfer.quantization.fp8_quantization import mxfp8_quantize

    q, sf = mxfp8_quantize(a.contiguous(), is_sf_swizzled_layout=True)
    rows = a.shape[0]
    sf = _unswizzle_sf_128x4(sf.view(torch.uint8).reshape(-1), rows, MXFP8_SF_COLS)
    return q.view(torch.float8_e4m3fn), sf


def flashinfer_nvfp4_activation(
    a: torch.Tensor, g_a: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer's own NVFP4 quantization of ``a`` -> (packed E2M1 ``[M, K/2]``, linear UE4M3 ``[M, K/16]``)."""
    from flashinfer.quantization.fp4_quantization import nvfp4_quantize
    from flashinfer.tllm_enums import SfLayout

    q, sf = nvfp4_quantize(
        a.contiguous(), g_a, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    rows = a.shape[0]
    q = q.view(torch.uint8).reshape(rows, NVFP4_PACKED_COLS)
    sf = _unswizzle_sf_128x4(sf.view(torch.uint8).reshape(-1), rows, NVFP4_SF_COLS)
    return q, sf


def kernel_mxfp8_activation(
    workspace_q, workspace_sf, rows
) -> Tuple[torch.Tensor, torch.Tensor]:
    padded_rows = (
        mxfp8_activation_scale_workspace_bytes(rows) // (MXFP8_SF_K_TILES * 512) * 128
    )
    sf = _unswizzle_sf_128x4(
        workspace_sf[: padded_rows * MXFP8_SF_COLS], padded_rows, MXFP8_SF_COLS
    )[:rows]
    return workspace_q[:rows], sf


def kernel_nvfp4_activation(
    workspace_q, workspace_sf, rows
) -> Tuple[torch.Tensor, torch.Tensor]:
    padded_rows = (
        nvfp4_activation_scale_workspace_bytes(rows) // (NVFP4_SF_K_TILES * 512) * 128
    )
    sf = _unswizzle_sf_128x4(
        workspace_sf[: padded_rows * NVFP4_SF_COLS], padded_rows, NVFP4_SF_COLS
    )[:rows]
    return workspace_q[:rows], sf


# --------------------------------------------------------------------------------------------
# Variant checks (shared with the standalone smoke script)
# --------------------------------------------------------------------------------------------


def run_bf16_case(rows: int, degree: int, model, device) -> Dict[str, float]:
    attn_out, idx, residual = make_inputs(rows, degree, device)
    out = minimax_h3_out_proj(attn_out, model["o_weight"], model["gate"], idx, residual)
    torch.cuda.synchronize()
    ref = minimax_h3_out_proj_reference(
        attn_out, model["o_weight"], model["gate"], idx, residual
    )
    return assert_within_budget(out, ref, f"bf16 P={degree} M={rows}")


def run_mxfp8_case(rows: int, degree: int, model, prepared, device) -> Dict[str, float]:
    attn_out, idx, residual = make_inputs(rows, degree, device)
    w_q, w_tiles, w_deq = prepared
    workspace_q = torch.empty(
        (rows, MINIMAX_H3_ATTN_DIM), dtype=torch.float8_e4m3fn, device=device
    )
    workspace_sf = torch.zeros(
        (mxfp8_activation_scale_workspace_bytes(rows),),
        dtype=torch.uint8,
        device=device,
    )
    out = minimax_h3_out_proj_mxfp8(
        attn_out,
        w_q,
        w_tiles,
        model["gate"],
        idx,
        residual,
        workspace_q=workspace_q,
        workspace_sf=workspace_sf,
    )
    torch.cuda.synchronize()
    a = minimax_h3_unpack_attn_out(attn_out)
    a_q, a_sf = kernel_mxfp8_activation(workspace_q, workspace_sf, rows)
    fi_q, fi_sf = flashinfer_mxfp8_activation(a)
    q_mismatch = int((a_q.view(torch.uint8) != fi_q.view(torch.uint8)).sum().item())
    sf_mismatch = int((a_sf != fi_sf).sum().item())
    assert q_mismatch == 0 and sf_mismatch == 0, (
        f"mxfp8 P={degree} M={rows}: quantized activation differs from mxfp8_quantize(A): "
        f"{q_mismatch} E4M3 codes, {sf_mismatch} scale bytes"
    )
    ref = _reference_from_operands(
        mxfp8_dequantize(a_q, a_sf), w_deq, model["gate"], idx, residual
    )
    stats = assert_within_budget(out, ref, f"mxfp8 P={degree} M={rows}")
    stats["activation_code_mismatches"] = q_mismatch
    stats["activation_scale_mismatches"] = sf_mismatch
    return stats


def run_nvfp4_case(rows: int, degree: int, model, prepared, device) -> Dict[str, float]:
    attn_out, idx, residual = make_inputs(rows, degree, device)
    w_q, w_tiles, w_scaled, g_w = prepared
    a = minimax_h3_unpack_attn_out(attn_out)
    # Static activation global scale calibrated from the activation of this shape.
    g_a = minimax_h3_nvfp4_global_scale(a)
    alpha = minimax_h3_nvfp4_alpha(g_a, g_w)
    workspace_q = torch.empty(
        (rows, NVFP4_PACKED_COLS), dtype=torch.uint8, device=device
    )
    workspace_sf = torch.zeros(
        (nvfp4_activation_scale_workspace_bytes(rows),),
        dtype=torch.uint8,
        device=device,
    )
    out = minimax_h3_out_proj_nvfp4(
        attn_out,
        g_a,
        w_q,
        w_tiles,
        alpha,
        model["gate"],
        idx,
        residual,
        workspace_q=workspace_q,
        workspace_sf=workspace_sf,
    )
    torch.cuda.synchronize()
    a_q, a_sf = kernel_nvfp4_activation(workspace_q, workspace_sf, rows)
    fi_q, fi_sf = flashinfer_nvfp4_activation(a, g_a)
    q_mismatch = int((a_q != fi_q).sum().item())
    sf_mismatch = int((a_sf != fi_sf).sum().item())
    assert q_mismatch == 0 and sf_mismatch == 0, (
        f"nvfp4 P={degree} M={rows}: quantized activation differs from nvfp4_quantize(A): "
        f"{q_mismatch} packed bytes, {sf_mismatch} scale bytes"
    )
    ref = _reference_from_operands(
        nvfp4_dequantize_scaled(a_q, a_sf),
        w_scaled,
        model["gate"],
        idx,
        residual,
        alpha=alpha.item(),
    )
    stats = assert_within_budget(out, ref, f"nvfp4 P={degree} M={rows}")
    stats["activation_byte_mismatches"] = q_mismatch
    stats["activation_scale_mismatches"] = sf_mismatch
    return stats


def prepare_mxfp8(model):
    """Prepared weight plus its FP32 dequantization (from FlashInfer's own quantization)."""
    w_q, w_tiles = prepare_minimax_h3_o_weight_mxfp8(model["o_weight"])
    fi_q, fi_sf = flashinfer_mxfp8_activation(model["o_weight"])
    assert torch.equal(w_q.view(torch.uint8), fi_q.view(torch.uint8))
    assert torch.equal(w_tiles, _weight_scale_tiles(fi_sf))
    return w_q, w_tiles, mxfp8_dequantize(fi_q, fi_sf)


def prepare_nvfp4(model):
    g_w = minimax_h3_nvfp4_global_scale(model["o_weight"])
    w_q, w_tiles = prepare_minimax_h3_o_weight_nvfp4(model["o_weight"], g_w)
    fi_q, fi_sf = flashinfer_nvfp4_activation(model["o_weight"], g_w)
    assert torch.equal(w_q, fi_q)
    assert torch.equal(w_tiles, _weight_scale_tiles(fi_sf))
    return w_q, w_tiles, nvfp4_dequantize_scaled(fi_q, fi_sf), g_w


# --------------------------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def model(device):
    return make_model(device)


@pytest.fixture(scope="module")
def prepared_mxfp8(model):
    return prepare_mxfp8(model)


@pytest.fixture(scope="module")
def prepared_nvfp4(model):
    return prepare_nvfp4(model)


def test_minimax_h3_weight_scale_tile_layout():
    """CPU check of the combined 256-row scale-tile order against its byte-offset formula."""
    cols = 4 * 7
    k_sets = cols // 4
    sf = torch.arange(MINIMAX_H3_HIDDEN * cols, dtype=torch.int64) % 251
    sf = sf.to(torch.uint8).reshape(MINIMAX_H3_HIDDEN, cols)
    tiles = _weight_scale_tiles(sf)
    assert tiles.numel() == (MINIMAX_H3_HIDDEN // 256) * k_sets * 2 * 512
    gen = torch.Generator().manual_seed(0)
    for _ in range(256):
        n = int(torch.randint(0, MINIMAX_H3_HIDDEN, (1,), generator=gen))
        c = int(torch.randint(0, cols, (1,), generator=gen))
        n_tile, within = divmod(n, 256)
        half, r = divmod(within, 128)
        k_set, kk = divmod(c, 4)
        offset = (
            ((n_tile * k_sets + k_set) * 2 + half) * 512
            + (r % 32) * 16
            + (r // 32) * 4
            + kk
        )
        assert int(tiles[offset]) == int(sf[n, c])


def test_minimax_h3_unpack_attn_out_layout():
    """CPU check of the receive-layout unpacking against the head mapping."""
    for degree in P_VALUES:
        rows = 3
        heads_local = MINIMAX_H3_NUM_HEADS // degree
        a = torch.arange(rows * MINIMAX_H3_ATTN_DIM, dtype=torch.float32).reshape(
            rows, MINIMAX_H3_ATTN_DIM
        )
        attn_out = (
            a.reshape(rows, degree, heads_local, MINIMAX_H3_HEAD_DIM)
            .permute(1, 0, 2, 3)
            .contiguous()
            .to(torch.bfloat16)
        )
        unpacked = minimax_h3_unpack_attn_out(attn_out)
        assert torch.equal(unpacked, a.to(torch.bfloat16))


@requires_blackwell
@pytest.mark.parametrize("degree", P_VALUES)
@pytest.mark.parametrize("rows", M_VALUES)
def test_minimax_h3_out_proj_bf16(rows, degree, model, device):
    run_bf16_case(rows, degree, model, device)


@requires_blackwell
@pytest.mark.parametrize("degree", P_VALUES)
@pytest.mark.parametrize("rows", M_VALUES)
def test_minimax_h3_out_proj_mxfp8(rows, degree, model, prepared_mxfp8, device):
    run_mxfp8_case(rows, degree, model, prepared_mxfp8, device)


@requires_blackwell
@pytest.mark.parametrize("degree", P_VALUES)
@pytest.mark.parametrize("rows", M_VALUES)
def test_minimax_h3_out_proj_nvfp4(rows, degree, model, prepared_nvfp4, device):
    run_nvfp4_case(rows, degree, model, prepared_nvfp4, device)


@requires_blackwell
def test_minimax_h3_out_proj_invalid_index_rows_pass_residual(model, device):
    rows = 300
    attn_out, idx, residual = make_inputs(rows, 2, device)
    idx = idx.clone()
    idx[:3] = torch.tensor(
        [-1, MINIMAX_H3_GATE_ROWS, -(2**31)], dtype=torch.int32, device=device
    )
    out = torch.full(
        (rows, MINIMAX_H3_HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
    )
    returned = minimax_h3_out_proj(
        attn_out, model["o_weight"], model["gate"], idx, residual, out=out
    )
    torch.cuda.synchronize()
    assert returned.data_ptr() == out.data_ptr()
    assert torch.equal(out[:3], residual[:3])
    ref = minimax_h3_out_proj_reference(
        attn_out, model["o_weight"], model["gate"], idx, residual
    )
    assert_within_budget(out, ref, "bf16 invalid-index probe")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_minimax_h3_out_proj_rejects_bad_inputs(model, device):
    attn_out, idx, residual = make_inputs(8, 4, device)
    args = (model["o_weight"], model["gate"], idx, residual)
    with pytest.raises(ValueError):
        minimax_h3_out_proj(attn_out.float(), *args)
    with pytest.raises(ValueError):
        minimax_h3_out_proj(attn_out.reshape(8, 4, 14, 128), *args)  # P must be first
    with pytest.raises(ValueError):
        minimax_h3_out_proj(
            attn_out.reshape(2, 8, 28, 128), *args
        )  # P=2 must hold 28 heads
    with pytest.raises(ValueError):
        minimax_h3_out_proj(attn_out[:, :, :7], *args)
    with pytest.raises(ValueError):
        minimax_h3_out_proj(
            attn_out, model["o_weight"][:, :64], model["gate"], idx, residual
        )
    with pytest.raises(ValueError):
        minimax_h3_out_proj(
            attn_out, model["o_weight"], model["gate"], idx.long(), residual
        )
    with pytest.raises(ValueError):
        minimax_h3_out_proj(
            attn_out,
            *args,
            out=torch.empty((8, 64), dtype=torch.bfloat16, device=device),
        )
