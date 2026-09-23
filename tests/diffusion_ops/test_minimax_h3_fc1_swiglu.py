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
"""Tests for the fused MiniMax-H3 RMSNorm + AdaLN + FC1 + SwiGLU operator (BF16 / MXFP8 / NVFP4).

Acceptance rule (shared by the three variants): the operator rounds to BF16 three times after the
FP32 accumulation (h, silu(gate), the product) and silu amplifies a 1-ulp change of a strongly
negative gate several times, so two correct implementations with different FP32 accumulation
orders legitimately disagree by a few BF16 ulps on a ~1e-6 fraction of the elements.  The check
therefore bounds the number of elements outside ``atol + rtol * |ref|`` (1e-2 / 1.6e-2) by
``max(4, 2e-7 * numel)``; a wrong tile or row produces thousands of violations.

For the quantized variants the quantized activation the kernel wrote is compared bit-exactly with
FlashInfer's own ``mxfp8_quantize`` / ``nvfp4_quantize`` of the BF16 modulated activation the BF16
kernel produced for the same inputs, and the output is compared with the reference math applied
to that quantized activation and the prepared (FlashInfer-quantized) weight.
"""

import math
from typing import Dict, Tuple

import pytest
import torch
import torch.nn.functional as F

from flashinfer.diffusion_ops import (
    minimax_h3_fc1_swiglu,
    minimax_h3_fc1_swiglu_mxfp8,
    minimax_h3_fc1_swiglu_nvfp4,
    prepare_minimax_h3_fc1_weight_mxfp8,
    prepare_minimax_h3_fc1_weight_nvfp4,
)
from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
    MINIMAX_H3_ADALN_ROWS,
    MINIMAX_H3_EPS,
    MINIMAX_H3_FC1_ROWS,
    MINIMAX_H3_FFN,
    MINIMAX_H3_HIDDEN,
    MXFP8_BLOCK,
    MXFP8_SF_COLS,
    NVFP4_BLOCK,
    NVFP4_PACKED_COLS,
    NVFP4_SF_COLS,
    _unswizzle_sf_128x4,
    minimax_h3_nvfp4_alpha,
    minimax_h3_nvfp4_global_scale,
    mxfp8_activation_scale_workspace_bytes,
    nvfp4_activation_scale_workspace_bytes,
)
from flashinfer.utils import get_compute_capability

ATOL = 1e-2
RTOL = 1.6e-2
MAX_VIOLATION_FRACTION = 2.0e-7
MAX_VIOLATIONS_FLOOR = 4
# One row, one partial pair of 128-row tiles (129 -> 2 tiles, 257 -> 3 tiles padded to 4) and the
# production Ulysses-8 token count.
M_VALUES = [1, 129, 257, 4824]
REFERENCE_CHUNK_ROWS = 1024
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


def make_model(device: torch.device, seed: int = 4611) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    def uniform(shape, lo, hi):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).uniform_(
            lo, hi, generator=g
        )

    def normal(shape, std):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).normal_(
            0.0, std, generator=g
        )

    return {
        "x_norm_weight": uniform((MINIMAX_H3_HIDDEN,), 0.9, 1.1),
        "adaln_scale": uniform((MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
        "adaln_shift": uniform((MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
        "fc1_weight": normal((MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN), 0.02),
    }


def make_inputs(
    rows: int, device: torch.device, seed: int = 4611
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed + 7919 * rows)
    x = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    ).normal_(0.0, 0.5, generator=g)
    # "production segments": nine contiguous AdaLN segments over the rows.
    r = torch.arange(rows, dtype=torch.int64, device=device)
    idx = torch.div(r * MINIMAX_H3_ADALN_ROWS, rows, rounding_mode="floor").clamp_max(
        MINIMAX_H3_ADALN_ROWS - 1
    )
    return x, idx.to(torch.int32)


# --------------------------------------------------------------------------------------------
# Reference math
# --------------------------------------------------------------------------------------------


def reference_modulated(
    x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps=MINIMAX_H3_EPS
):
    norm = F.rms_norm(x, (MINIMAX_H3_HIDDEN,), x_norm_weight, eps=eps).to(
        torch.bfloat16
    )
    idx = adaln_index.long()
    valid = (idx >= 0) & (idx < MINIMAX_H3_ADALN_ROWS)
    safe = idx.clamp(0, MINIMAX_H3_ADALN_ROWS - 1)
    a = torch.addcmul(
        adaln_shift.index_select(0, safe),
        norm,
        (adaln_scale.index_select(0, safe) + 1.0).to(torch.bfloat16),
    ).to(torch.bfloat16)
    return torch.where(valid[:, None], a, torch.zeros_like(a))


def swiglu_from_operands(a: torch.Tensor, w: torch.Tensor, alpha=None) -> torch.Tensor:
    """``y = BF16(BF16(silu(BF16(alpha * a @ w^T)[:, :FFN])) * BF16(...)[:, FFN:])`` with an FP32 GEMM
    (TF32 disabled), evaluated in row chunks."""
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        w_t = w.float().t()
        out = torch.empty(
            (a.shape[0], MINIMAX_H3_FFN), dtype=torch.bfloat16, device=a.device
        )
        for r0 in range(0, a.shape[0], REFERENCE_CHUNK_ROWS):
            r1 = min(a.shape[0], r0 + REFERENCE_CHUNK_ROWS)
            h = a[r0:r1].float() @ w_t
            if alpha is not None:
                h = h * float(alpha)
            h = h.to(torch.bfloat16)
            gate, up = h.chunk(2, dim=-1)
            out[r0:r1] = (F.silu(gate) * up).to(torch.bfloat16)
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def reference_bf16(x, model, adaln_index):
    a = reference_modulated(
        x,
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        adaln_index,
    )
    return swiglu_from_operands(a, model["fc1_weight"])


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

    q, sf = mxfp8_quantize(a, is_sf_swizzled_layout=True)
    rows = a.shape[0]
    sf = _unswizzle_sf_128x4(sf.view(torch.uint8).reshape(-1), rows, MXFP8_SF_COLS)
    return q.view(torch.float8_e4m3fn), sf


def flashinfer_nvfp4_activation(
    a: torch.Tensor, g_a: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer's own NVFP4 quantization of ``a`` -> (packed E2M1 ``[M, K/2]``, linear UE4M3 ``[M, K/16]``)."""
    from flashinfer.quantization.fp4_quantization import nvfp4_quantize
    from flashinfer.tllm_enums import SfLayout

    q, sf = nvfp4_quantize(a, g_a, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    rows = a.shape[0]
    q = q.view(torch.uint8).reshape(rows, NVFP4_PACKED_COLS)
    sf = _unswizzle_sf_128x4(sf.view(torch.uint8).reshape(-1), rows, NVFP4_SF_COLS)
    return q, sf


def kernel_mxfp8_activation(
    workspace_q, workspace_sf, rows
) -> Tuple[torch.Tensor, torch.Tensor]:
    padded_rows = (
        mxfp8_activation_scale_workspace_bytes(rows) // (MXFP8_SF_COLS // 4 * 512) * 128
    )
    sf = _unswizzle_sf_128x4(
        workspace_sf[: padded_rows * MXFP8_SF_COLS], padded_rows, MXFP8_SF_COLS
    )[:rows]
    return workspace_q[:rows], sf


def kernel_nvfp4_activation(
    workspace_q, workspace_sf, rows
) -> Tuple[torch.Tensor, torch.Tensor]:
    padded_rows = (
        nvfp4_activation_scale_workspace_bytes(rows) // (NVFP4_SF_COLS // 4 * 512) * 128
    )
    sf = _unswizzle_sf_128x4(
        workspace_sf[: padded_rows * NVFP4_SF_COLS], padded_rows, NVFP4_SF_COLS
    )[:rows]
    return workspace_q[:rows], sf


# --------------------------------------------------------------------------------------------
# Variant checks (shared with the standalone smoke script)
# --------------------------------------------------------------------------------------------


def run_bf16_case(rows: int, model, device) -> Dict[str, float]:
    x, idx = make_inputs(rows, device)
    out = minimax_h3_fc1_swiglu(
        x,
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        idx,
        model["fc1_weight"],
    )
    torch.cuda.synchronize()
    return assert_within_budget(out, reference_bf16(x, model, idx), f"bf16 M={rows}")


def modulated_activation_from_kernel(rows: int, model, device, x, idx) -> torch.Tensor:
    """The BF16 modulated activation the BF16 kernel writes (its workspace) for these inputs."""
    workspace = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    )
    minimax_h3_fc1_swiglu(
        x,
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        idx,
        model["fc1_weight"],
        workspace=workspace,
    )
    torch.cuda.synchronize()
    return workspace


def run_mxfp8_case(rows: int, model, prepared, device) -> Dict[str, float]:
    x, idx = make_inputs(rows, device)
    w_q, w_tiles, w_deq = prepared
    workspace_q = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.float8_e4m3fn, device=device
    )
    workspace_sf = torch.zeros(
        (mxfp8_activation_scale_workspace_bytes(rows),),
        dtype=torch.uint8,
        device=device,
    )
    out = minimax_h3_fc1_swiglu_mxfp8(
        x,
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        idx,
        w_q,
        w_tiles,
        workspace_q=workspace_q,
        workspace_sf=workspace_sf,
    )
    torch.cuda.synchronize()
    a_same = modulated_activation_from_kernel(rows, model, device, x, idx)
    a_q, a_sf = kernel_mxfp8_activation(workspace_q, workspace_sf, rows)
    fi_q, fi_sf = flashinfer_mxfp8_activation(a_same)
    q_mismatch = int((a_q.view(torch.uint8) != fi_q.view(torch.uint8)).sum().item())
    sf_mismatch = int((a_sf != fi_sf).sum().item())
    assert q_mismatch == 0 and sf_mismatch == 0, (
        f"mxfp8 M={rows}: quantized activation differs from mxfp8_quantize(a): "
        f"{q_mismatch} E4M3 codes, {sf_mismatch} scale bytes"
    )
    # Independent sanity of the modulated activation itself against the PyTorch chain.
    a_ref = reference_modulated(
        x, model["x_norm_weight"], model["adaln_scale"], model["adaln_shift"], idx
    )
    a_stats = violation_stats(a_same, a_ref)
    assert a_stats["violations"] <= a_stats["budget"], (
        f"mxfp8 M={rows} modulated activation: {a_stats}"
    )
    ref = swiglu_from_operands(mxfp8_dequantize(a_q, a_sf), w_deq)
    stats = assert_within_budget(out, ref, f"mxfp8 M={rows}")
    stats["activation_code_mismatches"] = q_mismatch
    stats["activation_scale_mismatches"] = sf_mismatch
    return stats


def run_nvfp4_case(rows: int, model, prepared, device) -> Dict[str, float]:
    x, idx = make_inputs(rows, device)
    w_q, w_tiles, w_scaled, g_w = prepared
    # Static activation global scale calibrated from the reference activation of this shape.
    a_ref = reference_modulated(
        x, model["x_norm_weight"], model["adaln_scale"], model["adaln_shift"], idx
    )
    g_a = minimax_h3_nvfp4_global_scale(a_ref)
    alpha = minimax_h3_nvfp4_alpha(g_a, g_w)
    workspace_q = torch.empty(
        (rows, NVFP4_PACKED_COLS), dtype=torch.uint8, device=device
    )
    workspace_sf = torch.zeros(
        (nvfp4_activation_scale_workspace_bytes(rows),),
        dtype=torch.uint8,
        device=device,
    )
    out = minimax_h3_fc1_swiglu_nvfp4(
        x,
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        idx,
        g_a,
        w_q,
        w_tiles,
        alpha,
        workspace_q=workspace_q,
        workspace_sf=workspace_sf,
    )
    torch.cuda.synchronize()
    a_same = modulated_activation_from_kernel(rows, model, device, x, idx)
    a_q, a_sf = kernel_nvfp4_activation(workspace_q, workspace_sf, rows)
    fi_q, fi_sf = flashinfer_nvfp4_activation(a_same, g_a)
    q_mismatch = int((a_q != fi_q).sum().item())
    sf_mismatch = int((a_sf != fi_sf).sum().item())
    assert q_mismatch == 0 and sf_mismatch == 0, (
        f"nvfp4 M={rows}: quantized activation differs from nvfp4_quantize(a): "
        f"{q_mismatch} packed bytes, {sf_mismatch} scale bytes"
    )
    ref = swiglu_from_operands(
        nvfp4_dequantize_scaled(a_q, a_sf), w_scaled, alpha=alpha.item()
    )
    stats = assert_within_budget(out, ref, f"nvfp4 M={rows}")
    stats["activation_byte_mismatches"] = q_mismatch
    stats["activation_scale_mismatches"] = sf_mismatch
    return stats


def prepare_mxfp8(model):
    """Prepared weight plus its FP32 dequantization (from FlashInfer's own quantization)."""
    w_q, w_tiles = prepare_minimax_h3_fc1_weight_mxfp8(model["fc1_weight"])
    fi_q, fi_sf = flashinfer_mxfp8_activation(model["fc1_weight"])
    assert torch.equal(w_q.view(torch.uint8), fi_q.view(torch.uint8))
    return w_q, w_tiles, mxfp8_dequantize(fi_q, fi_sf)


def prepare_nvfp4(model):
    g_w = minimax_h3_nvfp4_global_scale(model["fc1_weight"])
    w_q, w_tiles = prepare_minimax_h3_fc1_weight_nvfp4(model["fc1_weight"], g_w)
    fi_q, fi_sf = flashinfer_nvfp4_activation(model["fc1_weight"], g_w)
    assert torch.equal(w_q, fi_q)
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


@requires_blackwell
@pytest.mark.parametrize("rows", M_VALUES)
def test_minimax_h3_fc1_swiglu_bf16(rows, model, device):
    run_bf16_case(rows, model, device)


@requires_blackwell
@pytest.mark.parametrize("rows", M_VALUES)
def test_minimax_h3_fc1_swiglu_mxfp8(rows, model, prepared_mxfp8, device):
    run_mxfp8_case(rows, model, prepared_mxfp8, device)


@requires_blackwell
@pytest.mark.parametrize("rows", M_VALUES)
def test_minimax_h3_fc1_swiglu_nvfp4(rows, model, prepared_nvfp4, device):
    run_nvfp4_case(rows, model, prepared_nvfp4, device)


@requires_blackwell
def test_minimax_h3_fc1_swiglu_invalid_index_rows_are_zero(model, device):
    rows = 300
    x, idx = make_inputs(rows, device)
    idx = idx.clone()
    idx[:3] = torch.tensor(
        [-1, MINIMAX_H3_ADALN_ROWS, -(2**31)], dtype=torch.int32, device=device
    )
    out = torch.full(
        (rows, MINIMAX_H3_FFN), float("nan"), dtype=torch.bfloat16, device=device
    )
    returned = minimax_h3_fc1_swiglu(
        x,
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        idx,
        model["fc1_weight"],
        out=out,
    )
    torch.cuda.synchronize()
    assert returned.data_ptr() == out.data_ptr()
    assert (out[:3] == 0).all()
    assert_within_budget(out, reference_bf16(x, model, idx), "bf16 invalid-index probe")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_minimax_h3_fc1_swiglu_rejects_bad_inputs(model, device):
    x, idx = make_inputs(8, device)
    args = (
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        idx,
        model["fc1_weight"],
    )
    with pytest.raises(ValueError):
        minimax_h3_fc1_swiglu(x.float(), *args)
    with pytest.raises(ValueError):
        minimax_h3_fc1_swiglu(x[:, :64], *args)
    with pytest.raises(ValueError):
        minimax_h3_fc1_swiglu(x, *args, eps=1e-6)
    with pytest.raises(ValueError):
        minimax_h3_fc1_swiglu(
            x,
            model["x_norm_weight"],
            model["adaln_scale"],
            model["adaln_shift"],
            idx.long(),
            model["fc1_weight"],
        )
    with pytest.raises(ValueError):
        minimax_h3_fc1_swiglu(
            x, *args, out=torch.empty((8, 64), dtype=torch.bfloat16, device=device)
        )
