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

import pytest
import torch

from flashinfer import NVFP44Over6Config, SfLayout, nvfp4_quantize
from flashinfer.experimental.cake_nvfp4_per_token import cake_backend as cb
from flashinfer.experimental.cake_nvfp4_per_token.cake_jit import KERNELS, MODULES

FLT_MAX = 3.4028234663852886e38
GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)


# ---------------------------------------------------------------------------
# Host rules (CPU)
# ---------------------------------------------------------------------------


def test_padding_rules():
    assert cb.padded_rows(1) == 128 and cb.padded_rows(128) == 128
    assert cb.padded_rows(129) == 256 and cb.padded_rows(4097) == 4224
    assert cb.padded_sf_cols(7168) == 448 and cb.padded_sf_cols(2112) == 132
    assert cb.padded_sf_cols(16) == 4


def test_cta_config_rules():
    # Few CTAs: widest CTA, one CTA per SM for a single row, 256 threads with
    # two blocks per thread for K <= 8192, two CTAs per SM above 128 rows; a
    # single wave of rows (128 < M <= 148) keeps the widest CTA, more than one
    # wave with at most four blocks per 256-thread lane takes 256 x 2.
    assert cb.cta_config(7168, 1) == (512, 1)
    assert cb.cta_config(7168, 8) == (256, 3)
    assert cb.cta_config(7168, 130) == (256, 3)
    assert cb.cta_config(16384, 32) == (512, 1)
    assert cb.cta_config(16384, 130) == (512, 2)
    assert cb.cta_config(16384, 148) == (512, 2)
    assert cb.cta_config(16384, 149) == (256, 2)
    assert cb.cta_config(16384, 257) == (256, 2)
    assert cb.cta_config(18432, 257) == (512, 2)
    assert cb.cta_config(28672, 257) == (512, 2)
    # Many CTAs: narrow CTAs holding 4-8 blocks per thread.
    assert cb.cta_config(7168, 512) == (256, 4)
    assert cb.cta_config(16384, 2048) == (256, 4)
    assert cb.cta_config(28672, 1000) == (256, 3)
    assert cb.cta_config(7168, 8192) == (128, 8)
    assert cb.cta_config(16384, 8192) == (128, 5)
    assert cb.cta_config(28672, 8192) == (256, 3)


def test_quant_plan_grid_and_keys():
    plan = cb.quant_plan(257, 7168, True, False, "sm_100a", 148)
    assert plan.grid == 384 and plan.padded_rows == 384 and plan.padded_cols == 448
    assert plan.kernel_key == "quant:k7168_t256_mb3_bf16"
    fold = cb.quant_plan(257, 7168, False, True, "sm_100a", 148)
    assert fold.kernel_key == "quant:k7168_t256_mb3_f16_fold"
    wide = cb.quant_plan(8192, 7168, True, False, "sm_103a", 152)
    # One CTA per padded row regardless of the SM count.
    assert wide.threads == 128 and wide.min_blocks == 8 and wide.grid == 8192
    assert wide.kernel_key == "quant:k7168_t128_mb8_bf16"
    assert cb.quant_plan(1, 28672, True, False, "sm_100a", 148).grid == 128
    with pytest.raises(ValueError):
        cb.quant_plan(8, 100, True, False, "sm_100a", 148)


@pytest.mark.parametrize("arch", cb.ARCHES)
def test_required_kernel_keys_are_registered_when_programs_exist(arch):
    required = cb.required_kernel_keys(arch)
    assert any(key.startswith("quant:") for key in required)
    assert any(key.startswith("gemm:") for key in required)
    if arch in KERNELS:
        missing = sorted(set(required) - set(KERNELS[arch]))
        assert not missing, f"{arch} lacks {missing}"
        for module_name in KERNELS[arch].values():
            assert MODULES[module_name]["arch"] == arch


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------


def _require_program():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    device = torch.device("cuda", 0)
    capability = torch.cuda.get_device_capability(device)
    if capability not in cb.SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip("the cake per-token NVFP4 quantizer requires SM100/SM103")
    if not cb.generated_program_available(device):
        pytest.skip("no generated per-token NVFP4 program registered for this GPU")
    return device


def _cute_dsl_peer():
    try:
        from flashinfer.cute_dsl import is_cute_dsl_available

        if not is_cute_dsl_available():
            return None
        from flashinfer.quantization.kernels.nvfp4_quantize import (
            nvfp4_quantize_per_token_cute_dsl,
        )
    except Exception:  # noqa: BLE001 - the peer is optional evidence
        return None
    return nvfp4_quantize_per_token_cute_dsl


def _reference(x, gs_inv, out_scale=None):
    """FP32 recipe (rounding-mode agnostic): scaled values, E4M3, per-token scales."""
    xf = x.float()
    m, k = xf.shape
    row_amax = xf.abs().amax(dim=1)
    token_scale = row_amax * gs_inv.float().reshape(())
    encode = torch.where(
        row_amax == 0, torch.full_like(token_scale, FLT_MAX), 1.0 / token_scale
    )
    token_scale = torch.where(row_amax == 0, torch.zeros_like(token_scale), token_scale)
    blocks = xf.view(m, k // 16, 16)
    block_max = blocks.abs().amax(dim=2)
    sf = encode[:, None] * (block_max / 6.0)
    sf_e4m3 = sf.to(torch.float8_e4m3fn)
    sf_dec = sf_e4m3.float()
    output_scale = torch.where(
        sf_dec == 0, torch.zeros_like(sf_dec), 1.0 / (sf_dec / encode[:, None])
    )
    scaled = blocks * output_scale[:, :, None]
    if out_scale is not None:
        token_scale = token_scale * out_scale.float().reshape(())
    return scaled.reshape(m, k), sf_e4m3, token_scale


def _check_against_reference(x, gs_inv, out_scale, fp4, sf, scale):
    """Recipe check that is agnostic to the kernels' FP32 rounding.

    Both the cake and the CuTe-DSL quantizer form the per-token encode scale with
    ``rcp.approx.ftz`` and multiply in a different association than the FP32
    reference, so a block scale that sits within one FP32 ulp of an E4M3 rounding
    tie may land one E4M3 step away from the reference (about 1e-6 of the blocks
    on random data).  The kernel is held to the reference within one E4M3 ulp on
    the block scales and the FP4 codes are checked against the values the
    kernel's own block scales imply; bitwise agreement with the CuTe-DSL peer is
    asserted separately by the caller.
    """
    m, k = x.shape
    scaled_ref, sf_ref, scale_ref = _reference(x, gs_inv, out_scale)
    torch.testing.assert_close(scale, scale_ref, atol=1e-2, rtol=1e-2)
    logical = cb.sf_logical_offsets(cb.padded_rows(m), cb.padded_sf_cols(k), x.device)
    sf_logical = sf.reshape(-1)[logical]
    assert int(sf_logical[m:].sum()) == 0, "padding rows are not zero"
    assert int(sf_logical[:, k // 16 :].sum()) == 0, "padding columns are not zero"
    sf_kernel = sf_logical[:m, : k // 16].view(torch.float8_e4m3fn).float()
    sf_ref = sf_ref.float()
    e4m3_ulp = torch.exp2(torch.floor(torch.log2(sf_ref.clamp_min(2.0**-9))) - 3)
    sf_gap = (sf_kernel - sf_ref).abs()
    assert bool((sf_gap <= e4m3_ulp * 1.001).all()), (
        f"{int((sf_gap > e4m3_ulp * 1.001).sum())} block scales differ from the "
        "reference by more than one E4M3 ulp"
    )
    off_tie = int((sf_gap > 0).sum())
    assert off_tie <= max(8, m * (k // 16) // 10000), (
        f"{off_tie} block scales differ from the reference (one E4M3 step each)"
    )
    # FP4 codes: the kernel's own block scales define the quantisation grid.
    xf = x.float()
    row_amax = xf.abs().amax(dim=1)
    token_scale = row_amax * gs_inv.float().reshape(())
    encode = torch.where(
        row_amax == 0, torch.full_like(token_scale, FLT_MAX), 1.0 / token_scale
    )
    out_sc = torch.where(
        sf_kernel == 0, torch.zeros_like(sf_kernel), 1.0 / (sf_kernel / encode[:, None])
    )
    ref_codes = (xf.view(m, k // 16, 16) * out_sc[:, :, None]).view(m, k).clamp(-6, 6)
    nib = torch.stack([fp4 & 0xF, fp4 >> 4], dim=-1).reshape(m, k).long()
    e2m1 = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=x.device,
    )
    decoded = e2m1[nib]
    quantum = torch.where(
        ref_codes.abs() < 2, 0.5, torch.where(ref_codes.abs() < 4, 1.0, 2.0)
    )
    assert bool(((decoded - ref_codes).abs() <= quantum * 1.001).all())


# The validated matrix (cake_backend.validated_problems): bf16 activations on
# every K x M x fold; fp16 activations on the validated fp16 rows of K=7168 only.
_QUANT_CASES = [
    (m, k, torch.bfloat16, fold)
    for fold in (False, True)
    for k in (7168, 16384)
    for m in (1, 17, 130, 257, 4097)
] + [(m, 7168, torch.float16, False) for m in cb.VALIDATED_F16_INPUT_ROWS]


@pytest.mark.parametrize("m,k,dtype,fold", _QUANT_CASES)
def test_quantize_matches_reference_and_cute_dsl(m, k, dtype, fold):
    device = _require_program()
    g = torch.Generator(device=device).manual_seed(1000 + m + k)
    x = torch.randn(m, k, device=device, dtype=dtype, generator=g)
    gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)
    out_scale = (
        torch.tensor([0.37], dtype=torch.float32, device=device) if fold else None
    )
    fp4, sf, scale = nvfp4_quantize(
        x,
        gs_inv,
        sfLayout=SfLayout.layout_128x4,
        per_token_activation=True,
        backend="cake",
        out_scale=out_scale,
    )
    torch.cuda.synchronize()
    assert fp4.shape == (m, k // 2) and fp4.dtype == torch.uint8
    assert sf.shape == (cb.padded_rows(m), cb.padded_sf_cols(k))
    assert sf.dtype == torch.uint8
    assert scale.shape == (m,) and scale.dtype == torch.float32
    _check_against_reference(x, gs_inv, out_scale, fp4, sf, scale)
    peer = _cute_dsl_peer()
    if peer is not None:
        p_fp4, p_sf, p_scale = peer(x, gs_inv, 0, None, out_scale)
        torch.cuda.synchronize()
        assert torch.equal(p_fp4, fp4)
        assert torch.equal(p_sf.reshape(-1), sf.reshape(-1))
        assert torch.equal(p_scale, scale)


def test_prepared_runner_graph_replay_and_no_allocation():
    device = _require_program()
    m, k = 130, 7168
    g = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=g)
    gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)
    outputs = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
    runner = cb.prepare_nvfp4_per_token_quantize(x, gs_inv, outputs)
    assert runner.launch_count == 1
    runner()
    torch.cuda.synchronize()
    eager = tuple(t.clone() for t in outputs)
    before = torch.cuda.memory_stats()
    runner()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] - before["allocation.all.allocated"] == 0
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner()
    torch.cuda.synchronize()
    for _round_index in range(2):
        x.copy_(torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=g))
        for t in outputs:
            t.view(torch.uint8).fill_(0xFF)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        expected = nvfp4_quantize(x, gs_inv, per_token_activation=True, backend="cake")
        torch.cuda.synchronize()
        for got, want in zip(outputs, expected, strict=True):
            assert torch.equal(got.reshape(-1), want.reshape(-1))
    del eager


def test_rejections():
    device = _require_program()
    x = torch.randn(8, 7168, device=device, dtype=torch.bfloat16)
    gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="per_token_activation=True only"):
        nvfp4_quantize(x, gs_inv, backend="cake")
    with pytest.raises(ValueError, match="128x4"):
        nvfp4_quantize(
            x,
            gs_inv,
            sfLayout=SfLayout.layout_8x4,
            per_token_activation=True,
            backend="cake",
        )
    with pytest.raises(ValueError, match="128x4"):
        nvfp4_quantize(
            x, gs_inv, do_shuffle=True, per_token_activation=True, backend="cake"
        )
    with pytest.raises(ValueError, match="dependent launch"):
        nvfp4_quantize(
            x, gs_inv, per_token_activation=True, backend="cake", enable_pdl=False
        )
    with pytest.raises(ValueError, match="4over6"):
        nvfp4_quantize(
            x,
            gs_inv,
            per_token_activation=True,
            backend="cake",
            nvfp4_4over6=NVFP44Over6Config(),
        )
    outputs = cb.allocate_nvfp4_per_token_quantize_outputs(4, 7168, device)
    with pytest.raises(ValueError, match="fp4"):
        cb.prepare_nvfp4_per_token_quantize(x, gs_inv, outputs)
