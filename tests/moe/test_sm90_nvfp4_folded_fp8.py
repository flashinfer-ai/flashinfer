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

from __future__ import annotations

import pytest
import torch

from flashinfer.fused_moe.nvfp4_checkpoint import NVFP4Checkpoint
from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
    fold_nvfp4_checkpoint_to_fp8_blockscale,
    make_sm90_push_folded_fp8_weights_from_checkpoints,
)
from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
    nvfp4_weights as nvfp4_weight_impl,
)

_E2M1_REFERENCE = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _checkpoint(
    codes: torch.Tensor,
    per16_scale: torch.Tensor,
    alpha: torch.Tensor,
    *,
    mapping: tuple[int, ...] | None = None,
) -> NVFP4Checkpoint:
    experts, rows, columns = codes.shape
    packed = codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)
    return NVFP4Checkpoint(
        packed.contiguous(),
        per16_scale.to(torch.float8_e4m3fn).contiguous(),
        alpha.to(torch.float32).contiguous(),
        (experts, rows, columns),
        tuple(range(experts)) if mapping is None else mapping,
        "flashinfer.folded_fp8.test",
    )


def _reference_blockscale(
    checkpoint: NVFP4Checkpoint,
) -> tuple[torch.Tensor, torch.Tensor]:
    low = checkpoint.packed_e2m1.bitwise_and(0x0F)
    high = checkpoint.packed_e2m1.bitwise_right_shift(4).bitwise_and(0x0F)
    table = torch.tensor(_E2M1_REFERENCE, dtype=torch.float32)
    codes = torch.stack((low, high), dim=-1).reshape(checkpoint.physical_shape)
    dense = table[codes.to(torch.int64)]
    dense = dense.reshape(*dense.shape[:2], -1, 16)
    dense = dense * checkpoint.scale_e4m3_per16.to(torch.float32).unsqueeze(-1)
    dense = dense.flatten(-2)
    alpha = checkpoint.global_alpha
    multiplier = alpha.reshape(1, 1, 1) if alpha.ndim == 0 else alpha[:, None, None]
    dense = dense * multiplier
    _, logical_rows, logical_columns = checkpoint.logical_shape
    dense = dense[:, :logical_rows, :logical_columns].contiguous()
    dense = torch.where(dense == 0, torch.zeros_like(dense), dense)
    experts, rows, columns = dense.shape
    tiled = dense.reshape(experts, rows // 128, 128, columns // 128, 128)
    amax = tiled.abs().amax(dim=(2, 4))
    scales = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    quantized = (
        (tiled / scales[:, :, None, :, None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(experts, rows, columns)
    )
    return quantized, scales


def test_folded_fp8_matches_canonical_checkpoint_reference() -> None:
    codes = torch.arange(16, dtype=torch.uint8).repeat(2, 128, 8)
    per16 = torch.empty(2, 128, 8, dtype=torch.float32)
    per16[0] = torch.tensor(
        (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0), dtype=torch.float32
    )
    per16[1] = torch.tensor(
        (32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.0), dtype=torch.float32
    )
    checkpoint = _checkpoint(codes, per16, torch.tensor((0.3, 1.7)))

    actual_q, actual_sf = fold_nvfp4_checkpoint_to_fp8_blockscale(checkpoint)
    expected_q, expected_sf = _reference_blockscale(checkpoint)

    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_sf, expected_sf)
    assert not bool((actual_q.view(torch.uint8) == 0x80).any())


def test_folded_fp8_decodes_low_nibble_before_high_nibble() -> None:
    packed = torch.zeros(1, 128, 64, dtype=torch.uint8)
    packed[0, 0, 0] = 0x21
    checkpoint = NVFP4Checkpoint(
        packed,
        torch.ones(1, 128, 8, dtype=torch.float8_e4m3fn),
        torch.tensor(2.0, dtype=torch.float32),
        (1, 128, 128),
        (0,),
        "flashinfer.folded_fp8.test",
    )

    quantized, scales = fold_nvfp4_checkpoint_to_fp8_blockscale(checkpoint)

    assert torch.equal(quantized[0, 0, :2].float(), torch.tensor((224.0, 448.0)))
    assert not bool(quantized[0, 0, 2:].view(torch.uint8).any())
    assert scales.item() == torch.tensor(2.0 / 448.0, dtype=torch.float32).item()


def test_folded_fp8_ignores_physical_padding() -> None:
    codes = torch.full((1, 256, 256), 7, dtype=torch.uint8)
    codes[:, :128, :128] = 1
    per16 = torch.full((1, 256, 16), 8.0, dtype=torch.float32)
    padded = _checkpoint(codes, per16, torch.tensor(1.3))
    checkpoint = NVFP4Checkpoint(
        padded.packed_e2m1,
        padded.scale_e4m3_per16,
        padded.global_alpha,
        (1, 128, 128),
        (0,),
        padded.source_format_version,
    )
    cropped = _checkpoint(
        codes[:, :128, :128],
        per16[:, :128, :8],
        torch.tensor(1.3),
    )

    actual_q, actual_sf = fold_nvfp4_checkpoint_to_fp8_blockscale(checkpoint)
    expected_q, expected_sf = fold_nvfp4_checkpoint_to_fp8_blockscale(cropped)

    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_sf, expected_sf)


def test_folded_fp8_preserves_block_coordinates() -> None:
    codes = torch.ones(1, 256, 256, dtype=torch.uint8)
    per16 = torch.empty(1, 256, 16, dtype=torch.float32)
    per16[:, :128, :8] = 1.0
    per16[:, :128, 8:] = 2.0
    per16[:, 128:, :8] = 4.0
    per16[:, 128:, 8:] = 8.0
    checkpoint = _checkpoint(codes, per16, torch.ones(1))

    quantized, scales = fold_nvfp4_checkpoint_to_fp8_blockscale(checkpoint)
    expected_q, expected_sf = _reference_blockscale(checkpoint)

    assert torch.equal(quantized.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(scales, expected_sf)
    assert scales[0, 0, 0] < scales[0, 0, 1] < scales[0, 1, 0] < scales[0, 1, 1]


def test_folded_fp8_is_row_chunk_invariant(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(17)
    codes = torch.randint(0, 16, (2, 384, 256), dtype=torch.uint8, generator=generator)
    per16 = torch.randint(
        0, 64, (2, 384, 16), dtype=torch.int32, generator=generator
    ).to(torch.float32)
    checkpoint = _checkpoint(codes, per16, torch.tensor((0.5, 2.0)))

    monkeypatch.setattr(nvfp4_weight_impl, "_FOLDED_FP8_CHUNK_ROWS", 128)
    narrow_q, narrow_sf = fold_nvfp4_checkpoint_to_fp8_blockscale(checkpoint)
    monkeypatch.setattr(nvfp4_weight_impl, "_FOLDED_FP8_CHUNK_ROWS", 1024)
    wide_q, wide_sf = fold_nvfp4_checkpoint_to_fp8_blockscale(checkpoint)

    assert torch.equal(narrow_q.view(torch.uint8), wide_q.view(torch.uint8))
    assert torch.equal(narrow_sf, wide_sf)


def test_folded_fp8_zero_blocks_use_positive_unit_scale() -> None:
    codes = torch.full((1, 128, 128), 8, dtype=torch.uint8)
    per16 = torch.full((1, 128, 8), 448.0, dtype=torch.float32)
    checkpoint = _checkpoint(codes, per16, torch.tensor(3.0))

    quantized, scales = fold_nvfp4_checkpoint_to_fp8_blockscale(checkpoint)

    assert torch.equal(
        quantized.view(torch.uint8), torch.zeros_like(quantized.view(torch.uint8))
    )
    assert torch.equal(scales, torch.ones_like(scales))


@pytest.mark.parametrize(
    "alpha,code",
    (
        (torch.finfo(torch.float32).max, 7),
        (torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)).item(), 1),
    ),
)
def test_folded_fp8_rejects_unrepresentable_block_scales(
    alpha: float, code: int
) -> None:
    codes = torch.full((1, 128, 128), code, dtype=torch.uint8)
    per16 = torch.full((1, 128, 8), 448.0, dtype=torch.float32)
    checkpoint = _checkpoint(codes, per16, torch.tensor(alpha))

    with pytest.raises(ValueError, match="finite and positive"):
        fold_nvfp4_checkpoint_to_fp8_blockscale(checkpoint)


def test_folded_fp8_interleaves_gate_up_blocks_and_scales(monkeypatch) -> None:
    w13_codes = torch.empty(1, 512, 128, dtype=torch.uint8)
    w13_per16 = torch.empty(1, 512, 8, dtype=torch.float32)
    for block, (code, scale) in enumerate(((1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0))):
        begin = block * 128
        w13_codes[:, begin : begin + 128] = code
        w13_per16[:, begin : begin + 128] = scale
    w2_codes = torch.ones(1, 128, 256, dtype=torch.uint8)
    w2_per16 = torch.ones(1, 128, 16, dtype=torch.float32)
    w13 = _checkpoint(w13_codes, w13_per16, torch.ones(1))
    w2 = _checkpoint(w2_codes, w2_per16, torch.ones(1))

    plain = make_sm90_push_folded_fp8_weights_from_checkpoints(w13, w2)
    interleaved = make_sm90_push_folded_fp8_weights_from_checkpoints(
        w13, w2, interleave_gate_up=True
    )
    monkeypatch.setattr(nvfp4_weight_impl, "_FOLDED_FP8_CHUNK_ROWS", 128)
    chunked = make_sm90_push_folded_fp8_weights_from_checkpoints(
        w13, w2, interleave_gate_up=True
    )
    block_order = torch.tensor((0, 2, 1, 3))

    assert not plain.w13_interleaved
    assert interleaved.w13_interleaved
    assert torch.equal(
        interleaved.w13_fp8.reshape(1, 4, 128, 128).view(torch.uint8),
        plain.w13_fp8.reshape(1, 4, 128, 128).view(torch.uint8)[:, block_order],
    )
    assert torch.equal(interleaved.w13_sf, plain.w13_sf[:, block_order])
    assert torch.equal(
        interleaved.w2_fp8.view(torch.uint8), plain.w2_fp8.view(torch.uint8)
    )
    assert torch.equal(interleaved.w2_sf, plain.w2_sf)
    assert torch.equal(
        chunked.w13_fp8.view(torch.uint8), interleaved.w13_fp8.view(torch.uint8)
    )
    assert torch.equal(chunked.w13_sf, interleaved.w13_sf)
    assert torch.equal(
        chunked.w2_fp8.view(torch.uint8), interleaved.w2_fp8.view(torch.uint8)
    )
    assert torch.equal(chunked.w2_sf, interleaved.w2_sf)


def test_folded_fp8_modelopt_loader_folds_layers_sequentially(monkeypatch) -> None:
    from flashinfer.moe_ep import (
        load_sm90_push_nvfp4_modelopt_folded_fp8_weights,
    )

    w13 = _checkpoint(
        torch.ones(1, 256, 128, dtype=torch.uint8),
        torch.ones(1, 256, 8),
        torch.ones(1),
    )
    w2 = _checkpoint(
        torch.ones(1, 128, 128, dtype=torch.uint8),
        torch.ones(1, 128, 8),
        torch.ones(1),
    )
    state_dict = {
        "w13.weight": w13.packed_e2m1,
        "w13.weight_scale": w13.scale_e4m3_per16,
        "w13.weight_scale_2": w13.global_alpha,
        "w2.weight": w2.packed_e2m1,
        "w2.weight_scale": w2.scale_e4m3_per16,
        "w2.weight_scale_2": w2.global_alpha,
    }
    events: list[tuple[str, int]] = []
    original_fold = nvfp4_weight_impl._fold_nvfp4_checkpoint_to_fp8_blockscale

    def record_move(checkpoint, device):
        events.append(("move", checkpoint.logical_shape[1]))
        return checkpoint

    def record_fold(checkpoint, *, interleave_gate_up):
        events.append(("fold", checkpoint.logical_shape[1]))
        return original_fold(
            checkpoint,
            interleave_gate_up=interleave_gate_up,
        )

    monkeypatch.setattr(nvfp4_weight_impl, "_move_modelopt_checkpoint", record_move)
    monkeypatch.setattr(
        nvfp4_weight_impl,
        "_fold_nvfp4_checkpoint_to_fp8_blockscale",
        record_fold,
    )

    folded = load_sm90_push_nvfp4_modelopt_folded_fp8_weights(
        state_dict,
        w13_prefix="w13",
        w2_prefix="w2",
        device="cuda:0",
    )

    assert events == [("move", 256), ("fold", 256), ("move", 128), ("fold", 128)]
    assert folded.w13_fp8.shape == (1, 256, 128)
    assert folded.w2_fp8.shape == (1, 128, 128)


def test_folded_fp8_pair_rejects_inconsistent_shapes() -> None:
    w13 = _checkpoint(
        torch.ones(1, 256, 128, dtype=torch.uint8),
        torch.ones(1, 256, 8),
        torch.ones(1),
    )
    w2 = _checkpoint(
        torch.ones(1, 128, 256, dtype=torch.uint8),
        torch.ones(1, 128, 16),
        torch.ones(1),
    )

    with pytest.raises(ValueError, match="shapes"):
        make_sm90_push_folded_fp8_weights_from_checkpoints(w13, w2)


def test_folded_fp8_pair_rejects_inconsistent_mapping() -> None:
    w13 = _checkpoint(
        torch.ones(2, 256, 128, dtype=torch.uint8),
        torch.ones(2, 256, 8),
        torch.ones(2),
        mapping=(0, 1),
    )
    w2 = _checkpoint(
        torch.ones(2, 128, 128, dtype=torch.uint8),
        torch.ones(2, 128, 8),
        torch.ones(2),
        mapping=(1, 0),
    )

    with pytest.raises(ValueError, match="share an expert mapping"):
        make_sm90_push_folded_fp8_weights_from_checkpoints(w13, w2)


def test_folded_fp8_bundle_is_rejected_by_nvfp4_backend_contract() -> None:
    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4 import (
        validate_transformed_mega_weights,
    )

    w13 = _checkpoint(
        torch.ones(1, 256, 128, dtype=torch.uint8),
        torch.ones(1, 256, 8),
        torch.ones(1),
    )
    w2 = _checkpoint(
        torch.ones(1, 128, 128, dtype=torch.uint8),
        torch.ones(1, 128, 8),
        torch.ones(1),
    )
    folded = make_sm90_push_folded_fp8_weights_from_checkpoints(w13, w2)

    with pytest.raises(MoEEpConfigError, match="Sm90PushNvFp4Weights"):
        validate_transformed_mega_weights(
            folded,
            intermediate_size=128,
            hidden_size=128,
            num_local_experts=1,
            nvfp4_mode="w4a8",
            group_size=128,
            residual_scheme="generic",
        )


@pytest.mark.parametrize("shape", ((1, 127, 128), (1, 128, 127)))
def test_folded_fp8_rejects_unaligned_logical_shape(
    shape: tuple[int, int, int],
) -> None:
    experts, rows, columns = shape
    physical_k = ((columns + 15) // 16) * 16
    codes = torch.zeros(experts, rows, physical_k, dtype=torch.uint8)
    checkpoint = NVFP4Checkpoint(
        (codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)).contiguous(),
        torch.ones(
            experts, rows, physical_k // 16, dtype=torch.float8_e4m3fn
        ).contiguous(),
        torch.ones(experts, dtype=torch.float32),
        shape,
        tuple(range(experts)),
        "flashinfer.folded_fp8.test",
    )

    with pytest.raises(ValueError, match="divisible by 128"):
        fold_nvfp4_checkpoint_to_fp8_blockscale(checkpoint)


def test_folded_fp8_rejects_nonidentity_expert_mapping() -> None:
    codes = torch.zeros(2, 128, 128, dtype=torch.uint8)
    per16 = torch.ones(2, 128, 8, dtype=torch.float32)
    checkpoint = _checkpoint(codes, per16, torch.ones(2), mapping=(2, 3))

    with pytest.raises(ValueError, match="identity-ordered"):
        fold_nvfp4_checkpoint_to_fp8_blockscale(checkpoint)
