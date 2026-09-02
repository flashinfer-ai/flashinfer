# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

import pytest

from flashinfer.jit import cake_kda_packed_t1


@pytest.mark.parametrize(
    ("batch", "expected"),
    [
        (1, "register_tile16"),
        (14, "register_tile16"),
        (15, "register_tile8_interleaved"),
        (29, "register_tile8_interleaved"),
        (30, "register_tile16_warp"),
        (38, "register_tile16_warp"),
        (39, "cpasync_tile64_register_pipeline"),
        (41, "cpasync_tile64_register_pipeline"),
        (42, "cpasync_tile128_packed_state_v_private_prefetch"),
        (80, "cpasync_tile128_packed_state_v_private_prefetch"),
        (81, "cpasync_tile128_v_private_prefetch"),
        (101, "cpasync_tile128_v_private_prefetch"),
        (102, "cpasync_tile128_paired_row_pipeline"),
        (152, "cpasync_tile128_paired_row_pipeline"),
        (153, "cpasync_tile128_register_pipeline"),
        (65535, "cpasync_tile128_register_pipeline"),
    ],
)
def test_aligned_selector_matches_qualified_batch_bands(batch, expected):
    assert (
        cake_kda_packed_t1.select_cake_kda_packed_t1_variant(
            batch,
            state_aligned=True,
            aux_vec4_aligned=True,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("batch", "expected"),
    [
        (1, "cpasync_tile64_ilp4"),
        (24, "cpasync_tile64_ilp4"),
        (25, "cpasync_tile64"),
        (37, "cpasync_tile64"),
        (38, "cpasync_tile128_ilp4"),
        (39, None),
        (65535, None),
    ],
)
def test_scalar_aux_selector_fails_closed_outside_qualified_bands(batch, expected):
    assert (
        cake_kda_packed_t1.select_cake_kda_packed_t1_variant(
            batch,
            state_aligned=True,
            aux_vec4_aligned=False,
        )
        == expected
    )


def test_unaligned_state_uses_legacy_route():
    for batch in (1, 38, 512):
        assert (
            cake_kda_packed_t1.select_cake_kda_packed_t1_variant(
                batch,
                state_aligned=False,
                aux_vec4_aligned=True,
            )
            is None
        )
    with pytest.raises(ValueError, match="batch must be positive"):
        cake_kda_packed_t1.select_cake_kda_packed_t1_variant(
            0,
            state_aligned=True,
            aux_vec4_aligned=True,
        )


def test_binding_preserves_stream_stride_index_and_alignment_contracts():
    binding = (
        cake_kda_packed_t1._get_csrc_dir() / "cake_kda_packed_t1_binding.cuh"
    ).read_text()
    assert "CHECK_INPUT_TYPE(mixed_qkv, dl_bfloat16)" in binding
    assert "state_indices must have shape [B]" in binding
    assert "state.stride(0) % 8 == 0" in binding
    assert "state.data_ptr()) % 16 == 0" in binding
    assert "CAKE_KDA_PACKED_T1_REQUIRES_AUX_VEC4" in binding
    assert "mixed_qkv.stride(0)" in binding
    assert "raw_gate.stride(0)" in binding
    assert "raw_beta.stride(0)" in binding
    assert "reinterpret_cast<cudaStream_t>(cuda_stream)" in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run" in binding
