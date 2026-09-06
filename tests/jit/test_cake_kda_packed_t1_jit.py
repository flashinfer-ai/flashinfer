# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

import pytest

from flashinfer.jit import cake_kda_packed_t1
from flashinfer.jit import core as jit_core


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


@pytest.mark.parametrize("variant", cake_kda_packed_t1.CAKE_KDA_PACKED_T1_VARIANTS)
def test_jit_specs_bind_frozen_source_and_physical_launch_metadata(
    monkeypatch,
    tmp_path,
    variant,
):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "3a")},
    )
    monkeypatch.setattr(
        cake_kda_packed_t1.jit_env,
        "FLASHINFER_GEN_SRC_DIR",
        tmp_path,
    )
    cake_kda_packed_t1.gen_cake_kda_packed_t1_module.cache_clear()

    metadata = cake_kda_packed_t1.CAKE_KDA_PACKED_T1_VARIANT_METADATA[variant]
    spec = cake_kda_packed_t1.gen_cake_kda_packed_t1_module(variant, "sm100f")
    uri = cake_kda_packed_t1.get_cake_kda_packed_t1_uri(variant, "sm100f")

    assert spec.name == uri
    assert spec.sources == [tmp_path / uri / "cake_kda_packed_t1_binding.cu"]
    assert "-gencode=arch=compute_100f,code=sm_100f" in spec.extra_cuda_cflags
    assert "-DFLASHINFER_CAKE_KDA_PACKED_T1_TARGET_KIND=100" in spec.extra_cuda_cflags
    assert "-use_fast_math" in spec.extra_cuda_cflags
    assert "--maxrregcount=128" in spec.extra_cuda_cflags
    assert ("--ftz=false" in spec.extra_cuda_cflags) == (
        variant == "register_tile8_interleaved"
    )

    source = (cake_kda_packed_t1._get_csrc_dir() / metadata.body).read_text()
    assert metadata.symbol in source
    binding = spec.sources[0].read_text()
    assert f'#define CAKE_KDA_PACKED_T1_BODY_FILE "{metadata.body}"' in binding
    assert f"#define CAKE_KDA_PACKED_T1_KERNEL {metadata.symbol}" in binding
    assert f"#define CAKE_KDA_PACKED_T1_VALUE_TILES {metadata.value_tiles}" in binding
    assert f"#define CAKE_KDA_PACKED_T1_THREADS {metadata.threads}" in binding
    assert f"#define CAKE_KDA_PACKED_T1_SMEM_BYTES {metadata.smem_bytes}" in binding
    assert (
        "#define CAKE_KDA_PACKED_T1_REQUIRES_AUX_VEC4 "
        f"{int(metadata.requires_aux_vec4)}"
    ) in binding
    assert '#include "cake_kda_packed_t1_binding.cuh"' in binding
    cake_kda_packed_t1.gen_cake_kda_packed_t1_module.cache_clear()


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
