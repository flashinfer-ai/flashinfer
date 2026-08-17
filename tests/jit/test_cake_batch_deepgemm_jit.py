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

import hashlib

import pytest

from flashinfer.jit import core as jit_core
from flashinfer.jit.gemm import cake_batch_deepgemm


_EXPECTED_METADATA = {
    "n128_k512": cake_batch_deepgemm.CakeBatchDeepGemmMetadata(
        n=128,
        k=512,
        variant=0,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_n128_k512",
        source="cake_batch_deepgemm_fp8_n128_k512.cu",
        smem_bytes=103424,
        use_fast_math=True,
    ),
    "n512_k128": cake_batch_deepgemm.CakeBatchDeepGemmMetadata(
        n=512,
        k=128,
        variant=1,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_n512_k128",
        source="cake_batch_deepgemm_fp8_n512_k128.cu",
        smem_bytes=50176,
        use_fast_math=True,
    ),
    "n4096_k7168": cake_batch_deepgemm.CakeBatchDeepGemmMetadata(
        n=4096,
        k=7168,
        variant=2,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_n4096_k7168",
        source="cake_batch_deepgemm_fp8_n4096_k7168.cu",
        smem_bytes=203776,
        use_fast_math=False,
    ),
    "n7168_k2048": cake_batch_deepgemm.CakeBatchDeepGemmMetadata(
        n=7168,
        k=2048,
        variant=3,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_n7168_k2048",
        source="cake_batch_deepgemm_fp8_n7168_k2048.cu",
        smem_bytes=205824,
        use_fast_math=False,
    ),
}

_EXPORTED_SOURCE_SHA256 = {
    "n128_k512": "850207f601659685846747df228b70774134b0f81606f7af623acc0f14627c0d",
    "n512_k128": "e666768f9cf28391e813804dfe67531ba63b3a8a555e7595db765344103dc0f4",
    "n4096_k7168": "5d3917f4108bb4260410989b2913304c2df4e89bb0a4e371ebf0a2aac0467acb",
    "n7168_k2048": "5f59bd3cf2a7062e5445ec183934cdb476a7b2072c9fcbc4130a812b33f6e3be",
}


def test_cake_batch_deepgemm_metadata_and_exported_source_hashes():
    assert cake_batch_deepgemm.CAKE_BATCH_DEEPGEMM_METADATA == _EXPECTED_METADATA
    csrc_dir = cake_batch_deepgemm._get_csrc_dir()
    for shape, metadata in _EXPECTED_METADATA.items():
        source = csrc_dir / metadata.source
        assert source.is_file()
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest()
            == _EXPORTED_SOURCE_SHA256[shape]
        )
        assert metadata.symbol in source.read_text()


@pytest.mark.parametrize("shape", tuple(_EXPECTED_METADATA))
@pytest.mark.parametrize(
    ("target", "target_arch", "expected_flag", "target_kind"),
    [
        (
            "sm100a",
            (10, "0a"),
            "-gencode=arch=compute_100a,code=sm_100a",
            1000,
        ),
        (
            "sm103a",
            (10, "3a"),
            "-gencode=arch=compute_103a,code=sm_103a",
            1003,
        ),
    ],
)
def test_cake_batch_deepgemm_jit_spec(
    monkeypatch,
    tmp_path,
    shape,
    target,
    target_arch,
    expected_flag,
    target_kind,
):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {target_arch},
    )
    monkeypatch.setattr(
        cake_batch_deepgemm.jit_env,
        "FLASHINFER_GEN_SRC_DIR",
        tmp_path,
    )
    cake_batch_deepgemm.gen_cake_batch_deepgemm_module.cache_clear()

    metadata = _EXPECTED_METADATA[shape]
    uri = f"cake_batch_deepgemm_fp8_{shape}_{target}"
    spec = cake_batch_deepgemm.gen_cake_batch_deepgemm_module(shape, target)

    assert spec.name == uri
    assert spec.sources == [tmp_path / uri / f"{uri}_binding.cu"]
    assert spec.sources[0].is_file()
    assert expected_flag in spec.extra_cuda_cflags
    assert sum("-gencode=arch=compute_" in flag for flag in spec.extra_cuda_cflags) == 1
    forbidden_compute = "compute_103a" if target == "sm100a" else "compute_100a"
    assert not any(forbidden_compute in flag for flag in spec.extra_cuda_cflags)
    if metadata.use_fast_math:
        assert "--use_fast_math" in spec.extra_cuda_cflags
    else:
        assert "--use_fast_math" not in spec.extra_cuda_cflags

    binding = spec.sources[0].read_text()
    assert (
        f'#define FLASHINFER_CAKE_BATCH_DEEPGEMM_BODY_FILE "{metadata.source}"'
        in binding
    )
    assert f"#define FLASHINFER_CAKE_BATCH_DEEPGEMM_KERNEL {metadata.symbol}" in binding
    assert (
        f"#define FLASHINFER_CAKE_BATCH_DEEPGEMM_VARIANT {metadata.variant}"
        in binding
    )
    assert f"#define FLASHINFER_CAKE_BATCH_DEEPGEMM_N {metadata.n}" in binding
    assert f"#define FLASHINFER_CAKE_BATCH_DEEPGEMM_K {metadata.k}" in binding
    assert (
        f"#define FLASHINFER_CAKE_BATCH_DEEPGEMM_SMEM_BYTES {metadata.smem_bytes}"
        in binding
    )
    assert (
        f"#define FLASHINFER_CAKE_BATCH_DEEPGEMM_TARGET_KIND {target_kind}"
        in binding
    )
    assert '#include "cake_batch_deepgemm_fp8_binding.cuh"' in binding
    cake_batch_deepgemm.gen_cake_batch_deepgemm_module.cache_clear()


def test_cake_batch_deepgemm_rejects_unknown_shape_and_target():
    with pytest.raises(ValueError, match="unsupported Cake batch DeepGEMM shape"):
        cake_batch_deepgemm.gen_cake_batch_deepgemm_module("n256_k256", "sm100a")
    with pytest.raises(ValueError, match="unsupported Cake batch DeepGEMM target"):
        cake_batch_deepgemm.gen_cake_batch_deepgemm_module("n128_k512", "sm107a")


def test_cake_batch_deepgemm_binding_public_abi_and_boundary():
    binding = (
        cake_batch_deepgemm._get_csrc_dir() / "cake_batch_deepgemm_fp8_binding.cuh"
    ).read_text()

    assert "#include FLASHINFER_CAKE_BATCH_DEEPGEMM_BODY_FILE" in binding
    assert "kTargetSM100a = 1000" in binding
    assert "kTargetSM103a = 1003" in binding
    assert "major == 10 && minor == expected_minor" in binding
    assert "batch == 1 || batch == 4 || batch == 8 || batch == 64" in binding
    assert "batch == 128 || batch == 256" in binding
    assert "m == 128 || m == 256 || m == 512 || m == 1024" in binding
    assert "(m == 8192 || m == 16384) && batch * m <= 16384" in binding
    assert "a_scale must have shape [B,M,K/128]" in binding
    assert "b_scale must have shape [B,N/128,K/128]" in binding
    assert "masked_m must have shape [B]" in binding
    assert "expected_m must be in [0,M]" in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run" in binding
    assert "mask_values" not in binding
    assert "valid_rows" not in binding
    assert "tile_codes" not in binding
    assert ".cpu()" not in binding
    assert ".item()" not in binding
