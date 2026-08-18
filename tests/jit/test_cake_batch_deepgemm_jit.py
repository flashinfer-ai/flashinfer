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
from types import SimpleNamespace

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
    "large_nk": cake_batch_deepgemm.CakeBatchDeepGemmMetadata(
        n=7168,
        k=2048,
        variant=3,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_n7168_k2048",
        source="cake_batch_deepgemm_fp8_large_nk.cu",
        smem_bytes=205824,
        use_fast_math=False,
    ),
    "short_m_n6144_k7168": cake_batch_deepgemm.CakeBatchDeepGemmMetadata(
        n=6144,
        k=7168,
        variant=4,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_large_nk_cta1",
        source="cake_batch_deepgemm_fp8_short_m_n6144_k7168.cu",
        smem_bytes=203776,
        use_fast_math=False,
    ),
}

_EXPORTED_SOURCE_SHA256 = {
    "n128_k512": "850207f601659685846747df228b70774134b0f81606f7af623acc0f14627c0d",
    "n512_k128": "e666768f9cf28391e813804dfe67531ba63b3a8a555e7595db765344103dc0f4",
    "n4096_k7168": "5d3917f4108bb4260410989b2913304c2df4e89bb0a4e371ebf0a2aac0467acb",
    "large_nk": "e8b03b6cb00d397040030d35c8093e1cf28a2d944e52065ba900a7b25cfa5953",
    "short_m_n6144_k7168": "d58a9a5d0284307ed1887e4624c015a76f913e495e5c04c7e0dbbb5e00e9821d",
}
_FROZEN_BODY_BEGIN = "// BEGIN FROZEN CAKE EXPORT\n"
_FROZEN_BODY_END = "// END FROZEN CAKE EXPORT\n"


def test_cake_batch_deepgemm_metadata_and_exported_source_hashes():
    assert cake_batch_deepgemm.CAKE_BATCH_DEEPGEMM_METADATA == _EXPECTED_METADATA
    csrc_dir = cake_batch_deepgemm._get_csrc_dir()
    for shape, metadata in _EXPECTED_METADATA.items():
        source = csrc_dir / metadata.source
        assert source.is_file()
        source_text = source.read_text()
        _, begin_marker, remainder = source_text.partition(_FROZEN_BODY_BEGIN)
        frozen_body, end_marker, after_body = remainder.partition(_FROZEN_BODY_END)
        assert begin_marker == _FROZEN_BODY_BEGIN
        assert end_marker == _FROZEN_BODY_END
        assert (
            hashlib.sha256(frozen_body.encode()).hexdigest()
            == _EXPORTED_SOURCE_SHA256[shape]
        )
        assert metadata.symbol in frozen_body
        assert after_body.strip() == "// clang-format on"


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
        f"#define FLASHINFER_CAKE_BATCH_DEEPGEMM_VARIANT {metadata.variant}" in binding
    )
    assert f"#define FLASHINFER_CAKE_BATCH_DEEPGEMM_N {metadata.n}" in binding
    assert f"#define FLASHINFER_CAKE_BATCH_DEEPGEMM_K {metadata.k}" in binding
    assert (
        f"#define FLASHINFER_CAKE_BATCH_DEEPGEMM_SMEM_BYTES {metadata.smem_bytes}"
        in binding
    )
    assert (
        f"#define FLASHINFER_CAKE_BATCH_DEEPGEMM_TARGET_KIND {target_kind}" in binding
    )
    assert '#include "cake_batch_deepgemm_fp8_binding.cuh"' in binding
    cake_batch_deepgemm.gen_cake_batch_deepgemm_module.cache_clear()


def test_cake_batch_deepgemm_rejects_unknown_shape_and_target():
    with pytest.raises(ValueError, match="unsupported Cake batch DeepGEMM shape"):
        cake_batch_deepgemm.gen_cake_batch_deepgemm_module("n256_k256", "sm100a")
    with pytest.raises(ValueError, match="unsupported Cake batch DeepGEMM target"):
        cake_batch_deepgemm.gen_cake_batch_deepgemm_module("n128_k512", "sm107a")


@pytest.mark.parametrize(
    ("n", "k", "expected_m", "route"),
    [
        (128, 512, 64, "n128_k512"),
        (512, 128, 64, "n512_k128"),
        (4096, 7168, 64, "n4096_k7168"),
        (7168, 2048, 64, "large_nk"),
        (6144, 7168, 230, "large_nk"),
        (6144, 7168, 1228, "large_nk"),
        (6144, 7168, 24, "short_m_n6144_k7168"),
        (7168, 3072, 24, "large_nk"),
        (4096, 4096, 24, "large_nk"),
        (4096, 2048, 24, "large_nk"),
    ],
)
def test_cake_batch_deepgemm_source_routes(n, k, expected_m, route):
    assert cake_batch_deepgemm._select_route(n, k, expected_m) == route


@pytest.mark.parametrize(
    ("route", "expected_b_rows"),
    [
        ("large_nk", 64),
        ("short_m_n6144_k7168", 128),
    ],
)
def test_cake_batch_deepgemm_tensor_map_matches_route(
    monkeypatch,
    route,
    expected_b_rows,
):
    calls = []

    def fake_tensor_map_device(tensor, **kwargs):
        calls.append((tensor, kwargs))
        return len(calls)

    monkeypatch.setattr(
        cake_batch_deepgemm,
        "_tensor_map_device",
        fake_tensor_map_device,
    )
    a = SimpleNamespace(shape=(6, 4096, 7168))
    b = SimpleNamespace(shape=(6, 6144, 7168))
    out = SimpleNamespace(shape=(6, 4096, 6144))

    cake_batch_deepgemm._tensor_maps(a, b, out, route)

    assert calls[0][1]["box_dims"] == (128, 128, 2, 1)
    assert calls[1][1]["box_dims"] == (128, expected_b_rows, 2, 1)


def test_cake_batch_deepgemm_rejects_unknown_source_route():
    with pytest.raises(ValueError, match="unsupported Cake batch DeepGEMM shape"):
        cake_batch_deepgemm._select_route(256, 256, 64)


def test_cake_batch_deepgemm_binding_public_abi_and_boundary():
    binding = (
        cake_batch_deepgemm._get_csrc_dir() / "cake_batch_deepgemm_fp8_binding.cuh"
    ).read_text()

    assert "#include FLASHINFER_CAKE_BATCH_DEEPGEMM_BODY_FILE" in binding
    assert "kTargetSM100a = 1000" in binding
    assert "kTargetSM103a = 1003" in binding
    assert "major == 10 && minor == expected_minor" in binding
    assert "batch == 1 || batch == 4 || batch == 6 || batch == 8" in binding
    assert "batch == 32 || batch == 64" in binding
    assert "batch == 128 || batch == 256" in binding
    assert "m == 128 || m == 256 || m == 512 || m == 1024 || m == 4096" in binding
    assert "(m == 8192 || m == 16384) && batch * m <= 16384" in binding
    assert "(n == 6144 && k == 7168)" in binding
    assert "(n == 7168 && k == 3072)" in binding
    assert "(n == 4096 && k == 4096)" in binding
    assert "the short-M N6144/K7168 route requires expected_m=24" in binding
    assert "kMaxGenericPersistentCtas = 156" in binding
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
