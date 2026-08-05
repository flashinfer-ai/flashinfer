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

CPU-only integrity tests for the frozen CAKE MSA CUDA sources.
"""

import hashlib
import re
from pathlib import Path


_CAKE_MSA_CSRC_DIR = Path(__file__).resolve().parents[2] / "csrc" / "cake_msa"

_VARIANTS = (
    "decode_bf16_flat",
    "decode_bf16_paged",
    "decode_fp16_flat",
    "decode_fp16_paged",
    "decode_fp8_flat",
    "decode_fp8_paged",
    "decode_m16_bf16_flat",
    "decode_m16_bf16_paged",
    "prefill_m128_bf16_flat",
    "prefill_m128_bf16_gqa16_flat",
    "prefill_m128_bf16_gqa16_paged",
    "prefill_m128_bf16_paged",
    "prefill_m128_fp16_flat",
    "prefill_m128_fp16_paged",
    "prefill_m128_fp8_flat",
    "prefill_m128_fp8_paged",
    "prefill_m64_bf16_flat",
    "topk",
)

# Freeze the complete public source inventory, including each device body and
# host binding.
_SOURCE_SHA256 = {
    "cake_msa_decode_bf16_flat.cu": "1c2b6c95a0a3f16ade59dcd936a8495cafcccedb1d0a6f6585c29f0685792381",
    "cake_msa_decode_bf16_flat_binding.cu": "f8928306669683cca2b32d0383da941fd223faf25478a5b5c2bcb2bdefaf511f",
    "cake_msa_decode_bf16_paged.cu": "e6ac952bc77780efa28fb38bd90a687c1e422b983ff4036e7c7481efe21c6494",
    "cake_msa_decode_bf16_paged_binding.cu": "3276e49a018424f4e9c8097cd3aec3b0693deabc5ce6bada8b94d98bb8515f1b",
    "cake_msa_decode_fp16_flat.cu": "e94c19b626732dd925540bcdf7b006818200bc12660c4d2548acec678e8971ea",
    "cake_msa_decode_fp16_flat_binding.cu": "f25374e00b7645618b148ca1f3aab55568143ffa29abbc36480dcf6e275c1f44",
    "cake_msa_decode_fp16_paged.cu": "ed94d6f23528eb62a4c9e3217519f714d22115b5ddc661caa868a9339858d75b",
    "cake_msa_decode_fp16_paged_binding.cu": "a2c92ee7646527ebed5afacc7315f20e3f2e8b63c8961001b0a7742ba1877b99",
    "cake_msa_decode_fp8_flat.cu": "42a2016e362b30839e6e5b1fee08549cfe9c67e3f17ed2be46a42b3bf346f298",
    "cake_msa_decode_fp8_flat_binding.cu": "65246382bb9d4780a202a2bf404f99090ad508cb06c5074f9478852df6cd2c10",
    "cake_msa_decode_fp8_paged.cu": "c78a640e14e357f4b2329c5056dd1c6818628019c73552f615f8078f1379814b",
    "cake_msa_decode_fp8_paged_binding.cu": "181136cb7e22c18db0408180a0537a055b46b710a4d64d44c64ba08c771a1160",
    "cake_msa_decode_m16_bf16_flat.cu": "12d8893e92b2d0dbe3a7f45a894d1d98f1235922c58f09d9a69936f724343569",
    "cake_msa_decode_m16_bf16_flat_binding.cu": "7cce6013f914c642e5f3243d9ed94c2f28773bdd28f7df775aea46438924436e",
    "cake_msa_decode_m16_bf16_paged.cu": "d974c367cf2fcda56b4adb0940cd122a236e5963295a67699b49f75a598e732a",
    "cake_msa_decode_m16_bf16_paged_binding.cu": "3a05f2bfb8550b692eae38921931d90f604e3ecefc7c2f246a24683bf7fd352a",
    "cake_msa_prefill_m128_bf16_flat.cu": "4e1fc08163989f181ef0f2de316eccb9c6a7037bdd233ad2c5e6ed74b1b0f600",
    "cake_msa_prefill_m128_bf16_flat_binding.cu": "d8654b6215ce53f4315281c5dd74ecd7d45848fc26475b3036249e885a7276f0",
    "cake_msa_prefill_m128_bf16_gqa16_flat.cu": "d3dce4aa9b30d5350d7be6f2e01ababc4475116bf8cd5419c46c8473aaa33459",
    "cake_msa_prefill_m128_bf16_gqa16_flat_binding.cu": "bc9cf4c038a87738c8e583fdc17450bd5982ea6190373f8c065f9ee6446c16a5",
    "cake_msa_prefill_m128_bf16_gqa16_paged.cu": "3e6c60518c6666050ed8cb9792f6eab8a0265e0311b76b4235255df3e5477f34",
    "cake_msa_prefill_m128_bf16_gqa16_paged_binding.cu": "b0cd93d959adb957fecea0eb3744cc98c5838cc642ff088ec6c7a19bab5d157b",
    "cake_msa_prefill_m128_bf16_paged.cu": "11656c45d4e0ae5108486a199a11cb2bf005b2a792543e1503e5798252490ddb",
    "cake_msa_prefill_m128_bf16_paged_binding.cu": "f6c2ce95b6e78f94b7133ec8d9f2b3a0e2d7939a61be8c79e03e2bd2a4502df6",
    "cake_msa_prefill_m128_fp16_flat.cu": "c3d7c21398fba913bd84f2654f6b96cb27052f78c41976fb2d602e894b3f0fb2",
    "cake_msa_prefill_m128_fp16_flat_binding.cu": "c671e8d1e0fb3e7fb0b9ed768fff8e13c0ee262dafa15d3bb60f62ab6b7ead66",
    "cake_msa_prefill_m128_fp16_paged.cu": "7e7c507b0262cacfc598f99a47fa67a7ac8375ddd2ede7b1961e3047de254f66",
    "cake_msa_prefill_m128_fp16_paged_binding.cu": "e18556e08eba86c4442e49dbbe7f847ae1c82508e2318df531115698621c678a",
    "cake_msa_prefill_m128_fp8_flat.cu": "eb1c08bbf9e2f20d6579ea4397e79ede7d3e405bdff2df30114329d9cb20b4d0",
    "cake_msa_prefill_m128_fp8_flat_binding.cu": "b17fc25aeef6194c1a7b29cdd9538db619b8a04fc6f576fdc37722c977dd9c35",
    "cake_msa_prefill_m128_fp8_paged.cu": "26a8dd6df860cd18abee547fee51f1ebe97a0b5a55971615b1ddf805f3b087e6",
    "cake_msa_prefill_m128_fp8_paged_binding.cu": "3907d184e87291524f1765024e8af62bb65cd56fa59bbcbbfbc4c6342fc23907",
    "cake_msa_prefill_m64_bf16_flat.cu": "f8804ebddc44c7580e253abb29d05f7a4393bd90a8ecd13100476b6d34c432d2",
    "cake_msa_prefill_m64_bf16_flat_binding.cu": "f582a554274dd4e0ad03636eae98f31d955f68a740a5bfcd4033615dfecafada",
    "cake_msa_topk.cu": "4b4c6a2d6134980e7f6568ea1eaf07521eafddd9ed65f641097660a89f115043",
    "cake_msa_topk_binding.cu": "55e62a2f97c7645a08203814d8d283005677f1501acc451930aa4f063afbf640",
}

_FORBIDDEN_BINDING_APIS = ("EmbedCubin", "kernel.Launch", "GetKernel")
_FORBIDDEN_TMA_STORAGE_TOKENS = (
    "TmaDeviceArena",
    "TmaDeviceSlot",
    "CUdeviceptr",
    "cuCtxGetCurrent",
    "cuCtxGetDevice",
    "cuMemAlloc",
    "cudaMalloc",
    "cuMemcpyHtoD",
    "cudaMemcpy(",
    "cuStreamIsCapturing",
    "__device__ CUtensorMap",
    "__constant__ CUtensorMap",
    "fence.proxy.tensormap::generic.acquire",
)

_NEUTRAL_KERNEL_SYMBOL = re.compile(r"\bkernel_cake_msa_[a-z0-9_]+\b")
_DEVICE_KERNEL_DEFINITION = re.compile(
    r"__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+"
    r"(kernel_cake_msa_[a-z0-9_]+)\s*\("
)
_DEVICE_SOURCE_INCLUDE = re.compile(r'#include "(cake_msa_[a-z0-9_]+\.cu)"')
_CUDA_LAUNCH_SYMBOL = re.compile(
    r"cudaLaunchKernel\s*\(\s*reinterpret_cast<const void\*>\("
    r"(kernel_cake_msa_[a-z0-9_]+)\)"
)
_GRID_CONSTANT_TENSOR_MAP = re.compile(
    r"\bconst __grid_constant__ CakeMsaTensorMap ([A-Za-z0-9_]+)\b"
)
_ENCODED_HOST_TENSOR_MAP = re.compile(
    r"\bCUtensorMap h_([A-Za-z0-9_]+) = EncodeTma_[A-Za-z0-9_]+\("
)
_TENSOR_MAP_ENCODER_DEFINITION = re.compile(
    r"\binline CUtensorMap EncodeTma_([A-Za-z0-9_]+)\("
)
_DIRECT_TENSOR_MAP_LAUNCH_ARG = re.compile(r"&h_([A-Za-z0-9_]+)(?=,|})")
_CUDA_12_8_ILLEGAL_GLOBAL_VECTOR = re.compile(r"\b(?:ld|st)\.global\.v8\.b32\b")


def _device_name(variant):
    return f"cake_msa_{variant}.cu"


def _binding_name(variant):
    return f"cake_msa_{variant}_binding.cu"


def test_cake_msa_frozen_source_inventory_and_sha256():
    expected_device_names = {_device_name(variant) for variant in _VARIANTS}
    expected_binding_names = {_binding_name(variant) for variant in _VARIANTS}
    expected_names = expected_device_names | expected_binding_names
    actual_sources = {path.name: path for path in _CAKE_MSA_CSRC_DIR.glob("*.cu")}

    assert len(expected_device_names) == 18
    assert len(expected_binding_names) == 18
    assert set(_SOURCE_SHA256) == expected_names
    assert set(actual_sources) == expected_names

    device_names = {name for name in actual_sources if not name.endswith("_binding.cu")}
    binding_names = {name for name in actual_sources if name.endswith("_binding.cu")}
    assert device_names == expected_device_names
    assert binding_names == expected_binding_names

    actual_sha256 = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in actual_sources.items()
    }
    assert actual_sha256 == _SOURCE_SHA256


def test_cake_msa_device_binding_pairs_use_neutral_kernel_symbols():
    for variant in _VARIANTS:
        device_name = _device_name(variant)
        binding_name = _binding_name(variant)
        device_source = (_CAKE_MSA_CSRC_DIR / device_name).read_text(encoding="utf-8")
        binding_source = (_CAKE_MSA_CSRC_DIR / binding_name).read_text(encoding="utf-8")
        expected_symbol = f"kernel_cake_msa_{variant}"

        assert device_source.count("__global__") == 1
        assert _DEVICE_KERNEL_DEFINITION.findall(device_source) == [expected_symbol]
        assert set(_NEUTRAL_KERNEL_SYMBOL.findall(device_source)) == {expected_symbol}
        assert _DEVICE_SOURCE_INCLUDE.findall(binding_source) == [device_name]
        assert set(_NEUTRAL_KERNEL_SYMBOL.findall(binding_source)) == {expected_symbol}
        assert _CUDA_LAUNCH_SYMBOL.findall(binding_source) == [expected_symbol]

        forbidden_apis = [
            api for api in _FORBIDDEN_BINDING_APIS if api in binding_source
        ]
        assert not forbidden_apis, (
            f"{binding_name} uses forbidden generated-host APIs: {forbidden_apis}"
        )


def test_cake_msa_sources_use_cuda_12_8_global_vector_widths():
    illegal_sites = {}
    for variant in _VARIANTS:
        device_name = _device_name(variant)
        device_source = (_CAKE_MSA_CSRC_DIR / device_name).read_text(encoding="utf-8")
        matches = _CUDA_12_8_ILLEGAL_GLOBAL_VECTOR.findall(device_source)
        if matches:
            illegal_sites[device_name] = len(matches)

    assert not illegal_sites, (
        "CUDA 12.8 ptxas limits global b32 vectors to 128 bits; split each "
        f"v8 instruction into two v4 instructions: {illegal_sites}"
    )


def test_cake_msa_tma_variants_use_grid_constant_tensor_map_parameters():
    for variant in _VARIANTS:
        if variant == "topk":
            continue

        device_name = _device_name(variant)
        binding_name = _binding_name(variant)
        device_source = (_CAKE_MSA_CSRC_DIR / device_name).read_text(encoding="utf-8")
        binding_source = (_CAKE_MSA_CSRC_DIR / binding_name).read_text(encoding="utf-8")

        if variant.startswith("decode_"):
            expected_maps = {
                "Q",
                "Q_prefill",
                "K",
                "K_prefill_pair",
                "V",
                "V_prefill_pair",
                "KV",
            }
        else:
            expected_maps = {"q", "k", "v"}

        grid_constant_params = _GRID_CONSTANT_TENSOR_MAP.findall(device_source)
        assert set(grid_constant_params) == {f"{name}_value" for name in expected_maps}
        assert len(grid_constant_params) == len(expected_maps)

        encoded_host_maps = _ENCODED_HOST_TENSOR_MAP.findall(binding_source)
        encoder_definitions = _TENSOR_MAP_ENCODER_DEFINITION.findall(binding_source)
        direct_launch_maps = _DIRECT_TENSOR_MAP_LAUNCH_ARG.findall(binding_source)
        assert set(encoded_host_maps) == expected_maps
        assert len(encoded_host_maps) == len(expected_maps)
        assert set(encoder_definitions) == expected_maps
        assert len(encoder_definitions) == len(expected_maps)
        assert direct_launch_maps == encoded_host_maps

        assert "static_assert(sizeof(CUtensorMap) == 128);" in binding_source
        assert (
            "static_assert(sizeof(CakeMsaTensorMap) == sizeof(CUtensorMap));"
            in binding_source
        )

        combined_source = device_source + binding_source
        forbidden_storage = [
            token for token in _FORBIDDEN_TMA_STORAGE_TOKENS if token in combined_source
        ]
        assert not forbidden_storage, (
            f"{variant} uses forbidden tensor-map storage: {forbidden_storage}"
        )


def test_cake_msa_m128_prefill_masks_empty_register_halves():
    for variant in _VARIANTS:
        if not variant.startswith("prefill_m128_"):
            continue

        device_source = (_CAKE_MSA_CSRC_DIR / _device_name(variant)).read_text(
            encoding="utf-8"
        )
        assert device_source.count("body_valid > 0 && body_valid < 64") == 2
        assert device_source.count("valid_cols > 0 && tail_valid < 64") == 2
        assert "tail_valid > 0 && tail_valid < 64" not in device_source


def test_cake_msa_m64_prefill_masks_empty_register_halves():
    device_source = (
        _CAKE_MSA_CSRC_DIR / _device_name("prefill_m64_bf16_flat")
    ).read_text(encoding="utf-8")
    assert device_source.count("if (valid_cols > 0 && half_valid < 64)") == 2
    assert "half_valid > 0 && half_valid < 64" not in device_source
