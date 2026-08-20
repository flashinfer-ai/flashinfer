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

CPU-only integrity tests for the frozen Blackwell MSA CUDA sources.
"""

import hashlib
import re
from pathlib import Path


_BLACKWELL_MSA_CSRC_DIR = Path(__file__).resolve().parents[2] / "csrc" / "blackwell_msa"

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
    "blackwell_msa_decode_bf16_flat.cu": "e0127ebb17aa637c3347518f9aec476fa4838ff0b6bbfd2b7e5fdf8d1cb62f77",
    "blackwell_msa_decode_bf16_flat_binding.cu": "14475a4e1ba841520736b7c57e0d99fea28b10940be5b78548bd1543dbd617e2",
    "blackwell_msa_decode_bf16_paged.cu": "7c79ae4486ee4539fd79232303d4e07a1eb928fb7d1776ba64a2e8713559f644",
    "blackwell_msa_decode_bf16_paged_binding.cu": "e520d6ea011b423f3f42fc40af578f5a6d5d7dcfc7f7a30f2bd33997cc468b58",
    "blackwell_msa_decode_fp16_flat.cu": "dfb89ad38d688a638eb76f8f8653d393b9109a000ccb1d77b7498260a76dd7b6",
    "blackwell_msa_decode_fp16_flat_binding.cu": "f39fc6a250fc98728baac24a8ea714f2d0d80595a121d018e22d1291d97f26cf",
    "blackwell_msa_decode_fp16_paged.cu": "fb0d34923c846e04803805f8a613d8142c2783c1b27bad2c2bfbc5e8a8661ead",
    "blackwell_msa_decode_fp16_paged_binding.cu": "ed1bdddf7a907e24e46ab81d5822b9f9d61835008b3c3884ddd055acef8c45c6",
    "blackwell_msa_decode_fp8_flat.cu": "29c6887a86bc517a0a9844a14ed2294dedf473aa3032d7318e818298192e6fd5",
    "blackwell_msa_decode_fp8_flat_binding.cu": "c64c8469fed41db19c464926f14a89dc7aec1a4df2cdd59ebbee539693b32e53",
    "blackwell_msa_decode_fp8_paged.cu": "85a2b48ca6cb670840965429b286e4c97a5088880a85d7be716f23c5d9cf9f67",
    "blackwell_msa_decode_fp8_paged_binding.cu": "b2557af47464995126bab0736c1c52cb3d20c0e7b78c2806540239048d07e5f7",
    "blackwell_msa_decode_m16_bf16_flat.cu": "b7791ca358b1e9debf6e05351691a27bc3663b62a0d8ba5a2e977ea2fe64997d",
    "blackwell_msa_decode_m16_bf16_flat_binding.cu": "d339e955469765bd562f0f84abbe01f474e13abdb1fa117e87557a3a081e911a",
    "blackwell_msa_decode_m16_bf16_paged.cu": "bd46e6b49bc6d9f507fa20faa2619941463794b8b9dd7e89b5fd31885efe7689",
    "blackwell_msa_decode_m16_bf16_paged_binding.cu": "9105c5b1bc8d3101b0e6bc36fc509a65bad64454e33de473cb78d326be8d85fa",
    "blackwell_msa_prefill_m128_bf16_flat.cu": "83f9003707c0845600a0790eaadc401c6e28507ae3f533e72b1e21cf76ee1d10",
    "blackwell_msa_prefill_m128_bf16_flat_binding.cu": "57a5775f27ed23b5884d3eb716213c1803a32580691b11a4fc2edd4f3d9ccf19",
    "blackwell_msa_prefill_m128_bf16_gqa16_flat.cu": "3c9d4673d3d254d52923e4383fa665ecff34ea9edaa38330dc3095f9d50a8bf4",
    "blackwell_msa_prefill_m128_bf16_gqa16_flat_binding.cu": "9d2592e985abcc0c8e286d3079822b58fa87c972e7af4c53492bee5b72cec18f",
    "blackwell_msa_prefill_m128_bf16_gqa16_paged.cu": "8fcbe045f489d10039fdbf811622cfafb49445b61d6fb856566e6303828c7188",
    "blackwell_msa_prefill_m128_bf16_gqa16_paged_binding.cu": "01486a7729d1b2a9881128874b929e03052824dca40a31f1e3ac635e0c4674a5",
    "blackwell_msa_prefill_m128_bf16_paged.cu": "8995794afb84f6d8d9cf69762b93b4a7187e3bf0bf96765ecc402261f1b417c0",
    "blackwell_msa_prefill_m128_bf16_paged_binding.cu": "82b354fb45694df140c9ffae41243a239e452eb0bcaafc665ba68fb0ab38f097",
    "blackwell_msa_prefill_m128_fp16_flat.cu": "7d9f72d35b1838a08f4bf74930b8a2e918c0b25f3817c46082b9528547f123dd",
    "blackwell_msa_prefill_m128_fp16_flat_binding.cu": "9c54c9d623918adba6baa586ff59f241932be01d0f66b678302f208e38a01210",
    "blackwell_msa_prefill_m128_fp16_paged.cu": "5c68eb244cb924b45f83007a13b44a750b838e913155f750193393f19ce30858",
    "blackwell_msa_prefill_m128_fp16_paged_binding.cu": "636b72a84084ffdeaafbc8f4e8fe7fe8162d4309503191a377dff7a237f12056",
    "blackwell_msa_prefill_m128_fp8_flat.cu": "157381a895da42224d875bb0190e7b3325b6c6851e140620e94420795d35fb24",
    "blackwell_msa_prefill_m128_fp8_flat_binding.cu": "aec2bf31fba6372834eaab1261a331293fbd5cd968b85bf7b72c3e0ddb7f56df",
    "blackwell_msa_prefill_m128_fp8_paged.cu": "010443ddc665494c40e0391417693da5db5dac595c82da62e86012a05e2d9ac6",
    "blackwell_msa_prefill_m128_fp8_paged_binding.cu": "4c957594df4922a5956b10c450644e46cf3ec69ed486da0d0559be2352d5d1d1",
    "blackwell_msa_prefill_m64_bf16_flat.cu": "26944467b8f45a2a04607f417fd4173a83db9cce50aa1700eb28a50a6fb8f7c9",
    "blackwell_msa_prefill_m64_bf16_flat_binding.cu": "a8a9546dbd36bfba2e4ecaebc3faab8684c74a1a81a284f7115095d5b56e7ef7",
    "blackwell_msa_topk.cu": "bbc6c19643db7f9eccefa6af8ba0c80a734bd5b013fd3100e497934f27d80947",
    "blackwell_msa_topk_binding.cu": "be191ba3cc5451ee27d925a20f834ca843032c30a4512a9aa972f28dcfa797af",
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

_NEUTRAL_KERNEL_SYMBOL = re.compile(r"\bkernel_blackwell_msa_[a-z0-9_]+\b")
_DEVICE_KERNEL_DEFINITION = re.compile(
    r"__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+"
    r"(kernel_blackwell_msa_[a-z0-9_]+)\s*\("
)
_DEVICE_SOURCE_INCLUDE = re.compile(r'#include "(blackwell_msa_[a-z0-9_]+\.cu)"')
_CUDA_LAUNCH_SYMBOL = re.compile(
    r"cudaLaunchKernel\s*\(\s*reinterpret_cast<const void\*>\("
    r"(kernel_blackwell_msa_[a-z0-9_]+)\)"
)
_GRID_CONSTANT_TENSOR_MAP = re.compile(
    r"\bconst __grid_constant__ BlackwellMsaTensorMap ([A-Za-z0-9_]+)\b"
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
    return f"blackwell_msa_{variant}.cu"


def _binding_name(variant):
    return f"blackwell_msa_{variant}_binding.cu"


def test_blackwell_msa_frozen_source_inventory_and_sha256():
    expected_device_names = {_device_name(variant) for variant in _VARIANTS}
    expected_binding_names = {_binding_name(variant) for variant in _VARIANTS}
    expected_names = expected_device_names | expected_binding_names
    actual_sources = {path.name: path for path in _BLACKWELL_MSA_CSRC_DIR.glob("*.cu")}

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


def test_blackwell_msa_device_binding_pairs_use_neutral_kernel_symbols():
    for variant in _VARIANTS:
        device_name = _device_name(variant)
        binding_name = _binding_name(variant)
        device_source = (_BLACKWELL_MSA_CSRC_DIR / device_name).read_text(encoding="utf-8")
        binding_source = (_BLACKWELL_MSA_CSRC_DIR / binding_name).read_text(encoding="utf-8")
        expected_symbol = f"kernel_blackwell_msa_{variant}"

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


def test_blackwell_msa_sources_use_cuda_12_8_global_vector_widths():
    illegal_sites = {}
    for variant in _VARIANTS:
        device_name = _device_name(variant)
        device_source = (_BLACKWELL_MSA_CSRC_DIR / device_name).read_text(encoding="utf-8")
        matches = _CUDA_12_8_ILLEGAL_GLOBAL_VECTOR.findall(device_source)
        if matches:
            illegal_sites[device_name] = len(matches)

    assert not illegal_sites, (
        "CUDA 12.8 ptxas limits global b32 vectors to 128 bits; split each "
        f"v8 instruction into two v4 instructions: {illegal_sites}"
    )


def test_blackwell_msa_tma_variants_use_grid_constant_tensor_map_parameters():
    for variant in _VARIANTS:
        if variant == "topk":
            continue

        device_name = _device_name(variant)
        binding_name = _binding_name(variant)
        device_source = (_BLACKWELL_MSA_CSRC_DIR / device_name).read_text(encoding="utf-8")
        binding_source = (_BLACKWELL_MSA_CSRC_DIR / binding_name).read_text(encoding="utf-8")

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
            "static_assert(sizeof(BlackwellMsaTensorMap) == sizeof(CUtensorMap));"
            in binding_source
        )

        combined_source = device_source + binding_source
        forbidden_storage = [
            token for token in _FORBIDDEN_TMA_STORAGE_TOKENS if token in combined_source
        ]
        assert not forbidden_storage, (
            f"{variant} uses forbidden tensor-map storage: {forbidden_storage}"
        )


def test_blackwell_msa_m128_prefill_masks_empty_register_halves():
    for variant in _VARIANTS:
        if not variant.startswith("prefill_m128_"):
            continue

        device_source = (_BLACKWELL_MSA_CSRC_DIR / _device_name(variant)).read_text(
            encoding="utf-8"
        )
        assert device_source.count("body_valid > 0 && body_valid < 64") == 2
        assert device_source.count("valid_cols > 0 && tail_valid < 64") == 2
        assert "tail_valid > 0 && tail_valid < 64" not in device_source


def test_blackwell_msa_m64_prefill_masks_empty_register_halves():
    device_source = (
        _BLACKWELL_MSA_CSRC_DIR / _device_name("prefill_m64_bf16_flat")
    ).read_text(encoding="utf-8")
    assert device_source.count("if (valid_cols > 0 && half_valid < 64)") == 2
    assert "half_valid > 0 && half_valid < 64" not in device_source
