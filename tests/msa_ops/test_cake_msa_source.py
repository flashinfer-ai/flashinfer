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

# Freeze the complete public source inventory, including the device body and
# its host binding for every variant.  Keep these digests out of the
# public CUDA files so they do not expose generator provenance.
_SOURCE_SHA256 = {
    "cake_msa_decode_bf16_flat.cu": "f9e22803e3ab8e30c6001b92e8b7b7fedbbe78f2286f1c029285d4ae3a6f39d3",
    "cake_msa_decode_bf16_flat_binding.cu": "66ff0761fd9c54e34ef5d2fcb88e0428f1395441a4c0b88a57f9555183a0980d",
    "cake_msa_decode_bf16_paged.cu": "066ee8a226a87da2ea9cb274c84fc577b62d1ff339a8fc0ec299fbbb44e58cce",
    "cake_msa_decode_bf16_paged_binding.cu": "ebfab54b367e5d7d2430d254c6c9b4f225680fad6f7379472b1b31d067710228",
    "cake_msa_decode_fp16_flat.cu": "230adad70c29a7732584d77c4a79b1e5503e5fe34edd58573a6afcfc7e48dea7",
    "cake_msa_decode_fp16_flat_binding.cu": "b26fcf5c289e3088ad2c2488f9b89cff64860bd7d3f6ddeec7a21cb582579b94",
    "cake_msa_decode_fp16_paged.cu": "0379f8c4f868aeb5a2f13ed5d60b2a17e375109e292845aabaeb5bd55902c5c6",
    "cake_msa_decode_fp16_paged_binding.cu": "3c7056eaed79f61575bb5c936f1711f8ff41c8ac0dfe4586533ef9602a9a0105",
    "cake_msa_decode_fp8_flat.cu": "62befbfce102eafc3a3083927885f3f17d1b29ffa6f73c337f59f0725ecfd39c",
    "cake_msa_decode_fp8_flat_binding.cu": "65050ee6c1fa930ca356e7f4bf0221045c0d9adc2070a1cbdf14069de8f2fae2",
    "cake_msa_decode_fp8_paged.cu": "4cffc3198ee296d6921afee56914062bff42ec15f74386962d26367b84b9362b",
    "cake_msa_decode_fp8_paged_binding.cu": "422f65caa6415a18cebcb2e8882600b4b04ffdac9603eec745e9113ed4d37f5b",
    "cake_msa_decode_m16_bf16_flat.cu": "52591e583b89051bdfaa3add386d94a44051359846a8299713beb59bf7f450af",
    "cake_msa_decode_m16_bf16_flat_binding.cu": "8b83f001560ce21801efa4679423bdaa86aa0af0b2105c87a8cb34fb2c916c67",
    "cake_msa_decode_m16_bf16_paged.cu": "9b0a1d2a68e1747c68dffbc20e88ab1ef0be1146334d54ba23cc6fd286e94000",
    "cake_msa_decode_m16_bf16_paged_binding.cu": "58540beab6bce0f4957775152b169e013cb9636d3114666baa1734620ae493e6",
    "cake_msa_prefill_m128_bf16_flat.cu": "78a27286684792d3795ce268516cb53d91d24c2b9f4a0dc569436922d0b6d77c",
    "cake_msa_prefill_m128_bf16_flat_binding.cu": "87009e7f6219d732434d20e11166638b86df1dea505ccb0cae6941c9ae021777",
    "cake_msa_prefill_m128_bf16_gqa16_flat.cu": "5c26070b6f45699ccdae322c9c8937d7fc924741c039a15a1554c48bfec90cd5",
    "cake_msa_prefill_m128_bf16_gqa16_flat_binding.cu": "8a622e4ec003111077e21dffff82de77a272b617f05e4430c9135c53100a4987",
    "cake_msa_prefill_m128_bf16_gqa16_paged.cu": "6c45de021c993f6658a4d3241e8e9dd91a52f9b227fa88825acf7a0ab862c453",
    "cake_msa_prefill_m128_bf16_gqa16_paged_binding.cu": "78516fba73cacff8a1e528c5bf5fb481dbc3524f21077422b122a8b54c3d3d5f",
    "cake_msa_prefill_m128_bf16_paged.cu": "eb980b7428872f67b90497a12b2b680591b1d9ac83870c705162c658b90473b0",
    "cake_msa_prefill_m128_bf16_paged_binding.cu": "b3041e30fc992ff6a335e724ece8a1d5185bc2b085c03dd1660cbabe53536c8f",
    "cake_msa_prefill_m128_fp16_flat.cu": "f27c95b260c88a40a803704aa205965337ed91e1fb917653353d2268122e024b",
    "cake_msa_prefill_m128_fp16_flat_binding.cu": "d8817ad9ebe03bf28afe8ddc508ba07ce0f74cb1fbcb80bae1d3a76e8f8c34b1",
    "cake_msa_prefill_m128_fp16_paged.cu": "b627f2696ae310b9c7e39b543c02807fc1ca3f747aae42809a2b8f90fb048669",
    "cake_msa_prefill_m128_fp16_paged_binding.cu": "516777081d61538f4bca42d258bdeebe98b07f34731fae49dd58695333c7bf51",
    "cake_msa_prefill_m128_fp8_flat.cu": "9c3496faaf123ddf1a317a08e9dbdebb5a00d29574d31bf0bef27fb7c4b6d97a",
    "cake_msa_prefill_m128_fp8_flat_binding.cu": "47454f5a8f8a00c6d0b12e730297b39d86fa7e722624292e3967d816ec2223f4",
    "cake_msa_prefill_m128_fp8_paged.cu": "96d6d70973fb90fe33bc6babe464d0fc10ba257504b2bff584b5048929ec5917",
    "cake_msa_prefill_m128_fp8_paged_binding.cu": "c82066a290e3a070d519f2ced75ffea541b22130d579bae377d07ff5162ca32d",
    "cake_msa_prefill_m64_bf16_flat.cu": "f818ec3a96eaa96e3483b3aa3369e57fafc9f2a267635e61c06ef1b4cd7d4bc7",
    "cake_msa_prefill_m64_bf16_flat_binding.cu": "72e56d4eb4411c871da6ca86ae860ad1f97e6785ecbea6b036f861288ebc3f87",
    "cake_msa_topk.cu": "4b4c6a2d6134980e7f6568ea1eaf07521eafddd9ed65f641097660a89f115043",
    "cake_msa_topk_binding.cu": "55e62a2f97c7645a08203814d8d283005677f1501acc451930aa4f063afbf640",
}

_FORBIDDEN_BINDING_APIS = ("EmbedCubin", "kernel.Launch", "GetKernel")

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
