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

"""Host-side MLA KV cache record formats; imports no CuTe DSL."""


# ===========================================================================
# Storage formats and configuration
# ===========================================================================


def cache_record_size(cache_format: str, rope_dim: int) -> int:
    """Elements per token; quantized formats use uint8 elements."""
    return {
        "native": 512 + rope_dim,
        "nvfp4": 288,
        "nvfp4.fp8rope": 352,
        "fp8": 512,
    }[cache_format]


def validate_nvfp4_kv_formats(
    primary_format, extra_format, rope_dim, qkv_dtype, page_size
):
    """Reject unsupported storage/compute pairs before constructing tensor maps.

    Page size one is the internal physical-row view, not an external cache-page
    restriction. Native/native retains the caller's ordinary paged-cache path.
    A mixed pair has FP8 only in the primary SWA pool and NVFP4 in the extra.
    """
    formats = (primary_format, extra_format)
    allowed = (
        ("native", "native"),
        ("nvfp4", "nvfp4"),
        ("nvfp4.fp8rope", "nvfp4.fp8rope"),
        ("fp8", "nvfp4"),
    )
    if formats not in allowed:
        raise ValueError(
            "cache formats must be native/native, nvfp4/nvfp4, nvfp4.fp8rope/nvfp4.fp8rope, or fp8 primary/nvfp4 extra"
        )
    if primary_format == "native":
        return
    if qkv_dtype != "e4m3":
        raise ValueError("NVFP4 KV formats require E4M3 Q")
    if page_size != 1:
        raise ValueError("NVFP4 KV requires an internal page-size-one storage-row view")
    if primary_format == "nvfp4.fp8rope":
        if rope_dim != 64:
            raise ValueError("nvfp4.fp8rope requires 512 NoPE and 64 RoPE channels")
    elif rope_dim != 0:
        raise ValueError("nvfp4/fp8 require D512 QK/V without an extra RoPE tail")
