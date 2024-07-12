"""
Copyright (c) 2024 by FlashInfer team.

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

mask_mode_literal = {
    0: "MaskMode::kNone",
    1: "MaskMode::kCausal",
    2: "MaskMode::kCustom",
}

logits_hook_literal = {
    0: "LogitsPostHook::kNone",
    1: "LogitsPostHook::kSoftCap",
}

warp_layout_literal = {
    0: "WarpLayout::k4x1x2",
    1: "WarpLayout::k4x1x1",
    2: "WarpLayout::k1x4x1",
}

pos_encoding_mode_literal = {
    0: "PosEncodingMode::kNone",
    1: "PosEncodingMode::kRoPELlama",
    2: "PosEncodingMode::kALiBi",
}

dtype_literal = {
    "f16": "half",
    "bf16": "nv_bfloat16",
    "e4m3": "__nv_fp8_e4m3",
    "e5m2": "__nv_fp8_e5m2",
}

idtype_literal = {
    "i32": "int32_t",
    "u32": "uint32_t",
    "i64": "int64_t",
    "u64": "uint64_t",
}

bool_literal = {
    0: "false",
    1: "true",
}
