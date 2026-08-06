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
import re

from flashinfer.jit.gemm import blackwell_bf16_bmm


def test_blackwell_bf16_bmm_jit_spec_and_frozen_source():
    spec = blackwell_bf16_bmm.gen_blackwell_bf16_bmm_module()

    assert spec.name == "blackwell_bf16_bmm_cake_sm103a"
    assert [source.name for source in spec.sources] == [
        "blackwell_bf16_bmm.cu",
        "blackwell_bf16_bmm_sm103.cu",
    ]
    assert all(source.is_file() for source in spec.sources)
    assert [
        flag for flag in spec.extra_cuda_cflags if flag.startswith("-gencode=")
    ] == ["-gencode=arch=compute_103a,code=sm_103a"]
    assert "--use_fast_math" in spec.extra_cuda_cflags

    generated_text = spec.sources[1].read_text()
    assert (
        hashlib.sha256(generated_text.encode()).hexdigest()
        == "5b83ea431ce0398d31f73e01d63a91b48a1299bf6d907e15e9cf18324d40284b"
    )
    assert "Source commit: 850c3b728d731c9f201c5dc5aad5d1ee51156f57" in (generated_text)
    assert "typedef unsigned long long uint64_t" not in generated_text
    assert "LoomTensorMap" not in generated_text
    assert "typedef struct __align__(64)" not in generated_text
    assert (
        generated_text.count("#include <flashinfer/gemm/blackwell_bf16_bmm.cuh>") == 1
    )

    repo_root = spec.sources[0].parents[1]
    declarations = (
        repo_root / "include/flashinfer/gemm/blackwell_bf16_bmm.cuh"
    ).read_text()
    symbol_pattern = (
        r"kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_[A-Za-z0-9_]+"
    )
    symbols = re.findall(symbol_pattern, declarations)
    assert len(symbols) == 13
    assert len(set(symbols)) == 13
    for symbol in symbols:
        exact_symbol = rf"\b{re.escape(symbol)}\b"
        assert len(re.findall(exact_symbol, declarations)) == 1
        assert len(re.findall(exact_symbol, generated_text)) == 1
