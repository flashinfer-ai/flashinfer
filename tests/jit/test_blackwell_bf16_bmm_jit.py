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
import json
import re

import pytest

from flashinfer.aot import _gen_blackwell_bf16_bmm_aot_specs
from flashinfer.jit.gemm import blackwell_bf16_bmm


@pytest.mark.parametrize(
    "target,expected_gencode,expected_target_define",
    [
        (
            "sm100a",
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DFLASHINFER_BLACKWELL_BF16_BMM_TARGET_MINOR=0",
        ),
        (
            "sm103a",
            "-gencode=arch=compute_103a,code=sm_103a",
            "-DFLASHINFER_BLACKWELL_BF16_BMM_TARGET_MINOR=3",
        ),
    ],
)
def test_blackwell_bf16_bmm_jit_spec_and_frozen_source(
    target, expected_gencode, expected_target_define
):
    spec = blackwell_bf16_bmm.gen_blackwell_bf16_bmm_module(target)

    assert spec.name == f"blackwell_bf16_bmm_cake_{target}"
    closure = blackwell_bf16_bmm._CAKE_GENERATED_CLOSURES[target]
    source_root = spec.sources[0].parents[1]
    repo_root = source_root.parent
    assert [source.relative_to(source_root).as_posix() for source in spec.sources] == (
        closure["sources"]
    )
    assert spec.sources[0] == repo_root / closure["binding"]
    assert spec.sources[0].name == f"cake_bf16_bmm_binding_{target}.cu"
    assert len(spec.sources) > 1
    assert len(set(spec.sources)) == len(spec.sources)
    assert all(source.is_file() for source in spec.sources)
    assert [
        flag for flag in spec.extra_cuda_cflags if flag.startswith("-gencode=")
    ] == [expected_gencode]
    assert expected_target_define in spec.extra_cuda_cflags
    assert "--use_fast_math" in spec.extra_cuda_cflags
    assert all(flag in spec.extra_cuda_cflags for flag in closure["compile_flags"])

    declarations_path = repo_root / closure["header"]
    declarations = declarations_path.read_text()
    binding_text = spec.sources[0].read_text()
    assert binding_text.count(f'#include "{declarations_path.name}"') == 1

    # Verify the committed closure against every actual source and header;
    # the former single-file snapshot no longer describes the native build.
    identity_inputs = {
        path.relative_to(repo_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in [*spec.sources, declarations_path]
    }
    assert hashlib.sha256(
        json.dumps(identity_inputs, sort_keys=True).encode()
    ).hexdigest() == closure["identity"]

    symbol_pattern = r"\bkernel_cake_bf16_bmm_[0-9a-f]+\b"
    declared_symbols = re.findall(symbol_pattern, declarations)
    assert declared_symbols
    assert len(set(declared_symbols)) == len(declared_symbols)
    defined_symbols = []
    for source in spec.sources[1:]:
        generated_text = source.read_text()
        # The canonical prelude distinguishes NVRTC and native CUDA types and
        # retains compiler-enforced integer/tensor-map ABI size checks.
        assert "static_assert(sizeof(uint64_t) == 8," in generated_text
        assert "static_assert(sizeof(CUtensorMap) == 128," in generated_text
        assert "typedef struct __align__(64)" not in generated_text
        symbols = re.findall(symbol_pattern, generated_text)
        assert len(symbols) == 1
        defined_symbols.extend(symbols)
    assert sorted(defined_symbols) == sorted(declared_symbols)
    assert set(re.findall(symbol_pattern, binding_text)) == set(declared_symbols)


def test_blackwell_bf16_bmm_jit_rejects_unsupported_target():
    with pytest.raises(ValueError, match="unsupported CAKE BF16 BMM target"):
        blackwell_bf16_bmm.gen_blackwell_bf16_bmm_module("sm100f")


@pytest.mark.parametrize(
    "sm_capabilities,expected_names",
    [
        ({"sm100a_exact": True}, ["blackwell_bf16_bmm_cake_sm100a"]),
        ({"sm103a_exact": True}, ["blackwell_bf16_bmm_cake_sm103a"]),
        (
            {"sm100a_exact": True, "sm103a_exact": True},
            [
                "blackwell_bf16_bmm_cake_sm100a",
                "blackwell_bf16_bmm_cake_sm103a",
            ],
        ),
        ({"sm100": True, "sm103": True}, []),
    ],
)
def test_blackwell_bf16_bmm_aot_target_matrix(sm_capabilities, expected_names):
    specs = _gen_blackwell_bf16_bmm_aot_specs(sm_capabilities)
    assert [spec.name for spec in specs] == expected_names
