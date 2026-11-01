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

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def compilation_context_module():
    # Load the flag-generation module without importing optional GPU backends.
    path = Path(__file__).resolve().parents[2] / "flashinfer/compilation_context.py"
    name = "_test_compilation_context"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def context(monkeypatch, compilation_context_module):
    monkeypatch.setenv(
        "FLASHINFER_CUDA_ARCH_LIST", "8.0 9.0a 10.0a 10.3a 12.0a 12.0f 12.1a"
    )
    monkeypatch.setattr(
        compilation_context_module, "_nvcc_supports_sm107", lambda: True
    )
    monkeypatch.setattr(
        compilation_context_module, "cutlass_supports_sm107", lambda: True
    )
    return compilation_context_module.CompilationContext()


def _gencode_flags(flags):
    return [flag for flag in flags if flag.startswith("-gencode=")]


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        ((10, "0a"), "-gencode=arch=compute_100a,code=sm_100a"),
        ((10, "3a"), "-gencode=arch=compute_103a,code=sm_103a"),
        ((12, "0a"), "-gencode=arch=compute_120a,code=sm_120a"),
        ((12, "0f"), "-gencode=arch=compute_120f,code=sm_120f"),
        ((12, "1a"), "-gencode=arch=compute_121a,code=sm_121a"),
    ],
)
def test_exact_arch_filter_excludes_other_minor_versions_and_suffixes(
    context, arch, expected
):
    # An SM103-only kernel must not also compile for SM100; the same applies
    # to architecture-specific and family-specific targets within SM12x.
    assert _gencode_flags(context.get_nvcc_flags_list(supported_archs=[arch])) == [
        expected
    ]


def test_mixed_major_and_exact_filters(context):
    assert _gencode_flags(
        context.get_nvcc_flags_list(supported_archs=[9, (10, "3a")])
    ) == [
        "-gencode=arch=compute_90a,code=sm_90a",
        "-gencode=arch=compute_103a,code=sm_103a",
    ]


@pytest.mark.parametrize("argument", ["supported_archs", "supported_major_versions"])
def test_major_filter_preserves_all_minor_versions(context, argument):
    assert _gencode_flags(context.get_nvcc_flags_list(**{argument: [10]})) == [
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_103a,code=sm_103a",
    ]


@pytest.mark.parametrize(
    "kwargs", [{}, {"supported_archs": None}, {"supported_major_versions": []}]
)
def test_unfiltered_flags_preserve_targets_and_common_defines(context, kwargs):
    assert context.get_nvcc_flags_list(**kwargs) == [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_103a,code=sm_103a",
        "-gencode=arch=compute_120a,code=sm_120a",
        "-gencode=arch=compute_120f,code=sm_120f",
        "-gencode=arch=compute_121a,code=sm_121a",
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_FP4_E2M1",
    ]


@pytest.mark.parametrize("supported_archs", [[], [11], [(10, "1a")]])
def test_empty_or_unmatched_arch_filter_fails(context, supported_archs):
    with pytest.raises(RuntimeError, match="No supported CUDA architectures"):
        context.get_nvcc_flags_list(supported_archs=supported_archs)


@pytest.mark.parametrize(
    ("major_versions", "supported_archs", "expected"),
    [
        ([10], [9, (10, "3a")], "-gencode=arch=compute_103a,code=sm_103a"),
        ([9], [9, (10, "3a")], "-gencode=arch=compute_90a,code=sm_90a"),
        ([], [(10, "3a")], "-gencode=arch=compute_103a,code=sm_103a"),
    ],
)
def test_major_and_exact_filters_take_intersection(
    context, major_versions, supported_archs, expected
):
    assert _gencode_flags(
        context.get_nvcc_flags_list(major_versions, supported_archs=supported_archs)
    ) == [expected]


def test_disjoint_filters_fail_instead_of_compiling_union(context):
    with pytest.raises(RuntimeError, match="No supported CUDA architectures"):
        context.get_nvcc_flags_list([10], supported_archs=[(12, "1a")])


@pytest.mark.parametrize(
    ("nvcc_support", "cutlass_support", "map_sm107", "expected"),
    [
        (False, False, False, "-gencode=arch=compute_100f,code=sm_100f"),
        (False, True, True, "-gencode=arch=compute_100f,code=sm_100f"),
        (True, False, False, "-gencode=arch=compute_107a,code=sm_107a"),
        (True, False, True, "-gencode=arch=compute_100f,code=sm_100f"),
        (True, True, False, "-gencode=arch=compute_107a,code=sm_107a"),
        (True, True, True, "-gencode=arch=compute_107a,code=sm_107a"),
    ],
)
def test_exact_sm107_filter_preserves_toolchain_fallback(
    monkeypatch,
    compilation_context_module,
    nvcc_support,
    cutlass_support,
    map_sm107,
    expected,
):
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "10.0a 10.7a")
    monkeypatch.setattr(
        compilation_context_module, "_nvcc_supports_sm107", lambda: nvcc_support
    )
    monkeypatch.setattr(
        compilation_context_module, "cutlass_supports_sm107", lambda: cutlass_support
    )
    context = compilation_context_module.CompilationContext()

    assert _gencode_flags(
        context.get_nvcc_flags_list(
            map_sm107_to_100f=map_sm107, supported_archs=[(10, "7a")]
        )
    ) == [expected]


def test_exact_filter_matches_original_arch_before_sm107_mapping(
    monkeypatch, compilation_context_module
):
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "10.7a")
    monkeypatch.setattr(
        compilation_context_module, "_nvcc_supports_sm107", lambda: False
    )
    monkeypatch.setattr(
        compilation_context_module, "cutlass_supports_sm107", lambda: False
    )
    context = compilation_context_module.CompilationContext()

    with pytest.raises(RuntimeError, match="No supported CUDA architectures"):
        context.get_nvcc_flags_list(
            map_sm107_to_100f=True, supported_archs=[(10, "0f")]
        )


def test_existing_positional_major_and_mapping_arguments(
    monkeypatch, compilation_context_module
):
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "9.0a 10.7a")
    monkeypatch.setattr(
        compilation_context_module, "_nvcc_supports_sm107", lambda: True
    )
    monkeypatch.setattr(
        compilation_context_module, "cutlass_supports_sm107", lambda: False
    )
    context = compilation_context_module.CompilationContext()

    assert _gencode_flags(context.get_nvcc_flags_list([10], True)) == [
        "-gencode=arch=compute_100f,code=sm_100f"
    ]
