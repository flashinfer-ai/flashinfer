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

from types import SimpleNamespace

import pytest
from packaging.version import Version

from flashinfer.jit import core as jit_core
from flashinfer.jit import cake_kda_packed_t1
from flashinfer.jit import cake_flash_kda_packed_t1 as flash_kda_packed_t1


_FROZEN_BODY_BEGIN = "// BEGIN FROZEN GENERATED BODY\n"
_FROZEN_BODY_END = "// END FROZEN GENERATED BODY\n"


@pytest.mark.parametrize(
    ("variant", "target", "target_arch", "expected_flag", "target_kind"),
    [
        (
            "tile8",
            "sm100a",
            (10, "0a"),
            "-gencode=arch=compute_100a,code=sm_100a",
            1000,
        ),
        (
            "tile16",
            "sm100a",
            (10, "0a"),
            "-gencode=arch=compute_100a,code=sm_100a",
            1000,
        ),
        (
            "tile8",
            "sm100f",
            (10, "3a"),
            "-gencode=arch=compute_100f,code=sm_100f",
            100,
        ),
        (
            "tile16",
            "sm100f",
            (10, "3a"),
            "-gencode=arch=compute_100f,code=sm_100f",
            100,
        ),
    ],
)
def test_flash_kda_packed_t1_jit_spec_and_frozen_body(
    monkeypatch,
    tmp_path,
    variant,
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
        flash_kda_packed_t1.jit_env,
        "FLASHINFER_GEN_SRC_DIR",
        tmp_path,
    )
    flash_kda_packed_t1.gen_flash_kda_packed_t1_module.cache_clear()

    uri = flash_kda_packed_t1.get_flash_kda_packed_t1_uri(variant, target)
    spec = flash_kda_packed_t1.gen_flash_kda_packed_t1_module(variant, target)

    assert uri == f"flash_kda_packed_t1_{variant}_{target}"
    assert spec.name == uri
    assert spec.sources == [tmp_path / uri / "cake_flashkda_packed_t1_binding.cu"]
    assert spec.sources[0].is_file()
    assert expected_flag in spec.extra_cuda_cflags
    assert (
        f"-DFLASHINFER_FLASH_KDA_PACKED_T1_TARGET_KIND={target_kind}"
        in spec.extra_cuda_cflags
    )
    assert "-use_fast_math" in spec.extra_cuda_cflags
    assert "--maxrregcount=128" in spec.extra_cuda_cflags
    assert sum("-gencode=arch=compute_" in flag for flag in spec.extra_cuda_cflags) == 1
    forbidden_compute = "compute_100f" if target == "sm100a" else "compute_103a"
    assert not any(forbidden_compute in flag for flag in spec.extra_cuda_cflags)

    csrc_dir = flash_kda_packed_t1._get_csrc_dir()
    frozen_text = (csrc_dir / f"cake_flashkda_packed_t1_{variant}.cu").read_text()
    metadata = flash_kda_packed_t1.FLASH_KDA_PACKED_T1_VARIANT_METADATA[variant]
    _, begin_marker, remainder = frozen_text.partition(_FROZEN_BODY_BEGIN)
    frozen_body, end_marker, after_body = remainder.partition(_FROZEN_BODY_END)
    assert begin_marker == _FROZEN_BODY_BEGIN
    assert end_marker == _FROZEN_BODY_END
    assert metadata.symbol in frozen_body
    assert "FlashKDATensorMap" in frozen_body
    assert after_body.strip() == "// clang-format on"

    binding_text = spec.sources[0].read_text()
    assert (
        f'#define FLASHKDA_PACKED_T1_BODY_FILE "cake_flashkda_packed_t1_{variant}.cu"'
        in binding_text
    )
    assert f"#define FLASHKDA_PACKED_T1_KERNEL {metadata.symbol}" in binding_text
    assert (
        f"#define FLASHKDA_PACKED_T1_VALUE_SPLITS {metadata.value_splits}"
        in binding_text
    )
    assert '#include "cake_flashkda_packed_t1_binding.cuh"' in binding_text
    flash_kda_packed_t1.gen_flash_kda_packed_t1_module.cache_clear()


def test_flash_kda_packed_t1_metadata_and_batch_selector():
    assert flash_kda_packed_t1.FLASH_KDA_PACKED_T1_VARIANTS == (
        "tile8",
        "tile16",
    )
    expected_metadata = {
        "tile8": flash_kda_packed_t1.FlashKDAPackedT1VariantMetadata(
            value_splits=16,
            symbol="kernel_kimi_k3_kda_t1_packed",
            batch_min=1,
            batch_max=31,
        ),
        "tile16": flash_kda_packed_t1.FlashKDAPackedT1VariantMetadata(
            value_splits=8,
            symbol="kernel_kimi_k3_kda_t1_packed_tile16",
            batch_min=32,
            batch_max=None,
        ),
    }
    assert expected_metadata == flash_kda_packed_t1.FLASH_KDA_PACKED_T1_VARIANT_METADATA
    for batch in (1, 8, 16, 31):
        assert flash_kda_packed_t1._variant_for_batch(batch) == "tile8"
    for batch in (32, 64, 128, 65535):
        assert flash_kda_packed_t1._variant_for_batch(batch) == "tile16"
    for batch in (0, -1):
        with pytest.raises(ValueError, match="batch must be positive"):
            flash_kda_packed_t1._variant_for_batch(batch)


def test_flash_kda_packed_t1_variant_validation_and_getter(monkeypatch):
    with pytest.raises(ValueError, match="unsupported packed KDA T=1 variant"):
        flash_kda_packed_t1.get_flash_kda_packed_t1_uri("tile32", "sm100a")
    with pytest.raises(ValueError, match="unsupported packed KDA T=1 target"):
        flash_kda_packed_t1.get_flash_kda_packed_t1_uri("tile8", "sm103a")

    sentinel = object()
    monkeypatch.setattr(
        flash_kda_packed_t1,
        "load_flash_kda_packed_t1_module",
        lambda variant, target: (sentinel, variant, target),
    )
    assert flash_kda_packed_t1.get_flash_kda_packed_t1_module("tile16", "sm100f") == (
        sentinel,
        "tile16",
        "sm100f",
    )


def test_flash_kda_packed_t1_binding_contract():
    binding = (
        flash_kda_packed_t1._get_csrc_dir() / "cake_flashkda_packed_t1_binding.cuh"
    ).read_text()

    assert "#include FLASHKDA_PACKED_T1_BODY_FILE" in binding
    assert "kHeads = 12" in binding
    assert "kHeadDim = 128" in binding
    assert "kTargetFamily = 100" in binding
    assert "kTargetSM100a = 1000" in binding
    assert "minor == 0 || minor == 3" in binding
    assert "CHECK_LAST_DIM_CONTIGUOUS(mixed_qkv)" in binding
    assert "CHECK_INPUT_TYPE(mixed_qkv, dl_bfloat16)" in binding
    assert "mixed_qkv must have shape [B," in binding
    assert "state must have compact [H,V,K] blocks" in binding
    assert "state_indices must have shape [B]" in binding
    assert 'CheckNoOverlap(state, "state", out, "output")' in binding
    assert "state_base_mod8" in binding
    assert "state.stride(0)" in binding
    assert "reinterpret_cast<cudaStream_t>(cuda_stream)" in binding
    assert "torch.cuda.current_stream" not in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run" in binding


@pytest.mark.parametrize(
    ("target_archs", "cuda_version", "expected_sm100a", "expected_sm100f"),
    [
        ({(10, "0a")}, "12.8", True, False),
        ({(10, "0a")}, "13.0", False, True),
        ({(10, "3a")}, "12.8", False, False),
        ({(10, "3a")}, "12.9", False, True),
        ({(10, "0a"), (10, "3a")}, "13.0", False, True),
        ({(10, "0f")}, "13.0", False, True),
        ({(10, "3f")}, "13.0", False, True),
        ({(12, "0f")}, "13.0", False, False),
    ],
)
def test_aot_detects_flash_kda_packed_t1_targets(
    monkeypatch,
    target_archs,
    cuda_version,
    expected_sm100a,
    expected_sm100f,
):
    from flashinfer import aot

    class FakeCompilationContext:
        TARGET_CUDA_ARCHS = target_archs

        def get_nvcc_flags_list(self, supported_major_versions=None):
            del supported_major_versions
            return [
                f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
                for major, minor in sorted(self.TARGET_CUDA_ARCHS)
            ]

    monkeypatch.setattr(aot, "CompilationContext", FakeCompilationContext)
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version(cuda_version))

    capabilities = aot.detect_sm_capabilities()
    assert capabilities["flash_kda_packed_t1_sm100a"] is expected_sm100a
    assert capabilities["flash_kda_packed_t1_sm100f"] is expected_sm100f


@pytest.mark.parametrize(
    ("capabilities", "expected_targets"),
    [
        ({"flash_kda_packed_t1_sm100a": True}, ["sm100a"]),
        ({"flash_kda_packed_t1_sm100f": True}, ["sm100f"]),
        (
            {
                "flash_kda_packed_t1_sm100a": True,
                "flash_kda_packed_t1_sm100f": True,
            },
            ["sm100a", "sm100f"],
        ),
        ({"sm103": True}, []),
    ],
)
def test_aot_registers_flash_kda_packed_t1_portfolio(
    monkeypatch, capabilities, expected_targets
):
    from flashinfer import aot

    legacy_calls = []
    cake_calls = []

    def fake_flash_kda_packed_t1(variant, target):
        legacy_calls.append((variant, target))
        return SimpleNamespace(name=f"flash_kda_packed_t1_{variant}_{target}")

    def fake_cake_kda_packed_t1(variant, target):
        cake_calls.append((variant, target))
        return SimpleNamespace(name=f"cake_kda_packed_t1_{variant}_{target}")

    monkeypatch.setattr(
        aot,
        "gen_flash_kda_packed_t1_module",
        fake_flash_kda_packed_t1,
    )
    monkeypatch.setattr(
        aot,
        "gen_cake_kda_packed_t1_module",
        fake_cake_kda_packed_t1,
    )
    monkeypatch.setattr(
        aot, "gen_spdlog_module", lambda: SimpleNamespace(name="spdlog")
    )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(
        aot, "gen_cudnn_fmha_module", lambda: SimpleNamespace(name="cudnn")
    )

    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        capabilities,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )

    expected_legacy_calls = [
        (variant, target)
        for target in expected_targets
        for variant in flash_kda_packed_t1.FLASH_KDA_PACKED_T1_VARIANTS
    ]
    expected_cake_calls = [
        (variant, target)
        for target in expected_targets
        for variant in cake_kda_packed_t1.CAKE_KDA_PACKED_T1_VARIANTS
    ]
    assert legacy_calls == expected_legacy_calls
    assert cake_calls == expected_cake_calls
    assert [spec.name for spec in specs] == [
        "spdlog",
        *[
            name
            for target in expected_targets
            for name in [
                *(
                    f"flash_kda_packed_t1_{variant}_{target}"
                    for variant in flash_kda_packed_t1.FLASH_KDA_PACKED_T1_VARIANTS
                ),
                *(
                    f"cake_kda_packed_t1_{variant}_{target}"
                    for variant in cake_kda_packed_t1.CAKE_KDA_PACKED_T1_VARIANTS
                ),
            ]
        ],
        "cudnn",
    ]
