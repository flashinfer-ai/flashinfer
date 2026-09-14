"""Host tests for SM120 SVDQuant JIT specialization and generated sources."""

from pathlib import Path

import pytest

from flashinfer.jit import core
from flashinfer.jit.gemm import svdquant_sm120
from flashinfer.jit.gemm.svdquant_sm120_configs import SVDQUANT_SM120_CONFIGS


@pytest.fixture(autouse=True)
def isolated_jit_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(svdquant_sm120.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    monkeypatch.setattr(
        svdquant_sm120.jit_env,
        "FLASHINFER_CSRC_DIR",
        Path(__file__).resolve().parents[2] / "csrc",
    )
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "jit_spec_registry", core.JitSpecRegistry())
    monkeypatch.setattr(
        svdquant_sm120.current_compilation_context,
        "get_nvcc_flags_list",
        lambda supported_major_versions: ["-gencode=arch=compute_120f,code=sm_120f"],
    )


@pytest.mark.parametrize(
    "rank,name,rank_flags",
    [
        (32, "nvfp4_svdquant_gemm_cutlass_sm120", []),
        (
            64,
            "nvfp4_svdquant_gemm_cutlass_sm120_rank64",
            ["-DSVDQ_SM120_LORA_RANK=64"],
        ),
    ],
)
def test_lora_rank_specializes_module_identity(
    rank: int, name: str, rank_flags: list[str]
) -> None:
    spec = svdquant_sm120.gen_gemm_sm120_module_cutlass_nvfp4_svdquant(lora_rank=rank)

    assert isinstance(spec, core.JitSpecNvcc)
    assert spec.name == name
    assert spec.extra_cuda_cflags is not None
    assert [
        flag
        for flag in spec.extra_cuda_cflags
        if flag.startswith("-DSVDQ_SM120_LORA_RANK=")
    ] == rank_flags


@pytest.mark.parametrize("rank", [-32, 0, 1, 31, 33, 48])
def test_invalid_rank_is_rejected_before_writing_sources(
    rank: int, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="positive multiple of 32"):
        svdquant_sm120.gen_gemm_sm120_module_cutlass_nvfp4_svdquant(lora_rank=rank)

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("rank", [32, 64])
def test_module_generates_every_tactic_translation_unit(rank: int) -> None:
    spec = svdquant_sm120.gen_gemm_sm120_module_cutlass_nvfp4_svdquant(lora_rank=rank)

    assert isinstance(spec, core.JitSpecNvcc)
    assert [source.name for source in spec.sources[:2]] == [
        "nvfp4_svdquant_gemm_cutlass_sm120.cu",
        "nvfp4_smooth_quantize_sm100.cu",
    ]
    assert len(spec.sources) == 33
    assert all(source.is_file() for source in spec.sources)
    for source, config in zip(spec.sources[2:], SVDQUANT_SM120_CONFIGS, strict=True):
        assert source.name == f"nvfp4_svdquant_gemm_cutlass_sm120_{config}.cu"
        assert f"({config})" in source.read_text(encoding="utf-8")


def test_module_links_cublaslt() -> None:
    spec = svdquant_sm120.gen_gemm_sm120_module_cutlass_nvfp4_svdquant()

    assert isinstance(spec, core.JitSpecNvcc)
    assert spec.extra_ldflags == ["-lcublasLt"]
