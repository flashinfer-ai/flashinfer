"""Tests for the trtllm-gen flashinferMetaInfo.h architecture filter.

These are pure text-transform tests: no GPU, no network, no artifact download.
"""

import re

import pytest

from flashinfer.jit.trtllm_gen_metainfo import (
    BLACKWELL_CUBIN_ARCHS,
    RUBIN_CUBIN_ARCHS,
    MetaInfoFilterError,
    filter_metainfo,
)

HEADER = """\
#pragma once
#include <flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/BatchedGemmOptions.h>
namespace batchedGemm {
namespace tensorrt_llm {
namespace kernels {

static constexpr size_t tllmGenBatchedGemmListLen = 4;

static const batchedGemm::BatchedGemmConfig tllmGenBatchedGemmList[] = {
{nullptr, 0, 1, "bmm_a_sm100f", 512, "hash_a", "", nullptr, nullptr, nullptr, 0, { /* mA */ 1
, /* mB */ 2
 }, gemm::SmVersion::Sm100f},
{nullptr, 0, 2, "bmm_b_sm103a", 512, "hash_b", "", nullptr, nullptr, nullptr, 0, { /* mA */ 3
 }, gemm::SmVersion::Sm103a},
{nullptr, 0, 3, "bmm_c_sm107a", 512, "hash_c", "", nullptr, nullptr, nullptr, 0, { /* mA */ 4
 }, gemm::SmVersion::Sm107a},
{nullptr, 0, 4, "bmm_d_sm100a", 512, "hash_d", "", nullptr, nullptr, nullptr, 0, { /* mA */ 5
 }, gemm::SmVersion::Sm100a},
};

} // namespace kernels
} // namespace tensorrt_llm
} // namespace batchedGemm
"""


def _declared_len(source: str) -> int:
    return int(re.search(r"ListLen = (\d+)", source).group(1))


def _archs(source: str) -> list:
    return re.findall(r"gemm::SmVersion::(Sm\w+)\},", source)


def test_blackwell_keeps_only_blackwell_kernels():
    out, kept, dropped = filter_metainfo(HEADER, BLACKWELL_CUBIN_ARCHS)
    assert _archs(out) == ["Sm100f", "Sm103a", "Sm100a"]
    assert (kept, dropped) == (3, 1)
    assert "bmm_c_sm107a" not in out


def test_rubin_keeps_only_rubin_kernels():
    out, kept, dropped = filter_metainfo(HEADER, RUBIN_CUBIN_ARCHS)
    assert _archs(out) == ["Sm107a"]
    assert (kept, dropped) == (1, 3)
    # sm100f is NOT loadable on Rubin for BMM/GEMM (isArchCompatible in
    # csrc/trtllm_batched_gemm_runner.cu), so it must be dropped.
    assert "bmm_a_sm100f" not in out


@pytest.mark.parametrize("archs", [BLACKWELL_CUBIN_ARCHS, RUBIN_CUBIN_ARCHS])
def test_declared_length_matches_emitted_array(archs):
    out, kept, _ = filter_metainfo(HEADER, archs)
    assert _declared_len(out) == kept
    assert out.count("\n{nullptr, 0, ") == kept


def test_variants_partition_the_manifest():
    """Every entry lands in exactly one variant; none is lost or duplicated."""
    bw, n_bw, _ = filter_metainfo(HEADER, BLACKWELL_CUBIN_ARCHS)
    rb, n_rb, _ = filter_metainfo(HEADER, RUBIN_CUBIN_ARCHS)
    assert n_bw + n_rb == _declared_len(HEADER)
    assert set(_archs(bw)).isdisjoint(_archs(rb))


def test_non_entry_lines_are_preserved():
    out, _, _ = filter_metainfo(HEADER, BLACKWELL_CUBIN_ARCHS)
    for line in ("#pragma once", "namespace kernels {", "} // namespace batchedGemm"):
        assert line in out


def test_unknown_arch_selection_fails_loudly():
    with pytest.raises(MetaInfoFilterError, match="kept 0"):
        filter_metainfo(HEADER, ["Sm999z"])


def test_missing_list_declaration_fails_loudly():
    broken = HEADER.replace(
        "static const batchedGemm::BatchedGemmConfig tllmGenBatchedGemmList[] = {",
        "static const batchedGemm::BatchedGemmConfig someOtherName[] = {",
    )
    with pytest.raises(MetaInfoFilterError, match="kernel-list declaration"):
        filter_metainfo(broken, BLACKWELL_CUBIN_ARCHS)


def test_missing_list_length_fails_loudly():
    broken = HEADER.replace(
        "static constexpr size_t tllmGenBatchedGemmListLen = 4;", ""
    )
    with pytest.raises(MetaInfoFilterError, match="ListLen"):
        filter_metainfo(broken, BLACKWELL_CUBIN_ARCHS)


def test_unterminated_entry_fails_loudly():
    broken = HEADER.replace(" }, gemm::SmVersion::Sm100a},", " }, gemm::SmVersion::")
    with pytest.raises(MetaInfoFilterError, match="unterminated"):
        filter_metainfo(broken, BLACKWELL_CUBIN_ARCHS)


def test_gemm_manifest_uses_the_same_shape():
    gemm_header = HEADER.replace("tllmGenBatchedGemmList", "tllmGenGemmList").replace(
        "batchedGemm::BatchedGemmConfig", "gemm::GemmConfig"
    )
    out, kept, _ = filter_metainfo(gemm_header, RUBIN_CUBIN_ARCHS)
    assert kept == 1
    assert _declared_len(out) == 1
