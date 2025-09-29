from flashinfer.artifacts import (
    ArtifactPath,
    get_available_cubin_files,
    get_cubin_file_list,
)

import responses

from flashinfer.jit.cubin_loader import safe_urljoin


def test_sanity_check_urllib_behavior():
    # We use safe_urljoin which ensures the base is always treated as a directory
    # by adding a trailing slash if needed before calling urljoin.
    base_with_trailing_slash = "https://example.com/some/path/"
    base_without_trailing_slash = "https://example.com/some/path"
    single_segment = "file.txt"
    single_segment_with_leading_slash = "/file.txt"
    multiple_segments = "more/path/file.txt"
    intermediate_segments = "more/path/"

    joined = safe_urljoin(base_with_trailing_slash, single_segment)
    assert joined == "https://example.com/some/path/file.txt"

    # safe_urljoin adds trailing slash, so base is treated as directory
    joined = safe_urljoin(base_without_trailing_slash, single_segment)
    assert joined == "https://example.com/some/path/file.txt"

    joined = safe_urljoin(base_with_trailing_slash, single_segment_with_leading_slash)
    assert joined == "https://example.com/file.txt"

    joined = safe_urljoin(
        base_without_trailing_slash, single_segment_with_leading_slash
    )
    assert joined == "https://example.com/file.txt"

    joined = safe_urljoin(base_with_trailing_slash, multiple_segments)
    assert joined == "https://example.com/some/path/more/path/file.txt"

    joined = safe_urljoin(
        safe_urljoin(base_with_trailing_slash, intermediate_segments), single_segment
    )
    assert joined == "https://example.com/some/path/more/path/file.txt"

    joined = safe_urljoin(intermediate_segments, single_segment)
    assert joined == "more/path/file.txt"


# Fake but real-enough looking URL, these tests should not actually try to reach it.
test_cubin_repository = "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-unit-test"

artifact_paths = ArtifactPath()

success_gemm_response = """
<!DOCTYPE html>
<html>
    <head>
        <meta name="robots" content="noindex"/>
        <title>Index of sw-kernelinferencelibrary-public-generic-local/037e528e719ec3456a7d7d654f26b805e44c63b1/gemm-8704aa4-f91dc9e</title>
    </head>
    <body>
        <h1>Index of sw-kernelinferencelibrary-public-generic-local/037e528e719ec3456a7d7d654f26b805e44c63b1/gemm-8704aa4-f91dc9e</h1>
        <pre>Name                                                                                                                                   Last modified      Size</pre>
        <hr/>
        <pre>
            <a href="../">../</a>
            <a href="include/">include/</a>
            03-Sep-2025 03:44    -
<a href="Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin">Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin</a>
            03-Sep-2025 03:44  60.79 KB
<a href="Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128_s6_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin">Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128_s6_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin</a>
            03-Sep-2025 03:44  63.70 KB
<a href="Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128u2_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin">Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128u2_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin</a>
            03-Sep-2025 03:44  63.08 KB
<a href="LICENSE">LICENSE</a>
            03-Sep-2025 03:44  11.09 KB
<a href="target_path.txt">target_path.txt</a>
            03-Sep-2025 03:44  21 bytes

        </pre>
        <hr/>
        <address style="font-size:small;">Artifactory/7.117.14 Server</address>
    </body>
</html>
"""

# Expected GEMM cubin files from the mock response
expected_gemm_cubin_files = {
    "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
    "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128_s6_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
    "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128u2_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
}

success_fmha_response = """
<!DOCTYPE html>
<html>
    <head>
        <meta name="robots" content="noindex"/>
        <title>Index of sw-kernelinferencelibrary-public-generic-local/037e528e719ec3456a7d7d654f26b805e44c63b1/fmha/trtllm-gen</title>
    </head>
    <body>
        <h1>Index of sw-kernelinferencelibrary-public-generic-local/037e528e719ec3456a7d7d654f26b805e44c63b1/fmha/trtllm-gen</h1>
        <pre>Name                                                                                                                                   Last modified      Size</pre>
        <hr/>
        <pre>
            <a href="../">../</a>
            <a href="include/">include/</a>
            03-Sep-2025 03:45    -
<a href="fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext.cubin">fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext.cubin</a>
            03-Sep-2025 03:45  106.09 KB
<a href="fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128StaticContext.cubin">fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128StaticContext.cubin</a>
            03-Sep-2025 03:45  99.89 KB
<a href="fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext.cubin">fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext.cubin</a>
            03-Sep-2025 03:45  102.89 KB
<a href="LICENSE">LICENSE</a>
            03-Sep-2025 03:45  11.09 KB

        </pre>
        <hr/>
        <address style="font-size:small;">Artifactory/7.117.14 Server</address>
    </body>
</html>

"""

# Expected FMHA cubin files from the mock response
expected_fmha_cubin_files = {
    "fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext.cubin",
    "fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128StaticContext.cubin",
    "fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext.cubin",
}

success_bmm_response = """
<!DOCTYPE html>
<html>
    <head>
        <meta name="robots" content="noindex"/>
        <title>Index of sw-kernelinferencelibrary-public-generic-local/037e528e719ec3456a7d7d654f26b805e44c63b1/batched_gemm-8704aa4-ba3b00d</title>
    </head>
    <body>
        <h1>Index of sw-kernelinferencelibrary-public-generic-local/037e528e719ec3456a7d7d654f26b805e44c63b1/batched_gemm-8704aa4-ba3b00d</h1>
        <pre>Name                                                                                                                                                                                    Last modified      Size</pre>
        <hr/>
        <pre>
            <a href="../">../</a>
            <a href="include/">include/</a>
            03-Sep-2025 03:44    -
<a href="Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_bN_clmp_dynBatch_sm100f.cubin">Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_bN_clmp_dynBatch_sm100f.cubin</a>
            03-Sep-2025 03:44  108.73 KB
<a href="Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedS_bN_clmp_dynBatch_sm100f.cubin">Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedS_bN_clmp_dynBatch_sm100f.cubin</a>
            03-Sep-2025 03:44  89.20 KB
<a href="Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256u2_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_bN_clmp_dynBatch_sm100f.cubin">Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256u2_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_bN_clmp_dynBatch_sm100f.cubin</a>
            03-Sep-2025 03:44  112.02 KB
<a href="LICENSE">LICENSE</a>
            03-Sep-2025 03:44  11.09 KB
<a href="target_path.txt">target_path.txt</a>
            03-Sep-2025 03:44  29 bytes

        </pre>
        <hr/>
        <address style="font-size:small;">Artifactory/7.117.14 Server</address>
    </body>
</html>

"""

# Expected BMM cubin files from the mock response
expected_bmm_cubin_files = {
    "Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_bN_clmp_dynBatch_sm100f.cubin",
    "Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedS_bN_clmp_dynBatch_sm100f.cubin",
    "Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256u2_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_bN_clmp_dynBatch_sm100f.cubin",
}

success_deepgemm_response = """
<!DOCTYPE html>
<html>
    <head>
        <meta name="robots" content="noindex"/>
        <title>Index of sw-kernelinferencelibrary-public-generic-local/51d730202c9eef782f06ecc950005331d85c5d4b/deep-gemm</title>
    </head>
    <body>
        <h1>Index of sw-kernelinferencelibrary-public-generic-local/51d730202c9eef782f06ecc950005331d85c5d4b/deep-gemm</h1>
        <pre>Name                                          Last modified      Size</pre>
        <hr/>
        <pre>
            <a href="../">../</a>
            <a href="kernel.fp8_m_grouped_gemm.007404769193.cubin">kernel.fp8_m_grouped_gemm.007404769193.cubin</a>
            15-Sep-2025 23:32  54.94 KB
<a href="kernel.fp8_m_grouped_gemm.007d9ebdca7e.cubin">kernel.fp8_m_grouped_gemm.007d9ebdca7e.cubin</a>
            15-Sep-2025 23:32  103.99 KB
<a href="kernel.fp8_m_grouped_gemm.02acb2ba71fd.cubin">kernel.fp8_m_grouped_gemm.02acb2ba71fd.cubin</a>
            15-Sep-2025 23:32  256.61 KB
<a href="kernel.fp8_m_grouped_gemm.0457375eb02f.cubin">kernel.fp8_m_grouped_gemm.0457375eb02f.cubin</a>
            15-Sep-2025 23:32  75.47 KB
<a href="kernel_map.json">kernel_map.json</a>
            15-Sep-2025 23:32  107.83 KB
<a href="LICENSE">LICENSE</a>
            15-Sep-2025 23:32  11.09 KB

        </pre>
        <hr/>
        <address style="font-size:small;">Artifactory/7.117.14 Server</address>
    </body>
</html>

"""

# Expected BMM cubin files from the mock response
expected_deepgemm_cubin_files = {
    "kernel.fp8_m_grouped_gemm.007d9ebdca7e.cubin",
    "kernel.fp8_m_grouped_gemm.02acb2ba71fd.cubin",
    "kernel.fp8_m_grouped_gemm.0457375eb02f.cubin",
}


def _mock_file_index_responses():
    gemm_source = safe_urljoin(test_cubin_repository, artifact_paths.TRTLLM_GEN_GEMM)
    responses.add(responses.GET, gemm_source, body=success_gemm_response, status=200)
    fmha_source = safe_urljoin(test_cubin_repository, artifact_paths.TRTLLM_GEN_FMHA)
    responses.add(responses.GET, fmha_source, body=success_fmha_response, status=200)
    bmm_source = safe_urljoin(test_cubin_repository, artifact_paths.TRTLLM_GEN_BMM)
    responses.add(responses.GET, bmm_source, body=success_bmm_response, status=200)
    deepgemm_source = safe_urljoin(test_cubin_repository, artifact_paths.DEEPGEMM)
    responses.add(
        responses.GET, deepgemm_source, body=success_deepgemm_response, status=200
    )


@responses.activate
def test_get_available_cubin_files():
    _mock_file_index_responses()
    source = safe_urljoin(test_cubin_repository, artifact_paths.TRTLLM_GEN_GEMM)
    available_cubin_files = get_available_cubin_files(
        source, retries=3, delay=0, timeout=5
    )
    assert len(available_cubin_files) == 3

    # Check that all expected files are present
    actual_cubin_files = set(available_cubin_files)
    assert actual_cubin_files == expected_gemm_cubin_files, (
        f"Expected files: {expected_gemm_cubin_files}, but got: {actual_cubin_files}"
    )

    # Check that each individual expected file is in the results
    for expected_file in expected_gemm_cubin_files:
        assert expected_file in available_cubin_files, (
            f"Expected cubin file '{expected_file}' not found in results"
        )


@responses.activate
def test_get_available_cubin_files_non_200_response():
    """Test that non-200 response codes return an empty tuple."""
    gemm_path = "037e528e719ec3456a7d7d654f26b805e44c63b1/gemm-8704aa4-f91dc9e/"
    source = safe_urljoin(test_cubin_repository, gemm_path)

    # Test 404 Not Found
    responses.add(responses.GET, source, status=404)
    available_cubin_files = get_available_cubin_files(
        source, retries=1, delay=0, timeout=5
    )
    assert available_cubin_files == ()

    # Reset responses and test 500 Internal Server Error
    responses.reset()
    responses.add(responses.GET, source, status=500)
    available_cubin_files = get_available_cubin_files(
        source, retries=1, delay=0, timeout=5
    )
    assert available_cubin_files == ()

    # Reset responses and test 403 Forbidden
    responses.reset()
    responses.add(responses.GET, source, status=403)
    available_cubin_files = get_available_cubin_files(
        source, retries=1, delay=0, timeout=5
    )
    assert available_cubin_files == ()


@responses.activate
def test_get_cubin_file_list(monkeypatch):
    _mock_file_index_responses()
    from flashinfer import artifacts

    monkeypatch.setattr(
        artifacts, "FLASHINFER_CUBINS_REPOSITORY", test_cubin_repository
    )
    cubin_files = list(get_cubin_file_list())

    # Check that all the cubin's are in there.
    for expected_file_name in expected_gemm_cubin_files:
        expected_file_path = artifact_paths.TRTLLM_GEN_GEMM + "/" + expected_file_name
        assert any(expected_file_path in url for url in cubin_files), (
            f"Expected cubin file '{expected_file_path}' not found in cubin file list"
        )

    for expected_file_name in expected_fmha_cubin_files:
        expected_file_path = artifact_paths.TRTLLM_GEN_FMHA + "/" + expected_file_name
        assert any(expected_file_path in url for url in cubin_files), (
            f"Expected cubin file '{expected_file_path}' not found in cubin file list"
        )

    for expected_file_name in expected_bmm_cubin_files:
        expected_file_path = artifact_paths.TRTLLM_GEN_BMM + "/" + expected_file_name
        assert any(expected_file_path in url for url in cubin_files), (
            f"Expected cubin file '{expected_file_path}' not found in cubin file list"
        )

    for expected_file_name in expected_deepgemm_cubin_files:
        expected_file_path = artifact_paths.DEEPGEMM + "/" + expected_file_name
        assert any(expected_file_path in url for url in cubin_files), (
            f"Expected cubin file '{expected_file_path}' not found in cubin file list"
        )

    # Check that the meta info headers are included (note the inconsistent casing in the actual function)
    # Capitalization is inconsistent in the actual filenames, so we check for both variants.
    meta_info_headers = [
        url
        for url in cubin_files
        if "include/flashInferMetaInfo.h" in url
        or "include/flashinferMetaInfo.h" in url
    ]
    assert len(meta_info_headers) == 3, (
        f"Meta info headers count mismatch. Expected 3, got {len(meta_info_headers)}. Headers found: {meta_info_headers}"
    )
