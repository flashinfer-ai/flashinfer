from flashinfer.artifacts import get_artifacts_status, get_available_cubin_files, get_cubin, get_cubin_file_list

import responses

from urllib.parse import urljoin

def test_sanity_check_urllib_behavior():
    # We use urllib's url joining, just making sure it behaves as we expect.
    # In particular, it should handle slashes by itself.
    base_with_trailing_slash = "https://example.com/some/path/"
    base_without_trailing_slash = "https://example.com/some/path"
    single_segment = "file.txt"
    single_segment_with_leading_slash = "/file.txt"
    multiple_segments = "more/path/file.txt"
    intermediate_segments = "more/path/"

    joined = urljoin(base_with_trailing_slash, single_segment)
    assert joined == "https://example.com/some/path/file.txt"

    joined = urljoin(base_without_trailing_slash, single_segment)
    assert joined == "https://example.com/some/file.txt"

    joined = urljoin(base_with_trailing_slash, single_segment_with_leading_slash)
    assert joined == "https://example.com/file.txt"

    joined = urljoin(base_without_trailing_slash, single_segment_with_leading_slash)
    assert joined == "https://example.com/file.txt"

    joined = urljoin(base_with_trailing_slash, multiple_segments)
    assert joined == "https://example.com/some/path/more/path/file.txt"

    joined = urljoin(urljoin(base_with_trailing_slash, intermediate_segments), single_segment)
    assert joined == "https://example.com/some/path/more/path/file.txt"

success_response = """
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
<a href="Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128u2_s6_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin">Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128u2_s6_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin</a>
            03-Sep-2025 03:44  65.87 KB
<a href="Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256_s2_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin">Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256_s2_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin</a>
            03-Sep-2025 03:44  60.49 KB
<a href="Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin">Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin</a>
            03-Sep-2025 03:44  61.51 KB
<a href="Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256_s4_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_sm100f.cubin">Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256_s4_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_sm100f.cubin</a>
            03-Sep-2025 03:44  72.00 KB
<a href="Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256_s4_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin">Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256_s4_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin</a>
            03-Sep-2025 03:44  64.52 KB
<a href="Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256u2_s2_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin">Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256u2_s2_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin</a>
            03-Sep-2025 03:44  61.73 KB
<a href="Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256u2_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin">Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256u2_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin</a>
            03-Sep-2025 03:44  63.99 KB
<a href="Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256u2_s4_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_sm100f.cubin">Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256u2_s4_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_sm100f.cubin</a>
            03-Sep-2025 03:44  74.70 KB
<a href="Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256u2_s4_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin">Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256u2_s4_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin</a>
            03-Sep-2025 03:44  67.12 KB
<a href="Gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_cga1x1x1_16dp256b_TN_transOut_noShflA_dsFp8_schedS_sm100f.cubin">Gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_cga1x1x1_16dp256b_TN_transOut_noShflA_dsFp8_schedS_sm100f.cubin</a>
            03-Sep-2025 03:44  61.44 KB
<a href="Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_cga1x1x1_16dp256b_TN_transOut_noShflA_dsFp8_schedP2x2x1x3_sm100f.cubin">Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_cga1x1x1_16dp256b_TN_transOut_noShflA_dsFp8_schedP2x2x1x3_sm100f.cubin</a>
            03-Sep-2025 03:44  71.36 KB
<a href="Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_cga1x1x1_16dp256b_TN_transOut_noShflA_dsFp8_schedS_sm100f.cubin">Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_cga1x1x1_16dp256b_TN_transOut_noShflA_dsFp8_schedS_sm100f.cubin</a>
            03-Sep-2025 03:44  64.14 KB
<a href="Gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s6_et64x8_m64x8x32_cga1x1x1_16dp256b_TN_transOut_noShflA_dsFp8_schedP2x2x1x3_sm100f.cubin">Gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s6_et64x8_m64x8x32_cga1x1x1_16dp256b_TN_transOut_noShflA_dsFp8_schedP2x2x1x3_sm100f.cubin</a>
            03-Sep-2025 03:44  73.05 KB
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

# Fake but real-enough looking URL, these tests should not actually try to reach it.
test_cubin_repository = "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-unit-test"

@responses.activate
def test_get_available_cubin_files():
    gemm_path = "037e528e719ec3456a7d7d654f26b805e44c63b1/gemm-8704aa4-f91dc9e/"
    source = urljoin(test_cubin_repository, gemm_path)
    responses.add(responses.GET, source, body=success_response, status=200)
    available_cubin_files = get_available_cubin_files(source, retries=3, delay=0, timeout=5)
    assert len(available_cubin_files) == 16
    
    # Expected cubin files from the mock response
    expected_cubin_files = {
        "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
        "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128_s6_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
        "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128u2_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
        "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128u2_s6_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
        "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256_s2_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
        "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
        "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256_s4_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_sm100f.cubin",
        "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256_s4_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
        "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256u2_s2_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
        "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256u2_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
        "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256u2_s4_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_sm100f.cubin",
        "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x256u2_s4_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
        "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_cga1x1x1_16dp256b_TN_transOut_noShflA_dsFp8_schedS_sm100f.cubin",
        "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_cga1x1x1_16dp256b_TN_transOut_noShflA_dsFp8_schedP2x2x1x3_sm100f.cubin",
        "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_cga1x1x1_16dp256b_TN_transOut_noShflA_dsFp8_schedS_sm100f.cubin",
        "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s6_et64x8_m64x8x32_cga1x1x1_16dp256b_TN_transOut_noShflA_dsFp8_schedP2x2x1x3_sm100f.cubin",
    }
    
    # Check that all expected files are present
    actual_cubin_files = set(available_cubin_files)
    assert actual_cubin_files == expected_cubin_files, f"Expected files: {expected_cubin_files}, but got: {actual_cubin_files}"
    
    # Check that each individual expected file is in the results
    for expected_file in expected_cubin_files:
        assert expected_file in available_cubin_files, f"Expected cubin file '{expected_file}' not found in results"


@responses.activate
def test_get_available_cubin_files_non_200_response():
    """Test that non-200 response codes return an empty tuple."""
    gemm_path = "037e528e719ec3456a7d7d654f26b805e44c63b1/gemm-8704aa4-f91dc9e/"
    source = urljoin(test_cubin_repository, gemm_path)
    
    # Test 404 Not Found
    responses.add(responses.GET, source, status=404)
    available_cubin_files = get_available_cubin_files(source, retries=1, delay=0, timeout=5)
    assert available_cubin_files == ()
    
    # Reset responses and test 500 Internal Server Error
    responses.reset()
    responses.add(responses.GET, source, status=500)
    available_cubin_files = get_available_cubin_files(source, retries=1, delay=0, timeout=5)
    assert available_cubin_files == ()
    
    # Reset responses and test 403 Forbidden
    responses.reset()
    responses.add(responses.GET, source, status=403)
    available_cubin_files = get_available_cubin_files(source, retries=1, delay=0, timeout=5)
    assert available_cubin_files == ()


# def test_get_artifacts_status():
#     pass