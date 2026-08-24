from flashinfer.artifacts import (
    ArtifactPath,
    get_available_cubin_files,
    get_subdir_file_list,
)

import hashlib

import pytest
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
    # The Rubin pins list the *same* sm100f kernel filenames as their non-Rubin
    # counterparts (built from different sources), so reuse the same directory
    # index bodies. This is what makes bare-filename checksum keys collide.
    bmm_rubin_source = safe_urljoin(
        test_cubin_repository, artifact_paths.TRTLLM_GEN_BMM_RUBIN
    )
    responses.add(
        responses.GET, bmm_rubin_source, body=success_bmm_response, status=200
    )
    gemm_rubin_source = safe_urljoin(
        test_cubin_repository, artifact_paths.TRTLLM_GEN_GEMM_RUBIN
    )
    responses.add(
        responses.GET, gemm_rubin_source, body=success_gemm_response, status=200
    )
    deepgemm_rubin_source = safe_urljoin(
        test_cubin_repository, artifact_paths.DEEPGEMM_RUBIN
    )
    responses.add(
        responses.GET, deepgemm_rubin_source, body=success_deepgemm_response, status=200
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


# Directory index of a cute-dsl DSL_FMHA arch directory: the kernels there are
# TVM-FFI shared objects (.so), not .cubin files (#4432). Mixed with a stray
# .cubin and non-kernel files to check the enumerator keeps both kernel
# extensions and nothing else.
success_dsl_fmha_response = """
<!DOCTYPE html>
<html>
    <head>
        <meta name="robots" content="noindex"/>
        <title>Index of sw-kernelinferencelibrary-public-generic-local/5b34f84266cbc2135066ce96885b664992535670/fmha/cute-dsl/x86_64/sm_103a</title>
    </head>
    <body>
        <h1>Index of sw-kernelinferencelibrary-public-generic-local/5b34f84266cbc2135066ce96885b664992535670/fmha/cute-dsl/x86_64/sm_103a</h1>
        <pre>Name                                          Last modified      Size</pre>
        <hr/>
        <pre>
            <a href="../">../</a>
            <a href="cute_dsl_fmha_bf16_h128_causal_nonpersistent_varlen_lse_pdl_tvmffi.so">cute_dsl_fmha_bf16_h128_causal_nonpersistent_varlen_lse_pdl_tvmffi.so</a>
            03-Sep-2025 03:45  1.2 MB
<a href="cute_dsl_fmha_bf16_h128_causal_nonpersistent_varlen_tvmffi.so">cute_dsl_fmha_bf16_h128_causal_nonpersistent_varlen_tvmffi.so</a>
            03-Sep-2025 03:45  1.2 MB
<a href="some_kernel.cubin">some_kernel.cubin</a>
            03-Sep-2025 03:45  60.79 KB
<a href="checksums.txt">checksums.txt</a>
            03-Sep-2025 03:45  40.12 KB
<a href="LICENSE">LICENSE</a>
            03-Sep-2025 03:45  11.09 KB

        </pre>
        <hr/>
        <address style="font-size:small;">Artifactory/7.117.14 Server</address>
    </body>
</html>
"""

expected_dsl_fmha_kernel_files = {
    "cute_dsl_fmha_bf16_h128_causal_nonpersistent_varlen_lse_pdl_tvmffi.so",
    "cute_dsl_fmha_bf16_h128_causal_nonpersistent_varlen_tvmffi.so",
    "some_kernel.cubin",
}


@responses.activate
def test_get_available_cubin_files_matches_so():
    """Regression for #4432: kernel .so files (cute-dsl) must be enumerated
    alongside .cubin files; non-kernel files must still be excluded."""
    source = safe_urljoin(
        test_cubin_repository,
        safe_urljoin(artifact_paths.DSL_FMHA, "x86_64/sm_103a/"),
    )
    responses.add(responses.GET, source, body=success_dsl_fmha_response, status=200)
    available_files = get_available_cubin_files(source, retries=1, delay=0, timeout=5)
    assert set(available_files) == expected_dsl_fmha_kernel_files


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


def test_get_checksums_unreachable_pin_raises(monkeypatch, tmp_path):
    """An artifact pin whose checksums.txt cannot be fetched must fail loudly.

    Guards the diagnosis path exercised by #4280: a pin added to `cubin_dirs`
    without a published (or, in tests, mocked) manifest used to surface as a bare
    FileNotFoundError on a local cache path, which reads like a corrupt cache
    rather than an unreachable pin. `download_file` is stubbed rather than mocked
    over HTTP so the test does not pay its 4 retries of exponential backoff.
    """
    from flashinfer import artifacts

    monkeypatch.setattr(artifacts, "FLASHINFER_CUBIN_DIR", tmp_path / "cubins")
    monkeypatch.setattr(artifacts, "download_file", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError) as excinfo:
        artifacts.get_checksums([artifact_paths.DEEPGEMM_RUBIN])
    # The pin must be named -- that is the whole point of the error.
    assert artifact_paths.DEEPGEMM_RUBIN in str(excinfo.value)


def test_get_checksums_falls_back_to_cached_manifest(monkeypatch, tmp_path):
    """A failed refresh must not invalidate an already-cached manifest.

    Offline / FLASHINFER_NO_DOWNLOAD setups rely on the on-disk copy.
    """
    from flashinfer import artifacts

    cubin_dir = tmp_path / "cubins"
    monkeypatch.setattr(artifacts, "FLASHINFER_CUBIN_DIR", cubin_dir)
    monkeypatch.setattr(artifacts, "download_file", lambda *args, **kwargs: False)

    manifest_body = "abc123 kernel.fp8_m_grouped_gemm.007d9ebdca7e.cubin\n"
    cached = cubin_dir / safe_urljoin(artifact_paths.DEEPGEMM_RUBIN, "checksums.txt")
    cached.parent.mkdir(parents=True)
    cached.write_text(manifest_body)
    monkeypatch.setitem(
        artifacts.CheckSumHash.map_checksums,
        safe_urljoin(artifact_paths.DEEPGEMM_RUBIN, "checksums.txt"),
        hashlib.sha256(manifest_body.encode()).hexdigest(),
    )

    checksums = artifacts.get_checksums([artifact_paths.DEEPGEMM_RUBIN])
    assert checksums == {
        safe_urljoin(
            artifact_paths.DEEPGEMM_RUBIN,
            "kernel.fp8_m_grouped_gemm.007d9ebdca7e.cubin",
        ): "abc123"
    }


def test_get_checksums_rejects_tampered_manifest(monkeypatch, tmp_path):
    """A manifest that does not match its pinned SHA-256 must not be parsed.

    Its entries become download paths and per-file checksums, so a tampered
    manifest has to be rejected before parsing, not discovered afterwards.
    """
    from flashinfer import artifacts

    cubin_dir = tmp_path / "cubins"
    monkeypatch.setattr(artifacts, "FLASHINFER_CUBIN_DIR", cubin_dir)
    monkeypatch.setattr(artifacts, "download_file", lambda *args, **kwargs: False)

    cached = cubin_dir / safe_urljoin(artifact_paths.DEEPGEMM_RUBIN, "checksums.txt")
    cached.parent.mkdir(parents=True)
    cached.write_text("abc123 kernel.fp8_m_grouped_gemm.007d9ebdca7e.cubin\n")
    monkeypatch.setitem(
        artifacts.CheckSumHash.map_checksums,
        safe_urljoin(artifact_paths.DEEPGEMM_RUBIN, "checksums.txt"),
        "0" * 64,
    )

    with pytest.raises(RuntimeError) as excinfo:
        artifacts.get_checksums([artifact_paths.DEEPGEMM_RUBIN])
    assert "pinned SHA-256" in str(excinfo.value)


def test_get_checksums_rejects_traversal_filenames(monkeypatch, tmp_path):
    """Manifest entries are joined onto FLASHINFER_CUBIN_DIR; absolute paths
    and ``..`` segments must be rejected so a manifest can never direct a
    write outside the cubin cache."""
    from flashinfer import artifacts

    cubin_dir = tmp_path / "cubins"
    monkeypatch.setattr(artifacts, "FLASHINFER_CUBIN_DIR", cubin_dir)
    monkeypatch.setattr(artifacts, "download_file", lambda *args, **kwargs: False)

    for bad_name in ("../../outside.so", "/etc/evil.so", "a\\..\\b.cubin"):
        manifest_body = f"abc123 {bad_name}\n"
        cached = cubin_dir / safe_urljoin(
            artifact_paths.DEEPGEMM_RUBIN, "checksums.txt"
        )
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(manifest_body)
        monkeypatch.setitem(
            artifacts.CheckSumHash.map_checksums,
            safe_urljoin(artifact_paths.DEEPGEMM_RUBIN, "checksums.txt"),
            hashlib.sha256(manifest_body.encode()).hexdigest(),
        )

        with pytest.raises(RuntimeError) as excinfo:
            artifacts.get_checksums([artifact_paths.DEEPGEMM_RUBIN])
        assert "Unsafe filename" in str(excinfo.value)


@responses.activate
def test_get_subdir_file_list(monkeypatch, tmp_path):
    _mock_file_index_responses()
    from flashinfer import artifacts

    # Set up temporary directory for downloading checksums
    temp_cubin_dir = tmp_path / "cubins"
    temp_cubin_dir.mkdir(exist_ok=True)

    monkeypatch.setattr(
        artifacts, "FLASHINFER_CUBINS_REPOSITORY", test_cubin_repository
    )
    monkeypatch.setattr(artifacts, "FLASHINFER_CUBIN_DIR", temp_cubin_dir)

    # Mock checksums.txt files for each subdirectory
    checksums_fmha = """d26dbf837f40ff2dcd964094ab6e1b3f2424edda5979c313f5262655161fce98 include/flashInferMetaInfo.h
a1b2c3d4e5f6 fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext.cubin
b2c3d4e5f6a7 fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128StaticContext.cubin
c3d4e5f6a7b8 fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext.cubin
"""

    checksums_gemm = """bd5c3227bec4f8d7a7d3a27fd7628e010d99a5c42651d0a6b97e146803e63340 include/flashinferMetaInfo.h
d1e2f3a4b5c6 Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin
e2f3a4b5c6d7 Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128_s6_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin
f3a4b5c6d7e8 Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128u2_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin
"""

    checksums_bmm = """4a8ceeb356fc5339021acf884061e97e49e01da5c75dbf0f7cf4932c37a70152 include/flashinferMetaInfo.h
a4b5c6d7e8f9 Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_bN_clmp_dynBatch_sm100f.cubin
b5c6d7e8f9a0 Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedS_bN_clmp_dynBatch_sm100f.cubin
c6d7e8f9a0b1 Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256u2_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_bN_clmp_dynBatch_sm100f.cubin
"""

    # Rubin pins: identical kernel filenames to the non-Rubin pins above, but
    # every hash differs (they are built from different sources). Keying the
    # checksum map by bare filename would let whichever pin is processed last
    # overwrite the other's hashes.
    checksums_bmm_rubin = """1111111111111111111111111111111111111111111111111111111111111111 include/flashinferMetaInfo.h
aaaa111122223333 Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_bN_clmp_dynBatch_sm100f.cubin
bbbb111122223333 Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedS_bN_clmp_dynBatch_sm100f.cubin
cccc111122223333 Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256u2_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedP2x1x2x3_bN_clmp_dynBatch_sm100f.cubin
"""

    checksums_gemm_rubin = """2222222222222222222222222222222222222222222222222222222222222222 include/flashinferMetaInfo.h
dddd111122223333 Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin
eeee111122223333 Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128_s6_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin
ffff111122223333 Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128u2_s3_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin
"""

    checksums_deepgemm = """b4374f857c3066089c4ec6b5e79e785559fa2c05ce2623710b0b04bf86414a48 kernel_map.json
a0b1c2d3e4f5 kernel.fp8_m_grouped_gemm.007404769193.cubin
d7e8f9a0b1c2 kernel.fp8_m_grouped_gemm.007d9ebdca7e.cubin
e8f9a0b1c2d3 kernel.fp8_m_grouped_gemm.02acb2ba71fd.cubin
f9a0b1c2d3e4 kernel.fp8_m_grouped_gemm.0457375eb02f.cubin
"""

    checksums_deepgemm_rubin = """3333333333333333333333333333333333333333333333333333333333333333 kernel_map.json
1111aaaabbbbcccc kernel.fp8_m_grouped_gemm.007404769193.cubin
2222aaaabbbbcccc kernel.fp8_m_grouped_gemm.007d9ebdca7e.cubin
3333aaaabbbbcccc kernel.fp8_m_grouped_gemm.02acb2ba71fd.cubin
4444aaaabbbbcccc kernel.fp8_m_grouped_gemm.0457375eb02f.cubin
"""

    # Add mock responses for checksums.txt files
    fmha_checksums_url = safe_urljoin(
        test_cubin_repository,
        safe_urljoin(artifact_paths.TRTLLM_GEN_FMHA, "checksums.txt"),
    )
    responses.add(responses.GET, fmha_checksums_url, body=checksums_fmha, status=200)

    gemm_checksums_url = safe_urljoin(
        test_cubin_repository,
        safe_urljoin(artifact_paths.TRTLLM_GEN_GEMM, "checksums.txt"),
    )
    responses.add(responses.GET, gemm_checksums_url, body=checksums_gemm, status=200)

    bmm_checksums_url = safe_urljoin(
        test_cubin_repository,
        safe_urljoin(artifact_paths.TRTLLM_GEN_BMM, "checksums.txt"),
    )
    responses.add(responses.GET, bmm_checksums_url, body=checksums_bmm, status=200)

    bmm_rubin_checksums_url = safe_urljoin(
        test_cubin_repository,
        safe_urljoin(artifact_paths.TRTLLM_GEN_BMM_RUBIN, "checksums.txt"),
    )
    responses.add(
        responses.GET, bmm_rubin_checksums_url, body=checksums_bmm_rubin, status=200
    )

    gemm_rubin_checksums_url = safe_urljoin(
        test_cubin_repository,
        safe_urljoin(artifact_paths.TRTLLM_GEN_GEMM_RUBIN, "checksums.txt"),
    )
    responses.add(
        responses.GET, gemm_rubin_checksums_url, body=checksums_gemm_rubin, status=200
    )

    deepgemm_checksums_url = safe_urljoin(
        test_cubin_repository, safe_urljoin(artifact_paths.DEEPGEMM, "checksums.txt")
    )
    responses.add(
        responses.GET, deepgemm_checksums_url, body=checksums_deepgemm, status=200
    )

    deepgemm_rubin_checksums_url = safe_urljoin(
        test_cubin_repository,
        safe_urljoin(artifact_paths.DEEPGEMM_RUBIN, "checksums.txt"),
    )
    responses.add(
        responses.GET,
        deepgemm_rubin_checksums_url,
        body=checksums_deepgemm_rubin,
        status=200,
    )

    # Mock DSL_FMHA checksums + directory index for the host cpu_arch.
    # Pin to x86_64 so the test is deterministic regardless of the runner arch.
    # The cute-dsl kernels ship as .so, not .cubin (#4432).
    monkeypatch.setattr(artifacts, "_get_host_cpu_arch", lambda: "x86_64")
    checksums_dsl_fmha = """aabbccdd11223344 cute_dsl_fmha_bf16_h128_causal_nonpersistent_varlen_tvmffi.so
bbccddee22334455 cute_dsl_fmha_bf16_h128_causal_nonpersistent_varlen_lse_pdl_tvmffi.so
"""
    # Minimal directory index: an empty HTML page with no cubin/header hrefs.
    # Enumeration is driven by the checksums.txt manifest, so the index body
    # must not matter; this one is registered only so a regression back to
    # HTML scraping fails fast (empty listing) instead of retrying 404s.
    empty_dir_index = '<html><body><pre><a href="../">../</a></pre></body></html>'
    for sm_arch in artifact_paths.DSL_FMHA_ARCHS:
        subdir = safe_urljoin(artifact_paths.DSL_FMHA, f"x86_64/{sm_arch}/")
        responses.add(
            responses.GET,
            safe_urljoin(test_cubin_repository, safe_urljoin(subdir, "checksums.txt")),
            body=checksums_dsl_fmha,
            status=200,
        )
        responses.add(
            responses.GET,
            safe_urljoin(test_cubin_repository, subdir),
            body=empty_dir_index,
            status=200,
        )

    # get_checksums() refuses to parse a manifest that does not match its
    # pinned SHA-256, so pin every mocked manifest body for this test.
    mocked_manifests = {
        artifact_paths.TRTLLM_GEN_FMHA: checksums_fmha,
        artifact_paths.TRTLLM_GEN_GEMM: checksums_gemm,
        artifact_paths.TRTLLM_GEN_BMM: checksums_bmm,
        artifact_paths.TRTLLM_GEN_BMM_RUBIN: checksums_bmm_rubin,
        artifact_paths.TRTLLM_GEN_GEMM_RUBIN: checksums_gemm_rubin,
        artifact_paths.DEEPGEMM: checksums_deepgemm,
        artifact_paths.DEEPGEMM_RUBIN: checksums_deepgemm_rubin,
        **{
            safe_urljoin(
                artifact_paths.DSL_FMHA, f"x86_64/{sm_arch}/"
            ): checksums_dsl_fmha
            for sm_arch in artifact_paths.DSL_FMHA_ARCHS
        },
    }
    for manifest_subdir, manifest_body in mocked_manifests.items():
        monkeypatch.setitem(
            artifacts.CheckSumHash.map_checksums,
            safe_urljoin(manifest_subdir, "checksums.txt"),
            hashlib.sha256(manifest_body.encode()).hexdigest(),
        )

    cubin_files = list(get_subdir_file_list())

    # Extract just the file paths from the (path, checksum) tuples
    cubin_file_paths = [path for path, _ in cubin_files]

    # Check that all the cubin's are in there.
    for expected_file_name in expected_gemm_cubin_files:
        expected_file_path = safe_urljoin(
            artifact_paths.TRTLLM_GEN_GEMM, expected_file_name
        )
        assert any(expected_file_path in url for url in cubin_file_paths), (
            f"Expected cubin file '{expected_file_path}' not found in cubin file list"
        )

    for expected_file_name in expected_fmha_cubin_files:
        expected_file_path = safe_urljoin(
            artifact_paths.TRTLLM_GEN_FMHA, expected_file_name
        )
        assert any(expected_file_path in url for url in cubin_file_paths), (
            f"Expected cubin file '{expected_file_path}' not found in cubin file list"
        )

    for expected_file_name in expected_bmm_cubin_files:
        expected_file_path = safe_urljoin(
            artifact_paths.TRTLLM_GEN_BMM, expected_file_name
        )
        assert any(expected_file_path in url for url in cubin_file_paths), (
            f"Expected cubin file '{expected_file_path}' not found in cubin file list"
        )

    for expected_file_name in expected_deepgemm_cubin_files:
        expected_file_path = safe_urljoin(artifact_paths.DEEPGEMM, expected_file_name)
        assert any(expected_file_path in url for url in cubin_file_paths), (
            f"Expected cubin file '{expected_file_path}' not found in cubin file list"
        )

    # Check that the meta info headers are included (note the inconsistent casing in the actual function)
    # Capitalization is inconsistent in the actual filenames, so we check for both variants.
    meta_info_headers = [
        url
        for url in cubin_file_paths
        if "include/flashInferMetaInfo.h" in url
        or "include/flashinferMetaInfo.h" in url
    ]
    # FMHA, GEMM, BMM, GEMM_RUBIN, BMM_RUBIN.
    assert len(meta_info_headers) == 5, (
        f"Meta info headers count mismatch. Expected 5, got {len(meta_info_headers)}. Headers found: {meta_info_headers}"
    )

    # Regression: per-arch pins share kernel filenames but not hashes, so each
    # entry must carry the checksum from its own pin. Keying the checksum map
    # by bare filename let the pin processed last overwrite the earlier one,
    # which failed verification for every shared name.
    by_path = dict(cubin_files)
    assert len(by_path) == len(cubin_files), "duplicate paths in cubin file list"

    for shared_name, plain_dir, rubin_dir in (
        (
            "Bmm_Bfloat16_E2m1E2m1_Fp32_t128x16x256_s6_et128x16_m128x16x64_cga1x1x1_16dp256b_TN_transOut_schedS_bN_clmp_dynBatch_sm100f.cubin",
            artifact_paths.TRTLLM_GEN_BMM,
            artifact_paths.TRTLLM_GEN_BMM_RUBIN,
        ),
        (
            "Gemm_Bfloat16_E2m1E2m1_Fp32_t128x128x128_s6_et128x128_m128x128x64_cga1x1x1_16dp256b_TN_transOut_schedS_sm100f.cubin",
            artifact_paths.TRTLLM_GEN_GEMM,
            artifact_paths.TRTLLM_GEN_GEMM_RUBIN,
        ),
        (
            "kernel.fp8_m_grouped_gemm.007d9ebdca7e.cubin",
            artifact_paths.DEEPGEMM,
            artifact_paths.DEEPGEMM_RUBIN,
        ),
    ):
        plain_path = safe_urljoin(plain_dir, shared_name)
        rubin_path = safe_urljoin(rubin_dir, shared_name)
        assert plain_path in by_path, f"{plain_path} missing from cubin file list"
        assert rubin_path in by_path, f"{rubin_path} missing from cubin file list"
        assert by_path[plain_path] != by_path[rubin_path], (
            f"{shared_name} resolved to the same checksum for both pins "
            f"({by_path[plain_path]}) -- the per-pin hashes collided"
        )

    # Regression for #4432: the cute-dsl FMHA kernels are .so files, which the
    # old HTML-scraping enumerator (cubin/header regexes only) silently
    # skipped, so download_artifacts() reported success while every DSL kernel
    # was missing from the cache. Every manifest-listed .so must be enumerated
    # for every arch, carrying the checksum from its own manifest.
    for sm_arch in artifact_paths.DSL_FMHA_ARCHS:
        subdir = safe_urljoin(artifact_paths.DSL_FMHA, f"x86_64/{sm_arch}/")
        for so_name, so_sha in (
            (
                "cute_dsl_fmha_bf16_h128_causal_nonpersistent_varlen_tvmffi.so",
                "aabbccdd11223344",
            ),
            (
                "cute_dsl_fmha_bf16_h128_causal_nonpersistent_varlen_lse_pdl_tvmffi.so",
                "bbccddee22334455",
            ),
        ):
            so_path = safe_urljoin(subdir, so_name)
            assert so_path in by_path, (
                f"DSL FMHA kernel '{so_path}' not enumerated -- .so artifacts "
                f"would be silently skipped by download_artifacts() (#4432)"
            )
            assert by_path[so_path] == so_sha

    # Mixed-content directory: enumeration is manifest-driven, so files with
    # extensions the old scraper never anticipated (e.g. deepgemm's
    # kernel_map.json) must be enumerated too, not only .cubin/.h files.
    kernel_map_path = safe_urljoin(artifact_paths.DEEPGEMM, "kernel_map.json")
    assert kernel_map_path in by_path

    # Every entry of every manifest must be enumerated, so a file that is
    # listed but missing on the server now fails download_artifacts() loudly
    # instead of being silently skipped.
    manifest_entries = artifacts.get_checksums(
        [
            artifact_paths.TRTLLM_GEN_FMHA,
            artifact_paths.TRTLLM_GEN_BMM,
            artifact_paths.TRTLLM_GEN_GEMM,
            artifact_paths.TRTLLM_GEN_BMM_RUBIN,
            artifact_paths.TRTLLM_GEN_GEMM_RUBIN,
            artifact_paths.DEEPGEMM,
            artifact_paths.DEEPGEMM_RUBIN,
        ]
        + [
            safe_urljoin(artifact_paths.DSL_FMHA, f"x86_64/{sm_arch}/")
            for sm_arch in artifact_paths.DSL_FMHA_ARCHS
        ]
    )
    missing = set(manifest_entries) - set(by_path)
    assert not missing, f"manifest entries not enumerated for download: {missing}"
