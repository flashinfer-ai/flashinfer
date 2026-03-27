"""Test that get_artifact() and ensure_symlink() avoid network downloads when
files are already present at the canonical artifact path (e.g. from the
flashinfer-cubin package).

No network access or HTTP mocking required — download_file is patched to
raise on any call, so the tests prove correctness purely via local I/O.
"""

import hashlib
from unittest.mock import patch

from flashinfer.jit.cubin_loader import get_artifact, get_meta_hash, ensure_symlink
from flashinfer.jit.fused_moe import BMM_EXPORT_HEADERS


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download_file_should_not_be_called(*args, **kwargs):
    raise AssertionError(
        f"download_file was called — expected all files to be served locally.\n"
        f"  args={args}"
    )


# ---------------------------------------------------------------------------
# get_artifact — covers all artifact types (FMHA cubins, GEMM cubins, BMM cubins,
# DeepGEMM cubins, cuDNN SDPA cubins, checksums.txt, flashInferMetaInfo.h)
# ---------------------------------------------------------------------------


def test_get_artifact_no_download_when_file_present(monkeypatch, tmp_path):
    """get_artifact() returns local bytes without calling download_file."""
    cubin_dir = tmp_path / "cubins"
    cubin_dir.mkdir()

    import flashinfer.jit.cubin_loader as cl

    monkeypatch.setattr(cl, "FLASHINFER_CUBIN_DIR", cubin_dir)

    content = b"fake cubin data for testing"
    sha = _sha256(content)
    rel_path = "deadbeef123/fmha/trtllm-gen/SomeKernel.cubin"

    local_file = cubin_dir / rel_path
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_bytes(content)

    with patch.object(cl, "download_file", _download_file_should_not_be_called):
        result = get_artifact(rel_path, sha)

    assert result == content


def test_get_artifact_no_download_for_checksums(monkeypatch, tmp_path):
    """checksums.txt is served locally via get_artifact()."""
    cubin_dir = tmp_path / "cubins"
    cubin_dir.mkdir()

    import flashinfer.jit.cubin_loader as cl

    monkeypatch.setattr(cl, "FLASHINFER_CUBIN_DIR", cubin_dir)

    content = b"abc123 SomeKernel.cubin\ndef456 AnotherKernel.cubin\n"
    sha = _sha256(content)
    rel_path = "deadbeef123/fmha/trtllm-gen/checksums.txt"

    local_file = cubin_dir / rel_path
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_bytes(content)

    with patch.object(cl, "download_file", _download_file_should_not_be_called):
        result = get_artifact(rel_path, sha)

    assert result == content


def test_get_artifact_no_download_for_metainfo_header(monkeypatch, tmp_path):
    """flashInferMetaInfo.h is served locally via get_artifact()."""
    cubin_dir = tmp_path / "cubins"
    cubin_dir.mkdir()

    import flashinfer.jit.cubin_loader as cl

    monkeypatch.setattr(cl, "FLASHINFER_CUBIN_DIR", cubin_dir)

    content = b"// auto-generated kernel metadata\nstatic const int x = 1;\n"
    sha = _sha256(content)
    rel_path = "deadbeef123/fmha/trtllm-gen/include/flashInferMetaInfo.h"

    local_file = cubin_dir / rel_path
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_bytes(content)

    with patch.object(cl, "download_file", _download_file_should_not_be_called):
        result = get_artifact(rel_path, sha)

    assert result == content


def test_get_artifact_sha256_mismatch_triggers_download(monkeypatch, tmp_path):
    """get_artifact() must call download_file when the local sha256 doesn't match."""
    cubin_dir = tmp_path / "cubins"
    cubin_dir.mkdir()

    import flashinfer.jit.cubin_loader as cl

    monkeypatch.setattr(cl, "FLASHINFER_CUBIN_DIR", cubin_dir)

    content = b"stale content"
    rel_path = "deadbeef123/fmha/trtllm-gen/SomeKernel.cubin"

    local_file = cubin_dir / rel_path
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_bytes(content)

    wrong_sha = "0" * 64  # doesn't match content
    download_called = False

    def fake_download(*args, **kwargs):
        nonlocal download_called
        download_called = True

    with patch.object(cl, "download_file", fake_download):
        get_artifact(rel_path, wrong_sha)

    assert download_called, "download_file should be called on sha256 mismatch"


# ---------------------------------------------------------------------------
# BMM export headers — get_artifact() + ensure_symlink(), same pattern used by
# gen_trtllm_gen_fused_moe_sm100_module() and gen_moe_utils_module()
# ---------------------------------------------------------------------------


def _build_checksums_and_files(header_files):
    """Create fake header contents and a matching checksums.txt."""
    file_contents = {}
    checksum_lines = []
    for f in header_files:
        content = f"// fake header {f}\n".encode()
        file_contents[f] = content
        checksum_lines.append(f"{_sha256(content)} {f}")
    checksums_txt = "\n".join(checksum_lines).encode()
    return file_contents, checksums_txt


def _populate_artifact_dir(cubin_dir, header_path, header_files, file_contents):
    """Write header files at the canonical artifact path under cubin_dir."""
    for f in header_files:
        dest = cubin_dir / header_path / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(file_contents[f])


def _fetch_headers_and_symlink(cubin_dir, bmm_export_path, checksums_txt):
    """Reproduce the get_artifact + ensure_symlink pattern used by callers."""
    for header in BMM_EXPORT_HEADERS:
        h = get_artifact(
            f"{bmm_export_path}/{header}", get_meta_hash(checksums_txt, header)
        )
        assert h, f"{header} not found"
    ensure_symlink(
        cubin_dir / "flashinfer" / "trtllm" / "batched_gemm" / "trtllmGen_bmm_export",
        cubin_dir / bmm_export_path,
    )


def test_bmm_headers_no_download_when_files_present(monkeypatch, tmp_path):
    """BMM export headers served locally, symlink created correctly."""
    artifact_path = "deadbeef/batched_gemm-abc-def/"
    bmm_export_path = f"{artifact_path}include/trtllmGen_bmm_export"
    cubin_dir = tmp_path / "cubins"
    cubin_dir.mkdir()

    file_contents, checksums_txt = _build_checksums_and_files(BMM_EXPORT_HEADERS)
    _populate_artifact_dir(
        cubin_dir, bmm_export_path, BMM_EXPORT_HEADERS, file_contents
    )

    import flashinfer.jit.cubin_loader as cl

    monkeypatch.setattr(cl, "FLASHINFER_CUBIN_DIR", cubin_dir)

    header_dest_dir = (
        cubin_dir / "flashinfer" / "trtllm" / "batched_gemm" / "trtllmGen_bmm_export"
    )

    with patch.object(cl, "download_file", _download_file_should_not_be_called):
        _fetch_headers_and_symlink(cubin_dir, bmm_export_path, checksums_txt)

    assert header_dest_dir.is_symlink()
    assert header_dest_dir.resolve() == (cubin_dir / bmm_export_path).resolve()

    for f in BMM_EXPORT_HEADERS:
        assert (header_dest_dir / f).exists(), f"Header {f} not accessible via symlink"
        assert (header_dest_dir / f).read_bytes() == file_contents[f]


def test_bmm_headers_stale_directory_replaced(monkeypatch, tmp_path):
    """A pre-existing real directory at symlink path gets replaced."""
    artifact_path = "deadbeef/batched_gemm-abc-def/"
    bmm_export_path = f"{artifact_path}include/trtllmGen_bmm_export"
    cubin_dir = tmp_path / "cubins"
    cubin_dir.mkdir()

    file_contents, checksums_txt = _build_checksums_and_files(BMM_EXPORT_HEADERS)
    _populate_artifact_dir(
        cubin_dir, bmm_export_path, BMM_EXPORT_HEADERS, file_contents
    )

    import flashinfer.jit.cubin_loader as cl

    monkeypatch.setattr(cl, "FLASHINFER_CUBIN_DIR", cubin_dir)

    header_dest_dir = (
        cubin_dir / "flashinfer" / "trtllm" / "batched_gemm" / "trtllmGen_bmm_export"
    )
    header_dest_dir.mkdir(parents=True, exist_ok=True)
    (header_dest_dir / "stale_file.h").write_text("old")

    with patch.object(cl, "download_file", _download_file_should_not_be_called):
        _fetch_headers_and_symlink(cubin_dir, bmm_export_path, checksums_txt)

    assert header_dest_dir.is_symlink()
    assert not (header_dest_dir / "stale_file.h").exists()


def test_bmm_headers_idempotent(monkeypatch, tmp_path):
    """Calling the fetch+symlink pattern twice is a no-op the second time."""
    artifact_path = "deadbeef/batched_gemm-abc-def/"
    bmm_export_path = f"{artifact_path}include/trtllmGen_bmm_export"
    cubin_dir = tmp_path / "cubins"
    cubin_dir.mkdir()

    file_contents, checksums_txt = _build_checksums_and_files(BMM_EXPORT_HEADERS)
    _populate_artifact_dir(
        cubin_dir, bmm_export_path, BMM_EXPORT_HEADERS, file_contents
    )

    import flashinfer.jit.cubin_loader as cl

    monkeypatch.setattr(cl, "FLASHINFER_CUBIN_DIR", cubin_dir)

    with patch.object(cl, "download_file", _download_file_should_not_be_called):
        _fetch_headers_and_symlink(cubin_dir, bmm_export_path, checksums_txt)
        _fetch_headers_and_symlink(cubin_dir, bmm_export_path, checksums_txt)

    header_dest_dir = (
        cubin_dir / "flashinfer" / "trtllm" / "batched_gemm" / "trtllmGen_bmm_export"
    )
    assert header_dest_dir.is_symlink()
