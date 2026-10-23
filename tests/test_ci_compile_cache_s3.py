"""Tests for ci/compile_cache_s3.sh, the shared S3 layer of the CI compile cache.

The S3 bucket is stood in for by a local HTTP server (anonymous reads, with
ETags) and a fake ``aws`` CLI on PATH that writes uploads into the server's
directory.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci" / "compile_cache_s3.sh"
IMAGE = "flashinfer/flashinfer-ci-cu129:20260930-b1420f1"
IMAGE_KEY = "flashinfer_flashinfer-ci-cu129_20260930-b1420f1"
LANE = "H100/all"
JIT = "flashinfer/.cache/flashinfer/0.6.0/90a/cached_ops"
META = {"arch": "sm_90a", "cute_dsl_version": "4.7.0", "source_sha256": "a" * 64}

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("curl") is None,
    reason="needs GNU userland and curl",
)


class _Handler(http.server.SimpleHTTPRequestHandler):
    not_modified = 0

    def do_GET(self):
        path = Path(self.directory) / self.path.lstrip("/")
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        etag = '"%s"' % hashlib.md5(data).hexdigest()
        if self.headers.get("If-None-Match") == etag:
            type(self).not_modified += 1
            self.send_response(304)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("ETag", etag)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def bucket(tmp_path):
    root = tmp_path / "bucket"
    root.mkdir()
    _Handler.not_modified = 0

    def handler(*args, **kwargs):
        return _Handler(*args, directory=str(root), **kwargs)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}/compile-cache/main"
    server.shutdown()


@pytest.fixture
def fake_aws(tmp_path, bucket):
    """An ``aws`` on PATH that stores ``s3 cp <file> s3://b/<key>`` in the bucket."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "aws.log"
    aws = bin_dir / "aws"
    aws.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> "{log}"\n'
        'src="${@: -2:1}"; uri="${@: -1}"; key="${uri#s3://*/}"\n'
        f'mkdir -p "$(dirname "{bucket[0]}/$key")" && cp "$src" "{bucket[0]}/$key"\n'
    )
    aws.chmod(0o755)
    return bin_dir, log


def run(*args, env=None):
    result = subprocess.run(
        ["bash", str(SCRIPT), *args],
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def upload(src, fake_aws, **env):
    return run(
        "upload",
        "s3://bucket/compile-cache/main",
        IMAGE,
        LANE,
        str(src),
        env={
            "PATH": f"{fake_aws[0]}:{os.environ['PATH']}",
            "AWS_ACCESS_KEY_ID": "test",
            **env,
        },
    )


def write(path: Path, data: bytes | str = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode()
    path.write_bytes(data)
    return path


def cute_module(root: Path, name: str, kernels, meta=META) -> Path:
    module = root / JIT / f"{name}_sm90a_cute_dsl"
    write(module / "meta.json", json.dumps(meta))
    for kernel in kernels:
        write(module / f"{kernel}.o", f"{name}/{kernel}")
    return module


def populate(root: Path) -> None:
    cute_module(root, "gemm", ["k1", "k2"])
    write(root / JIT / "kda_tirx_abc" / "kernel.so")
    write(root / JIT / "kda_ptx" / "k.cubin")
    write(root / "flashinfer/.cache/flashinfer/cubins/abc/x.cubin")
    write(root / "triton/HASH/kernel.cubin")
    # Never shared: nvcc modules, tuning, generated sources, temporaries.
    write(root / JIT / "norm" / "build.ninja")
    write(root / JIT / "norm" / "norm.so")
    write(root / "flashinfer/.cache/flashinfer/0.6.0/90a/generated/x.cu")
    write(root / "flashinfer/.cache/flashinfer/autotune/x.json")
    write(root / JIT / "gemm_sm90a_cute_dsl" / "k3.o.tmp.123")
    write(root / JIT / "gemm_sm90a_cute_dsl" / "k3.lock")


def tarball(bucket_root: Path) -> Path:
    return bucket_root / "compile-cache/main" / IMAGE_KEY / f"{LANE}.tar.gz"


def members(path: Path) -> set[str]:
    with tarfile.open(path) as tar:
        return {m.name for m in tar.getmembers() if m.isfile()}


def test_writer_uploads_only_content_keyed_artifacts(tmp_path, bucket, fake_aws):
    writer = tmp_path / "writer"
    out = run("restore", bucket[1], IMAGE, LANE, str(writer), "--writer")
    assert "No shared compile cache" in out
    populate(writer)

    out = upload(writer, fake_aws)
    assert "Uploaded" in out
    assert members(tarball(bucket[0])) == {
        f"{JIT}/gemm_sm90a_cute_dsl/meta.json",
        f"{JIT}/gemm_sm90a_cute_dsl/k1.o",
        f"{JIT}/gemm_sm90a_cute_dsl/k2.o",
        f"{JIT}/kda_tirx_abc/kernel.so",
        f"{JIT}/kda_ptx/k.cubin",
        "flashinfer/.cache/flashinfer/cubins/abc/x.cubin",
        "triton/HASH/kernel.cubin",
    }


def test_writer_restores_and_skips_upload_without_new_builds(
    tmp_path, bucket, fake_aws
):
    first = tmp_path / "first"
    run("restore", bucket[1], IMAGE, LANE, str(first), "--writer")
    populate(first)
    upload(first, fake_aws)
    uploads = fake_aws[1].read_text().count("s3 cp")

    second = tmp_path / "second"
    out = run("restore", bucket[1], IMAGE, LANE, str(second), "--writer")
    assert "Restored shared compile cache" in out
    assert (second / JIT / "gemm_sm90a_cute_dsl" / "k1.o").read_text() == "gemm/k1"
    assert not (second / JIT / "norm").exists()
    assert "compiled nothing new" in upload(second, fake_aws)
    assert fake_aws[1].read_text().count("s3 cp") == uploads

    write(second / JIT / "gemm_sm90a_cute_dsl" / "k9.o")
    assert "Uploaded" in upload(second, fake_aws)
    assert f"{JIT}/gemm_sm90a_cute_dsl/k9.o" in members(tarball(bucket[0]))


def test_writer_refuses_a_directory_with_local_files(tmp_path, bucket):
    writer = tmp_path / "writer"
    write(writer / JIT / "planted_sm90a_cute_dsl" / "meta.json")
    out = run("restore", bucket[1], IMAGE, LANE, str(writer), "--writer")
    assert "is not empty" in out
    assert not (writer / ".s3-manifest").exists()


def test_merge_adds_only_what_the_runner_lacks(tmp_path, bucket, fake_aws):
    writer = tmp_path / "writer"
    run("restore", bucket[1], IMAGE, LANE, str(writer), "--writer")
    cute_module(writer, "same", ["k1", "k2"])
    cute_module(writer, "stale", ["k1", "k2"])
    cute_module(writer, "missing", ["k1"])
    write(writer / "triton/A/kernel.cubin", "s3")
    write(writer / "triton/B/kernel.cubin", "s3")
    upload(writer, fake_aws)

    local = tmp_path / "local"
    write(cute_module(local, "same", ["k1"]) / "k1.o", "local")
    other_meta = {**META, "source_sha256": "b" * 64}
    cute_module(local, "stale", ["k1"], meta=other_meta)
    write(local / "triton/A/kernel.cubin", "local")
    run("restore", bucket[1], IMAGE, LANE, str(local))

    same = local / JIT / "same_sm90a_cute_dsl"
    assert (same / "k1.o").read_text() == "local"
    assert (same / "k2.o").read_text() == "same/k2"
    # A module whose meta.json differs gets nothing: its objects would be
    # loaded under the local meta.
    stale = local / JIT / "stale_sm90a_cute_dsl"
    assert not (stale / "k2.o").exists()
    assert json.loads((stale / "meta.json").read_text()) == other_meta
    assert (local / JIT / "missing_sm90a_cute_dsl" / "k1.o").exists()
    assert (local / "triton/A/kernel.cubin").read_text() == "local"
    assert (local / "triton/B/kernel.cubin").read_text() == "s3"
    assert not list(tmp_path.glob("local.s3-restore.*"))

    # An unchanged object is not downloaded again.
    out = run("restore", bucket[1], IMAGE, LANE, str(local))
    assert "unchanged" in out
    assert _Handler.not_modified == 1


def test_oversized_upload_keeps_only_this_jobs_builds(tmp_path, bucket, fake_aws):
    writer = tmp_path / "writer"
    run("restore", bucket[1], IMAGE, LANE, str(writer), "--writer")
    write(writer / "triton/OLD/kernel.cubin", os.urandom(3 << 20))
    cute_module(writer, "gemm", ["k1"])
    upload(writer, fake_aws)

    second = tmp_path / "second"
    run("restore", bucket[1], IMAGE, LANE, str(second), "--writer")
    write(second / JIT / "gemm_sm90a_cute_dsl" / "k2.o")
    out = upload(second, fake_aws, CI_COMPILE_CACHE_S3_MAX_MB="1")
    assert "uploading only this job's builds" in out
    assert members(tarball(bucket[0])) == {
        f"{JIT}/gemm_sm90a_cute_dsl/meta.json",
        f"{JIT}/gemm_sm90a_cute_dsl/k2.o",
    }


@pytest.mark.parametrize("lane", ["../escape", "H100/..", "a b", "/abs"])
def test_invalid_lane_is_skipped(tmp_path, bucket, lane):
    out = run("restore", bucket[1], IMAGE, lane, str(tmp_path / "dir"))
    assert "invalid lane" in out
    assert not (tmp_path / "dir").exists()


def test_failures_only_warn(tmp_path, fake_aws):
    # Nothing listens on port 9; the job continues cold.
    out = run("restore", "http://127.0.0.1:9", IMAGE, LANE, str(tmp_path / "dir"))
    assert "::warning" in out

    populate(tmp_path / "writer")
    out = run(
        "upload",
        "s3://bucket/compile-cache/main",
        IMAGE,
        LANE,
        str(tmp_path / "writer"),
        env={"PATH": f"{fake_aws[0]}:{os.environ['PATH']}", "AWS_ACCESS_KEY_ID": ""},
    )
    assert "no AWS credentials" in out
    assert not fake_aws[1].exists()
