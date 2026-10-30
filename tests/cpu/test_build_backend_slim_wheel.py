"""The wheel post-processing of build_backend: level-9 re-deflate, dropping the
vendored 3rdparty Python trees, and the release-only exclusion of experimental
kernel sources (flashinfer-ai/flashinfer#7977)."""

import base64
import csv
import hashlib
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def build_backend():
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    spec = importlib.util.spec_from_file_location(
        "flashinfer_build_backend", _ROOT / "build_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIST_INFO = "flashinfer_python-0.0.0.dist-info"
FILES = {
    "flashinfer/__init__.py": b"print('hi')\n",
    "flashinfer/experimental/__init__.py": b"",
    "flashinfer/experimental/foo/__init__.py": b"",
    "flashinfer/experimental/foo/build.json": b'{"programs": []}\n',
    "flashinfer/experimental/foo/csrc/kernel.cu": b"__global__ void k() {}\n" * 200,
    "flashinfer/experimental/foo/csrc/sub/kernel2.cu": b"// generated\n" * 50,
    "flashinfer/data/csrc/experimental/bar/generated/gemm.cu": b"// deepgemm\n" * 50,
    "flashinfer/data/csrc/norm/rmsnorm.cu": b"// stable kernel\n" * 50,
    "flashinfer/data/include/flashinfer/utils.cuh": b"#pragma once\n",
    "flashinfer/data/cutlass/include/cutlass/cutlass.h": b"#pragma once\n",
    "flashinfer/data/cutlass/tools/util/include/cutlass/util/host_tensor.h": b"#pragma once\n",
    "flashinfer/data/cutlass/python/cutlass_library/generator.py": b"def gen(): pass\n",
    "flashinfer/data/cutlass/examples/00_basic_gemm/basic_gemm.cu": b"// example\n",
    "flashinfer/data/cccl/cub/cub/cub.cuh": b"#pragma once\n",
    "flashinfer/data/cccl/libcudacxx/include/cuda/std/version": b"// version\n",
    "flashinfer/data/cccl/thrust/thrust/version.h": b"#pragma once\n",
    "flashinfer/data/cccl/python/cuda_cccl/setup.py": b"# setup\n",
    "flashinfer/data/spdlog/include/spdlog/spdlog.h": b"#pragma once\n",
    f"{DIST_INFO}/METADATA": b"Metadata-Version: 2.1\nName: flashinfer-python\nVersion: 0.0.0\n",
    f"{DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
}


def _record_hash(data: bytes) -> str:
    return (
        "sha256="
        + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    )


def _write_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        rows = []
        for name, data in FILES.items():
            zf.writestr(name, data)
            rows.append((name, _record_hash(data), str(len(data))))
        rows.append((f"{DIST_INFO}/RECORD", "", ""))
        text = io.StringIO()
        csv.writer(text, lineterminator="\n").writerows(rows)
        zf.writestr(f"{DIST_INFO}/RECORD", text.getvalue())


def _entries(path: Path) -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as zf:
        return {i.filename: i for i in zf.infolist()}


def _check_record(path: Path) -> set[str]:
    """RECORD lists exactly the entries, with matching hashes and sizes."""
    with zipfile.ZipFile(path) as zf:
        all_names = [i.filename for i in zf.infolist()]
        assert len(all_names) == len(set(all_names)), "duplicate zip entries"
        names = set(all_names)
        record = zf.read(f"{DIST_INFO}/RECORD").decode()
        listed = {}
        for row in csv.reader(io.StringIO(record)):
            listed[row[0]] = (row[1], row[2])
        assert set(listed) == names
        for name, (digest, size) in listed.items():
            if name == f"{DIST_INFO}/RECORD":
                assert digest == "" and size == ""
                continue
            data = zf.read(name)
            assert digest == _record_hash(data), name
            assert int(size) == len(data), name
        # RECORD is the last entry, as the wheel tools write it.
        assert zf.infolist()[-1].filename == f"{DIST_INFO}/RECORD"
    return names


def test_default_keeps_kernel_sources_and_drops_vendored_python(
    tmp_path, monkeypatch, build_backend
):
    monkeypatch.delenv(build_backend._EXPERIMENTAL_SOURCES_FLAG, raising=False)
    wheel = tmp_path / "flashinfer_python-0.0.0-py3-none-any.whl"
    _write_wheel(wheel)
    stats = build_backend._slim_wheel(wheel)
    names = _check_record(wheel)
    # vendored 3rdparty outside the header trees is gone
    assert "flashinfer/data/cutlass/python/cutlass_library/generator.py" not in names
    assert "flashinfer/data/cutlass/examples/00_basic_gemm/basic_gemm.cu" not in names
    assert "flashinfer/data/cccl/python/cuda_cccl/setup.py" not in names
    # header trees stay
    for kept in (
        "flashinfer/data/cutlass/include/cutlass/cutlass.h",
        "flashinfer/data/cutlass/tools/util/include/cutlass/util/host_tensor.h",
        "flashinfer/data/cccl/cub/cub/cub.cuh",
        "flashinfer/data/cccl/libcudacxx/include/cuda/std/version",
        "flashinfer/data/cccl/thrust/thrust/version.h",
        "flashinfer/data/spdlog/include/spdlog/spdlog.h",
    ):
        assert kept in names
    # kernel sources, experimental included, stay by default
    assert "flashinfer/experimental/foo/csrc/kernel.cu" in names
    assert "flashinfer/experimental/foo/csrc/sub/kernel2.cu" in names
    assert "flashinfer/data/csrc/experimental/bar/generated/gemm.cu" in names
    assert "flashinfer/data/csrc/norm/rmsnorm.cu" in names
    assert build_backend._EXPERIMENTAL_SOURCES_MARKER not in names
    assert stats["dropped_entries"] == 3
    # every entry is deflated and the content is unchanged
    for name, info in _entries(wheel).items():
        assert info.compress_type == zipfile.ZIP_DEFLATED, name
    with zipfile.ZipFile(wheel) as zf:
        assert (
            zf.read("flashinfer/data/csrc/norm/rmsnorm.cu")
            == FILES["flashinfer/data/csrc/norm/rmsnorm.cu"]
        )


def test_release_switch_drops_experimental_kernel_sources(
    tmp_path, monkeypatch, build_backend
):
    monkeypatch.setenv(build_backend._EXPERIMENTAL_SOURCES_FLAG, "1")
    wheel = tmp_path / "flashinfer_python-0.0.0-py3-none-any.whl"
    _write_wheel(wheel)
    stats = build_backend._slim_wheel(wheel)
    names = _check_record(wheel)
    assert "flashinfer/experimental/foo/csrc/kernel.cu" not in names
    assert "flashinfer/experimental/foo/csrc/sub/kernel2.cu" not in names
    assert "flashinfer/data/csrc/experimental/bar/generated/gemm.cu" not in names
    # the package stays importable: modules and manifests remain
    assert "flashinfer/experimental/foo/__init__.py" in names
    assert "flashinfer/experimental/foo/build.json" in names
    # stable kernel sources and headers are untouched
    assert "flashinfer/data/csrc/norm/rmsnorm.cu" in names
    assert "flashinfer/data/include/flashinfer/utils.cuh" in names
    # the marker explains the omission
    assert build_backend._EXPERIMENTAL_SOURCES_MARKER in names
    with zipfile.ZipFile(wheel) as zf:
        marker = zf.read(build_backend._EXPERIMENTAL_SOURCES_MARKER).decode()
    assert (
        build_backend._EXPERIMENTAL_SOURCES_FLAG in marker
        and "flashinfer.ai/whl" in marker
    )
    assert stats["dropped_entries"] == 6


def test_slimming_is_idempotent(tmp_path, monkeypatch, build_backend):
    monkeypatch.setenv(build_backend._EXPERIMENTAL_SOURCES_FLAG, "1")
    wheel = tmp_path / "flashinfer_python-0.0.0-py3-none-any.whl"
    _write_wheel(wheel)
    build_backend._slim_wheel(wheel)
    first = wheel.read_bytes()
    second_stats = build_backend._slim_wheel(wheel)
    assert second_stats["dropped_entries"] == 0
    with zipfile.ZipFile(io.BytesIO(first)) as zf:
        first_names = set(zf.namelist())
    assert set(_entries(wheel)) == first_names
    _check_record(wheel)


def test_entry_rules(build_backend):
    dropped = build_backend._wheel_entry_dropped
    assert dropped("flashinfer/data/cutlass/python/x.py", False)
    assert not dropped("flashinfer/data/cutlass/include/cutlass/x.h", False)
    assert not dropped("flashinfer/experimental/foo/csrc/k.cu", False)
    assert dropped("flashinfer/experimental/foo/csrc/k.cu", True)
    assert dropped("flashinfer/data/csrc/experimental/bar/k.cu", True)
    assert not dropped("flashinfer/experimental/foo/build.json", True)
    assert not dropped("flashinfer/experimental/foo/cake_jit.py", True)
    assert not dropped("flashinfer/data/csrc/norm/k.cu", True)
    assert not dropped("flashinfer/cake_gdp_backend/csrc/k.cu", True)
