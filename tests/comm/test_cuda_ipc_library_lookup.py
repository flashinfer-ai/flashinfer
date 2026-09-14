"""Tests for the /proc/self/maps library lookup in flashinfer.comm.cuda_ipc.

Pure string matching against a mocked maps table -- no GPU, MPI, or CUDA runtime required beyond
importing the module.

Run with: pytest tests/comm/test_cuda_ipc_library_lookup.py -vv
"""

from unittest import mock

import pytest

from flashinfer.comm.cuda_ipc import _is_library_filename, find_loaded_library

CUDART = "/usr/local/cuda/lib64/libcudart.so.12"
# torch ships a hash-suffixed copy, which the lookup's own comment calls out as a case it handles.
CUDART_HASHED = "/site-packages/nvidia/cuda_runtime/lib/libcudart-d0da41ae.so.11.0"
# tilelang ships this stub. It exports only a subset of the runtime, so binding it makes a later
# symbol lookup fail with an opaque "undefined symbol: cudaDeviceReset".
STUB = "/site-packages/tilelang/lib/libcudart_stub.so"
# `.so` also occurs inside these names, so checking only the text before it is not sufficient.
DOT_SOMETHING = "/opt/vendor/libcudart.something"
DOT_BACKUP = "/opt/vendor/libcudart.so.backup"
# The library name appears only in a directory component here.
DIR_ONLY = "/opt/libcudart/lib/libfoo.so"

ACCEPTED_FILENAMES = [
    "libcudart.so",
    "libcudart.so.12",
    "libcudart.so.11.0",
    "libcudart-d0da41ae.so.11.0",
]
REJECTED_FILENAMES = [
    "libcudart_stub.so",
    "libcudart_mock.so",
    "libcudartfoo.so.1",
    "libcudart.something",
    "libcudart.so.backup",
    "libcudart.so.12a",
]


def _maps_line(path: str, address: str = "7f0000000000-7f0000001000") -> str:
    return f"{address} r-xp 00000000 08:01 1234567 {path}\n"


def _mock_maps(*paths: str):
    """Patch open() so find_loaded_library reads a synthetic /proc/self/maps."""
    contents = "".join(
        _maps_line(path, f"7f{index}000000000-7f{index}000001000") for index, path in enumerate(paths)
    )
    return mock.patch("builtins.open", mock.mock_open(read_data=contents))


@pytest.mark.parametrize("filename", ACCEPTED_FILENAMES)
def test_accepts_real_cuda_runtime_filenames(filename):
    assert _is_library_filename(_maps_line(f"/usr/local/cuda/lib64/{filename}"), "libcudart")


@pytest.mark.parametrize("filename", REJECTED_FILENAMES)
def test_rejects_other_libraries_whose_names_contain_the_query(filename):
    assert not _is_library_filename(_maps_line(f"/usr/local/cuda/lib64/{filename}"), "libcudart")


def test_rejects_anonymous_mappings():
    """Lines with no path cannot name a library."""
    assert not _is_library_filename("7f0000000000-7f0000001000 rw-p 00000000 00:00 0 \n", "libcudart")


def test_finds_the_runtime_when_a_stub_is_mapped_first():
    """The regression this guards.

    /proc/self/maps is walked in address order and the scan stops at the first hit, so a stub at a
    lower address used to shadow the real runtime.
    """
    with _mock_maps(STUB, CUDART):
        assert find_loaded_library("libcudart") == CUDART


def test_finds_the_runtime_when_a_stub_is_mapped_second():
    """The result must not depend on load order."""
    with _mock_maps(CUDART, STUB):
        assert find_loaded_library("libcudart") == CUDART


def test_returns_none_when_only_a_stub_is_loaded():
    """A missing runtime is a different, and much clearer, failure than binding the wrong one."""
    with _mock_maps(STUB):
        assert find_loaded_library("libcudart") is None


def test_finds_a_hash_suffixed_runtime():
    with _mock_maps(CUDART_HASHED):
        assert find_loaded_library("libcudart") == CUDART_HASHED


def test_ignores_the_name_in_a_directory_component():
    with _mock_maps(DIR_ONLY):
        assert find_loaded_library("libcudart") is None


@pytest.mark.parametrize("decoy", [DOT_SOMETHING, DOT_BACKUP])
def test_ignores_names_that_only_contain_a_dot_so(decoy):
    """`.so` occurs inside `.something`, and `.so.backup` trails it, so neither is a real match."""
    with _mock_maps(decoy):
        assert find_loaded_library("libcudart") is None
    with _mock_maps(decoy, CUDART):
        assert find_loaded_library("libcudart") == CUDART


def test_returns_none_when_the_library_is_not_mapped():
    with _mock_maps("/usr/lib/libc.so.6"):
        assert find_loaded_library("libcudart") is None
