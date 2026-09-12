"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import shutil
import subprocess
import textwrap
import time
from pathlib import Path

import pytest


_HELPER_SOURCE = r"""
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cache_file_utils.h"

using deep_gemm::jit::CacheFileLock;
using deep_gemm::jit::isRegularNonEmptyFile;

void mark(std::filesystem::path const& path) {
  std::ofstream file(path);
  file << "1";
}

void holdLock(char const* lock_path, char const* ready_path, char const* acquired_path) {
  mark(ready_path);
  CacheFileLock lock(lock_path);
  mark(acquired_path);

  std::string command;
  std::getline(std::cin, command);
  if (command == "throw") {
    throw std::runtime_error("test exception");
  }
}

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "valid") {
    return isRegularNonEmptyFile(argv[2]) ? 0 : 1;
  }
  if (argc != 5 || std::string(argv[1]) != "lock") {
    return 2;
  }

  try {
    holdLock(argv[2], argv[3], argv[4]);
  } catch (std::runtime_error const&) {
    std::string command;
    std::getline(std::cin, command);
  }
  return 0;
}
"""


@pytest.fixture(scope="module")
def cache_file_helper(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.fail("A host C++ compiler is required for the cache-file unit test")

    build_dir = tmp_path_factory.mktemp("deep_gemm_cache_file_utils")
    source = build_dir / "cache_file_helper.cpp"
    executable = build_dir / "cache_file_helper"
    source.write_text(textwrap.dedent(_HELPER_SOURCE))

    include_dir = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "nv_internal"
        / "tensorrt_llm"
        / "deep_gemm"
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(include_dir),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


def _wait_for(path, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for {path.name}")


def _start_lock_process(executable, lock_path, state_dir, name):
    ready = state_dir / f"{name}.ready"
    acquired = state_dir / f"{name}.acquired"
    process = subprocess.Popen(
        [str(executable), "lock", str(lock_path), str(ready), str(acquired)],
        stdin=subprocess.PIPE,
        text=True,
    )
    return process, ready, acquired


def _send(process, command):
    assert process.stdin is not None
    process.stdin.write(f"{command}\n")
    process.stdin.flush()


def _release(process, command="release"):
    _send(process, command)
    assert process.wait(timeout=5) == 0


@pytest.mark.skipif(os.name == "nt", reason="CacheFileLock is POSIX-only")
def test_cache_file_lock_serializes_per_key_and_releases_on_exception(
    cache_file_helper, tmp_path
):
    lock_a = tmp_path / "a.lock"
    lock_b = tmp_path / "b.lock"
    holder, _, holder_acquired = _start_lock_process(
        cache_file_helper, lock_a, tmp_path, "holder"
    )
    waiter = independent = None

    try:
        _wait_for(holder_acquired)

        import fcntl

        with lock_a.open("a+") as probe, pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

        waiter, waiter_ready, waiter_acquired = _start_lock_process(
            cache_file_helper, lock_a, tmp_path, "waiter"
        )
        _wait_for(waiter_ready)

        independent, _, independent_acquired = _start_lock_process(
            cache_file_helper, lock_b, tmp_path, "independent"
        )
        _wait_for(independent_acquired)
        assert not waiter_acquired.exists()

        _send(holder, "throw")
        _wait_for(waiter_acquired)
        assert holder.poll() is None
        _release(holder)
        _release(waiter)
        _release(independent)
    finally:
        for process in (holder, waiter, independent):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


def test_regular_nonempty_cache_artifact(cache_file_helper, tmp_path):
    missing = tmp_path / "missing"
    directory = tmp_path / "directory"
    empty = tmp_path / "empty"
    nonempty = tmp_path / "nonempty"
    directory.mkdir()
    empty.touch()
    nonempty.write_bytes(b"cubin")

    for invalid in (missing, directory, empty):
        assert subprocess.run([cache_file_helper, "valid", invalid]).returncode == 1
    assert subprocess.run([cache_file_helper, "valid", nonempty]).returncode == 0
