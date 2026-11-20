# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise sparse-cache locking with native locks and real CuTe dependencies."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("cutlass")

_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "flashinfer/cute_dsl/sparse/bsa_utils/cache_utils.py"
)
_SPEC = importlib.util.spec_from_file_location("sparse_cache_lock_contract", _SOURCE)
_CACHE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CACHE)


@pytest.mark.cpu_only
def test_cache_lock_shared_readers_exclude_writer_and_preserve_identity(tmp_path):
    path = tmp_path / "kernel.lock"
    with _CACHE.FileLock(path, exclusive=False):
        identity = path.stat().st_ino
        with _CACHE.FileLock(path, exclusive=False, timeout=0.1):
            pass
        with (
            pytest.raises(RuntimeError, match="Timed out.*exclusive kernel"),
            _CACHE.FileLock(path, exclusive=True, timeout=0.1, label="kernel"),
        ):
            pytest.fail("contender entered while the cache lock was held")
    with _CACHE.FileLock(path, exclusive=True, timeout=0.1):
        assert path.stat().st_ino == identity


@pytest.mark.cpu_only
def test_cache_lock_preserves_body_error_and_releases(tmp_path):
    path = tmp_path / "kernel.lock"
    original = BlockingIOError("unrelated operation inside critical section")
    with (
        pytest.raises(BlockingIOError) as caught,
        _CACHE.FileLock(path, exclusive=True),
    ):
        raise original
    assert caught.value is original
    with _CACHE.FileLock(path, exclusive=True, timeout=0.1):
        pass


@pytest.mark.cpu_only
def test_cache_lock_does_not_retry_permanent_open_error(tmp_path):
    with pytest.raises(OSError), _CACHE.FileLock(tmp_path, exclusive=True, timeout=0):
        pytest.fail("a directory cannot serve as the cache lock file")
