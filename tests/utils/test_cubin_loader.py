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

import pathlib

import pytest

from flashinfer.jit.cubin_loader import ensure_symlink


def test_ensure_symlink_tolerates_concurrent_same_target(monkeypatch, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "include" / "target"
    original_symlink_to = pathlib.Path.symlink_to

    def create_link_then_raise(self, symlink_target, *args, **kwargs):
        original_symlink_to(self, symlink_target, *args, **kwargs)
        raise FileExistsError

    monkeypatch.setattr(pathlib.Path, "symlink_to", create_link_then_raise)

    ensure_symlink(link, target)

    assert link.is_symlink()
    assert link.resolve() == target.resolve()


def test_ensure_symlink_reraises_concurrent_different_target(monkeypatch, tmp_path):
    target = tmp_path / "target"
    other_target = tmp_path / "other_target"
    target.mkdir()
    other_target.mkdir()
    link = tmp_path / "include" / "target"
    original_symlink_to = pathlib.Path.symlink_to

    def create_wrong_link_then_raise(self, symlink_target, *args, **kwargs):
        original_symlink_to(self, other_target, *args, **kwargs)
        raise FileExistsError

    monkeypatch.setattr(pathlib.Path, "symlink_to", create_wrong_link_then_raise)

    with pytest.raises(FileExistsError):
        ensure_symlink(link, target)
