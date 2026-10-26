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

import flashinfer.deep_gemm as dg


def _patch_loader(monkeypatch, tmp_path, cubin_name):
    """Register one fake cubin and count artifact fetches; no GPU needed."""
    fetches = []
    artifact_dir = tmp_path / "deepgemm"
    artifact_dir.mkdir()
    (artifact_dir / f"{cubin_name}.cubin").write_bytes(b"")
    monkeypatch.setattr(dg, "KERNEL_MAP", {cubin_name: ("fake_symbol", "0" * 64)})
    monkeypatch.setattr(dg, "RUNTIME_CACHE", {})
    monkeypatch.setattr(dg, "FLASHINFER_CUBIN_DIR", tmp_path)
    monkeypatch.setattr(dg, "get_deepgemm_artifact_path", lambda: "deepgemm")
    monkeypatch.setattr(dg, "get_artifact", lambda path, sha: fetches.append(path))
    return fetches


def test_load_reuses_cached_runtime(monkeypatch, tmp_path):
    name, code = "fake_gemm", "// code"
    cubin_name = f"kernel.{name}.{dg.hash_to_hex(f'{name}$${code}')}"
    fetches = _patch_loader(monkeypatch, tmp_path, cubin_name)

    first = dg.load(name, code)
    second = dg.load(name, code)

    assert second is first
    assert len(fetches) == 1
    assert list(dg.RUNTIME_CACHE) == [cubin_name]


def test_load_all_then_load_hits_cache(monkeypatch, tmp_path):
    name, code = "fake_gemm", "// code"
    cubin_name = f"kernel.{name}.{dg.hash_to_hex(f'{name}$${code}')}"
    fetches = _patch_loader(monkeypatch, tmp_path, cubin_name)

    dg.load_all()
    runtime = dg.load(name, code)

    assert runtime is dg.RUNTIME_CACHE[cubin_name]
    assert len(fetches) == 1
