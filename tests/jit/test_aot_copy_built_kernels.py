# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``copy_built_kernels`` must package what the build wrote, not the AOT copy.

The two questions are different and only one of them may follow the prebuilt
preference: a runtime consumer loads ``get_library_path()`` (the AOT artifact
when one is installed), while packaging asks for the artifact this build just
produced.  When ``skip_prebuilt=False`` refreshes the JIT library, selecting the
AOT path would silently package the stale prebuilt library instead.
"""

from __future__ import annotations

import pathlib


from flashinfer import aot
from flashinfer.jit import core as jit_core
from flashinfer.jit import env as jit_env


def _spec(tmp_path: pathlib.Path, name: str = "fi_copy_probe") -> jit_core.JitSpecNvcc:
    return jit_core.JitSpecNvcc(
        name=name,
        sources=[],
        extra_cflags=[],
        extra_cuda_cflags=[],
        extra_include_dirs=[],
        artifact_version="copy-probe",
    )


def test_packaging_takes_the_built_library_when_an_aot_artifact_exists(
    tmp_path, monkeypatch
):
    """A spec with a prebuilt artifact still packages its own rebuilt library."""
    jit_dir = tmp_path / "jit"
    aot_dir = tmp_path / "aot"
    aot_dir.mkdir()
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", jit_dir)

    spec = _spec(tmp_path)
    # Two distinguishable artifacts: the built one and the prebuilt one.
    spec.jit_library_path.parent.mkdir(parents=True)
    spec.jit_library_path.write_bytes(b"jit-build")
    aot_library = aot_dir / f"{spec.name}.so"
    aot_library.write_bytes(b"prebuilt-aot")
    monkeypatch.setattr(jit_env, "get_aot_artifacts", lambda name: (object(),))
    monkeypatch.setattr(jit_env, "get_aot_path", lambda name: aot_library)

    # The runtime preference is unchanged: consumers still get the AOT artifact.
    assert spec.is_aot
    assert spec.get_library_path() == aot_library
    assert spec.get_built_library_path() == spec.jit_library_path

    out_dir = tmp_path / "package"
    aot.copy_built_kernels([spec], out_dir)

    packaged = out_dir / spec.name / f"{spec.name}.so"
    assert packaged.read_bytes() == b"jit-build"


def test_packaging_takes_the_built_library_without_an_aot_artifact(
    tmp_path, monkeypatch
):
    """Without a prebuilt artifact the two selectors agree."""
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setattr(jit_env, "get_aot_artifacts", lambda name: ())

    spec = _spec(tmp_path, name="fi_copy_probe_plain")
    spec.jit_library_path.parent.mkdir(parents=True)
    spec.jit_library_path.write_bytes(b"jit-only")

    assert not spec.is_aot
    assert spec.get_built_library_path() == spec.get_library_path()

    out_dir = tmp_path / "package-plain"
    aot.copy_built_kernels([spec], out_dir)
    assert (out_dir / spec.name / f"{spec.name}.so").read_bytes() == b"jit-only"
