"""
Copyright (c) 2023 by FlashInfer team.

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
from pathlib import Path

from setuptools import build_meta as orig
from flashinfer.build_utils import get_git_version

_root = Path(__file__).parent.resolve()
_data_dir = _root / "flashinfer" / "data"


def write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def get_version():
    package_version = (_root / "version.txt").read_text().strip()
    dev_suffix = os.environ.get("FLASHINFER_DEV_RELEASE_SUFFIX", "")
    if dev_suffix:
        package_version = f"{package_version}.dev{dev_suffix}"
    return package_version


def generate_build_meta() -> None:
    build_meta_str = f"__version__ = {get_version()!r}\n"
    build_meta_str += f"__git_version__ = {get_git_version(cwd=_root)!r}\n"
    write_if_different(_root / "flashinfer" / "_build_meta.py", build_meta_str)


def _create_data_dir():
    _data_dir.mkdir(parents=True, exist_ok=True)

    def ln(source: str, target: str) -> None:
        src = _root / source
        dst = _data_dir / target
        if dst.exists():
            if dst.is_symlink():
                dst.unlink()
            elif dst.is_dir():
                dst.rmdir()
        dst.symlink_to(src, target_is_directory=True)

    ln("3rdparty/cutlass", "cutlass")
    ln("3rdparty/spdlog", "spdlog")
    ln("csrc", "csrc")
    ln("include", "include")


def _prepare_for_wheel():
    # Remove data directory
    if _data_dir.exists():
        shutil.rmtree(_data_dir)


def _prepare_for_editable():
    _create_data_dir()


def _prepare_for_sdist():
    # Remove data directory
    if _data_dir.exists():
        shutil.rmtree(_data_dir)


def get_requires_for_build_wheel(config_settings=None):
    generate_build_meta()
    _prepare_for_wheel()
    return []


def get_requires_for_build_sdist(config_settings=None):
    generate_build_meta()
    _prepare_for_sdist()
    return []


def get_requires_for_build_editable(config_settings=None):
    generate_build_meta()
    _prepare_for_editable()
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _prepare_for_wheel()
    return orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _prepare_for_editable()
    return orig.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_for_editable()
    return orig.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _prepare_for_sdist()
    return orig.build_sdist(sdist_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_for_wheel()
    return orig.build_wheel(wheel_directory, config_settings, metadata_directory)
