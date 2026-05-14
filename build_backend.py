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

import base64
import csv
import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from setuptools import build_meta as orig
from build_utils import get_git_version

_root = Path(__file__).parent.resolve()
_data_dir = _root / "flashinfer" / "data"
_CUTLASS_DSL_PACKAGE = "nvidia-cutlass-dsl"
_CUTLASS_DSL_MIN_VERSION = ">=4.5.0"
_CUTLASS_DSL_BASE_REQUIREMENT = f"{_CUTLASS_DSL_PACKAGE}{_CUTLASS_DSL_MIN_VERSION}"
_CUTLASS_DSL_CU13_REQUIREMENT = (
    f"{_CUTLASS_DSL_PACKAGE}[cu13]{_CUTLASS_DSL_MIN_VERSION}"
)


def _create_build_metadata():
    """Create build metadata file with version information."""
    version_file = _root / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "0.0.0+unknown"

    # Add dev suffix if specified
    dev_suffix = os.environ.get("FLASHINFER_DEV_RELEASE_SUFFIX", "")
    if dev_suffix:
        version = f"{version}.dev{dev_suffix}"

    # Get git version
    git_version = get_git_version(cwd=_root)

    # Append local version suffix if available
    local_version = os.environ.get("FLASHINFER_LOCAL_VERSION")
    if local_version:
        # Use + to create a local version identifier that will appear in wheel name
        version = f"{version}+{local_version}"

    # Create build metadata in the source tree
    package_dir = Path(__file__).parent / "flashinfer"
    build_meta_file = package_dir / "_build_meta.py"

    # Check if we're in a git repository
    git_dir = Path(__file__).parent / ".git"
    in_git_repo = git_dir.exists()

    # If file exists and not in git repo (installing from sdist), keep existing file
    if build_meta_file.exists() and not in_git_repo:
        print("Build metadata file already exists (not in git repo), keeping it")
        return version

    # In git repo (editable) or file doesn't exist, create/update it
    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for flashinfer package."""\n')
        f.write(f'__version__ = "{version}"\n')
        f.write(f'__git_version__ = "{git_version}"\n')

    print(f"Created build metadata file with version {version}")
    return version


# Create build metadata as soon as this module is imported
_create_build_metadata()


def write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _parse_cuda_version(raw_version: str) -> Version:
    value = raw_version.strip().lower()
    if value.startswith("cuda"):
        value = value.removeprefix("cuda").strip("-_ ")
    if value.startswith("cu"):
        cuda_digits = value.removeprefix("cu")
        if not cuda_digits.isdigit():
            raise InvalidVersion(raw_version)
        if len(cuda_digits) <= 2:
            value = cuda_digits
        else:
            value = f"{cuda_digits[:-1]}.{cuda_digits[-1]}"
    return Version(value)


def _detect_cuda_version_from_env() -> Version | None:
    for env_name in ("FLASHINFER_CUDA_VERSION", "CUDA_VERSION"):
        raw_version = os.environ.get(env_name)
        if not raw_version:
            continue
        try:
            return _parse_cuda_version(raw_version)
        except InvalidVersion as exc:
            raise RuntimeError(
                f"{env_name}={raw_version!r} is not a valid CUDA version. "
                "Expected values like '13.0', 'cu130', or 'cu13'."
            ) from exc
    return None


def _detect_cuda_version_from_nvcc() -> Version | None:
    candidate_paths: list[Path] = []
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.environ.get(env_name)
        if cuda_home:
            candidate_paths.append(Path(cuda_home) / "bin" / "nvcc")

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        candidate_paths.append(Path(nvcc_path))

    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        try:
            output = subprocess.check_output(
                [str(candidate), "--version"],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        match = re.search(r"release\s+(\d+\.\d+)", output)
        if match:
            return _parse_cuda_version(match.group(1))

    return None


def _detect_cuda_version_from_torch() -> Version | None:
    try:
        import torch
    except Exception:
        return None

    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not torch_cuda_version:
        return None
    try:
        return _parse_cuda_version(torch_cuda_version)
    except InvalidVersion:
        return None


def _detect_cuda_version() -> Version | None:
    return (
        _detect_cuda_version_from_env()
        or _detect_cuda_version_from_nvcc()
        or _detect_cuda_version_from_torch()
    )


def _selected_cutlass_dsl_requirement() -> str:
    cuda_version = _detect_cuda_version()
    if cuda_version is not None and cuda_version.major >= 13:
        return _CUTLASS_DSL_CU13_REQUIREMENT
    return _CUTLASS_DSL_BASE_REQUIREMENT


def _is_unmarked_cutlass_dsl_requirement(requirement: str) -> bool:
    parsed_requirement = Requirement(requirement)
    return (
        canonicalize_name(parsed_requirement.name)
        == canonicalize_name(_CUTLASS_DSL_PACKAGE)
        and parsed_requirement.marker is None
    )


def _patch_metadata_content(content: str) -> str:
    selected_requirement = _selected_cutlass_dsl_requirement()
    patched_lines = []
    for line in content.splitlines(keepends=True):
        prefix = "Requires-Dist: "
        if line.startswith(prefix):
            requirement = line[len(prefix) :].strip()
            try:
                if _is_unmarked_cutlass_dsl_requirement(requirement):
                    newline = "\n" if line.endswith("\n") else ""
                    line = f"{prefix}{selected_requirement}{newline}"
            except Exception:
                pass
        patched_lines.append(line)
    return "".join(patched_lines)


def _patch_metadata_file(metadata_file: Path) -> None:
    content = metadata_file.read_text()
    patched_content = _patch_metadata_content(content)
    write_if_different(metadata_file, patched_content)


def _patch_dist_info_metadata(metadata_directory: str, dist_info: str) -> None:
    metadata_file = Path(metadata_directory) / dist_info / "METADATA"
    if metadata_file.exists():
        _patch_metadata_file(metadata_file)


def _record_hash(data: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(data).digest()
    encoded_digest = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded_digest}", str(len(data))


def _rewrite_record(
    record: bytes,
    metadata_name: str,
    metadata: bytes,
    record_name: str,
) -> bytes:
    rows = list(csv.reader(io.StringIO(record.decode("utf-8"))))
    metadata_hash, metadata_size = _record_hash(metadata)
    found_metadata = False
    found_record = False

    for row in rows:
        if not row:
            continue
        if row[0] == metadata_name:
            row[1] = metadata_hash
            row[2] = metadata_size
            found_metadata = True
        elif row[0] == record_name:
            row[1] = ""
            row[2] = ""
            found_record = True

    if not found_metadata:
        rows.append([metadata_name, metadata_hash, metadata_size])
    if not found_record:
        rows.append([record_name, "", ""])

    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode("utf-8")


def _patch_wheel_metadata(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path, "r") as wheel:
        infos = wheel.infolist()
        contents = {info.filename: wheel.read(info.filename) for info in infos}

    metadata_files = [name for name in contents if name.endswith(".dist-info/METADATA")]
    if len(metadata_files) != 1:
        return

    metadata_name = metadata_files[0]
    metadata = contents[metadata_name]
    patched_metadata = _patch_metadata_content(metadata.decode("utf-8")).encode("utf-8")
    if patched_metadata == metadata:
        return

    dist_info_dir = metadata_name.rsplit("/", 1)[0]
    record_name = f"{dist_info_dir}/RECORD"
    contents[metadata_name] = patched_metadata
    if record_name in contents:
        contents[record_name] = _rewrite_record(
            contents[record_name], metadata_name, patched_metadata, record_name
        )

    fd, temp_path = tempfile.mkstemp(
        prefix=f"{wheel_path.stem}.", suffix=".whl", dir=wheel_path.parent
    )
    os.close(fd)
    temp_wheel_path = Path(temp_path)
    try:
        with zipfile.ZipFile(temp_wheel_path, "w") as patched_wheel:
            for info in infos:
                patched_wheel.writestr(info, contents[info.filename])
        temp_wheel_path.replace(wheel_path)
    finally:
        if temp_wheel_path.exists():
            temp_wheel_path.unlink()


def _create_data_dir(use_symlinks=True):
    _data_dir.mkdir(parents=True, exist_ok=True)

    def ln(source: str, target: str) -> None:
        src = _root / source
        dst = _data_dir / target
        if dst.exists():
            if dst.is_symlink():
                dst.unlink()
            elif dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        if use_symlinks:
            dst.symlink_to(src, target_is_directory=True)
        else:
            # For wheel/sdist, copy actual files instead of symlinks
            if src.exists():
                shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True)

    ln("3rdparty/cutlass", "cutlass")
    ln("3rdparty/spdlog", "spdlog")
    ln("3rdparty/cccl", "cccl")
    ln("csrc", "csrc")
    ln("include", "include")


def _prepare_for_wheel():
    # For wheel, copy actual files instead of symlinks so they are included in the wheel
    if _data_dir.exists():
        shutil.rmtree(_data_dir)
    _create_data_dir(use_symlinks=False)

    # Copy license files from licenses/ to root to avoid nested path in wheel
    licenses_dir = _root / "licenses"
    if licenses_dir.exists():
        for license_file in licenses_dir.glob("*.txt"):
            shutil.copy2(
                license_file,
                _root / f"LICENSE.{license_file.stem.removeprefix('LICENSE.')}.txt",
            )


def _prepare_for_editable():
    # For editable install, use symlinks so changes are reflected immediately
    if _data_dir.exists():
        shutil.rmtree(_data_dir)
    _create_data_dir(use_symlinks=True)


def _prepare_for_sdist():
    # For sdist, copy actual files instead of symlinks so they are included in the tarball
    if _data_dir.exists():
        shutil.rmtree(_data_dir)
    _create_data_dir(use_symlinks=False)


def get_requires_for_build_wheel(config_settings=None):
    _prepare_for_wheel()
    return []


def get_requires_for_build_sdist(config_settings=None):
    _prepare_for_sdist()
    return []


def get_requires_for_build_editable(config_settings=None):
    _prepare_for_editable()
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _prepare_for_wheel()
    dist_info = orig.prepare_metadata_for_build_wheel(
        metadata_directory, config_settings
    )
    _patch_dist_info_metadata(metadata_directory, dist_info)
    return dist_info


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _prepare_for_editable()
    dist_info = orig.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )
    _patch_dist_info_metadata(metadata_directory, dist_info)
    return dist_info


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_for_editable()
    wheel_name = orig.build_editable(
        wheel_directory, config_settings, metadata_directory
    )
    _patch_wheel_metadata(Path(wheel_directory) / wheel_name)
    return wheel_name


def build_sdist(sdist_directory, config_settings=None):
    _prepare_for_sdist()
    return orig.build_sdist(sdist_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_for_wheel()
    wheel_name = orig.build_wheel(wheel_directory, config_settings, metadata_directory)
    _patch_wheel_metadata(Path(wheel_directory) / wheel_name)
    return wheel_name
