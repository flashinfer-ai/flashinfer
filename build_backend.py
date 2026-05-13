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
import subprocess
import sys
from pathlib import Path

from setuptools import build_meta as orig
from build_utils import get_git_version

_root = Path(__file__).parent.resolve()
_data_dir = _root / "flashinfer" / "data"

# moe_ep build infra: gated by BUILD_NVEP=1 env var.
# When set, _build_nvep_if_enabled() runs meson on 3rdparty/nixl and make on
# 3rdparty/nccl, then stages the produced .so files under flashinfer/moe_ep/.
_BUILD_NVEP = os.environ.get("BUILD_NVEP", "0") == "1"
_nvep_build_root = _root / "build_nvep"
_moe_ep_pkg = _root / "flashinfer" / "moe_ep"


def _detect_cuda_major() -> int:
    """Best-effort detection of the CUDA major version on the host."""
    try:
        out = subprocess.check_output(["nvcc", "--version"]).decode()
        for line in out.splitlines():
            if "release" in line:
                # e.g. "Cuda compilation tools, release 13.0, V13.0.48"
                token = line.split("release", 1)[1].split(",", 1)[0].strip()
                return int(token.split(".")[0])
    except Exception:
        pass
    return 13  # default — pyproject's nvep extras pin cu13 packages


def _apply_patches(submodule_dir: Path, patches_dir: Path) -> None:
    """Apply every *.patch in patches_dir to the submodule working tree.

    No-ops if patches_dir doesn't exist. Idempotent: skips patches that are
    already applied by checking `git apply --reverse --check` first.
    """
    if not patches_dir.is_dir():
        return
    for patch in sorted(patches_dir.glob("*.patch")):
        # Already applied?
        already = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=submodule_dir, capture_output=True,
        )
        if already.returncode == 0:
            print(f"[BUILD_NVEP] patch already applied, skipping: {patch.name}")
            continue
        # Check we *can* apply, then apply.
        subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=submodule_dir, check=True,
        )
        subprocess.run(
            ["git", "apply", str(patch)],
            cwd=submodule_dir, check=True,
        )
        print(f"[BUILD_NVEP] applied patch: {patch.name}")


def _build_nixl_ep() -> None:
    src = _root / "3rdparty" / "nixl"
    build = _nvep_build_root / "nixl"
    prefix = _nvep_build_root / "nixl_install"
    _apply_patches(src, _root / "3rdparty_patches" / "nixl")

    if not build.exists():
        subprocess.run([
            "meson", "setup", str(build), str(src),
            "-Dbuild_nixl_ep=true",
            "-Dbuild_examples=true",
            f"-Dprefix={prefix}",
            "--buildtype=release",
        ], check=True)
    subprocess.run(["ninja", "-C", str(build), "install"], check=True)

    dst = _moe_ep_pkg / "nixl_ep" / "_libs"
    dst.mkdir(parents=True, exist_ok=True)

    nixl_lib_src = prefix / "lib" / "x86_64-linux-gnu"
    if nixl_lib_src.exists():
        shutil.copytree(nixl_lib_src, dst / "nixl_lib", dirs_exist_ok=True)

    # The torch extension lands either in build/ or build/examples/device/ep/
    for cand in (build / "examples/device/ep").glob("nixl_ep_cpp*.so"):
        shutil.copy(cand, dst / cand.name)
        print(f"[BUILD_NVEP] staged: {cand.name}")

    # Vendor the python wrapper sources so we can import from
    # flashinfer.moe_ep.nixl_ep._vendored (Step B5).
    vendored_src = src / "examples/device/ep/nixl_ep"
    if vendored_src.exists():
        shutil.copytree(
            vendored_src,
            _moe_ep_pkg / "nixl_ep" / "_vendored",
            dirs_exist_ok=True,
        )


def _build_nccl_ep() -> None:
    src = _root / "3rdparty" / "nccl"
    build = _nvep_build_root / "nccl"
    _apply_patches(src, _root / "3rdparty_patches" / "nccl")

    subprocess.run(
        ["make", "src.build", f"BUILDDIR={build}", "-j"],
        cwd=src, check=True,
    )
    subprocess.run(
        ["make", "-C", "contrib/nccl_ep", f"BUILDDIR={build}", "-j"],
        cwd=src, check=True,
    )

    dst = _moe_ep_pkg / "nccl_ep" / "_libs"
    dst.mkdir(parents=True, exist_ok=True)
    for soname in ("libnccl.so.2", "libnccl_ep.so"):
        sopath = build / "lib" / soname
        if sopath.exists():
            shutil.copy(sopath, dst / soname)
            print(f"[BUILD_NVEP] staged: {soname}")

    # Editable-install the ctypes wrapper from contrib/nccl_ep/python so
    # `import nccl_ep` resolves on the user's env.
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-e",
        str(src / "contrib/nccl_ep/python"),
    ], check=True)

    # Editable-install nccl4py — gives Cython bindings + Communicator(ptr=...)
    # which is how the moe_ep NCCL backend bridges a torch.distributed
    # process group's raw ncclComm_t pointer into ncclEpCreateGroup.
    cuda_extra = f"cu{_detect_cuda_major()}"
    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-e",
        f"{src / 'bindings' / 'nccl4py'}[{cuda_extra}]",
    ], env=env, check=True)


def _fix_rpaths() -> None:
    """Rewrite RPATHs on staged .so files so they find siblings without LD_LIBRARY_PATH."""
    patchelf_ok = shutil.which("patchelf") is not None
    if not patchelf_ok:
        print("[BUILD_NVEP] patchelf not found; skipping RPATH fix-up")
        return
    rpath = "$ORIGIN:$ORIGIN/_libs:$ORIGIN/_libs/nixl_lib"
    for so in _moe_ep_pkg.rglob("*.so*"):
        # Skip symlinks
        if so.is_symlink():
            continue
        subprocess.run(
            ["patchelf", "--set-rpath", rpath, str(so)],
            check=False,
        )


def _build_nvep_if_enabled() -> None:
    if not _BUILD_NVEP:
        return
    print("[BUILD_NVEP] BUILD_NVEP=1 — building NIXL-EP + NCCL-EP from submodules")
    # Make sure submodules are present (sdist installs won't have them
    # initialized automatically).
    if not (_root / "3rdparty/nixl/meson.build").exists():
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=_root, check=True,
        )
    _build_nixl_ep()
    _build_nccl_ep()
    _fix_rpaths()
    print("[BUILD_NVEP] done")


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
    _build_nvep_if_enabled()
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
    _build_nvep_if_enabled()
    if _data_dir.exists():
        shutil.rmtree(_data_dir)
    _create_data_dir(use_symlinks=True)


def _prepare_for_sdist():
    # For sdist, copy actual files instead of symlinks so they are included in the tarball
    # NOTE: do NOT build moe_ep here — submodules + patches travel in the sdist
    # itself and get built during the *install* of the sdist.
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
