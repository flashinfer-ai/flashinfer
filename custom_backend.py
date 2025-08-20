import shutil
from pathlib import Path

from setuptools import build_meta as orig

_root = Path(__file__).parent.resolve()
_data_dir = _root / "flashinfer" / "data"
_aot_ops_dir = _root / "aot-ops"
_aot_ops_package_dir = _root / "build" / "aot-ops-package-dir"

_requires_for_aot = ["torch", "ninja", "numpy", "requests"]


def _rm_aot_ops_package_dir():
    if _aot_ops_package_dir.is_symlink():
        _aot_ops_package_dir.unlink()
    elif _aot_ops_package_dir.exists():
        shutil.rmtree(_aot_ops_package_dir)


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
    ln("tvm_binding", "tvm_binding")


def _prepare_for_wheel():
    # Remove data directory
    if _data_dir.exists():
        shutil.rmtree(_data_dir)

    # Link AOT ops directory to "aot-ops"
    _rm_aot_ops_package_dir()
    if not _aot_ops_dir.exists():
        _aot_ops_dir.mkdir()
    num_ops = len(list(_aot_ops_dir.glob("*/*.so")))
    print(f"{num_ops} AOT ops found in {_aot_ops_dir}")
    _aot_ops_package_dir.parent.mkdir(parents=True, exist_ok=True)
    _aot_ops_package_dir.symlink_to(_aot_ops_dir)


def _prepare_for_editable():
    _create_data_dir()

    _rm_aot_ops_package_dir()
    _aot_ops_dir.mkdir(parents=True, exist_ok=True)
    _aot_ops_package_dir.parent.mkdir(parents=True, exist_ok=True)
    _aot_ops_package_dir.symlink_to(_aot_ops_dir)


def _prepare_for_sdist():
    # Remove data directory
    if _data_dir.exists():
        shutil.rmtree(_data_dir)

    # Create an empty directory for AOT ops
    _rm_aot_ops_package_dir()
    _aot_ops_package_dir.parent.mkdir(parents=True, exist_ok=True)
    _aot_ops_package_dir.mkdir(parents=True)


def get_requires_for_build_wheel(config_settings=None):
    _prepare_for_wheel()
    return _requires_for_aot


def get_requires_for_build_sdist(config_settings=None):
    _prepare_for_sdist()
    return []


def get_requires_for_build_editable(config_settings=None):
    _prepare_for_editable()
    return _requires_for_aot


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
