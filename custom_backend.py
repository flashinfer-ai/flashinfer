import shutil
from pathlib import Path

from setuptools.build_meta import *  # noqa: F403

_root = Path(__file__).parent.resolve()
_data_dir = _root / "flashinfer" / "data"
_aot_ops_dir = _root / "aot-ops"
_aot_ops_package_dir = _root / "build" / "aot-ops-package-dir"

_requires_for_aot = ["torch", "ninja", "numpy"]


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
    ln("csrc", "csrc")
    ln("include", "include")
    ln("tvm_binding", "tvm_binding")


def get_requires_for_build_wheel(config_settings=None):
    # Remove data directory
    if _data_dir.exists():
        shutil.rmtree(_data_dir)

    # Link AOT ops directory to "aot-ops"
    _rm_aot_ops_package_dir()
    if len(list(_aot_ops_dir.glob("*/*.so"))) == 0:
        raise RuntimeError(f"No AOT ops found in {_aot_ops_dir}")
    _aot_ops_package_dir.parent.mkdir(parents=True, exist_ok=True)
    _aot_ops_package_dir.symlink_to(_aot_ops_dir)

    return _requires_for_aot


def get_requires_for_build_sdist(config_settings=None):
    # Remove data directory
    if _data_dir.exists():
        shutil.rmtree(_data_dir)

    # Create an empty directory for AOT ops
    _rm_aot_ops_package_dir()
    _aot_ops_package_dir.parent.mkdir(parents=True, exist_ok=True)
    _aot_ops_package_dir.mkdir(parents=True)

    return []


def get_requires_for_build_editable(config_settings=None):
    _create_data_dir()

    _rm_aot_ops_package_dir()
    _aot_ops_dir.mkdir(parents=True, exist_ok=True)
    _aot_ops_package_dir.parent.mkdir(parents=True, exist_ok=True)
    _aot_ops_package_dir.symlink_to(_aot_ops_dir)

    return _requires_for_aot
