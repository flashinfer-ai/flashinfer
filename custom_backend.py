import shutil
from pathlib import Path

from setuptools import build_meta as orig

_root = Path(__file__).parent.resolve()
_data_dir = _root / "flashinfer" / "data"


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


def _prepare_for_editable():
    _create_data_dir()


def _prepare_for_sdist():
    # Remove data directory
    if _data_dir.exists():
        shutil.rmtree(_data_dir)


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
