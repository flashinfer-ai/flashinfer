import os
from pathlib import Path

from setuptools import build_meta as orig
from setuptools.build_meta import *  # noqa: F403


def _get_requires_for_build():
    requires = []
    if os.environ.get("FLASHINFER_ENABLE_AOT", "0") == "1":
        requires += ["torch", "ninja"]
    return requires


def get_requires_for_build_wheel(config_settings=None):
    return _get_requires_for_build()


def get_requires_for_build_editable(config_settings=None):
    return _get_requires_for_build()


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    root = Path(__file__).parent.resolve()
    data_dir = root / "flashinfer" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    def ln(src: str, dst: str) -> None:
        src: Path = root / src
        dst: Path = data_dir / dst
        if dst.exists():
            if dst.is_symlink():
                dst.unlink()
            elif dst.is_dir():
                dst.rmdir()
        dst.symlink_to(src, target_is_directory=True)

    ln("3rdparty/cutlass", "cutlass")
    ln("csrc", "csrc")
    ln("include", "include")
    return orig.build_editable(wheel_directory, config_settings, metadata_directory)
