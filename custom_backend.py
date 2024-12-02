import os

from setuptools.build_meta import *  # noqa: F403


def _get_requires_for_build():
    return ["torch"] if os.environ.get("FLASHINFER_ENABLE_AOT", "0") == "1" else []


def get_requires_for_build_wheel(config_settings=None):
    return _get_requires_for_build()


def get_requires_for_build_editable(config_settings=None):
    return _get_requires_for_build()
