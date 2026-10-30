"""The former mHC training names stay importable as aliases of the det names: every
old module name and API name is the same object as its ``cake_mhc_det_train`` name."""

import importlib

import pytest

import flashinfer

MODULES = {
    "flashinfer.cake_mhc_train": "flashinfer.cake_mhc_det_train",
    "flashinfer.cake_mhc_train_backend": "flashinfer.cake_mhc_det_train_backend",
    "flashinfer.cake_mhc_train_head_backend": "flashinfer.cake_mhc_det_train_head_backend",
}


def _new_name(old: str) -> str:
    return old.replace("cake_mhc_train", "cake_mhc_det_train").replace(
        "CakeMHCTrain", "CakeMHCDetTrain"
    )


@pytest.mark.parametrize("old_module", sorted(MODULES))
def test_old_module_names_are_the_det_objects(old_module):
    old = importlib.import_module(old_module)
    new = importlib.import_module(MODULES[old_module])
    assert sorted(_new_name(name) for name in old.__all__) == sorted(new.__all__)
    for name in old.__all__:
        assert getattr(old, name) is getattr(new, _new_name(name)), name


def test_top_level_old_names_are_the_det_objects():
    facade = importlib.import_module("flashinfer.cake_mhc_det_train")
    old_names = [
        name
        for name in dir(flashinfer)
        if "cake_mhc_train" in name or "CakeMHCTrain" in name
    ]
    assert old_names
    for name in old_names:
        assert getattr(flashinfer, name) is getattr(facade, _new_name(name)), name
        assert getattr(flashinfer, _new_name(name)) is getattr(flashinfer, name), name
