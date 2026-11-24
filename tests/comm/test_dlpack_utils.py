# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DLPack storage ownership, including views that drop Python tensor attributes."""

import gc
import importlib.util
import weakref

import pytest
import torch

from flashinfer.comm import dlpack_utils


@pytest.fixture
def cpu_capsules(monkeypatch):
    create = dlpack_utils.create_dlpack_capsule

    def create_cpu(*args, **kwargs):
        wrapper = create(*args, **kwargs)
        wrapper._managed_tensor.dl_tensor.device.device_type = 1
        return wrapper

    monkeypatch.setattr(dlpack_utils, "create_dlpack_capsule", create_cpu)
    return create_cpu


@pytest.mark.parametrize("stride", [16, 32])
@pytest.mark.parametrize("retained", ["view", "cycle", "storage", "dlpack"])
def test_dlpack_owner_survives_dtype_views(cpu_capsules, stride, retained):
    for _ in range(10):
        backing = torch.arange(2 * stride, dtype=torch.uint8)
        owner_ref = weakref.ref(backing)
        tensor = dlpack_utils.pack_strided_memory(
            backing.data_ptr(), 16, stride, 2, torch.uint8, 0, owner=backing
        )
        descriptor_ref = weakref.ref(tensor._capsule_wrapper._managed_tensor)
        expected = backing.view(2, stride)[:, :16].clone().view(torch.int32)
        view = tensor.view(torch.int32)
        del tensor, backing
        gc.collect()
        assert owner_ref() is not None
        assert descriptor_ref() is not None
        torch.testing.assert_close(view, expected)

        if retained == "cycle":
            cycle = [view]
            cycle.append(cycle)
            del view, cycle
        elif retained == "storage":
            storage = view.untyped_storage()
            del view
            gc.collect()
            assert owner_ref() is not None
            del storage
        elif retained == "dlpack":
            other = torch.from_dlpack(view)
            del view
            gc.collect()
            torch.testing.assert_close(other, expected)
            assert owner_ref() is not None
            del other
        else:
            del view
        gc.collect()
        assert owner_ref() is None
        assert descriptor_ref() is None


def test_unconsumed_capsule_releases_owner(cpu_capsules):
    backing = torch.zeros(16, dtype=torch.uint8)
    owner_ref = weakref.ref(backing)
    wrapper = cpu_capsules(backing.data_ptr(), 16, 16, 1, torch.uint8, 0, owner=backing)
    descriptor_ref = weakref.ref(wrapper._managed_tensor)
    capsule = wrapper.capsule
    del backing, wrapper
    gc.collect()
    assert owner_ref() is not None
    assert descriptor_ref() is not None
    del capsule
    gc.collect()
    assert owner_ref() is None
    assert descriptor_ref() is None


def test_failed_consumer_releases_owner(cpu_capsules, monkeypatch):
    refs = []

    def reject(capsule):
        raise RuntimeError("consumer rejected the capsule")

    monkeypatch.setattr(torch.utils.dlpack, "from_dlpack", reject)

    def pack():
        backing = torch.zeros(16, dtype=torch.uint8)
        refs.append(weakref.ref(backing))
        dlpack_utils.pack_strided_memory(
            backing.data_ptr(), 16, 16, 1, torch.uint8, 0, owner=backing
        )

    with pytest.raises(RuntimeError, match="consumer rejected"):
        pack()
    gc.collect()
    assert refs[0]() is None


def test_unconsumed_capsule_retains_destructor():
    spec = importlib.util.spec_from_file_location(
        "dlpack_cleanup_test", dlpack_utils.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backing = torch.zeros(16, dtype=torch.uint8)
    wrapper = module.create_dlpack_capsule(
        backing.data_ptr(), 16, 16, 1, torch.uint8, 0, owner=backing
    )
    callback_ref = weakref.ref(module._delete_capsule)
    owner_ref = weakref.ref(backing)
    capsule = wrapper.capsule
    del wrapper, backing
    module.__dict__.clear()
    gc.collect()
    assert callback_ref() is not None
    del capsule
    gc.collect()
    assert owner_ref() is None
    assert callback_ref() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_dlpack_dtype_view_retains_allocation():
    backing = torch.arange(32, device="cuda", dtype=torch.int32)
    expected = backing.clone()
    owner_ref = weakref.ref(backing)
    tensor = dlpack_utils.pack_strided_memory(
        backing.data_ptr(),
        128,
        128,
        1,
        torch.uint8,
        torch.cuda.current_device(),
        owner=backing,
    )
    view = tensor.view(torch.int32).flatten()
    del backing, tensor
    gc.collect()
    torch.testing.assert_close(view, expected)
    assert owner_ref() is not None
    torch.cuda.synchronize()
    del view
    gc.collect()
    assert owner_ref() is None
