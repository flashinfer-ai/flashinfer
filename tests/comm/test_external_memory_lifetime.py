# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""External views must release metadata safely, including at Python shutdown."""

import os
import subprocess
import sys
import textwrap

import pytest

from flashinfer.comm import dlpack_utils


_CHILD = r"""
import gc
import sys

import torch

from flashinfer.comm import dlpack_utils
from flashinfer.comm.mnnvl import create_tensor_from_cuda_memory

assert dlpack_utils.__file__ == sys.argv[1], dlpack_utils.__file__
print("source", dlpack_utils.__file__, flush=True)
mode, api, dtype_name = sys.argv[2:]
backing = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
pointer = backing.data_ptr()
device = torch.cuda.current_device()

if mode == "geometry":
    observations = 0
    for name in ("uint8", "uint64", "float16", "float32", "int32",
                 "bfloat16", "float8_e4m3fn"):
        dtype = getattr(torch, name)
        itemsize = torch.empty(0, dtype=dtype).element_size()
        layouts = (
            (2, 64, 64, 128),
            (2, 64, 96, 160),
            (1, 64, 96, 64),
            (2, 64, 0, 64),
            (0, 64, 96, 0),
            (2, 0, 96, 0),
            (2, 65, 99, {1: 164, 2: 162, 4: 160, 8: 160}[itemsize]),
        )
        for rows, size, stride, nbytes in layouts:
            tensor = dlpack_utils.pack_strided_memory(
                pointer, size, stride, rows, dtype, device
            )
            reference = backing.view(dtype).as_strided(
                (rows, size // itemsize), (stride // itemsize, 1)
            )
            assert tensor.dtype == dtype
            assert tensor.shape == reference.shape
            assert tensor.stride() == reference.stride()
            assert tensor.device == backing.device
            assert tensor.untyped_storage().data_ptr() == pointer
            assert tensor.untyped_storage().nbytes() == nbytes
            assert torch.equal(tensor.view(torch.uint8), reference.view(torch.uint8))
            if tensor.numel():
                backing[0] = 37
                assert tensor.view(torch.uint8)[0, 0].item() == 37
            torch.cuda.synchronize()
            del tensor
            gc.collect()
            observations += 1
    for args in ((-1, 64, 64, 2), (pointer, -1, 64, 2),
                 (pointer, 8, 8, 2**62)):
        try:
            dlpack_utils.pack_strided_memory(*args, torch.uint8, device)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid geometry was accepted")
    print("geometry", observations, flush=True)
else:
    dtype = getattr(torch, dtype_name)
    itemsize = torch.empty(0, dtype=dtype).element_size()
    if api == "strided":
        tensor = dlpack_utils.pack_strided_memory(
            pointer, 64, 96, 2, dtype, device
        )
        reference = backing.as_strided((2, 64), (96, 1))
        expected_stride, expected_nbytes = 96 // itemsize, 160
    else:
        tensor = create_tensor_from_cuda_memory(
            pointer, (2, 64 // itemsize), dtype, device
        )
        reference = backing[:128].view(2, 64)
        expected_stride, expected_nbytes = 64 // itemsize, 128
    assert tensor.dtype == dtype
    assert tensor.shape == (2, 64 // itemsize)
    assert tensor.stride() == (expected_stride, 1)
    assert tensor.device == backing.device and tensor.data_ptr() == pointer
    assert tensor.untyped_storage().nbytes() == expected_nbytes
    assert torch.equal(tensor.view(torch.uint8), reference)
    tensor.view(torch.uint8)[0, 0] = 29
    assert backing[0].item() == 29
    backing[1] = 31
    assert tensor.view(torch.uint8)[0, 1].item() == 31
    torch.cuda.synchronize()
    print("validated", flush=True)
    if mode == "slice":
        derived = tensor[:, 1:]
        del tensor
        gc.collect()
        assert torch.equal(derived.view(torch.uint8), reference[:, itemsize:])
        torch.cuda.synchronize()
        del derived
    elif mode == "shutdown":
        CACHE = {"tensor": tensor}
        del tensor
    else:
        del tensor
    gc.collect()

backing.fill_(13)
assert backing[0].item() == 13
torch.cuda.synchronize()
print("completed", flush=True)
"""


def _run_child(mode, api="strided", dtype="uint8"):
    env = {**os.environ, "PYTHONMALLOC": "debug", "PYTHONFAULTHANDLER": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(_CHILD),
            dlpack_utils.__file__,
            mode,
            api,
            dtype,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert "completed" in result.stdout


@pytest.mark.parametrize("mode", ["ordinary", "slice", "shutdown"])
@pytest.mark.parametrize("api", ["strided", "contiguous"])
@pytest.mark.parametrize("dtype", ["uint8", "uint64"])
def test_external_memory_survives_view_deletion(mode, api, dtype):
    _run_child(mode, api, dtype)


def test_external_memory_preserves_geometry_and_dtype():
    _run_child("geometry")
