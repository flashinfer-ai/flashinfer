# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from flashinfer.gemm.gemm_base import _check_mm_mxfp8_problem_size


def _mxfp8_metadata():
    def tensor(shape, dtype):
        return SimpleNamespace(
            shape=shape, ndim=len(shape), dtype=dtype, device=torch.device("cuda:1")
        )

    return [
        tensor((4, 128), torch.float8_e4m3fn),
        tensor((128, 128), torch.float8_e4m3fn),
        tensor((512,), torch.uint8),
        tensor((512,), torch.uint8),
        tensor((4, 128), torch.bfloat16),
    ]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("index", range(5))
def test_problem_size_rejects_misplaced_tensor(index, device):
    args = _mxfp8_metadata()
    args[index].device = torch.device(device)
    # When a alone moves to cuda:0, b is the first mismatch against a.
    message = ["b must", "b must", "a_descale must", "b_descale must", "Output device"]
    expected = "a must" if (index, device) == (0, "cpu") else message[index]
    with pytest.raises(ValueError, match=expected):
        _check_mm_mxfp8_problem_size(*args, use_8x4_sf_layout=False)


@pytest.mark.parametrize("provide_out", [False, True])
def test_problem_size_accepts_one_noncurrent_gpu(monkeypatch, provide_out):
    def unexpected_query():
        pytest.fail("operand validation queried the ambient CUDA device")

    monkeypatch.setattr(torch.cuda, "current_device", unexpected_query)
    args = _mxfp8_metadata()
    if not provide_out:
        args[-1] = None
    assert _check_mm_mxfp8_problem_size(*args, use_8x4_sf_layout=False)


@pytest.mark.parametrize("backend", ["auto", "specialized"])
def test_public_rejects_host_scale_before_routing(monkeypatch, backend):
    from flashinfer import mm_mxfp8
    from flashinfer.gemm import gemm_base as gemm

    def unexpected_route(*args, **kwargs):
        pytest.fail("invalid operands reached backend selection")

    monkeypatch.setattr(gemm, "specialized_mm_mxfp8_eligible", unexpected_route)
    monkeypatch.setattr(gemm, "get_compute_capability", unexpected_route)
    args = _mxfp8_metadata()
    args[3].device = torch.device("cpu")
    with pytest.raises(ValueError, match="b_descale must"):
        mm_mxfp8(*args[:4], out=args[4], backend=backend, use_8x4_sf_layout=False)
