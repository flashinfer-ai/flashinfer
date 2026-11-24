"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Public-API tests of ``flashinfer.prefill.prepare_nvfp4_attention`` and
``flashinfer.prefill.nvfp4_attention_workspace_bytes``, the ``backend="cake"``
entries of the experimental NVFP4 attention package; the package's own suite
is ``test_cake_nvfp4_attention.py``.
"""

import pytest
import torch

from flashinfer.prefill import nvfp4_attention_workspace_bytes, prepare_nvfp4_attention
from tests.cake.test_helpers.cake_package import cake_package

SUPPORTED = {(10, 0), (10, 3), (12, 0)}


def _skip_unless_supported():
    if (
        not torch.cuda.is_available()
        or tuple(torch.cuda.get_device_capability()) not in SUPPORTED
    ):
        pytest.skip("an SM100, SM103 or SM120 GPU is required")
    from flashinfer.jit.cpp_ext import get_cuda_version

    cake_build = cake_package("nvfp4_attention", "cake_build")
    loader = cake_package("nvfp4_attention", "cake_jit")
    arch = loader.arch_for_capability(tuple(torch.cuda.get_device_capability()))
    version = get_cuda_version()
    local = (int(version.major), int(version.minor))
    for name, entry in cake_build.libraries_by_name().items():
        floor = tuple(map(int, entry["min_cuda"].split(".")))
        if (
            entry["arch"] == arch
            and local < floor
            and not cake_build.prebuilt_path(name).is_file()
        ):
            # The sm100_sm103 quantizer needs CUDA 13.0: ptxas 12.9 miscompiles
            # its E2M1 decode.
            pytest.skip(f"{name} needs CUDA {entry['min_cuda']} or newer")


def _reference(q, k, v):
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            scores = (q[b, h].float() @ k[b, h].float().T) * (128**-0.5)
            out[b, h] = torch.softmax(scores, dim=-1) @ v[b, h].float()
    return out


def test_nvfp4_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        prepare_nvfp4_attention(None, None, None, None, backend="unknown")


def test_nvfp4_attention_public_entry_runs_the_cake_route():
    _skip_unless_supported()
    torch.manual_seed(5)
    q = torch.randn((2, 5, 1100, 128), dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = torch.empty_like(q)
    attention = prepare_nvfp4_attention(q, k, v, out, backend="cake")
    assert attention() is out
    torch.testing.assert_close(out.float(), _reference(q, k, v), atol=1.0, rtol=0.1)
    assert attention.workspace_bytes == nvfp4_attention_workspace_bytes(
        2, 5, 1100, q.device
    )
    assert attention.route_metadata["records"]


def test_nvfp4_attention_public_entry_takes_a_pooled_workspace_and_a_scale():
    _skip_unless_supported()
    torch.manual_seed(6)
    q = torch.randn((1, 5, 257, 128), dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = torch.empty_like(q)
    pool = torch.empty(
        nvfp4_attention_workspace_bytes(1, 5, 257, q.device),
        dtype=torch.uint8,
        device="cuda",
    )
    attention = prepare_nvfp4_attention(
        q, k, v, out, backend="cake", softmax_scale=0.1, workspace=pool
    )
    assert attention.workspace is pool
    assert attention() is out and torch.isfinite(out).all()
