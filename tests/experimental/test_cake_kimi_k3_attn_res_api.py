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

flashinfer public-API tests of the Cake Kimi-K3 AttnRes package: the
``flashinfer.kimi_k3_attn_res`` facade.  Its entry points are pass-throughs to
``flashinfer.experimental.cake_kimi_k3_attn_res.cake_backend`` that carry the
experimental marking and accept ``backend="cake"`` only; this module checks that
identity (the same plan and the same bits as the backend, the caller's ``out``
returned, the ``backend`` keyword) and leaves the numerical contract to
``test_cake_kimi_k3_attn_res.py``, which imports only the backend package.
"""

import pytest
import torch

import flashinfer.kimi_k3_attn_res as facade
from flashinfer.experimental.cake_kimi_k3_attn_res import cake_backend as backend


def _device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    capability = tuple(torch.cuda.get_device_capability(device))
    if backend.SUPPORTED_COMPUTE_CAPABILITIES.get(capability) is None:
        pytest.skip("SM100 / SM103 required")
    return device


def _make_inputs(M, K, *, seed=31000):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = backend.HIDDEN_SIZE
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    return dict(
        prefix=torch.empty((M, hidden), **bf16).uniform_(-1.0, 1.0, generator=g),
        delta=torch.empty((M, hidden), **bf16).uniform_(
            -0.015625, 0.015625, generator=g
        ),
        blocks=torch.empty((M, backend.MAX_BLOCKS, hidden), **bf16).uniform_(
            -1.0, 1.0, generator=g
        ),
        norm_weight=torch.empty(hidden, **bf16).normal_(1.0, 0.05, generator=g),
        qk_weight=torch.empty(hidden, **bf16).normal_(0.0, 0.02, generator=g),
        output_norm_weight=torch.empty(hidden, **bf16).normal_(1.0, 0.05, generator=g),
        out=torch.full((M, hidden), float("nan"), **bf16),
    )


def _clone(inputs):
    return {k: v.clone() for k, v in inputs.items()}


def _assert_same_bits(inputs, mirror):
    for name in ("out", "prefix", "blocks"):
        assert torch.equal(
            inputs[name].view(torch.int16), mirror[name].view(torch.int16)
        ), name


def test_facade_rejects_other_backends():
    _device()
    inputs = _make_inputs(2, 4)
    with pytest.raises(ValueError, match="backend='cake'"):
        facade.prepare_kimi_k3_attn_res(**inputs, num_blocks=4, backend="triton")
    with pytest.raises(ValueError, match="backend='cake'"):
        facade.kimi_k3_attn_res(**inputs, num_blocks=4, backend="triton")


def test_facade_prepare_passes_through_to_the_backend():
    device = _device()
    M, K = 64, 4
    if not backend.generated_program_available(device, M, K):
        pytest.skip("program not registered for this cell")
    assert facade.generated_program_available(
        device, M, K
    ) == backend.generated_program_available(device, M, K)
    inputs = _make_inputs(M, K)
    mirror = _clone(inputs)
    runner = facade.prepare_kimi_k3_attn_res(**inputs, num_blocks=K)
    expected = backend.prepare_kimi_k3_attn_res(**mirror, num_blocks=K)
    assert type(runner) is backend.KimiK3AttnResRunner
    assert runner.plan == expected.plan
    assert runner.launch() is inputs["out"]
    expected.launch()
    torch.cuda.synchronize()
    _assert_same_bits(inputs, mirror)


@pytest.mark.parametrize("pdl", [False, True])
def test_facade_one_shot_call_matches_the_backend(pdl):
    device = _device()
    M, K = 64, 4
    if not backend.generated_program_available(device, M, K, enable_pdl=pdl):
        pytest.skip("program not registered for this cell")
    inputs = _make_inputs(M, K, seed=31001)
    mirror = _clone(inputs)
    out = facade.kimi_k3_attn_res(**inputs, num_blocks=K, enable_pdl=pdl)
    assert out is inputs["out"]
    backend.kimi_k3_attn_res(**mirror, num_blocks=K, enable_pdl=pdl)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    _assert_same_bits(inputs, mirror)
