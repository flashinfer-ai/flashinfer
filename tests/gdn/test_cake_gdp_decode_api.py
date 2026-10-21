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

Tests of the flashinfer public surface of the Cake Gated DeltaProduct (GDP)
decode kernels for Nemotron 4 (SM100 / SM103): the ``flashinfer.cake_gdp``
facade and its re-export from the ``flashinfer`` top-level package.  The
facade's entry point is a pass-through to ``flashinfer.cake_gdp_backend``;
this module checks that identity (same objects at the top level, the
``gated_delta_rule_mtp`` parameter list, same bits as the backend) and leaves
the numerical contract to ``test_cake_gdp_decode.py``, which imports only the
backend package.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import flashinfer
import flashinfer.cake_gdp_backend as backend
from flashinfer.cake_gdp import cake_gated_delta_product_mtp, is_cake_gdp_supported
from flashinfer.cake_gdp_backend import cake_jit as loader
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
K = 128
V = 64
SEED = 760
PARAMETERS = (
    "q",
    "k",
    "v",
    "initial_state",
    "initial_state_indices",
    "A_log",
    "a",
    "dt_bias",
    "b",
    "scale",
    "output",
    "ssm_state_indices",
    "disable_state_update",
    "use_qk_l2norm",
    "output_state_indices",
)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in SUPPORTED_CAPABILITIES
        and is_cake_gdp_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_gdp needs an SM100 / SM103 device with exported routes",
)


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(get_compute_capability(_device()))
        flashinfer.jit.build_jit_specs(loader.specs_for_arch(arch), verbose=False)
    yield


def test_function_is_exported() -> None:
    assert flashinfer.cake_gated_delta_product_mtp is cake_gated_delta_product_mtp
    assert flashinfer.cake_gdp.is_cake_gdp_supported is backend.is_cake_gdp_supported
    assert flashinfer.cake_gdp.NUM_HOUSEHOLDER == (1, 2, 3, 4)
    assert (flashinfer.cake_gdp.HEAD_SIZE_K, flashinfer.cake_gdp.HEAD_SIZE_V) == (K, V)


def test_signature_follows_gated_delta_rule_mtp() -> None:
    """The facade takes the parameters of ``flashinfer.gated_delta_rule_mtp`` in its order and with its
    defaults (``k`` / ``v`` / ``b`` carry the Householder axis)."""
    signature = inspect.signature(cake_gated_delta_product_mtp)
    assert tuple(signature.parameters) == PARAMETERS
    defaults = {
        name: p.default
        for name, p in signature.parameters.items()
        if p.default is not p.empty
    }
    assert defaults == {
        "scale": None,
        "output": None,
        "ssm_state_indices": None,
        "disable_state_update": None,
        "use_qk_l2norm": True,
        "output_state_indices": None,
    }
    assert inspect.signature(backend.cake_gated_delta_product_mtp) == signature


def _inputs(n: int, T: int, generator: torch.Generator) -> dict:
    device = _device()
    B, H = 4, 6

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, device=device, generator=generator, dtype=torch.float32
        )

    return dict(
        q=randn(B, T, H, K).to(torch.bfloat16),
        k=randn(B, T, n, H, K).to(torch.bfloat16),
        v=randn(B, T, n, H, V).to(torch.bfloat16),
        initial_state=randn(9, H, V, K).mul_(0.1),
        initial_state_indices=torch.tensor(
            (7, 2, -1, 4), device=device, dtype=torch.int32
        ),
        A_log=randn(H).mul_(0.1),
        a=randn(B, T, H).mul_(0.1).to(torch.bfloat16),
        dt_bias=randn(H).mul_(0.1),
        b=randn(B, T, n, H).mul_(0.1).to(torch.bfloat16),
    )


@pytest.mark.parametrize("n,T", ((1, 1), (3, 4), (4, 2)))
def test_facade_matches_the_backend(n: int, T: int) -> None:
    """The facade forwards every argument: output and pool bits equal the backend's for the same inputs."""
    if not loader.route_applies(
        device_capability=get_compute_capability(_device()),
        num_householder=n,
        num_tokens=T,
    ):
        pytest.skip(
            f"num_householder={n}, T={T} has no exported program on this device"
        )
    inputs = _inputs(
        n, T, torch.Generator(device=_device()).manual_seed(SEED + 10 * n + T)
    )
    pool_before = inputs["initial_state"].clone()
    write = torch.tensor((1, 3, 5, 6), device=_device(), dtype=torch.int32)
    out, pool = cake_gated_delta_product_mtp(**inputs, output_state_indices=write)
    assert pool is inputs["initial_state"]
    assert out.shape == (4, T, 6, V) and out.dtype == torch.bfloat16
    assert torch.equal(out[2], torch.zeros_like(out[2]))
    pool_facade = pool.clone()
    inputs["initial_state"].copy_(pool_before)
    out_backend, _ = backend.cake_gated_delta_product_mtp(
        **inputs, output_state_indices=write
    )
    assert torch.equal(out, out_backend)
    assert torch.equal(pool_facade, inputs["initial_state"])
    assert torch.equal(pool_facade[7], pool_before[7]) and torch.equal(
        pool_facade[5], pool_before[5]
    )


def test_unsupported_device_is_reported() -> None:
    assert is_cake_gdp_supported(torch.device("cpu")) is False
    inputs = _inputs(3, 1, torch.Generator(device=_device()).manual_seed(SEED))
    cpu = {name: value.cpu() for name, value in inputs.items()}
    with pytest.raises(ValueError):
        cake_gated_delta_product_mtp(**cpu)
