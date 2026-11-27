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
decode kernels (SM90 / SM100 / SM103 / SM107; FP32 or FP16 state pool): the
``flashinfer.cake_gdp`` facade and its re-export from the
``flashinfer`` top-level package.  The facade's entry point is a pass-through
to ``flashinfer.cake_gdp_backend``;
this module checks that identity (same objects at the top level, the
``gated_delta_rule_mtp`` parameter list and the trailing stochastic-rounding
parameters, same bits as the backend) and leaves
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

SUPPORTED_CAPABILITIES = ((9, 0), (10, 0), (10, 3), (10, 7))
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
    "rand_seed",
    "philox_rounds",
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
    reason="cake_gdp needs an SM90 / SM100 / SM103 / SM107 device with exported routes",
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
    defaults (``k`` / ``v`` / ``b`` carry the Householder axis), then the stochastic rounding of an FP16 pool
    (``rand_seed`` None: round to nearest even; ``philox_rounds`` 10, as FlashInfer's Mamba kernels)."""
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
        "rand_seed": None,
        "philox_rounds": 10,
    }
    assert inspect.signature(backend.cake_gated_delta_product_mtp) == signature
    supported = inspect.signature(is_cake_gdp_supported).parameters["state_dtype"]
    assert supported.kind is inspect.Parameter.KEYWORD_ONLY
    assert supported.default is torch.float32


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality of two 2-byte tensors of the same dtype and shape (a signed zero differs)."""
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.view(torch.int16), b.view(torch.int16))
    )


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


@pytest.mark.parametrize("seeded", (False, True))
def test_facade_matches_the_backend_fp16(seeded: bool) -> None:
    """Over an FP16 pool, to nearest or with stochastic rounding: the facade forwards the seed and the round count,
    so output and pool bits equal the backend's."""
    if not loader.route_applies(
        device_capability=get_compute_capability(_device()),
        num_householder=3,
        num_tokens=4,
        state_dtype=torch.float16,
    ):
        pytest.skip("no FP16-pool program on this device")
    inputs = _inputs(3, 4, torch.Generator(device=_device()).manual_seed(SEED + 7))
    inputs["initial_state"] = inputs["initial_state"].half()
    if seeded:
        inputs["rand_seed"] = torch.tensor(
            [-12345], dtype=torch.int64, device=_device()
        )
        inputs["philox_rounds"] = 7
    pool_before = inputs["initial_state"].clone()
    out, pool = cake_gated_delta_product_mtp(**inputs)
    assert pool is inputs["initial_state"] and pool.dtype == torch.float16
    pool_facade = pool.clone()
    inputs["initial_state"].copy_(pool_before)
    out_backend, _ = backend.cake_gated_delta_product_mtp(**inputs)
    assert _same_bits(out, out_backend)
    assert _same_bits(pool_facade, inputs["initial_state"])
    assert not torch.equal(pool_facade, pool_before)


# The Philox round count from which the runtime n = 3 snapshot_noise sibling serves a stochastic snapshot call
# below 3 x SM state heads (within (3 x SM, 4 x SM] it serves from 11 on SM100 / SM103 and from 16 on SM107, and
# above 124 x SM from 11 on SM107).  On SM90 it serves within (2 x SM, 3 x SM] only: the test's 24 state heads (at
# most 2 x SM) run the row-split runtime program of n = 3 in every launch class (None).
SNAPSHOT_NOISE_RUNTIME_ROUNDS = {
    "sm_90a": None,
    "sm_100a": 10,
    "sm_103a": 10,
    "sm_107a": 10,
}


@pytest.mark.parametrize(
    ("seeded", "rounds"), ((False, 0), (True, 5), (True, 9), (True, 10), (True, 16))
)
def test_facade_fp16_launch_class_picks_the_snapshot_noise_sibling(
    seeded: bool, rounds: int, monkeypatch
) -> None:
    """A facade call over an FP16 pool that stores per-token snapshots at T = 10 (the runtime program of three
    Householder updates) launches that program's ``snapshot_noise`` sibling when it rounds stochastically with at
    least the architecture's round count (at these state heads 10 on SM100 / SM103 / SM107), and the plain program
    below it (5 and 9 rounds) or when it rounds to nearest even; on SM90 these state heads launch the row-split
    runtime program (two CTAs per state head) in every launch class; output and pool bits equal the backend's."""
    capability = get_compute_capability(_device())
    if not loader.route_applies(
        device_capability=capability,
        num_householder=3,
        num_tokens=10,
        state_dtype=torch.float16,
    ):
        pytest.skip("no FP16-pool program on this device")
    edge = SNAPSHOT_NOISE_RUNTIME_ROUNDS[loader.arch_for_capability(capability)]
    launched: list[str] = []
    original = loader.run_module

    def recording(name, values, **kwargs):
        launched.append(name)
        return original(name, values, **kwargs)

    monkeypatch.setattr(loader, "run_module", recording)
    inputs = _inputs(3, 10, torch.Generator(device=_device()).manual_seed(SEED + 11))
    inputs["initial_state"] = inputs["initial_state"].half()
    ssm = torch.full((4, 10), -1, dtype=torch.int32, device=_device())
    ssm[:, -1] = torch.tensor((1, 3, 6, 5), dtype=torch.int32, device=_device())
    ssm[0, 4] = 8
    inputs["ssm_state_indices"] = ssm
    if seeded:
        inputs["rand_seed"] = torch.tensor(
            [-12345], dtype=torch.int64, device=_device()
        )
        inputs["philox_rounds"] = rounds
    pool_before = inputs["initial_state"].clone()
    out, pool = cake_gated_delta_product_mtp(**inputs)
    pool_facade = pool.clone()
    inputs["initial_state"].copy_(pool_before)
    out_backend, _ = backend.cake_gated_delta_product_mtp(**inputs)
    assert len(launched) == 2 and launched[0] == launched[1], launched
    record = loader.MODULES[launched[0]]
    sibling = bool(record["build"].get("snapshot_noise"))
    if edge is None:
        assert not sibling and record["grid_rule"]["ctas_per_state"] == 2, launched
        assert record["build"]["t_steps"] is None, launched
    else:
        assert sibling == (seeded and rounds >= edge), launched
    assert _same_bits(out, out_backend)
    assert _same_bits(pool_facade, inputs["initial_state"])
    assert not torch.equal(pool_facade[8], pool_before[8])


def test_unsupported_state_dtype_is_reported() -> None:
    """A state pool dtype outside float32 / float16 is refused by both predicates with ``False`` (no raise)."""
    assert is_cake_gdp_supported(_device(), state_dtype=torch.bfloat16) is False
    assert (
        loader.route_applies(
            device_capability=get_compute_capability(_device()),
            num_householder=3,
            num_tokens=1,
            state_dtype=torch.bfloat16,
        )
        is False
    )


def test_unsupported_device_is_reported() -> None:
    assert is_cake_gdp_supported(torch.device("cpu")) is False
    inputs = _inputs(3, 1, torch.Generator(device=_device()).manual_seed(SEED))
    cpu = {name: value.cpu() for name, value in inputs.items()}
    with pytest.raises(ValueError):
        cake_gated_delta_product_mtp(**cpu)
