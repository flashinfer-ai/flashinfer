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

Tests of the flashinfer public surface of the Cake four-stream mHC training
kernels (SM100 / SM103 / SM107): the ``flashinfer.cake_mhc_train`` facade and
its re-exports from the ``flashinfer`` top-level package.  The facade's entry
points are pass-throughs to ``flashinfer.cake_mhc_train_backend``; this module
checks that identity (same objects at the top level, same bits as the backend,
documented keyword arguments) and leaves the numerical contract to
``test_cake_mhc_train.py``, which imports only the backend package.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import flashinfer
import flashinfer.cake_mhc_train_backend as backend
from flashinfer import cake_mhc_train as facade
from flashinfer.cake_mhc_train_backend import cake_jit as loader
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
ENTRY_POINTS = (
    "cake_mhc_train_forward_prepare",
    "cake_mhc_train_forward_stats",
    "cake_mhc_train_forward_finish",
    "cake_mhc_train_backward_prepare",
    "cake_mhc_train_backward_finish",
    "cake_mhc_train_prepare",
    "cake_mhc_train_finish",
)
HELPERS = (
    "cake_mhc_train_forward_workspace_bytes",
    "cake_mhc_train_backward_workspace_bytes",
    "CakeMHCTrainPrepareFunction",
    "CakeMHCTrainFinishFunction",
)
SEED = 2032


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in SUPPORTED_CAPABILITIES
        and facade.is_cake_mhc_train_supported(_device())
    )


def test_top_level_exports_are_the_facade_objects() -> None:
    for name in ENTRY_POINTS + HELPERS:
        assert getattr(flashinfer, name) is getattr(facade, name), name
    for name in HELPERS:
        assert getattr(facade, name) is getattr(backend, name), name
    assert facade.SUPPORTED_COMPUTE_CAPABILITIES == SUPPORTED_CAPABILITIES


def test_entry_points_document_every_keyword() -> None:
    """Every facade entry point is a real ``def`` with a NumPy-style docstring naming each parameter, and its
    keyword-only switches carry defaults (every behaviour switch is a keyword argument with a documented default)."""
    for name in ENTRY_POINTS:
        function = getattr(facade, name)
        doc = inspect.getdoc(function) or ""
        assert "Parameters" in doc and "Returns" in doc, name
        signature = inspect.signature(function)
        backend_signature = inspect.signature(getattr(backend, name))
        assert list(signature.parameters) == list(backend_signature.parameters), name
        for parameter in signature.parameters.values():
            assert parameter.name in doc, (name, parameter.name)
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                assert parameter.default is not inspect.Parameter.empty, (
                    name,
                    parameter.name,
                )
    for name in ("rms_eps", "hc_eps", "sinkhorn_iters"):
        parameter = inspect.signature(facade.cake_mhc_train_forward_prepare).parameters[
            name
        ]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


pytestmark_gpu = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(get_compute_capability(_device()))
        flashinfer.jit.build_jit_specs(loader.specs_for_arch(arch), verbose=False)
    yield


def _inputs(T: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=_device()).manual_seed(SEED)

    def randn(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(
            *shape, generator=generator, device=_device(), dtype=torch.float32
        ).to(dtype)

    return {
        "X": randn(T, 4, 4096, dtype=torch.bfloat16),
        "fn": (randn(24, 16384) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3),
        "base": 0.1 * randn(24),
        "B": randn(T, 4096, dtype=torch.bfloat16),
        "dY": randn(T, 4, 4096, dtype=torch.bfloat16),
        "dU": randn(T, 4096, dtype=torch.bfloat16),
    }


@pytestmark_gpu
@pytest.mark.parametrize("T", (129,))
def test_facade_entry_points_match_the_backend(T: int) -> None:
    """The facade forwards every argument: each entry point returns the same bits as the backend package's."""
    i = _inputs(T)
    U, stats = facade.cake_mhc_train_forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"]
    )
    U_b, stats_b = backend.cake_mhc_train_forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"]
    )
    assert torch.equal(U, U_b) and torch.equal(stats, stats_b)
    assert torch.equal(
        facade.cake_mhc_train_forward_stats(i["X"], i["fn"], i["scale"], i["base"]),
        stats,
    )
    Y = facade.cake_mhc_train_forward_finish(i["X"], i["B"], stats)
    assert torch.equal(Y, backend.cake_mhc_train_forward_finish(i["X"], i["B"], stats))
    dB, gstats = facade.cake_mhc_train_backward_prepare(i["dY"], i["B"], i["X"], stats)
    dB_b, gstats_b = backend.cake_mhc_train_backward_prepare(
        i["dY"], i["B"], i["X"], stats
    )
    assert torch.equal(dB, dB_b) and torch.equal(gstats, gstats_b)
    args = (
        i["X"],
        i["dY"],
        i["dU"],
        i["B"],
        i["fn"],
        i["scale"],
        i["base"],
        stats,
        gstats,
    )
    outputs = facade.cake_mhc_train_backward_finish(*args)
    outputs_b = backend.cake_mhc_train_backward_finish(*args)
    assert all(torch.equal(a, b) for a, b in zip(outputs, outputs_b, strict=False))
    assert facade.cake_mhc_train_forward_workspace_bytes(
        T, _device()
    ) == backend.cake_mhc_train_forward_workspace_bytes(T, _device())


@pytestmark_gpu
def test_autograd_wrappers_match_the_backend() -> None:
    T = 65
    i = _inputs(T)
    grads = []
    for module in (facade, backend):
        X = i["X"].clone().requires_grad_(True)
        fn = i["fn"].clone().requires_grad_(True)
        scale = i["scale"].clone().requires_grad_(True)
        base = i["base"].clone().requires_grad_(True)
        U, handle = module.cake_mhc_train_prepare(X, fn, scale, base)
        B = (U.float() + i["B"].float()).to(torch.bfloat16)
        Y = module.cake_mhc_train_finish(X, B, handle)
        (Y.float() * i["dY"].float()).sum().backward()
        grads.append(
            (X.grad.clone(), fn.grad.clone(), scale.grad.clone(), base.grad.clone())
        )
    assert all(torch.equal(a, b) for a, b in zip(*grads, strict=False))
