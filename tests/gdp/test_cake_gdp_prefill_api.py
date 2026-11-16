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
prefill kernel (SM90): the ``backend="cake"`` route of
:func:`flashinfer.gdp_prefill.chunk_gated_delta_product`.  The route is a
pass-through to ``flashinfer.cake_gdp_backend.cake_chunk_gated_delta_product``;
this module checks the dispatch (same bits as the backend, every argument
forwarded, the backend's errors surfaced, the refusal on a device without an
exported route naming what is exported), the ``backend`` validation (the
message names ``"cake"``), that ``"auto"`` still dispatches to cuDNN and the
parameter list; the numerical contract is ``test_cake_gdp_prefill.py``, which
imports only the backend package.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import flashinfer
import flashinfer.cake_gdp_backend as backend
import flashinfer.gdp_prefill as gdp_prefill
from flashinfer.cake_gdp_backend import cake_jit as loader
from flashinfer.gdp_prefill import chunk_gated_delta_product
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability

SUPPORTED_CAPABILITIES = ((9, 0),)
K = 128
V = 64
SEED = 760
PARAMETERS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "num_householder",
    "scale",
    "initial_state",
    "output_final_state",
    "cu_seqlens",
    "use_qk_l2norm_in_kernel",
    "output",
    "output_state",
    "backend",
    "state_indices",
    "cu_seqlens_cpu",
)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in SUPPORTED_CAPABILITIES
        and backend.is_cake_gdp_prefill_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_gdp prefill needs an SM90 device with exported routes",
)


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(get_compute_capability(_device()))
        prefill = {
            loader.library_name(name, arch)
            for name, record in loader.MODULES.items()
            if record["role"] == "prefill" and arch in record["arches"]
        }
        prefixes = tuple(f"{library}_" for library in prefill)
        specs = [
            spec
            for spec in loader.specs_for_arch(arch)
            if any(spec.name[: len(p)] == p for p in prefixes)
        ]
        flashinfer.jit.build_jit_specs(specs, verbose=False)
    yield


def test_signature_adds_the_cake_backend() -> None:
    """The public function keeps its parameter list and order; ``backend`` admits ``"cake"``; ``cu_seqlens_cpu`` is
    keyword-only, optional and a tensor."""
    signature = inspect.signature(chunk_gated_delta_product)
    assert tuple(signature.parameters) == PARAMETERS
    assert signature.parameters["backend"].default == "auto"
    assert signature.parameters["backend"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "cake" in str(signature.parameters["backend"].annotation)
    host = signature.parameters["cu_seqlens_cpu"]
    assert host.kind is inspect.Parameter.KEYWORD_ONLY and host.default is None
    assert "Tensor" in str(host.annotation)
    backend_signature = inspect.signature(backend.cake_chunk_gated_delta_product)
    assert tuple(backend_signature.parameters) == tuple(
        p for p in PARAMETERS if p != "backend"
    )


def _inputs(n: int, lens: tuple[int, ...], H: int, generator: torch.Generator) -> dict:
    device = _device()
    T = sum(lens)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, device=device, generator=generator, dtype=torch.float32
        )

    offsets = [0]
    for length in lens:
        offsets.append(offsets[-1] + length)
    return dict(
        q=torch.nn.functional.normalize(randn(T, H, K), dim=-1).to(torch.bfloat16),
        k=torch.nn.functional.normalize(randn(T * n, H, K), dim=-1).to(torch.bfloat16),
        v=randn(T * n, H, V).to(torch.bfloat16),
        g=1.0 - 0.5 * torch.rand(T, H, device=device, generator=generator),
        beta=torch.rand(T * n, H, device=device, generator=generator),
        num_householder=n,
        initial_state=randn(len(lens), H, V, K),
        output_final_state=True,
        cu_seqlens=torch.tensor(offsets, device=device, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "n,lens,H", ((1, (129, 65), 3), (3, (300, 0, 1, 129), 6), (4, (64,), 2))
)
def test_cake_backend_matches_the_package(
    n: int, lens: tuple[int, ...], H: int
) -> None:
    """``backend="cake"`` forwards every argument: output and final-state bits equal the package entry's."""
    inputs = _inputs(
        n, lens, H, torch.Generator(device=_device()).manual_seed(SEED + n)
    )
    out, state = chunk_gated_delta_product(**inputs, backend="cake")
    out_pkg, state_pkg = backend.cake_chunk_gated_delta_product(**inputs)
    assert out.shape == (sum(lens), H, V) and out.dtype == torch.bfloat16
    assert torch.equal(out, out_pkg) and torch.equal(state, state_pkg)
    only = chunk_gated_delta_product(
        **{**inputs, "output_final_state": False}, backend="cake"
    )
    assert isinstance(only, torch.Tensor) and torch.equal(only, out)


def test_cake_backend_forwards_the_pool_and_the_scalars(monkeypatch) -> None:
    """``state_indices`` and every scalar reach the package entry unchanged (recorded through a stub)."""
    recorded = {}

    def stub(q, k, v, g, beta, num_householder, scale, **kwargs):
        recorded.update(kwargs, num_householder=num_householder, scale=scale)
        return torch.zeros((q.shape[0], q.shape[1], V), device=q.device, dtype=q.dtype)

    monkeypatch.setattr(backend, "cake_chunk_gated_delta_product", stub)
    inputs = _inputs(
        2, (129, 65), 3, torch.Generator(device=_device()).manual_seed(SEED)
    )
    slots = torch.tensor([1, 0], device=_device(), dtype=torch.int32)
    pool = torch.zeros((4, 3, V, K), device=_device())
    out = chunk_gated_delta_product(
        **{
            **inputs,
            "initial_state": pool,
            "output_final_state": False,
            "scale": 0.5,
            "use_qk_l2norm_in_kernel": True,
        },
        backend="cake",
        state_indices=slots,
    )
    assert out.shape == (194, 3, V)
    assert recorded["state_indices"] is slots and recorded["initial_state"] is pool
    assert recorded["scale"] == 0.5 and recorded["use_qk_l2norm_in_kernel"] is True
    assert recorded["num_householder"] == 2 and recorded["output_final_state"] is False
    assert recorded["cu_seqlens"] is inputs["cu_seqlens"]


def test_unit_gate_calls_dispatch_to_the_unit_programs(monkeypatch) -> None:
    """``g=None`` launches the unit-gate program and ``beta=None`` the unit-beta program (the unread formal is a
    one-row placeholder), from a zero state; with an ``initial_state`` the call is refused (no SM90 program reads
    one with unit gates), and ``g=None`` together with ``beta=None`` is refused."""
    launched = []

    def record(name, values, **kwargs):
        launched.append((name, dict(values)))

    monkeypatch.setattr(loader, "run_module", record)
    inputs = _inputs(
        3, (129, 65), 2, torch.Generator(device=_device()).manual_seed(SEED)
    )
    arch = loader.arch_for_capability(torch.cuda.get_device_capability(_device()))
    T, n = inputs["q"].shape[0], 3
    for g, beta, flags in (
        (None, inputs["beta"], {"unit_gates": True, "unit_beta": False}),
        (inputs["g"], None, {"unit_gates": False, "unit_beta": True}),
    ):
        launched.clear()
        out = chunk_gated_delta_product(
            **{
                **inputs,
                "g": g,
                "beta": beta,
                "initial_state": None,
                "output_final_state": False,
            },
            backend="cake",
        )
        assert out.shape == (T, 2, V) and len(launched) == 1
        name, values = launched[0]
        build = loader.MODULES[name]["build"]
        assert {key: build[key] for key in flags} == flags
        assert (
            not build["use_initial_state"]
            and not build["use_state_indices"]
            and build["store_final_state"]
        )
        assert name == loader.selected_prefill_module(
            arch,
            use_initial_state=False,
            use_state_indices=False,
            store_final_state=True,
            **flags,
        )
        assert tuple(values["gate"].shape) == ((1, 2) if g is None else (T, 2))
        assert tuple(values["beta"].shape) == ((1, 2) if beta is None else (T * n, 2))
        launched.clear()
        with pytest.raises(
            ValueError,
            match="no exported Cake GDP prefill program on sm_90a for this call",
        ):
            chunk_gated_delta_product(
                **{**inputs, "g": g, "beta": beta}, backend="cake"
            )
        assert not launched
    with pytest.raises(
        ValueError,
        match="no exported Cake GDP prefill program for g=None with beta=None",
    ):
        chunk_gated_delta_product(**{**inputs, "g": None, "beta": None}, backend="cake")
    assert not launched


def test_cu_seqlens_cpu_reaches_the_cake_backend(monkeypatch) -> None:
    """``cu_seqlens_cpu`` (the host copy of ``cu_seqlens``) is forwarded to the Cake backend, where the body route
    and the split rule read it: a merged-class batch (two sequences of 384 chunks over ten heads, the pinned row of
    the route) selects the split program of the merged body; without it the backend synchronizes the copy and
    selects the same program; a device tensor is refused."""
    inputs = _inputs(
        3, (8192, 8192), 10, torch.Generator(device=_device()).manual_seed(SEED)
    )
    launched = []
    monkeypatch.setattr(
        loader, "run_module", lambda name, values, **kwargs: launched.append(name)
    )
    host = inputs["cu_seqlens"].cpu()
    chunk_gated_delta_product(**inputs, backend="cake", cu_seqlens_cpu=host)
    assert len(launched) == 1
    assert loader.MODULES[launched[0]]["body"] == loader.ROUTE_BODY_P3
    assert loader.MODULES[launched[0]]["build"].get(loader.PREFILL_SPLIT_FLAG) is True
    chunk_gated_delta_product(**inputs, backend="cake")
    assert launched == [launched[0]] * 2
    with pytest.raises(
        ValueError, match="cu_seqlens_cpu must be a CPU int32/int64 1-D tensor"
    ):
        chunk_gated_delta_product(
            **inputs, backend="cake", cu_seqlens_cpu=inputs["cu_seqlens"]
        )
    assert len(launched) == 2


@pytest.mark.parametrize("lens", ((4097, 1000), (129, 65)))
def test_results_without_cu_seqlens_cpu_equal_results_with_it(
    lens: tuple[int, ...],
) -> None:
    """A call without ``cu_seqlens_cpu`` (the backend synchronizes the host copy) returns the bits of the call with
    it, int32 or int64, on a batch the split rule cuts and on one it leaves whole."""
    inputs = _inputs(3, lens, 3, torch.Generator(device=_device()).manual_seed(SEED))
    out, state = chunk_gated_delta_product(**inputs, backend="cake")
    host = inputs["cu_seqlens"].cpu()
    for cu_seqlens_cpu in (host, host.to(torch.int64)):
        out_host, state_host = chunk_gated_delta_product(
            **inputs, backend="cake", cu_seqlens_cpu=cu_seqlens_cpu
        )
        assert torch.equal(out_host, out) and torch.equal(state_host, state)


def test_cudnn_backends_accept_and_ignore_cu_seqlens_cpu(monkeypatch) -> None:
    """``"auto"`` and ``"cudnn"`` take ``cu_seqlens_cpu`` and do not pass it on; the Cake package is not reached."""
    cudnn = pytest.importorskip("flashinfer.cudnn")
    calls = []

    def stub(q, k, v, g, beta, num_householder, scale, **kwargs):
        calls.append(kwargs)
        return torch.zeros((q.shape[0], q.shape[1], V), device=q.device, dtype=q.dtype)

    monkeypatch.setattr(cudnn, "cudnn_chunk_gated_delta_product", stub)
    monkeypatch.setattr(
        loader,
        "run_module",
        lambda name, values, **kwargs: pytest.fail(
            "the cake backend must not be reached"
        ),
    )
    inputs = _inputs(1, (64,), 2, torch.Generator(device=_device()).manual_seed(SEED))
    host = inputs["cu_seqlens"].cpu()
    for which in ("auto", "cudnn"):
        out = chunk_gated_delta_product(
            **{**inputs, "output_final_state": False},
            backend=which,
            cu_seqlens_cpu=host,
        )
        assert out.shape == (64, 2, V)
    assert len(calls) == 2 and all("cu_seqlens_cpu" not in kwargs for kwargs in calls)


def test_backend_value_is_validated() -> None:
    inputs = _inputs(1, (64,), 2, torch.Generator(device=_device()).manual_seed(SEED))
    with pytest.raises(ValueError, match='"auto", "cudnn" or "cake"') as info:
        chunk_gated_delta_product(**inputs, backend="cutlass")
    assert "cake" in str(info.value) and "'cutlass'" in str(info.value)
    with pytest.raises(ValueError, match="cu_seqlens is required"):
        chunk_gated_delta_product(**{**inputs, "cu_seqlens": None}, backend="cake")


def test_other_devices_are_refused_naming_the_exported_architectures(
    monkeypatch,
) -> None:
    """``backend="cake"`` on a device without an exported prefill route: ``ValueError`` names the capabilities and
    architectures the kernel is exported for (SM90) and nothing is launched."""
    inputs = _inputs(
        3, (129, 65), 2, torch.Generator(device=_device()).manual_seed(SEED)
    )
    launched = []
    monkeypatch.setattr(
        loader, "run_module", lambda name, values, **kwargs: launched.append(name)
    )
    for capability in ((10, 0), (10, 3), (12, 0), (8, 0)):
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda device=None, cap=capability: cap
        )
        assert not backend.is_cake_gdp_prefill_supported(_device())
        with pytest.raises(
            ValueError, match="no exported Cake GDP prefill program"
        ) as info:
            chunk_gated_delta_product(**inputs, backend="cake")
        assert (
            str(capability) in str(info.value)
            and "(9, 0)" in str(info.value)
            and "sm_90a" in str(info.value)
        )
    assert not launched


def test_auto_still_dispatches_to_cudnn(monkeypatch) -> None:
    """``"auto"`` and ``"cudnn"`` call cuDNN's engine as before; only ``"cake"`` reaches the Cake package."""
    cudnn = pytest.importorskip("flashinfer.cudnn")
    calls = []

    def stub(q, k, v, g, beta, num_householder, scale, **kwargs):
        calls.append(kwargs)
        return torch.zeros((q.shape[0], q.shape[1], V), device=q.device, dtype=q.dtype)

    monkeypatch.setattr(cudnn, "cudnn_chunk_gated_delta_product", stub)

    def cake_stub(*args, **kwargs):
        raise AssertionError("the cake backend must not be reached")

    monkeypatch.setattr(backend, "cake_chunk_gated_delta_product", cake_stub)
    inputs = _inputs(1, (64,), 2, torch.Generator(device=_device()).manual_seed(SEED))
    for which in ("auto", "cudnn"):
        out = chunk_gated_delta_product(
            **{**inputs, "output_final_state": False}, backend=which
        )
        assert out.shape == (64, 2, V)
    assert len(calls) == 2


def test_backend_errors_surface_through_the_route() -> None:
    inputs = _inputs(
        3, (129, 65), 3, torch.Generator(device=_device()).manual_seed(SEED)
    )
    with pytest.raises(ValueError, match="num_householder must be one of"):
        chunk_gated_delta_product(**{**inputs, "num_householder": 5}, backend="cake")
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        chunk_gated_delta_product(
            **{
                name: (value.cpu() if isinstance(value, torch.Tensor) else value)
                for name, value in inputs.items()
            },
            backend="cake",
        )


def test_docstring_names_the_cake_backend() -> None:
    doc = (
        chunk_gated_delta_product.__doc__
        or gdp_prefill.chunk_gated_delta_product.__wrapped__.__doc__
    )
    assert '"cake"' in doc and "SM90" in doc
