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

Tests of the flashinfer public surface of the Cake fused BF16 RMSNorm
training kernels (SM100 / SM103 / SM107): the ``flashinfer.cake_rmsnorm_det_train``
facade and its re-export from the ``flashinfer`` top-level package.  The
facade's three entry points are pass-throughs to
``flashinfer.cake_rmsnorm_det_train_backend``; this module checks that identity
(same objects at the top level, same bits as the backend) and leaves the
numerical contract to ``test_cake_rmsnorm_det_train.py``, which imports only the
backend package.
"""

from __future__ import annotations

import re

import pytest
import torch

import flashinfer
import flashinfer.cake_rmsnorm_det_train_backend as backend
from flashinfer.cake_rmsnorm_det_train import (
    CakeRMSNormDetFunction,
    cake_rmsnorm_det,
    cake_rmsnorm_det_train_backward,
    cake_rmsnorm_det_train_backward_workspace_bytes,
    cake_rmsnorm_det_train_forward,
    is_cake_rmsnorm_det_train_supported,
    program_windows,
    weight_dtypes,
)
from flashinfer.cake_rmsnorm_det_train_backend import cake_jit as loader
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTHS = (
    (6144, 1e-5),
    (5120, 1e-5),
    (4096, 1e-5),
    (2048, 1e-6),
    (1536, 1e-6),
    (512, 1e-6),
    (256, 1e-6),
)
HIDDEN_SIZES = tuple(hidden for hidden, _eps in WIDTHS)
EPS = {hidden: eps for hidden, eps in WIDTHS}
SEED = 760


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in SUPPORTED_CAPABILITIES
        and is_cake_rmsnorm_det_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_det_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(get_compute_capability(_device()))
        flashinfer.jit.build_jit_specs(loader.specs_for_arch(arch), verbose=False)
    yield


def _skip_unless_hidden(hidden: int) -> None:
    if not is_cake_rmsnorm_det_train_supported(_device(), hidden):
        pytest.skip(f"hidden={hidden} has no exported route on this device")


def _randn(*shape: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        *shape, dtype=torch.float32, device=_device(), generator=generator
    ).to(torch.bfloat16)


def test_function_signature_is_exported() -> None:
    assert flashinfer.cake_rmsnorm_det is cake_rmsnorm_det
    assert flashinfer.cake_rmsnorm_det_train_forward is cake_rmsnorm_det_train_forward
    assert flashinfer.cake_rmsnorm_det_train_backward is cake_rmsnorm_det_train_backward
    assert (
        flashinfer.cake_rmsnorm_det_train_backward_workspace_bytes
        is cake_rmsnorm_det_train_backward_workspace_bytes
    )
    assert flashinfer.CakeRMSNormDetFunction is CakeRMSNormDetFunction
    assert program_windows is backend.program_windows is loader.program_windows
    assert weight_dtypes is backend.weight_dtypes is loader.weight_dtypes


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_facade_entry_points_match_the_backend(hidden: int) -> None:
    """The facade forwards every argument: its forward, backward and autograd
    entry points return the same bits as the backend package's."""
    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    generator = torch.Generator(device=_device()).manual_seed(SEED + hidden)
    x = _randn(129, hidden, generator=generator)
    g = _randn(129, hidden, generator=generator)
    w = _randn(hidden, generator=generator)
    y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
    y_backend, rstd_backend = backend.cake_rmsnorm_det_train_forward(x, w, eps)
    assert torch.equal(y, y_backend) and torch.equal(rstd, rstd_backend)
    dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd)
    dx_backend, dw_backend = backend.cake_rmsnorm_det_train_backward(g, x, w, rstd)
    assert torch.equal(dx, dx_backend) and torch.equal(dw, dw_backend)
    assert torch.equal(cake_rmsnorm_det(x, w, eps), backend.cake_rmsnorm_det(x, w, eps))


# "Beside kernels resident on other streams" in the facade docstring lists, per
# architecture, the calls whose kernels need a whole SM.  A list item names an
# architecture set ("SM100 / SM103:") and the calls: "the residual-fused H <h>
# backward" (every token count) or "the forward / backward of H <h> above <t>
# tokens" (every program window above ``t``).
_WHOLE_SM_ITEM = re.compile(
    r"the residual-fused H (\d+) backward|the (forward|backward) of H (\d+) above (\d+) tokens"
)


def _documented_whole_sm() -> dict[str, set[tuple[str, int, bool, int]]]:
    """``arch -> {(kind, hidden, residual, above)}`` from the facade docstring."""
    import flashinfer.cake_rmsnorm_det_train as facade

    out: dict[str, set[tuple[str, int, bool, int]]] = {}
    for item in re.split(r"\n- ", facade.__doc__ or "")[1:]:
        text = " ".join(item.split("\n\n")[0].split())
        head = re.match(r"((?:SM1\d\d)(?: / SM1\d\d)*):", text)
        if head is None:
            continue
        for sm in re.findall(r"SM1\d\d", head.group(1)):
            entries = out.setdefault(f"sm_{sm[2:]}a", set())
            for residual_hidden, kind, hidden, above in _WHOLE_SM_ITEM.findall(text):
                if residual_hidden:
                    entries.add(("bwd", int(residual_hidden), True, 0))
                else:
                    entries.add(
                        (
                            "fwd" if kind == "forward" else "bwd",
                            int(hidden),
                            False,
                            int(above),
                        )
                    )
    return out


def test_documented_whole_sm_calls_match_the_program_table() -> None:
    """The facade's whole-SM lists name every architecture of the package, each
    listed call names a shipped route of that architecture whose token
    threshold is an edge of the route's program windows (with BF16 and FP32
    weights alike), and the lists are the stall test's ``WHOLE_SM`` table, which
    checks them on the device."""
    from tests.norm.test_cake_rmsnorm_det_train_stall import WHOLE_SM

    documented = _documented_whole_sm()
    assert set(documented) == set(loader.ARCHES)
    assert all(documented.values())
    for arch, entries in documented.items():
        assert entries == set(WHOLE_SM[arch]), arch
        for kind, hidden, residual, above in entries:
            dtypes = loader.weight_dtypes(arch, hidden, kind, residual)
            assert dtypes, (arch, kind, hidden, residual)
            for w_dtype in dtypes:
                edges = {
                    low
                    for low, _high, _library in program_windows(
                        arch, hidden, kind, residual, w_dtype
                    )
                }
                assert above in edges, (
                    arch,
                    kind,
                    hidden,
                    residual,
                    w_dtype,
                    above,
                    edges,
                )
