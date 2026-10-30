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

Tests of the Cake mHC training head (``flashinfer.cake_mhc_det_train``): the stream
mean ``Y = X.mean(dim=1)`` of BF16 ``X`` ``[T, 4, H]`` and its gradient, bit
for bit against torch (signed zeros and NaN encodings included) at hidden
4096, 5120 and 7168, through the forward / backward entry points, the
accumulating backward, the autograd node, activation checkpointing and CUDA
graph capture.
"""

from __future__ import annotations

import pytest
import torch
import torch.utils.checkpoint

import flashinfer
from flashinfer import cake_mhc_det_train as facade
from flashinfer.utils import get_compute_capability

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
HIDDEN = (4096, 5120, 7168)
TOKENS = (1, 2, 3, 7, 65, 129, 257, 1000, 2049, 4096)
BF16 = torch.bfloat16
SUBNORMAL = 2.0**-133
SPECIAL = (
    -0.0,
    0.0,
    float("inf"),
    float("-inf"),
    float("nan"),
    torch.finfo(BF16).max,
    -torch.finfo(BF16).max,
    SUBNORMAL,
    -SUBNORMAL,
    2 * SUBNORMAL,
    3 * SUBNORMAL,
    6 * SUBNORMAL,
    2.0**-126,
    2.0**24,
    -(2.0**24),
    1.0,
    3.0,
)
SEED = 7046


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported(hidden: int | None = None) -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in SUPPORTED_CAPABILITIES
        and facade.is_cake_mhc_det_train_head_supported(_device(), hidden)
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="the Cake mHC training head needs an SM100 / SM103 / SM107 GPU",
)


def _draw(
    shape: tuple[int, ...], kind: str, generator: torch.Generator
) -> torch.Tensor:
    """BF16 values: ``randn``, ``spread`` (``randn`` x ``2^k``, ``k`` in [-8, 8]) or ``special`` (half from SPECIAL)."""
    x = torch.randn(shape, device=_device(), generator=generator)
    if kind == "spread":
        x = x * torch.exp2(
            torch.randint(-8, 9, shape, device=_device(), generator=generator).float()
        )
    elif kind == "special":
        values = torch.tensor(SPECIAL, device=_device())
        pick = values[
            torch.randint(0, len(SPECIAL), shape, device=_device(), generator=generator)
        ]
        x = torch.where(
            torch.rand(shape, device=_device(), generator=generator) < 0.5, pick, x
        )
    return x.to(BF16)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int16)


def _assert_bitwise(got: torch.Tensor, want: torch.Tensor) -> None:
    assert got.shape == want.shape and got.dtype == want.dtype == BF16
    mismatch = int((_bits(got) != _bits(want)).sum())
    assert mismatch == 0, f"{mismatch} of {got.numel()} elements differ from torch"


def _torch_grad(X: torch.Tensor, dY: torch.Tensor) -> torch.Tensor:
    leaf = X.detach().requires_grad_(True)
    (dX,) = torch.autograd.grad(leaf.mean(dim=1), leaf, dY)
    return dX


def _torch_accumulated(
    X: torch.Tensor, dY: torch.Tensor, dX0: torch.Tensor
) -> torch.Tensor:
    leaf = X.detach().requires_grad_(True)
    leaf.grad = dX0.clone()
    leaf.mean(dim=1).backward(dY)
    return leaf.grad


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("T", TOKENS)
@pytest.mark.parametrize("kind", ["randn", "spread", "special"])
def test_forward_backward_bitwise(hidden: int, T: int, kind: str) -> None:
    if not _supported(hidden):
        pytest.skip(f"hidden {hidden} is not exported on this device")
    g = torch.Generator(device=_device()).manual_seed(SEED + T * 31 + hidden)
    X = _draw((T, 4, hidden), kind, g)
    dY = _draw((T, hidden), kind, g)
    dX0 = _draw((T, 4, hidden), kind, g)

    _assert_bitwise(facade.cake_mhc_det_train_head_forward(X), X.mean(dim=1))
    _assert_bitwise(facade.cake_mhc_det_train_head_backward(dY), _torch_grad(X, dY))
    dX = dX0.clone()
    out = facade.cake_mhc_det_train_head_backward(dY, dX=dX, accumulate=True)
    assert out is dX
    _assert_bitwise(dX, _torch_accumulated(X, dY, dX0))


@pytest.mark.parametrize("hidden", HIDDEN)
def test_preallocated_outputs_and_zero_tokens(hidden: int) -> None:
    if not _supported(hidden):
        pytest.skip(f"hidden {hidden} is not exported on this device")
    g = torch.Generator(device=_device()).manual_seed(SEED)
    X = _draw((300, 4, hidden), "randn", g)
    dY = _draw((300, hidden), "randn", g)
    Y = torch.full((300, hidden), float("nan"), dtype=BF16, device=_device())
    assert facade.cake_mhc_det_train_head_forward(X, out=Y) is Y
    _assert_bitwise(Y, X.mean(dim=1))
    dX = torch.full((300, 4, hidden), float("nan"), dtype=BF16, device=_device())
    assert facade.cake_mhc_det_train_head_backward(dY, dX=dX) is dX
    _assert_bitwise(dX, _torch_grad(X, dY))

    empty = torch.empty((0, 4, hidden), dtype=BF16, device=_device())
    assert facade.cake_mhc_det_train_head_forward(empty).shape == (0, hidden)
    assert facade.cake_mhc_det_train_head_backward(empty[:, 0]).shape == (0, 4, hidden)


def test_argument_checks() -> None:
    X = torch.zeros((8, 4, 4096), dtype=BF16, device=_device())
    with pytest.raises(ValueError, match="hidden"):
        facade.cake_mhc_det_train_head_forward(
            torch.zeros((8, 4, 4000), dtype=BF16, device=_device())
        )
    with pytest.raises(ValueError, match="float32|bfloat16"):
        facade.cake_mhc_det_train_head_forward(X.float())
    with pytest.raises(ValueError, match="contiguous"):
        facade.cake_mhc_det_train_head_forward(
            torch.zeros((4, 8, 4096), dtype=BF16, device=_device()).transpose(0, 1)
        )
    with pytest.raises(ValueError, match="aligned"):
        facade.cake_mhc_det_train_head_forward(
            torch.zeros(8 * 4 * 4096 + 8, dtype=BF16, device=_device())[8:].view(
                8, 4, 4096
            )
        )
    with pytest.raises(ValueError, match=r"\[T, 4, hidden\]"):
        facade.cake_mhc_det_train_head_forward(
            torch.zeros((8, 3, 4096), dtype=BF16, device=_device())
        )
    with pytest.raises(ValueError, match="contiguous"):
        facade.cake_mhc_det_train_head_forward(
            X, out=torch.empty((8, 4096 * 2), dtype=BF16, device=_device())[:, :4096]
        )
    with pytest.raises(ValueError, match="shape"):
        facade.cake_mhc_det_train_head_forward(
            X, out=torch.empty((8, 5120), dtype=BF16, device=_device())
        )
    dY = torch.zeros((8, 4096), dtype=BF16, device=_device())
    with pytest.raises(ValueError, match="accumulate"):
        facade.cake_mhc_det_train_head_backward(dY, accumulate=True)
    with pytest.raises(ValueError, match="shape"):
        facade.cake_mhc_det_train_head_backward(
            dY, dX=torch.empty((8, 4, 5120), dtype=BF16, device=_device())
        )
    with pytest.raises(ValueError, match="overlap"):
        facade.cake_mhc_det_train_head_backward(
            X.view(-1)[: 8 * 4096].view(8, 4096), dX=X
        )
    with pytest.raises(ValueError, match="overlap"):
        facade.cake_mhc_det_train_head_forward(
            X, out=X.view(-1)[: 8 * 4096].view(8, 4096)
        )


@pytest.mark.parametrize("hidden", HIDDEN)
def test_autograd_matches_torch(hidden: int) -> None:
    if not _supported(hidden):
        pytest.skip(f"hidden {hidden} is not exported on this device")
    g = torch.Generator(device=_device()).manual_seed(SEED + 1)
    X = _draw((513, 4, hidden), "spread", g)
    dY = _draw((513, hidden), "spread", g)
    ours = X.detach().requires_grad_(True)
    ref = X.detach().requires_grad_(True)
    Y = facade.cake_mhc_det_train_head(ours)
    _assert_bitwise(Y, ref.mean(dim=1))
    # Two backward passes: the second accumulates into X.grad exactly as torch's does.
    for _ in range(2):
        facade.cake_mhc_det_train_head(ours).backward(dY)
        ref.mean(dim=1).backward(dY)
        _assert_bitwise(ours.grad, ref.grad)
    # A non-contiguous incoming gradient (the expanded gradient of a sum) is read through a copy.
    ours.grad = ref.grad = None
    facade.cake_mhc_det_train_head(ours).float().sum().backward()
    ref.mean(dim=1).float().sum().backward()
    _assert_bitwise(ours.grad, ref.grad)
    assert flashinfer.cake_mhc_det_train_head is facade.cake_mhc_det_train_head


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_activation_checkpoint(use_reentrant: bool) -> None:
    g = torch.Generator(device=_device()).manual_seed(SEED + 2)
    X = _draw((257, 4, 4096), "randn", g)
    W = torch.randn((4096,), device=_device(), generator=g).to(BF16)
    dY = _draw((257, 4096), "randn", g)

    def block(x: torch.Tensor, head) -> torch.Tensor:
        return head(x * 2) * W

    ours = X.detach().requires_grad_(True)
    ref = X.detach().requires_grad_(True)
    out = torch.utils.checkpoint.checkpoint(
        block, ours, facade.cake_mhc_det_train_head, use_reentrant=use_reentrant
    )
    want = block(ref, lambda x: x.mean(dim=1))
    _assert_bitwise(out, want)
    out.backward(dY)
    want.backward(dY)
    _assert_bitwise(ours.grad, ref.grad)


def test_cuda_graph_capture() -> None:
    T, hidden = 1000, 7168 if _supported(7168) else 4096
    g = torch.Generator(device=_device()).manual_seed(SEED + 3)
    X = _draw((T, 4, hidden), "randn", g)
    dY = _draw((T, hidden), "randn", g)
    dXa = _draw((T, 4, hidden), "randn", g)
    Y = torch.empty((T, hidden), dtype=BF16, device=_device())
    dX = torch.empty((T, 4, hidden), dtype=BF16, device=_device())
    acc0 = dXa.clone()

    def step() -> None:
        facade.cake_mhc_det_train_head_forward(X, out=Y)
        facade.cake_mhc_det_train_head_backward(dY, dX=dX)
        facade.cake_mhc_det_train_head_backward(dY, dX=dXa, accumulate=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()  # loads the programs outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    dXa.copy_(acc0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _replay in range(2):
        X.copy_(_draw((T, 4, hidden), "special", g))
        dY.copy_(_draw((T, hidden), "special", g))
        before = dXa.clone()
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(Y, X.mean(dim=1))
        _assert_bitwise(dX, _torch_grad(X, dY))
        _assert_bitwise(dXa, _torch_accumulated(X, dY, before))
