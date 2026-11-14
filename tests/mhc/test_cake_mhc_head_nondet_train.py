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

Tests of the non-deterministic Cake mHC training head
(``flashinfer.cake_mhc_head_nondet_train``): the stream mean ``Y = X.mean(dim=1)``
of BF16 ``X`` ``[T, 4, H]`` and its gradient, through the forward / backward
entry points, the accumulating backward, the autograd node, activation
checkpointing and CUDA graph capture, at every hidden size the device's
architecture serves (4096, 5120 and 7168 on SM100 / SM103, 4096 on SM107).

The reference budget is the deterministic head test's: bit for bit torch
(signed zeros and NaN encodings included).  Every result is also compared with
the deterministic head module ``flashinfer.cake_mhc_head_det_train`` (the same
kernels under other library names).  The head has no order-varying step, so
under ``torch.use_deterministic_algorithms(True)`` (with and without
``warn_only``) no entry point raises or warns.  The ``nondet`` contract does not
promise run-to-run bits; this implementation runs the deterministic head's
kernels in a fixed order, so repeated calls and the checkpoint replay are
checked against the deterministic head's bits within one version.

The hidden sizes a case runs at come from the architecture table below, not
from the module's support query, and a separate case checks that query: a
support query that wrongly says "unsupported" fails instead of skipping every
case.
"""

from __future__ import annotations

import gc
import warnings

import pytest
import torch
import torch.utils.checkpoint

import flashinfer
from flashinfer import cake_mhc_head_det_train as det_facade
from flashinfer import cake_mhc_head_nondet_train as facade
from flashinfer.utils import get_compute_capability

# The hidden sizes each supported compute capability serves.
HIDDEN_BY_CAPABILITY = {
    (10, 0): (4096, 5120, 7168),
    (10, 3): (4096, 5120, 7168),
    (10, 7): (4096,),
}
HIDDEN = (4096, 5120, 7168)
TOKENS = (1, 2, 3, 7, 65, 129, 257, 1000, 2049, 4096, 7169)
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
SEED = 8068
# Repeated calls: the launches of each entry point on identical inputs, and the token counts (with two long calls).
REPEATS = 5
REPEAT_TOKENS = (1, 257, 14928, 16263)


@pytest.fixture(autouse=True)
def collect_graphs():
    """Free leftover CUDA graphs before and after each case (see the deterministic head's test)."""
    gc.collect()
    yield
    gc.collect()


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    return tuple(get_compute_capability(_device()))


def _hidden_sizes() -> tuple[int, ...]:
    """The hidden sizes the device's architecture serves (empty off SM100 / SM103 / SM107)."""
    return HIDDEN_BY_CAPABILITY.get(_capability(), ())


pytestmark = pytest.mark.skipif(
    not _hidden_sizes(),
    reason="the Cake mHC training head needs an SM100 / SM103 / SM107 GPU",
)


def _require(hidden: int) -> None:
    if hidden not in _hidden_sizes():
        pytest.skip(
            f"hidden {hidden} is not served on compute capability {_capability()}"
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


def _assert_bitwise(got: torch.Tensor, want: torch.Tensor, what: str = "torch") -> None:
    assert got.shape == want.shape and got.dtype == want.dtype == BF16
    mismatch = int((_bits(got) != _bits(want)).sum())
    assert mismatch == 0, f"{mismatch} of {got.numel()} elements differ from {what}"


def _complement(tensor: torch.Tensor) -> torch.Tensor:
    """A new BF16 tensor whose every element has the inverted bits of ``tensor``'s: an output buffer in which an
    element the kernel does not write differs from ``tensor``."""
    return (~_bits(tensor)).view(BF16).view(tensor.shape)


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


def test_support_query_follows_the_architecture() -> None:
    """The support query names exactly the hidden sizes of the architecture table, as the deterministic head's does."""
    dev = _device()
    assert facade.is_cake_mhc_head_nondet_train_supported(dev)
    for hidden in (*HIDDEN, 2048, 4000, 6144):
        served = hidden in _hidden_sizes()
        assert facade.is_cake_mhc_head_nondet_train_supported(dev, hidden) == served
        assert det_facade.is_cake_mhc_head_det_train_supported(dev, hidden) == served
    assert tuple(facade.SUPPORTED_COMPUTE_CAPABILITIES) == tuple(
        sorted(HIDDEN_BY_CAPABILITY)
    )
    assert tuple(facade.ARCHES) == ("sm_100a", "sm_103a", "sm_107a")


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("T", TOKENS)
@pytest.mark.parametrize("kind", ["randn", "spread", "special"])
def test_forward_backward_bitwise(hidden: int, T: int, kind: str) -> None:
    _require(hidden)
    g = torch.Generator(device=_device()).manual_seed(SEED + T * 31 + hidden)
    X = _draw((T, 4, hidden), kind, g)
    dY = _draw((T, hidden), kind, g)
    dX0 = _draw((T, 4, hidden), kind, g)

    Y = facade.cake_mhc_head_nondet_train_forward(X)
    _assert_bitwise(Y, X.mean(dim=1))
    _assert_bitwise(Y, det_facade.cake_mhc_head_det_train_forward(X), "the det head")
    dX = facade.cake_mhc_head_nondet_train_backward(dY)
    _assert_bitwise(dX, _torch_grad(X, dY))
    _assert_bitwise(dX, det_facade.cake_mhc_head_det_train_backward(dY), "the det head")
    acc = dX0.clone()
    out = facade.cake_mhc_head_nondet_train_backward(dY, dX=acc, accumulate=True)
    assert out is acc
    _assert_bitwise(acc, _torch_accumulated(X, dY, dX0))
    det_acc = dX0.clone()
    det_facade.cake_mhc_head_det_train_backward(dY, dX=det_acc, accumulate=True)
    _assert_bitwise(acc, det_acc, "the det head")


@pytest.mark.parametrize("hidden", HIDDEN)
def test_preallocated_outputs_and_zero_tokens(hidden: int) -> None:
    _require(hidden)
    g = torch.Generator(device=_device()).manual_seed(SEED)
    X = _draw((301, 4, hidden), "randn", g)
    dY = _draw((301, hidden), "randn", g)
    Y = torch.full((301, hidden), float("nan"), dtype=BF16, device=_device())
    assert facade.cake_mhc_head_nondet_train_forward(X, out=Y) is Y
    _assert_bitwise(Y, X.mean(dim=1))
    dX = torch.full((301, 4, hidden), float("nan"), dtype=BF16, device=_device())
    assert facade.cake_mhc_head_nondet_train_backward(dY, dX=dX) is dX
    _assert_bitwise(dX, _torch_grad(X, dY))

    empty = torch.empty((0, 4, hidden), dtype=BF16, device=_device())
    assert facade.cake_mhc_head_nondet_train_forward(empty).shape == (0, hidden)
    assert facade.cake_mhc_head_nondet_train_backward(empty[:, 0]).shape == (
        0,
        4,
        hidden,
    )
    assert (
        facade.cake_mhc_head_nondet_train_backward(
            empty[:, 0], dX=empty, accumulate=True
        )
        is empty
    )


def test_argument_checks() -> None:
    X = torch.zeros((8, 4, 4096), dtype=BF16, device=_device())
    with pytest.raises(ValueError, match="hidden"):
        facade.cake_mhc_head_nondet_train_forward(
            torch.zeros((8, 4, 4000), dtype=BF16, device=_device())
        )
    with pytest.raises(ValueError, match="float32|bfloat16"):
        facade.cake_mhc_head_nondet_train_forward(X.float())
    with pytest.raises(ValueError, match="contiguous"):
        facade.cake_mhc_head_nondet_train_forward(
            torch.zeros((4, 8, 4096), dtype=BF16, device=_device()).transpose(0, 1)
        )
    with pytest.raises(ValueError, match="aligned"):
        facade.cake_mhc_head_nondet_train_forward(
            torch.zeros(8 * 4 * 4096 + 8, dtype=BF16, device=_device())[8:].view(
                8, 4, 4096
            )
        )
    with pytest.raises(ValueError, match=r"\[T, 4, hidden\]"):
        facade.cake_mhc_head_nondet_train_forward(
            torch.zeros((8, 3, 4096), dtype=BF16, device=_device())
        )
    with pytest.raises(ValueError, match="contiguous"):
        facade.cake_mhc_head_nondet_train_forward(
            X, out=torch.empty((8, 4096 * 2), dtype=BF16, device=_device())[:, :4096]
        )
    with pytest.raises(ValueError, match="shape"):
        facade.cake_mhc_head_nondet_train_forward(
            X, out=torch.empty((8, 5120), dtype=BF16, device=_device())
        )
    dY = torch.zeros((8, 4096), dtype=BF16, device=_device())
    with pytest.raises(ValueError, match="accumulate"):
        facade.cake_mhc_head_nondet_train_backward(dY, accumulate=True)
    with pytest.raises(ValueError, match="shape"):
        facade.cake_mhc_head_nondet_train_backward(
            dY, dX=torch.empty((8, 4, 5120), dtype=BF16, device=_device())
        )
    with pytest.raises(ValueError, match="aligned"):
        facade.cake_mhc_head_nondet_train_backward(
            torch.zeros(8 * 4096 + 8, dtype=BF16, device=_device())[8:].view(8, 4096)
        )
    with pytest.raises(ValueError, match="overlap"):
        facade.cake_mhc_head_nondet_train_backward(
            X.view(-1)[: 8 * 4096].view(8, 4096), dX=X
        )
    with pytest.raises(ValueError, match="overlap"):
        facade.cake_mhc_head_nondet_train_forward(
            X, out=X.view(-1)[: 8 * 4096].view(8, 4096)
        )
    # The mode is part of the name: no entry point takes a ``deterministic`` keyword.
    with pytest.raises(TypeError):
        facade.cake_mhc_head_nondet_train_forward(X, deterministic=False)
    with pytest.raises(TypeError):
        facade.cake_mhc_head_nondet_train_backward(dY, deterministic=False)
    with pytest.raises(TypeError):
        facade.cake_mhc_head_nondet(X, deterministic=False)


@pytest.mark.parametrize("hidden", HIDDEN)
def test_autograd_matches_torch(hidden: int) -> None:
    _require(hidden)
    g = torch.Generator(device=_device()).manual_seed(SEED + 1)
    X = _draw((513, 4, hidden), "spread", g)
    dY = _draw((513, hidden), "spread", g)
    ours = X.detach().requires_grad_(True)
    det = X.detach().requires_grad_(True)
    ref = X.detach().requires_grad_(True)
    Y = facade.cake_mhc_head_nondet(ours)
    assert Y.grad_fn is not None
    _assert_bitwise(Y, ref.mean(dim=1))
    # Two backward passes: the second accumulates into X.grad exactly as torch's does.
    for _ in range(2):
        facade.cake_mhc_head_nondet(ours).backward(dY)
        det_facade.cake_mhc_head_det(det).backward(dY)
        ref.mean(dim=1).backward(dY)
        _assert_bitwise(ours.grad, ref.grad)
        _assert_bitwise(ours.grad, det.grad, "the det head")
    # A non-contiguous incoming gradient (the expanded gradient of a sum) is read through a copy.
    ours.grad = ref.grad = None
    facade.cake_mhc_head_nondet(ours).float().sum().backward()
    ref.mean(dim=1).float().sum().backward()
    _assert_bitwise(ours.grad, ref.grad)
    assert facade.CakeMHCHeadNondetFunction.apply(X).shape == (513, hidden)
    # The mode has no top-level name: the module is imported by its full name.
    assert not hasattr(flashinfer, "cake_mhc_head_nondet")
    assert not hasattr(flashinfer, "cake_mhc_head_nondet_train_forward")
    assert "CakeMHCHeadNondetFunction" not in flashinfer._LAZY_EXPORTS


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("use_reentrant", [False, True])
def test_activation_checkpoint(hidden: int, use_reentrant: bool) -> None:
    _require(hidden)
    g = torch.Generator(device=_device()).manual_seed(SEED + 2)
    X = _draw((257, 4, hidden), "randn", g)
    W = torch.randn((hidden,), device=_device(), generator=g).to(BF16)
    dY = _draw((257, hidden), "randn", g)

    def block(x: torch.Tensor, head) -> torch.Tensor:
        return head(x * 2) * W

    ours = X.detach().requires_grad_(True)
    det = X.detach().requires_grad_(True)
    ref = X.detach().requires_grad_(True)
    out = torch.utils.checkpoint.checkpoint(
        block, ours, facade.cake_mhc_head_nondet, use_reentrant=use_reentrant
    )
    det_out = torch.utils.checkpoint.checkpoint(
        block, det, det_facade.cake_mhc_head_det, use_reentrant=use_reentrant
    )
    want = block(ref, lambda x: x.mean(dim=1))
    _assert_bitwise(out, want)
    _assert_bitwise(out, det_out, "the det head")
    out.backward(dY)
    det_out.backward(dY)
    want.backward(dY)
    _assert_bitwise(ours.grad, ref.grad)
    _assert_bitwise(ours.grad, det.grad, "the det head")


def test_cuda_graph_capture() -> None:
    T, hidden = 1000, max(_hidden_sizes())
    g = torch.Generator(device=_device()).manual_seed(SEED + 3)
    X = _draw((T, 4, hidden), "randn", g)
    dY = _draw((T, hidden), "randn", g)
    dXa = _draw((T, 4, hidden), "randn", g)
    Y = torch.empty((T, hidden), dtype=BF16, device=_device())
    dX = torch.empty((T, 4, hidden), dtype=BF16, device=_device())
    acc0 = dXa.clone()

    def step() -> None:
        facade.cake_mhc_head_nondet_train_forward(X, out=Y)
        facade.cake_mhc_head_nondet_train_backward(dY, dX=dX)
        facade.cake_mhc_head_nondet_train_backward(dY, dX=dXa, accumulate=True)

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


@pytest.mark.parametrize("warn_only", [False, True], ids=["alert", "warn_only"])
def test_deterministic_algorithms_raise_and_warn_nothing(warn_only: bool) -> None:
    """With ``torch.use_deterministic_algorithms(True)`` every entry point -- forward, writing and accumulating
    backward, the autograd node and its checkpointed recompute -- runs without raising PyTorch's alert and, with
    ``warn_only=True``, without a warning (every warning is an error here), and returns the reference bits."""
    hidden = max(_hidden_sizes())
    g = torch.Generator(device=_device()).manual_seed(SEED + 4)
    X = _draw((257, 4, hidden), "spread", g)
    dY = _draw((257, hidden), "spread", g)
    dX0 = _draw((257, 4, hidden), "spread", g)
    want_Y, want_dX = X.mean(dim=1), _torch_grad(X, dY)
    want_acc = _torch_accumulated(X, dY, dX0)
    # Load the programs before the mode is on, so that a first-use build is not inside the check.
    facade.cake_mhc_head_nondet_train_forward(X)
    facade.cake_mhc_head_nondet_train_backward(dY, dX=dX0.clone(), accumulate=True)
    facade.cake_mhc_head_nondet_train_backward(dY)
    torch.cuda.synchronize()

    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Y = facade.cake_mhc_head_nondet_train_forward(X)
            dX = facade.cake_mhc_head_nondet_train_backward(dY)
            acc = dX0.clone()
            facade.cake_mhc_head_nondet_train_backward(dY, dX=acc, accumulate=True)
            node = X.detach().requires_grad_(True)
            facade.cake_mhc_head_nondet(node).backward(dY)
            checkpointed = X.detach().requires_grad_(True)
            torch.utils.checkpoint.checkpoint(
                facade.cake_mhc_head_nondet, checkpointed, use_reentrant=False
            ).backward(dY)
            torch.cuda.synchronize()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)
    _assert_bitwise(Y, want_Y)
    _assert_bitwise(dX, want_dX)
    _assert_bitwise(acc, want_acc)
    _assert_bitwise(node.grad, want_dX)
    _assert_bitwise(checkpointed.grad, want_dX)


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("T", REPEAT_TOKENS)
def test_repeats_give_the_det_head_bits(hidden: int, T: int) -> None:
    """The forward, the backward and the accumulating backward each run ``REPEATS`` times on identical inputs, and
    every run equals the deterministic head's result bit for bit (the kernels are the deterministic head's, in a
    fixed order; the ``nondet`` contract itself does not promise run-to-run bits).  A run writes into a buffer holding
    the inverted bits of that result, so an element the forward or the writing backward leaves unwritten fails too;
    the accumulating backward reads its buffer, so each run adds into a fresh copy of the same ``dX0``."""
    _require(hidden)
    g = torch.Generator(device=_device()).manual_seed(SEED + 5 + T * 31 + hidden)
    X = _draw((T, 4, hidden), "special", g)
    dY = _draw((T, hidden), "special", g)
    dX0 = _draw((T, 4, hidden), "special", g)

    Y = det_facade.cake_mhc_head_det_train_forward(X)
    dX = det_facade.cake_mhc_head_det_train_backward(dY)
    acc = dX0.clone()
    det_facade.cake_mhc_head_det_train_backward(dY, dX=acc, accumulate=True)
    for run in range(REPEATS):
        out = _complement(Y)
        assert facade.cake_mhc_head_nondet_train_forward(X, out=out) is out
        _assert_bitwise(out, Y, f"the det head's forward (run {run})")
        out = _complement(dX)
        assert facade.cake_mhc_head_nondet_train_backward(dY, dX=out) is out
        _assert_bitwise(out, dX, f"the det head's backward (run {run})")
        out = dX0.clone()
        facade.cake_mhc_head_nondet_train_backward(dY, dX=out, accumulate=True)
        _assert_bitwise(out, acc, f"the det head's accumulating backward (run {run})")


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("T", (257, 16263))
def test_checkpoint_replay_bitwise(hidden: int, T: int) -> None:
    """Under ``torch.utils.checkpoint(..., use_reentrant=False)`` around a small graph (``head(X * 2) * W``), the
    checkpoint's forward and its recompute in the backward, the output and the gradients of ``X`` and ``W`` equal
    the same graph's run without the checkpoint bit for bit (``W``'s gradient reads the recomputed head output), and
    that run equals the deterministic head's."""
    _require(hidden)
    g = torch.Generator(device=_device()).manual_seed(SEED + 6 + T * 31 + hidden)
    X = _draw((T, 4, hidden), "spread", g)
    W = _draw((hidden,), "randn", g)
    dOut = _draw((T, hidden), "randn", g)
    heads: list[torch.Tensor] = []

    def run(head, checkpointed: bool):
        def block(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            y = head(x * 2)
            heads.append(y.detach().clone())
            return y * w

        heads.clear()
        x = X.detach().requires_grad_(True)
        w = W.detach().requires_grad_(True)
        if checkpointed:
            out = torch.utils.checkpoint.checkpoint(block, x, w, use_reentrant=False)
        else:
            out = block(x, w)
        out.backward(dOut)
        return out.detach(), x.grad, w.grad, list(heads)

    det = run(det_facade.cake_mhc_head_det, checkpointed=False)
    plain = run(facade.cake_mhc_head_nondet, checkpointed=False)
    replay = run(facade.cake_mhc_head_nondet, checkpointed=True)
    assert len(plain[3]) == 1 and len(replay[3]) == 2, (len(plain[3]), len(replay[3]))
    _assert_bitwise(plain[3][0], det[3][0], "the det head's output")
    for y in replay[3]:
        _assert_bitwise(y, plain[3][0], "the head output without the checkpoint")
    for name, got, want, det_want in zip(
        ("output", "dX", "dW"), replay[:3], plain[:3], det[:3], strict=True
    ):
        _assert_bitwise(got, want, f"the {name} without the checkpoint")
        _assert_bitwise(want, det_want, f"the det head's {name}")
