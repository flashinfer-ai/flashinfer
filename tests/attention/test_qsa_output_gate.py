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
"""

import pytest
import torch

import flashinfer


def _reference(attention: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """The elementwise chain this kernel replaces.

    The attention value is already at the buffer's dtype, so its rounding has
    happened; the gate is widened, the product is formed at float, and the
    result is rounded once on the way back.
    """
    product = attention.to(torch.float32) * torch.sigmoid(gate.to(torch.float32))
    return product.to(attention.dtype)


# Every FlashInfer module is compiled with -use_fast_math, which maps expf to
# the approximate intrinsic, so the logistic here and the one torch computes are
# not the same function to the last bit. What is fixed is the order -- the
# attention value rounded first, the gate widened, the product at float, one
# store -- and the distance, which is one representable value.
#
# The bound comes from test_qsa_output_gate_exhaustive.py, which sweeps every
# bfloat16 a gate can hold rather than sampling; this file reuses it on the
# shapes and layouts the kernel has to handle.
_MAX_ULP = 1


def _ordered(values: torch.Tensor) -> torch.Tensor:
    """A float's bits as an integer that sorts the way the float does."""
    bits = values.view(torch.int16).to(torch.int64)
    return torch.where(bits < 0, torch.tensor(-0x8000, device=bits.device) - bits, bits)


def _assert_matches(out: torch.Tensor, expected: torch.Tensor) -> None:
    """Within one representable value, counted on the bits rather than estimated."""
    assert out.dtype == expected.dtype
    distance = (_ordered(out) - _ordered(expected)).abs()
    worst = int(distance.max())
    assert worst <= _MAX_ULP, f"{worst} ulp apart"


def _batch(rows, num_heads, head_dim, dtype, device, seed=0, pad=0):
    generator = torch.Generator(device=device).manual_seed(seed)
    attention = torch.randn(
        rows + pad, num_heads, head_dim, dtype=dtype, device=device, generator=generator
    )
    gate = torch.randn(
        rows, num_heads, head_dim, dtype=dtype, device=device, generator=generator
    )
    return attention, gate


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 7, 128, 2048])
@pytest.mark.parametrize("num_heads,head_dim", [(1, 64), (16, 128), (4, 100)])
def test_the_gate_matches_the_chain_it_replaces(dtype, rows, num_heads, head_dim):
    """One kernel, same result as the elementwise sequence."""
    device = torch.device("cuda")
    attention, gate = _batch(rows, num_heads, head_dim, dtype, device, seed=rows)
    out = flashinfer.qsa_output_gate(attention, gate)
    _assert_matches(out, _reference(attention, gate))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_the_padding_rows_of_the_attention_buffer_are_not_read(dtype):
    """A fixed-row plan pads its batch; the gate reads only the live rows.

    The padding is filled with a value that would be obvious in the output, and
    the output is one row per gate row.
    """
    device = torch.device("cuda")
    rows, num_heads, head_dim = 96, 8, 128
    attention, gate = _batch(rows, num_heads, head_dim, dtype, device, seed=5, pad=32)
    attention[rows:].fill_(1000.0)
    out = flashinfer.qsa_output_gate(attention, gate)
    assert out.shape == gate.shape
    _assert_matches(out, _reference(attention[:rows], gate))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_the_gate_applies_in_place(dtype):
    """``out=attention`` scales the buffer the caller already has."""
    device = torch.device("cuda")
    attention, gate = _batch(64, 8, 128, dtype, device, seed=11)
    expected = _reference(attention, gate)
    returned = flashinfer.qsa_output_gate(attention, gate, out=attention)
    assert returned.data_ptr() == attention.data_ptr()
    _assert_matches(attention, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_gate_that_is_a_view_of_a_fused_projection(dtype):
    """Only the head axis is contiguous when the gate rides beside the query.

    A fused QKV projection hands over a slice of a wider row, so the gate's row
    stride is the projection's width and its head stride is the head size. The
    kernel takes both from the tensor rather than assuming a packed layout.
    """
    device = torch.device("cuda")
    rows, num_heads, head_dim = 48, 8, 128
    generator = torch.Generator(device=device).manual_seed(3)
    fused = torch.randn(
        rows, 2 * num_heads * head_dim, dtype=dtype, device=device, generator=generator
    )
    gate = fused[:, num_heads * head_dim :].view(rows, num_heads, head_dim)
    assert not gate.is_contiguous()
    attention = torch.randn(
        rows, num_heads, head_dim, dtype=dtype, device=device, generator=generator
    )
    before = fused.clone()

    out = flashinfer.qsa_output_gate(attention, gate)

    _assert_matches(out, _reference(attention, gate))
    # The gate is an input; nothing writes through the view.
    torch.testing.assert_close(fused, before, rtol=0, atol=0)


def test_the_gate_allocates_nothing_under_graph_capture():
    """Capture has to add no tensor of the kernel's own.

    The elementwise chain this replaces forms float intermediates, and under
    capture those land in the graph's private pool for the life of the process.
    This asserts the property directly -- no allocation while the capture is
    open -- rather than comparing two captures, whose reserved bytes depend on
    what else the allocator is holding. The size of the difference is measured
    separately, in fresh processes, by graph_pool_probe.py.
    """
    device = torch.device("cuda")
    rows, num_heads, head_dim = 2048, 16, 128
    attention, gate = _batch(rows, num_heads, head_dim, torch.bfloat16, device, seed=9)
    out = torch.empty_like(gate)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            flashinfer.qsa_output_gate(attention, gate, out=out)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    # Eagerly first: with the output supplied, the call holds nothing.
    before = torch.cuda.memory_allocated()
    flashinfer.qsa_output_gate(attention, gate, out=out)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "the call allocated a tensor"

    # And under capture it costs what an ordinary copy costs. The bytes a
    # capture books for itself are the same either way; what the elementwise
    # chain adds on top of them is in the graph's private pool, which
    # graph_pool_probe.py measures in fresh processes.
    def capture(step):
        graph = torch.cuda.CUDAGraph()
        start = torch.cuda.memory_allocated()
        with torch.cuda.graph(graph):
            step()
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() - start, graph

    baseline, _held = capture(lambda: out.copy_(gate))
    allocated, graph = capture(
        lambda: flashinfer.qsa_output_gate(attention, gate, out=out)
    )
    assert allocated <= baseline, f"the capture allocated {allocated - baseline} bytes"

    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches(out, _reference(attention, gate))


def test_a_shorter_attention_buffer_is_refused():
    """Padding may make the attention taller, never shorter than the output."""
    device = torch.device("cuda")
    attention, gate = _batch(8, 4, 64, torch.bfloat16, device, seed=1)
    with pytest.raises(RuntimeError, match="at least the output's rows"):
        flashinfer.qsa_output_gate(attention[:4], gate)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_the_logistic_is_not_rounded_before_the_product(dtype):
    """The order, pinned against the shortcut that gets it wrong.

    ``out.mul_(torch.sigmoid(gate))`` rounds the logistic to the output dtype
    and only then multiplies, which moves the result. The bound above is too
    loose to separate the two on its own, so this compares which of them sits
    closer to a float64 evaluation of the same expression.
    """
    device = torch.device("cuda")
    attention, gate = _batch(512, 16, 128, dtype, device, seed=23)

    exact = attention.to(torch.float64) * torch.sigmoid(gate.to(torch.float64))
    kernel = flashinfer.qsa_output_gate(attention, gate).to(torch.float64)
    shortcut = (attention * torch.sigmoid(gate)).to(torch.float64)

    assert not torch.equal(shortcut, kernel), "the shortcut cannot be indistinguishable"
    assert (kernel - exact).abs().sum() < (shortcut - exact).abs().sum()
