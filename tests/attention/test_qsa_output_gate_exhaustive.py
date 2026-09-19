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
from flashinfer.jit.qsa_output_gate import gen_qsa_output_gate_module

# A bfloat16 has sixteen bits, so every value a gate can hold fits in one sweep.
# Nothing here is sampled: the claim this file makes is over the whole domain of
# the gate, against attention values chosen to cover the exponent range rather
# than drawn from a normal around one.
_ALL_BF16 = 1 << 16
_ATTENTION_VALUES = (
    0.0,
    -0.0,
    2.0**-133,  # subnormal in bfloat16
    -(2.0**-133),
    2.0**-126,  # the smallest normal float
    1.0,
    -1.0,
    0.5,
    -3.75,
    2.0**8,
    2.0**64,
    -(2.0**64),
    2.0**127,  # near the top of the exponent range
    float("inf"),
    float("-inf"),
    float("nan"),
)


def _reference(attention: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """The elementwise chain the kernel replaces, at the same precision.

    The attention value is already at the buffer's dtype; the gate is widened,
    the product is formed at float, and the result is rounded once on the way
    back. The kernel is built without fast math, so this is not an
    approximation of what it computes -- it is what it computes.
    """
    product = attention.to(torch.float32) * torch.sigmoid(gate.to(torch.float32))
    return product.to(attention.dtype)


@pytest.fixture(scope="module")
def sweep():
    """Every bfloat16 gate against every attention value above."""
    device = torch.device("cuda")
    gate_row = (
        torch.arange(_ALL_BF16, dtype=torch.int32, device=device)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    rows = len(_ATTENTION_VALUES)
    gate = gate_row.reshape(1, 1, _ALL_BF16).expand(rows, 1, _ALL_BF16).contiguous()
    attention = (
        torch.tensor(_ATTENTION_VALUES, dtype=torch.bfloat16, device=device)
        .reshape(rows, 1, 1)
        .expand(rows, 1, _ALL_BF16)
        .contiguous()
    )
    out = flashinfer.qsa_output_gate(attention, gate)
    return attention, gate, out, _reference(attention, gate)


def test_the_module_is_built_without_fast_math():
    """The flag every claim in this file rests on.

    Fast math would swap the logistic for its approximate intrinsic and turn on
    flush-to-zero. The second is not a rounding difference: a gate below about
    -87 drives the logistic subnormal, and flushing it takes the product to zero
    instead of to a small number, which at the top of the attention range is a
    difference of thousands of representable values.
    """
    spec = gen_qsa_output_gate_module()
    assert not [flag for flag in spec.extra_cuda_cflags if "fast_math" in flag]

    # And the exemption is this module's alone: turning it off everywhere would
    # be a different change, and a silent one.
    from flashinfer.jit.sparse_route import gen_sparse_route_module

    other = gen_sparse_route_module()
    assert [flag for flag in other.extra_cuda_cflags if "fast_math" in flag]


def _ordered(values: torch.Tensor) -> torch.Tensor:
    """A float's bits as an integer that sorts the way the float does."""
    bits = values.view(torch.int16).to(torch.int64)
    return torch.where(bits < 0, torch.tensor(-0x8000, device=bits.device) - bits, bits)


def test_every_bfloat16_gate_stays_within_one_representable_value(sweep):
    """The whole gate domain, against the expression a caller would write.

    This is the portable claim: precise math, no flush to zero, the order fixed,
    and a result at most one representable value from the reference. Whether a
    given architecture and toolkit round the logistic to exactly the same bits
    is a measurement, made below, not a promise.
    """
    attention, gate, out, expected = sweep
    real = attention.isfinite() & gate.isfinite()
    assert bool(real.any())
    distance = (_ordered(out[real]) - _ordered(expected[real])).abs()
    assert int(distance.max()) <= 1


def test_the_measured_distance_on_this_build(sweep):
    """What the sweep actually found, recorded rather than assumed.

    On the build this runs against, every value a bfloat16 gate can hold gives
    the same bits as the reference. A future toolkit that rounds the logistic
    differently would fail here while the contract above still holds, which is
    the point of keeping the two apart.
    """
    attention, gate, out, expected = sweep
    real = attention.isfinite() & gate.isfinite()
    differing = int((out[real] != expected[real]).sum())
    assert differing == 0, (
        f"{differing} of {int(real.sum())} values differ from the reference; "
        "the contract above still holds, but this build no longer matches it "
        "bit for bit"
    )


def test_the_zeros_and_the_subnormals_survive(sweep):
    """Nothing is flushed on the way through.

    An attention value of zero stays zero with its sign, and a subnormal one
    scales like any other value rather than collapsing. A gate low enough to
    drive the logistic subnormal keeps its small product too.
    """
    attention, gate, out, expected = sweep
    real = gate[0].isfinite()
    for index, value in enumerate(_ATTENTION_VALUES):
        if value != value or value in (float("inf"), float("-inf")):
            continue
        assert torch.equal(out[index][real], expected[index][real]), f"row {index}"

    # The band a flush-to-zero build loses: around a gate of -90 the logistic is
    # below the smallest normal float, and the product is still kept.
    deep = real & (gate[0] < -88.0) & (gate[0] > -100.0)
    assert bool(deep.any())
    top = _ATTENTION_VALUES.index(2.0**127)
    assert bool((out[top][deep] != 0).any()), "a subnormal logistic was flushed"
    del attention


def test_a_non_finite_attention_value_propagates(sweep):
    """Neither an infinity nor a NaN is produced from real inputs, or swallowed."""
    attention, gate, out, _expected = sweep
    del attention
    real_gate = gate[0].isfinite()
    for index, value in enumerate(_ATTENTION_VALUES):
        row = out[index][real_gate]
        if value != value:
            assert row.isnan().all(), "a NaN attention value has to stay NaN"
        elif value in (float("inf"), float("-inf")):
            # The logistic is zero only where the gate saturates, and inf times
            # zero is NaN; everywhere else the sign of the infinity survives.
            saturated = torch.sigmoid(gate[index][real_gate].to(torch.float32)) == 0.0
            assert row[saturated].isnan().all()
            assert row[~saturated].isinf().all()
            assert bool(((row[~saturated] > 0) == (value > 0)).all())
        else:
            assert row.isfinite().all()


def test_a_non_finite_gate_propagates(sweep):
    """The gate's own infinities and NaNs, asserted rather than passed over."""
    attention, gate, out, _expected = sweep
    column = attention[:, 0, 0]
    rows = (column.isfinite() & (column != 0)).nonzero().flatten()
    assert rows.numel()

    # The gate has one head of one row, so the masks below are over features.
    gate_row = gate[0, 0]
    nan_gate = gate_row.isnan()
    assert bool(nan_gate.any())
    assert out[rows][:, :, nan_gate].isnan().all(), "a NaN gate has to stay NaN"

    # sigmoid(+inf) is one, so the attention value passes through unchanged;
    # sigmoid(-inf) is zero, so the product is zero.
    positive = gate_row == float("inf")
    negative = gate_row == float("-inf")
    assert bool(positive.any()) and bool(negative.any())
    for row in rows.tolist():
        torch.testing.assert_close(
            out[row][:, positive], attention[row][:, positive], rtol=0, atol=0
        )
        assert bool((out[row][:, negative] == 0).all())


def test_a_saturating_gate_reaches_one_and_zero(sweep):
    """The two ends of the logistic, where the product is the input or nothing."""
    attention, gate, out, expected = sweep
    del attention
    one = _ATTENTION_VALUES.index(1.0)
    row_gate, row_out, row_expected = gate[one], out[one], expected[one]
    large = row_gate.isfinite() & (row_gate > 20.0)
    assert bool(large.any())
    torch.testing.assert_close(
        row_out[large], torch.ones_like(row_out[large]), rtol=0, atol=0
    )
    small = row_gate.isfinite() & (row_gate < -200.0)
    assert bool(small.any())
    torch.testing.assert_close(
        row_out[small], torch.zeros_like(row_out[small]), rtol=0, atol=0
    )
    real = row_gate.isfinite()
    assert torch.equal(row_out[real], row_expected[real])


def test_the_gate_may_not_overlap_the_tensor_being_written():
    """`out=attention` is supported; a gate sharing storage with `out` is not.

    The kernel writes `out` while reading `gate`, so overlapping those two would
    read values that had already been replaced. Sharing a starting address is
    the obvious case; a shifted view of the same storage is the one worth a
    test.
    """
    device = torch.device("cuda")
    attention = torch.randn(8, 4, 64, dtype=torch.bfloat16, device=device)
    both = torch.randn(9, 4, 64, dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError, match="overlap"):
        flashinfer.qsa_output_gate(attention, both[:8], out=both[:8])
    # Shifted by a row: a different starting address, the same storage.
    with pytest.raises(RuntimeError, match="overlap"):
        flashinfer.qsa_output_gate(attention, both[1:], out=both[:8])


def test_scaling_in_place_matches_scaling_into_a_separate_buffer():
    """The in-place path is the aliasing one; it has to agree with the other."""
    device = torch.device("cuda")
    attention = torch.randn(129, 7, 128, dtype=torch.bfloat16, device=device)
    gate = torch.randn(129, 7, 128, dtype=torch.bfloat16, device=device)
    separate = flashinfer.qsa_output_gate(attention, gate)
    in_place = flashinfer.qsa_output_gate(attention, gate, out=attention)
    assert in_place.data_ptr() == attention.data_ptr()
    torch.testing.assert_close(in_place, separate, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(4, 0, 64), (4, 2, 0), (0, 2, 64)])
def test_an_empty_axis_is_a_no_op(shape):
    """Nothing to scale, and nothing to check.

    The overlap check reads the last element of each axis, which does not exist
    when an axis is empty, so the entry point has to return before it.
    """
    device = torch.device("cuda")
    attention = torch.empty(shape, dtype=torch.bfloat16, device=device)
    gate = torch.empty(shape, dtype=torch.bfloat16, device=device)
    out = flashinfer.qsa_output_gate(attention, gate)
    assert out.shape == gate.shape


def test_out_has_to_be_attention_itself_or_hold_none_of_it():
    """A shifted view of the same storage is the dangerous case.

    Scaling in place is safe because each thread reads the element it writes.
    An `out` offset against `attention` is not: a thread can overwrite a value
    another thread has yet to read, and which of the two happens is a race.
    """
    device = torch.device("cuda")
    storage = torch.randn(10, 4, 64, dtype=torch.bfloat16, device=device)
    gate = torch.randn(8, 4, 64, dtype=torch.bfloat16, device=device)

    # Off by a row, and off by a single element: both overlap.
    with pytest.raises(RuntimeError, match="attention itself"):
        flashinfer.qsa_output_gate(storage[1:9], gate, out=storage[:8])
    shifted = storage.flatten()[1:].view(-1)[: 8 * 4 * 64].view(8, 4, 64)
    with pytest.raises(RuntimeError, match="attention itself"):
        flashinfer.qsa_output_gate(storage[:8], gate, out=shifted)

    # The prefix of a padded attention buffer is the supported in-place shape.
    padded = torch.randn(10, 4, 64, dtype=torch.bfloat16, device=device)
    flashinfer.qsa_output_gate(padded, gate, out=padded[:8])
