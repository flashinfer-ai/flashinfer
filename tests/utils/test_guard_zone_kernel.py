# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guard-zone integration with a real pre-allocated-output FlashInfer call.

``flashinfer.silu_and_mul(input, out=...)`` (``flashinfer/activation.py``) is
an existing public API that writes into a caller-provided output tensor, so it
is a genuine ``out=`` injection point: the guarded payload *is* the tensor the
kernel is handed, and the guards sit around it in the same allocation.

The hidden sizes below cover both vector widths the activation launcher can
pick: at ``hidden=512`` it takes the full 16-byte width, and at
``hidden=3420`` the y-half of each row starts at byte 6840, only 8-byte
aligned, so it falls back to a 4-element vector.  Both are legal only while
the guarded payload keeps its 16-byte alignment, which is the property the
helper must not break.

Work order cases: A3-12 (a real accepting-output kernel protects the actual
buffer) and its A3-04 flavour (payload correct while the guard fails).
"""

import pytest
import torch

import flashinfer
from flashinfer.utils import has_flashinfer_jit_cache

from tests.test_helpers.guard_zone import (
    PREFIX_SENTINEL,
    GuardZoneCorruption,
    allocate_guarded,
    clear_guards,
    inject_out_of_bounds_write,
    reset_guards,
    verify_guards,
)

HIDDEN_SIZES = [512, 3420]


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        [flashinfer.activation.gen_act_and_mul_module("silu")],
        verbose=False,
    )
    yield


@pytest.fixture(autouse=True)
def isolate_guard_registry():
    yield
    clear_guards()


def assert_matches_silu_reference(out: torch.Tensor, x: torch.Tensor) -> None:
    """Compare against an independent fp32 reference at the output's dtype."""
    hidden = x.shape[-1] // 2
    gate = torch.nn.functional.silu(x[..., :hidden].float())
    reference = gate * x[..., hidden:].float()
    torch.testing.assert_close(out, reference.to(out.dtype), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_silu_and_mul_writes_only_the_guarded_output(dtype, hidden):
    """A3-12: the real kernel writes the guarded payload and nothing beside it."""
    buffer = allocate_guarded((8, hidden), dtype, "cuda", alignment=16)
    x = torch.randn(8, 2 * hidden, dtype=dtype, device="cuda")
    assert buffer.payload.data_ptr() % 16 == 0
    reset_guards()

    out = flashinfer.silu_and_mul(x, out=buffer.payload, enable_pdl=False)

    assert out is buffer.payload
    assert_matches_silu_reference(out, x)
    verify_guards()


def test_guarded_output_reports_an_overrun_after_a_real_call():
    """A3-12: the passing verify above is not vacuous -- the guard is live."""
    buffer = allocate_guarded((8, 512), torch.float16, "cuda", alignment=16)
    x = torch.randn(8, 1024, dtype=torch.float16, device="cuda")
    reset_guards()
    flashinfer.silu_and_mul(x, out=buffer.payload, enable_pdl=False)
    verify_guards()

    inject_out_of_bounds_write(buffer, "suffix", size=16)

    with pytest.raises(GuardZoneCorruption) as excinfo:
        verify_guards()
    (hit,) = excinfo.value.hits
    assert (hit.region, hit.offset, hit.corrupted_bytes) == ("suffix", 0, 16)


def test_real_kernel_stays_correct_while_the_guard_reports_corruption():
    """A3-04 on the real call: correct output and guard failure are separate."""
    buffer = allocate_guarded((8, 512), torch.float16, "cuda", alignment=16)
    x = torch.randn(8, 1024, dtype=torch.float16, device="cuda")
    reset_guards()

    out = flashinfer.silu_and_mul(x, out=buffer.payload, enable_pdl=False)
    assert_matches_silu_reference(out, x)

    inject_out_of_bounds_write(buffer, "prefix", size=16)

    # Same output, still correct: the guard is the only witness.
    assert_matches_silu_reference(out, x)
    with pytest.raises(GuardZoneCorruption) as excinfo:
        verify_guards()
    (hit,) = excinfo.value.hits
    assert (hit.region, hit.expected) == ("prefix", PREFIX_SENTINEL)
