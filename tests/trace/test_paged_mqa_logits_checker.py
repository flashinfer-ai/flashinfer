# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The paged-MQA trace checker must reject output it cannot vouch for.

This checker is the oracle for both the FP8 and FP4 trace templates, so a
false pass here silently disarms both. It previously returned True in two
cases where it had compared nothing:

  * every non-finite position was dropped from `valid` -- including positions
    where the KERNEL emitted NaN -- so an all-NaN output emptied `valid` and
    `valid.any()` short-circuited to True;
  * `rel_l2` was set to 0.0 when the reference norm was zero, and no threshold
    can exceed 0.0, so an all-zero reference accepted any actual.

Raised by coderabbitai and endorsed by the reviewer on PR #4365
(r3725824140). These tests drive the checker directly rather than through a
kernel, so they run anywhere and fail for exactly one reason.
"""

import pytest
import torch

from flashinfer.trace.templates.attn_scores import _paged_mqa_logits_masked_check

DEVICE = "cpu"  # the checker is pure tensor math; no GPU needed


def _ctx(rows=2, max_len=8, next_n=1):
    """context_lens/next_n that make the whole row causally valid."""
    return dict(
        context_lens=torch.full((rows,), max_len, dtype=torch.int32, device=DEVICE),
        next_n=next_n,
    )


def test_all_nan_actual_is_rejected():
    """A kernel emitting NaN everywhere must fail, not pass vacuously."""
    ref = torch.randn(2, 8, device=DEVICE)
    act = torch.full((2, 8), float("nan"), device=DEVICE)
    assert not _paged_mqa_logits_masked_check(ref, act, **_ctx())


def test_partial_nan_actual_is_rejected():
    """A single NaN inside the causal region is still a kernel failure.

    The old filter removed just that position and compared the rest, so a
    kernel that was correct except for one poisoned element passed.
    """
    ref = torch.randn(2, 8, device=DEVICE)
    act = ref.clone()
    act[1, 3] = float("nan")
    assert not _paged_mqa_logits_masked_check(ref, act, **_ctx())


def test_inf_actual_is_rejected():
    ref = torch.randn(2, 8, device=DEVICE)
    act = ref.clone()
    act[0, 0] = float("inf")
    assert not _paged_mqa_logits_masked_check(ref, act, **_ctx())


def test_nonzero_actual_against_all_zero_reference_is_rejected():
    """An all-zero reference must not accept an arbitrary actual.

    With rnorm == 0 the relative error is undefined; substituting 0.0 made
    every actual pass.
    """
    ref = torch.zeros(2, 8, device=DEVICE)
    act = torch.full((2, 8), 5.0, device=DEVICE)
    assert not _paged_mqa_logits_masked_check(ref, act, **_ctx())


def test_zero_actual_against_all_zero_reference_is_accepted():
    """The legitimate all-zero case must still pass."""
    ref = torch.zeros(2, 8, device=DEVICE)
    act = torch.zeros(2, 8, device=DEVICE)
    assert _paged_mqa_logits_masked_check(ref, act, **_ctx())


def test_matching_output_is_accepted():
    """Guard against the fix over-rejecting: identical tensors must pass."""
    ref = torch.randn(2, 8, device=DEVICE)
    assert _paged_mqa_logits_masked_check(ref, ref.clone(), **_ctx())


def test_quantization_noise_is_accepted():
    """Small fp8/fp4-scale noise must still pass, or the oracle is useless."""
    ref = torch.randn(2, 8, device=DEVICE) * 10.0
    act = ref + torch.randn(2, 8, device=DEVICE) * 1e-3
    assert _paged_mqa_logits_masked_check(ref, act, **_ctx())


def test_nonfinite_reference_is_still_tolerated():
    """A non-finite REFERENCE stays tolerated: that is the fp8/fp4 rationale.

    Only the actual side is treated as a failure signal.
    """
    ref = torch.randn(2, 8, device=DEVICE)
    ref[0, 5] = float("inf")
    act = ref.clone()
    act[0, 5] = 1.0  # kernel produced something finite where the ref blew up
    assert _paged_mqa_logits_masked_check(ref, act, **_ctx())


def test_nonfinite_actual_outside_the_causal_region_is_ignored():
    """Beyond a row's causal limit the kernel may write anything, incl. -inf."""
    rows, max_len = 2, 8
    ref = torch.randn(rows, max_len, device=DEVICE)
    act = ref.clone()
    # context_lens=4 with next_n=1 makes positions >4 non-causal
    act[:, 6:] = float("-inf")
    ref[:, 6:] = float("-inf")
    ctx = dict(
        context_lens=torch.full((rows,), 4, dtype=torch.int32, device=DEVICE),
        next_n=1,
    )
    assert _paged_mqa_logits_masked_check(ref, act, **ctx)


def test_systematic_scale_error_is_rejected():
    """The rel-L2 floor must still catch a mis-scaled kernel."""
    ref = torch.randn(2, 8, device=DEVICE) * 10.0
    assert not _paged_mqa_logits_masked_check(ref, ref * 1.5, **_ctx())


def test_shape_mismatch_is_rejected():
    ref = torch.randn(2, 8, device=DEVICE)
    act = torch.randn(2, 9, device=DEVICE)
    assert not _paged_mqa_logits_masked_check(ref, act, **_ctx())


@pytest.mark.parametrize("wrap", [list, tuple])
def test_accepts_sequence_outputs(wrap):
    """Templates may hand the checker a 1-tuple/list; behaviour must not change."""
    ref = torch.randn(2, 8, device=DEVICE)
    act = torch.full((2, 8), float("nan"), device=DEVICE)
    assert not _paged_mqa_logits_masked_check(wrap([ref]), wrap([act]), **_ctx())
