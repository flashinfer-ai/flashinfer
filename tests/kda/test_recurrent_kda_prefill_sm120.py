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
"""SM120a ordinary multi-token prefill, through the public API.

Two layers, and the split matters. Most of the file drives
:func:`flashinfer.recurrent_kda` -- the whole chain from the public signature
through eligibility, dispatch, output and state adaptation to the kernel --
because a backend that is correct but never selected is a backend that does not
work. A smaller set pins ``decomp`` and ``fused`` individually through the
internal entry point, since ``auto`` picks one per shape and a suite that only
went through ``auto`` would leave whichever variant the runner's SM count does
not choose entirely untested.

The reference is deliberately self-contained and deliberately slow: a
token-serial recurrence in float32, rounding to bfloat16 at exactly the points
the public ABI rounds. It fixes the *contract*, not the schedule, so it is not
an implementation-shaped reference and it must not be replaced by one -- two
references that share a derivation cannot disagree, which is the whole reason
for having a second one.

On a runner that is not SM120a everything here still collects, the host-only
eligibility and facade cases still run, and the kernel cases skip with a reason
that names what was missing.
"""

from __future__ import annotations

import gc
import os
import random
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.utils import get_compute_capability, is_sm120a_supported

HEAD_DIM = 128

#: Elementwise tolerance against the reference below, as
#: ``abs(got - ref) <= ATOL + RTOL * abs(ref)``.
#:
#: Output and final state get their own numbers on purpose, and the state's are
#: looser for a reason that is worth writing down rather than discovering
#: again. The reference here is *contract*-shaped: it walks tokens one at a
#: time and rounds to bfloat16 where the public ABI does. The kernels are
#: chunk-shaped: they accumulate a chunk in float32 and round at the chunk
#: boundary. Both are correct implementations of the same contract and they do
#: not associate identically, so a state element that has absorbed 128 tokens
#: can land one or two bfloat16 ULP apart -- measured at 3.9e-3 on ~5 of
#: 196608 elements at magnitude 0.26, which is exactly 2 ULP there.
#:
#: The numbers below admit about three times that and nothing more. They are
#: not a way to make a failure go away: the same kernels agree *bitwise* with
#: the standalone implementation used during validation, and their float64
#: error is identical to that implementation's, so anything this would newly
#: catch is a change in the kernel rather than in the rounding.
OUTPUT_RTOL, OUTPUT_ATOL = 1.0e-2, 1.0e-3
STATE_RTOL, STATE_ATOL = 2.0e-2, 6.0e-3

#: Overridable so a CI failure can be reproduced exactly.
SEED = int(os.environ.get("SEED", "0"))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _skip_if_not_sm120() -> None:
    if not torch.cuda.is_available():
        pytest.skip("SM120a KDA prefill requires CUDA")
    if get_compute_capability(torch.device("cuda")) != (12, 0):
        pytest.skip("SM120 KDA prefill requires compute capability 12.0")
    if not is_sm120a_supported(torch.device("cuda")):
        pytest.skip("SM120 KDA prefill requires a CUDA toolkit with sm_120a support")
    from flashinfer.kda_kernels import can_implement_kda_prefill_sm120

    if can_implement_kda_prefill_sm120 is None:
        pytest.skip("the SM120 KDA prefill backend failed to import")


def _sm120_prefill():
    """The internal package, for the variant-pinned cases."""
    from flashinfer.kda_kernels import sm120_prefill

    return sm120_prefill


# ---------------------------------------------------------------------------
# Inputs and reference
# ---------------------------------------------------------------------------


def _make_inputs_sm120(
    *,
    seq_lens,
    num_heads: int,
    packed: bool,
    initial_state: bool = False,
    seed: int = 0,
    offsets_dtype: torch.dtype = torch.int32,
):
    """Inputs matching the SM120 contract: bfloat16 throughout, equal heads.

    ``q`` and ``k`` are pre-normalized because the kernel L2-normalizes them
    itself; feeding already-unit rows keeps the reference's normalization from
    being the thing under test.
    """
    _seed_everything(seed)
    if packed:
        batch_size = 1
        seq_len = sum(seq_lens)
    else:
        if len(set(seq_lens)) != 1:
            raise ValueError("fixed inputs require equal sequence lengths")
        batch_size = len(seq_lens)
        seq_len = seq_lens[0]

    shape = (batch_size, seq_len, num_heads, HEAD_DIM)

    def normalized():
        raw = torch.randn(shape, dtype=torch.float32, device="cuda")
        return F.normalize(raw, p=2, dim=-1).to(torch.bfloat16)

    q = normalized()
    k = normalized()
    v = torch.randn(shape, dtype=torch.float32, device="cuda").to(torch.bfloat16)
    g = (0.1 * torch.randn(shape, dtype=torch.float32, device="cuda")).to(
        torch.bfloat16
    )
    beta = torch.randn(
        (batch_size, seq_len, num_heads), dtype=torch.float32, device="cuda"
    ).to(torch.bfloat16)
    A_log = 0.1 * torch.randn(num_heads, dtype=torch.float32, device="cuda")
    dt_bias = 0.1 * torch.randn(
        (num_heads, HEAD_DIM), dtype=torch.float32, device="cuda"
    )

    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)

    state = None
    if initial_state:
        state = (
            0.1
            * torch.randn(
                (len(seq_lens), num_heads, HEAD_DIM, HEAD_DIM),
                dtype=torch.float32,
                device="cuda",
            )
        ).to(torch.bfloat16)

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "initial_state": state,
        "cu_seqlens": (
            torch.tensor(offsets, dtype=offsets_dtype, device="cuda")
            if packed
            else None
        ),
        "seq_lens": list(seq_lens),
    }


def _reference_kda_prefill(inputs, *, lower_bound=-5.0, scale=None):
    """Token-serial KDA prefill, rounding where the public ABI rounds.

    The state crosses a bfloat16 boundary once per token and the output once
    per row; everything between is float32. That is the contract the kernel
    implements, and reproducing it here is what makes the comparison a test of
    the kernel rather than of bfloat16.
    """
    q = inputs["q"]
    batch_size, seq_len, num_heads, head_dim = q.shape
    scale = head_dim**-0.5 if scale is None else scale

    q_flat = F.normalize(q.float(), dim=-1).reshape(-1, num_heads, head_dim)
    k_flat = F.normalize(inputs["k"].float(), dim=-1).reshape(-1, num_heads, head_dim)
    v_flat = inputs["v"].float().reshape(-1, num_heads, head_dim)
    g_flat = inputs["g"].float().reshape(-1, num_heads, head_dim)
    beta_flat = torch.sigmoid(inputs["beta"].float().reshape(-1, num_heads))

    gate = lower_bound * torch.sigmoid(
        torch.exp(inputs["A_log"]).reshape(1, num_heads, 1)
        * (g_flat + inputs["dt_bias"].reshape(1, num_heads, head_dim))
    )
    decay = torch.exp(gate)

    if inputs["cu_seqlens"] is None:
        offsets = [index * seq_len for index in range(batch_size + 1)]
    else:
        offsets = [int(value) for value in inputs["cu_seqlens"].tolist()]

    if inputs["initial_state"] is None:
        state = torch.zeros(
            (len(offsets) - 1, num_heads, head_dim, head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
    else:
        state = inputs["initial_state"].clone()

    out = torch.empty_like(q_flat)
    for sequence in range(len(offsets) - 1):
        for token in range(offsets[sequence], offsets[sequence + 1]):
            state_f32 = state[sequence].float()
            decayed = state_f32 * decay[token].unsqueeze(1)
            predicted = torch.einsum("hk,hvk->hv", k_flat[token], decayed)
            residual = beta_flat[token].unsqueeze(-1) * (v_flat[token] - predicted)
            updated = decayed + residual.unsqueeze(-1) * k_flat[token].unsqueeze(1)
            state[sequence] = updated.to(torch.bfloat16)
            projected = torch.einsum(
                "hk,hvk->hv", q_flat[token], state[sequence].float()
            )
            out[token] = (scale * projected).to(torch.bfloat16)
    return out.reshape_as(q), state


def _assert_elementwise(got, want, name, *, rtol, atol):
    """Compare with a per-element allowance and report the worst offender.

    ``torch.testing.assert_close`` would do the comparison; it would not say
    *which* element failed or by how much relative to its own allowance, and on
    a [1, 4096, 96, 128] tensor that is the difference between a diagnosable
    failure and a rerun.
    """
    a, b = got.float(), want.float()
    assert a.shape == b.shape, f"{name}: {tuple(a.shape)} vs {tuple(b.shape)}"
    assert not torch.isnan(a).any(), f"{name} has NaN"
    assert not torch.isinf(a).any(), f"{name} has Inf"
    tolerance = atol + rtol * b.abs()
    bad = (a - b).abs() > tolerance
    if bool(bad.any()):
        index = tuple(int(i) for i in torch.nonzero(bad)[0])
        raise AssertionError(
            f"{name}: {int(bad.sum())} of {a.numel()} outside tolerance; "
            f"max_abs={(a - b).abs().max().item():.3e} "
            f"max_normalized={((a - b).abs() / tolerance).max().item():.3e} "
            f"first at {index} got={a[index].item():.6e} ref={b[index].item():.6e}"
        )


def _call(inputs, **overrides):
    """Public-API call with this file's fixed gate settings."""
    kwargs = dict(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        beta_is_logit=True,
        lower_bound=-5.0,
        cu_seqlens=inputs["cu_seqlens"],
        initial_state=inputs["initial_state"],
    )
    kwargs.update(overrides)
    return flashinfer.recurrent_kda(**kwargs)


# ===========================================================================
# Eligibility. Host-only where it can be: an SM120 predicate must be testable
# without an SM120 device, or it can only be tested on the machine that least
# needs the answer.
# ===========================================================================


@pytest.fixture
def as_sm120_host(monkeypatch):
    """Make CPU tensors look like SM120 CUDA tensors to the predicate.

    Everything the rejection reason checks after the device is structural and
    device-independent, but the device check runs first -- so without this the
    thirty structural conditions could only be exercised on the one machine
    that least needs them checked.
    """
    from flashinfer import kda_prefill

    monkeypatch.setattr(kda_prefill, "get_compute_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        kda_prefill,
        "_is_contiguous_cuda_tensor",
        lambda tensor, *, dtype, device: (
            isinstance(tensor, torch.Tensor)
            and tensor.dtype == dtype
            and tensor.is_contiguous()
        ),
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    return kda_prefill


def _eligibility_kwargs(**overrides):
    """A structurally valid CPU-tensor call, for the rejection-reason tests.

    CPU tensors are the point: everything except the device check is structural
    and runs anywhere, so this exercises thirty conditions on a laptop.
    """
    heads = 2
    tokens = 16
    shape = (1, tokens, heads, HEAD_DIM)
    kwargs = dict(
        q=torch.zeros(shape, dtype=torch.bfloat16),
        k=torch.zeros(shape, dtype=torch.bfloat16),
        v=torch.zeros(shape, dtype=torch.bfloat16),
        g=torch.zeros(shape, dtype=torch.bfloat16),
        beta=torch.zeros((1, tokens, heads), dtype=torch.bfloat16),
        A_log=torch.zeros(heads),
        dt_bias=torch.zeros((heads, HEAD_DIM)),
        scale=None,
        initial_state=None,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        cu_seqlens=None,
        ssm_state_indices=None,
        num_spec_tokens=None,
        num_accepted_tokens=None,
        output=None,
        initial_state_source=None,
        initial_state_indices=None,
        beta_is_logit=True,
        seq_order=None,
        prefill_workspace=None,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=0,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("backend", ["auto", "cute-dsl"])
def test_public_sm120_dispatch_uses_cute_dsl_backend(monkeypatch, backend):
    """SM120 is selected by both automatic and explicit CuTe DSL routing."""
    from flashinfer import kda_prefill

    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill,
        "_sm120_kda_prefill_is_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill,
        "_run_sm120_kda_prefill",
        lambda **kwargs: sentinel,
    )

    assert (
        flashinfer.recurrent_kda(**_eligibility_kwargs(), backend=backend) is sentinel
    )


def test_public_sm120_dispatch_preserves_explicit_cake(monkeypatch):
    """An explicit Cake request must not probe or run the SM120 CuTe DSL path."""
    from flashinfer import kda_prefill

    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill,
        "_sm120_kda_prefill_is_eligible",
        lambda **kwargs: pytest.fail("backend='cake' must not probe SM120 CuTe DSL"),
    )
    monkeypatch.setattr(
        kda_prefill,
        "_flash_kda_prefill_is_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill,
        "_run_flash_kda_prefill",
        lambda **kwargs: sentinel,
    )

    assert flashinfer.recurrent_kda(**_eligibility_kwargs(), backend="cake") is sentinel


def test_public_sm120_capture_requires_preallocated_output(monkeypatch):
    """All recurrent prefill backends share one graph-stable output contract."""
    from flashinfer import kda_prefill

    call = _eligibility_kwargs()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(
        RuntimeError, match=r"capture requires a preallocated output tensor"
    ):
        kda_prefill._run_sm120_kda_prefill(
            q=call["q"],
            k=call["k"],
            v=call["v"],
            g=call["g"],
            beta=call["beta"],
            A_log=call["A_log"],
            dt_bias=call["dt_bias"],
            scale=call["scale"],
            initial_state=call["initial_state"],
            output_final_state=False,
            lower_bound=call["lower_bound"],
            cu_seqlens=call["cu_seqlens"],
            output=None,
            prefill_workspace=None,
        )


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"num_spec_tokens": 2}, "ordinary multi-token prefill"),
        ({"num_accepted_tokens": torch.zeros(1)}, "speculative-decode"),
        ({"ssm_state_indices": torch.zeros(1, dtype=torch.int32)}, "state pooling"),
        ({"seq_order": torch.zeros(1, dtype=torch.int32)}, "seq_order"),
        ({"checkpoint_every_n_tokens": 64}, "checkpoint"),
        ({"use_qk_l2norm_in_kernel": False}, "use_qk_l2norm_in_kernel"),
        ({"use_gate_in_kernel": False}, "use_gate_in_kernel"),
        ({"beta_is_logit": False}, "beta_is_logit"),
        ({"lower_bound": None}, "lower_bound"),
        ({"lower_bound": -6.0}, "lower_bound"),
        ({"lower_bound": 0.0}, "lower_bound"),
        ({"scale": float("inf")}, "scale"),
    ],
)
def test_sm120_rejection_reason_names_the_condition(overrides, expected):
    """Every refusal says which condition it was, in words a caller can act on."""
    from flashinfer import kda_prefill

    reason = kda_prefill._sm120_kda_prefill_rejection_reason(
        **_eligibility_kwargs(**overrides)
    )
    assert reason is not None
    assert expected in reason


def test_sm120_rejects_fixed_single_token():
    """``T=1`` is decode, and decode is not this backend's."""
    from flashinfer import kda_prefill

    shape = (1, 1, 2, HEAD_DIM)
    reason = kda_prefill._sm120_kda_prefill_rejection_reason(
        **_eligibility_kwargs(
            q=torch.zeros(shape, dtype=torch.bfloat16),
            k=torch.zeros(shape, dtype=torch.bfloat16),
            v=torch.zeros(shape, dtype=torch.bfloat16),
            g=torch.zeros(shape, dtype=torch.bfloat16),
            beta=torch.zeros((1, 1, 2), dtype=torch.bfloat16),
        )
    )
    assert reason is not None


def test_sm120_rejects_gqa_and_wrong_head_dim(as_sm120_host):
    """GQA and D != 128 are refused: this backend is equal-head and 128-wide."""
    kda_prefill = as_sm120_host
    heads, tokens = 4, 16
    q = torch.zeros((1, tokens, heads, HEAD_DIM), dtype=torch.bfloat16)
    v_gqa = torch.zeros((1, tokens, heads // 2, HEAD_DIM), dtype=torch.bfloat16)
    reason = kda_prefill._sm120_kda_prefill_rejection_reason(
        **_eligibility_kwargs(
            q=q,
            k=q,
            v=v_gqa,
            g=q,
            beta=torch.zeros((1, tokens, heads), dtype=torch.bfloat16),
            A_log=torch.zeros(heads),
            dt_bias=torch.zeros((heads, HEAD_DIM)),
        )
    )
    assert reason is not None and "GQA" in reason

    narrow = torch.zeros((1, tokens, heads, 64), dtype=torch.bfloat16)
    reason = kda_prefill._sm120_kda_prefill_rejection_reason(
        **_eligibility_kwargs(
            q=narrow,
            k=narrow,
            v=narrow,
            g=narrow,
            beta=torch.zeros((1, tokens, heads), dtype=torch.bfloat16),
            A_log=torch.zeros(heads),
            dt_bias=torch.zeros((heads, HEAD_DIM)),
        )
    )
    assert reason is not None and "head dimension" in reason


def test_sm120_rejects_float32_gate_and_state(as_sm120_host):
    """The first version publishes bfloat16 only, for both ``g`` and the state."""
    kda_prefill = as_sm120_host
    reason = kda_prefill._sm120_kda_prefill_rejection_reason(
        **_eligibility_kwargs(g=torch.zeros((1, 16, 2, HEAD_DIM)))
    )
    assert reason is not None and "g must be" in reason

    reason = kda_prefill._sm120_kda_prefill_rejection_reason(
        **_eligibility_kwargs(
            initial_state=torch.zeros((1, 2, HEAD_DIM, HEAD_DIM), dtype=torch.float32)
        )
    )
    assert reason is not None and "initial_state" in reason


def test_sm120_accepts_the_supported_shape_on_a_mocked_device(as_sm120_host):
    """The structural half passes on a call that differs only by device.

    Mocked rather than skipped: the question "does the predicate accept a valid
    call?" has nothing to do with which GPU is in the box, and a CI fleet with
    no SM120 runner would otherwise never ask it.
    """
    assert (
        as_sm120_host._sm120_kda_prefill_rejection_reason(**_eligibility_kwargs())
        is None
    )


@pytest.mark.parametrize("tokens", [2, 15, 16, 17])
def test_sm120_accepts_short_ordinary_prefill(tokens, as_sm120_host):
    """A short ``T`` is prefill unless it carries spec-decode metadata.

    The distinction is the metadata, not the length: a short prompt is an
    ordinary prefill, and refusing it because speculative decode happens to use
    similar lengths would give up real shapes for nothing.
    """
    kda_prefill = as_sm120_host
    heads = 2
    shape = (1, tokens, heads, HEAD_DIM)
    kwargs = _eligibility_kwargs(
        q=torch.zeros(shape, dtype=torch.bfloat16),
        k=torch.zeros(shape, dtype=torch.bfloat16),
        v=torch.zeros(shape, dtype=torch.bfloat16),
        g=torch.zeros(shape, dtype=torch.bfloat16),
        beta=torch.zeros((1, tokens, heads), dtype=torch.bfloat16),
    )
    assert kda_prefill._sm120_kda_prefill_rejection_reason(**kwargs) is None

    kwargs["num_spec_tokens"] = tokens - 1
    assert kda_prefill._sm120_kda_prefill_rejection_reason(**kwargs) is not None


def test_sm120_is_not_eligible_on_other_architectures(as_sm120_host, monkeypatch):
    """SM100 and SM103 keep Cake; nothing here can take a call from them."""
    for capability in ((10, 0), (10, 3), (9, 0)):
        monkeypatch.setattr(
            as_sm120_host, "get_compute_capability", lambda device, c=capability: c
        )
        reason = as_sm120_host._sm120_kda_prefill_rejection_reason(
            **_eligibility_kwargs()
        )
        assert reason is not None and "compute capability" in reason


def test_sm120_facade_imports_without_device_code():
    """Importing the facade must not load a device module or compile anything.

    This is what makes ``import flashinfer`` cheap for the majority of callers
    who never run KDA prefill on SM120, and it is easy to lose: one
    module-level ``from . import decomp`` would undo it and nothing else in the
    suite would notice.
    """
    import sys

    from flashinfer import kda_kernels

    assert hasattr(kda_kernels, "can_implement_kda_prefill_sm120")
    loaded = {name for name in sys.modules if "sm120_prefill" in name}
    assert "flashinfer.kda_kernels.sm120_prefill.decomp" not in loaded
    assert "flashinfer.kda_kernels.sm120_prefill.fused" not in loaded


def test_sm120_can_implement_is_fail_closed_on_cpu():
    """A CPU tensor is not eligible, and asking costs nothing."""
    from flashinfer.kda_kernels import can_implement_kda_prefill_sm120

    if can_implement_kda_prefill_sm120 is None:
        pytest.skip("SM120 backend unavailable")
    assert not can_implement_kda_prefill_sm120(q=torch.zeros(1, 2, 1, HEAD_DIM))


def test_sm120_variant_policy_reports_tuned_or_fallback():
    """The policy line must say whether this device's thresholds were measured.

    A time quoted under fallback thresholds is not a time quoted under tuned
    ones, and nothing else in a benchmark's output distinguishes them.
    """
    sm120_prefill = _sm120_prefill()

    tuned = sm120_prefill.describe_variant_policy(156)
    assert "measured on this device" in tuned

    fallback = sm120_prefill.describe_variant_policy(999)
    assert "FALLBACK" in fallback


def test_sm120_variant_choice_is_a_pure_function_of_shape():
    """``choose_variant`` needs no device, so its table can be tested anywhere."""
    sm120_prefill = _sm120_prefill()
    choose = sm120_prefill.choose_variant

    # Every measured row, at the CTA value on each side of its own threshold.
    # Written as a table rather than as prose assertions because the previous
    # version described `H >= 96 or T <= 48 or CTA >= 512` -- thresholds two
    # re-fits out of date -- and still passed, since none of its cases sat near
    # a boundary that had moved.
    #
    # (sm_count, cta_threshold): the profile's own CTA line.
    for sm_count, threshold in ((110, 128), (156, 144), (188, 144)):
        # CTA = 2 * batch * heads, so batch = threshold // (2 * heads) puts the
        # shape exactly on the line, and one less batch puts it just under.
        heads = 8
        on = threshold // (2 * heads)
        assert choose(on, heads, 1024, sm_count=sm_count) == "fused", sm_count
        assert choose(on - 1, heads, 1024, sm_count=sm_count) == "decomp", sm_count

        # The tokens term is 32 on every row, and it fires independently of CTA.
        assert choose(1, 1, 32, sm_count=sm_count) == "fused", sm_count
        assert choose(1, 1, 33, sm_count=sm_count) == "decomp", sm_count

        # No row has a heads term, and it cannot be tested through `choose`:
        # CTA is 2 * batch * heads, so the smallest CTA any H >= 96 shape can
        # have is 192, already over every threshold above.  That is the reason
        # the term is None -- it could never fire -- so assert the field.
        assert sm120_prefill.AUTO_PROFILES[sm_count].heads is None, sm_count

    # The two larger parts sit at 144 and the 110-SM part at 128, so CTA 128
    # separates them.  This is the case the re-fit turned on, and the one a
    # single shared row would get wrong.
    assert choose(8, 8, 1024, sm_count=110) == "fused"  # CTA 128 >= 128
    assert choose(8, 8, 1024, sm_count=156) == "decomp"  # CTA 128 < 144
    assert choose(8, 8, 1024, sm_count=188) == "decomp"

    # CTA 144 is the shape FlashInfer benchmarks (six sequences, H=12) and the
    # reason the threshold moved off 192.  All three rows must take it now.
    for sm_count in (110, 156, 188):
        assert choose(6, 12, 8192, sm_count=sm_count) == "fused", sm_count

    # An unmeasured SM count falls back rather than extrapolating.
    assert choose(1, 96, 1024, sm_count=999) == choose(1, 96, 1024, sm_count=156)


def test_sm120_bound_workspace_rejects_an_explicit_variant_change():
    """A bound workspace must not silently ignore a later explicit variant."""
    sm120_prefill = _sm120_prefill()
    resources = SimpleNamespace(variant="decomp")

    assert (
        sm120_prefill._resolve_variant("auto", None, None, True, resources, device=None)
        == "decomp"
    )
    with pytest.raises(ValueError, match=r"already bound.*decomp.*fused"):
        sm120_prefill._resolve_variant(
            "fused", None, None, True, resources, device=None
        )


def test_sm120_resolved_cache_rejects_a_recycled_tensor_identity(monkeypatch):
    """An address-key match is not enough when the owning tensor changed."""
    sm120_prefill = _sm120_prefill()
    sm120_prefill._RESOLVED.clear()
    sm120_prefill._RESOLVED_LAST = None
    monkeypatch.setattr(sm120_prefill, "current_stream_ptr", lambda device: 7)
    # Simulate two allocations receiving the same address and layout without
    # depending on the CPU allocator to recycle one during the test.
    monkeypatch.setattr(
        sm120_prefill,
        "tensor_identity",
        lambda tensor: None if tensor is None else ("recycled-address",),
    )

    first = torch.zeros(1)
    second = torch.zeros(1)
    scalars = (1.0, -5.0, "auto", True)
    value = (object(), object())
    try:
        sm120_prefill._remember_call(
            torch.device("cpu"), (first, None), scalars, None, value
        )
        assert (
            sm120_prefill._resolved_call(
                torch.device("cpu"), (first, None), scalars, None
            )
            is value
        )
        assert (
            sm120_prefill._resolved_call(
                torch.device("cpu"), (second, None), scalars, None
            )
            is None
        )
        assert not sm120_prefill._RESOLVED
    finally:
        sm120_prefill._RESOLVED.clear()
        sm120_prefill._RESOLVED_LAST = None


def test_sm120_resolved_fast_path_rejects_rebound_inference_storage(monkeypatch):
    """Rebinding ``.data`` must invalidate the one-entry facade fast path."""
    sm120_prefill = _sm120_prefill()
    sm120_prefill._RESOLVED.clear()
    sm120_prefill._RESOLVED_LAST = None
    monkeypatch.setattr(sm120_prefill, "current_stream_ptr", lambda device: 7)

    scalars = (1.0, -5.0, "auto", True)
    value = (object(), object())
    try:
        with torch.inference_mode():
            tensor = torch.zeros(1)
            replacement = torch.ones(2)
            sm120_prefill._remember_call(
                torch.device("cpu"), (tensor, None), scalars, None, value
            )
            assert (
                sm120_prefill._resolved_call(
                    torch.device("cpu"), (tensor, None), scalars, None
                )
                is value
            )

            tensor.data = replacement
            assert (
                sm120_prefill._resolved_call(
                    torch.device("cpu"), (tensor, None), scalars, None
                )
                is None
            )
    finally:
        sm120_prefill._RESOLVED.clear()
        sm120_prefill._RESOLVED_LAST = None


def test_sm120_flat_output_range_is_checked_on_the_host():
    """The flat output is bounded before a launch, not after it.

    Two limits meet one element apart. The tail store writes a partial chunk
    element-wise through ``(token * H + head) * DV + d``, which the device
    builds and consumes as INT32, so it needs the largest *index* -- ``T_total *
    H * DV - 1`` -- to fit. The DSL packs the flat view's extent as INT32 when
    it crosses into the compiled entry, so it needs the *count*. The count is
    therefore the bound, and it is the one asserted here: on hardware the shape
    one element inside the index limit does not launch, it raises
    ``OverflowError: Value overflow: 2147483648 exceeds range of l`` out of
    ``build_memref_desc``, which names neither the tensor nor the shape.
    """
    from flashinfer.kda_kernels.sm120_prefill import runtime

    heads = 1024
    largest_tokens = runtime.INT32_MAX // (heads * runtime.DV)
    assert largest_tokens * heads * runtime.DV <= runtime.INT32_MAX
    assert (largest_tokens + 1) * heads * runtime.DV > runtime.INT32_MAX

    runtime.check_flat_output_range(largest_tokens, heads)
    # A state-only call has no store, so it has nothing to bound.
    runtime.check_flat_output_range(0, heads)

    # One token more is the count the DSL cannot pack; two more is the index
    # the device cannot build. Both are the backend's error, not a stray one.
    for excess in (1, 2):
        with pytest.raises(runtime.KDAPrefillValidationError, match="INT32"):
            runtime.check_flat_output_range(largest_tokens + excess, heads)


def test_sm120_variant_policy_reads_the_input_device(monkeypatch):
    """A host holding two CC 12.0 parts must not drive both from one row.

    The measured rows disagree at CTA 128: the 110-SM part takes the fused
    kernel there and the 188-SM part takes the decomposed one. Reading device 0
    would apply the wrong row to every call on the other card, and the only
    symptom is a time nobody can attribute.
    """
    sm120_prefill = _sm120_prefill()

    def get_device_properties(device):
        index = device if isinstance(device, int) else torch.device(device).index
        return SimpleNamespace(
            multi_processor_count=110 if index == 0 else 188,
            name=f"<part {index}>",
        )

    sm120_prefill._sm_count.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    try:
        choose = sm120_prefill.choose_variant
        # CTA = 2 * batch * heads = 128: on the 110-SM row's threshold and
        # under the 188-SM row's.
        assert choose(8, 8, 1024, device=torch.device("cuda", 0)) == "fused"
        assert choose(8, 8, 1024, device=torch.device("cuda", 1)) == "decomp"
        # No device named still means the current one.
        assert choose(8, 8, 1024) == "fused"
        # And the policy line names the card it actually read.
        assert "188 SMs" in sm120_prefill.describe_variant_policy(
            device=torch.device("cuda", 1)
        )
    finally:
        sm120_prefill._sm_count.cache_clear()


def test_sm120_pinned_staging_carries_the_event_it_reuses_against():
    """A descriptor build may not refill a buffer whose transfer is pending.

    ``upload_bytes`` copies out of one pinned buffer per size with
    ``non_blocking=True``, so the transfer is queued rather than finished when
    it returns. Two cold builds of one size are ordinary rather than a corner --
    the size is a function of the descriptor count, so any two cold calls of a
    shape collide -- and without the event the second one's host-side refill
    overwrites bytes the first one's DMA has not read yet, which reaches the
    device as a descriptor made of two.

    The event is what that costs, so the event is what is asserted: the byte
    comparison below would pass on the racing version too, most of the time,
    which is exactly why it cannot be the guard.
    """
    _skip_if_not_sm120()
    from flashinfer.kda_kernels.sm120_prefill import runtime

    device = torch.device("cuda")
    first_payload = bytes(range(128)) * 2
    second_payload = bytes(reversed(range(128))) * 2

    runtime.clear_pinned_staging()
    first = runtime.upload_bytes(first_payload, device)
    staging, event = runtime._PINNED_STAGING[len(first_payload)]
    assert event is not None, "the pooled buffer must carry its upload's event"

    second = runtime.upload_bytes(second_payload, device)
    pooled, _ = runtime._PINNED_STAGING[len(second_payload)]
    assert pooled is staging, "one buffer per size is the point of the pool"

    torch.cuda.synchronize()
    assert bytes(first.cpu().tolist()) == first_payload
    assert bytes(second.cpu().tolist()) == second_payload


def test_sm120_clear_pinned_staging_waits_for_pending_uploads():
    """Cache clearing cannot drop a pinned buffer before its H2D completes."""
    from flashinfer.kda_kernels.sm120_prefill import runtime

    class Pending:
        synchronized = False

        def synchronize(self):
            self.synchronized = True

    pending = Pending()
    runtime.clear_pinned_staging()
    runtime._PINNED_STAGING[17] = (object(), pending)
    try:
        runtime.clear_pinned_staging()
        assert pending.synchronized
        assert not runtime._PINNED_STAGING
    finally:
        runtime._PINNED_STAGING.clear()


def test_sm120_tma_alignment_constant_matches_the_runtime():
    """The adapter's alignment gate and the descriptor's must be one number.

    ``flashinfer/kda_prefill.py`` refuses a misaligned base before the backend
    is reached; ``TensorMapSpec.validate`` refuses it again at the descriptor.
    They are separate constants in separate modules, and nothing but this
    assertion stops one from drifting -- which would turn a clear refusal into
    a driver error a long way from its cause.
    """
    from flashinfer import kda_prefill
    from flashinfer.kda_kernels.sm120_prefill import runtime

    assert kda_prefill._SM120_TMA_BASE_ALIGN == runtime.GLOBAL_BASE_ALIGN


def test_sm120_call_memos_hold_weak_references():
    """The three per-call memos may not be what keeps a caller's buffers alive.

    Each is keyed on tensor identity, and the obvious way to re-check that
    identity on the next call is to hold the tensors. Holding them puts one
    whole activation set -- q, k, v, g and out -- into steady-state device
    memory and keeps those blocks away from the caching allocator until the
    next call replaces them. The LRU behind each memo already used weak
    references; the fast paths in the facade and in both variants did not.

    Structural rather than an allocator measurement, and deliberately so: a
    live plan legitimately retains the buffers its descriptors and views
    address, at the C level where ``gc`` cannot see it, so a byte count here
    would be measuring the plan cache instead of these memos.
    """
    _skip_if_not_sm120()
    sm120_prefill = _sm120_prefill()

    inputs = _make_inputs_sm120(seq_lens=[64], num_heads=4, packed=False, seed=SEED)
    _call(inputs)

    memos = {"facade": sm120_prefill._RESOLVED_LAST}
    for name, module in sm120_prefill._MODULES.items():
        memos[name] = module._LAST
    assert any(memo is not None for memo in memos.values()), "no memo was written"

    for name, memo in memos.items():
        if memo is None:
            continue
        for held in memo[0]:
            assert held is None or isinstance(held, weakref.ref), (
                f"the {name} memo holds a strong reference: {type(held).__name__}"
            )


def test_sm120_cache_clear_iterates_a_module_snapshot(monkeypatch):
    """A variant published during cleanup cannot invalidate the iteration."""
    sm120_prefill = _sm120_prefill()
    cleared = []
    late = SimpleNamespace(clear_caches=lambda: cleared.append("late"))

    def clear_first():
        cleared.append("first")
        sm120_prefill._MODULES["late"] = late

    monkeypatch.setattr(
        sm120_prefill,
        "_MODULES",
        {"first": SimpleNamespace(clear_caches=clear_first)},
    )
    sm120_prefill.clear_kda_prefill_sm120_caches()

    assert cleared == ["first"]
    assert sm120_prefill._MODULES["late"] is late


def test_sm120_workspace_resources_are_created_once_under_threads():
    """Two first calls on one workspace must not each build their own state.

    The loser of that race would run against an orphan: its own lock, so
    nothing serializes the launch sequence the lock exists for; its own scratch,
    so device memory doubles; and its own capture flag, so a capture is recorded
    where nobody will look for it.
    """
    _skip_if_not_sm120()
    from flashinfer import kda_prefill

    workspace = kda_prefill.RecurrentKDAPrefillWorkspace(torch.device("cuda"))
    threads_count = 8
    barrier = threading.Barrier(threads_count)
    seen: list = []
    lock = threading.Lock()

    def _create():
        barrier.wait()
        resources = kda_prefill._sm120_prefill_resources(workspace, workspace.device)
        with lock:
            seen.append(resources)

    threads = [threading.Thread(target=_create) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(seen) == threads_count
    assert all(resources is seen[0] for resources in seen)
    assert seen[0] is workspace._sm120_state


def test_sm120_spent_workspace_check_and_scratch_share_one_hold():
    """A spent workspace must be refused before its scratch can be replaced.

    The check and the replacement are two steps on one object, so splitting
    them across the lock leaves a window: a thread that read the flag as False
    can install a new ``state_scratch`` after another thread's capture has
    already recorded the old buffer's address, and the replay then reads a
    buffer the workspace no longer references. The backend re-checks the flag,
    which orders the launches, but it cannot undo a replacement.

    Driven through the public API from several threads because that is the only
    way the two steps interleave at all; the assertion is that the refusal wins
    every time, whichever thread gets there first.
    """
    _skip_if_not_sm120()
    from flashinfer import kda_prefill

    workspace = kda_prefill.RecurrentKDAPrefillWorkspace(torch.device("cuda"))
    inputs = _make_inputs_sm120(seq_lens=[64], num_heads=4, packed=False, seed=SEED)
    output = torch.empty_like(inputs["v"])
    _call(inputs, output=output, prefill_workspace=workspace, output_final_state=True)

    resources = workspace._sm120_state
    assert resources is not None
    scratch = resources.state_scratch

    # Spend it exactly as a capture would, then let several threads in at once.
    resources.captured = True
    workspace._captured = True

    threads_count = 8
    barrier = threading.Barrier(threads_count)
    outcomes: list = []
    guard = threading.Lock()

    def _reuse():
        barrier.wait()
        try:
            _call(
                inputs,
                output=output,
                prefill_workspace=workspace,
                output_final_state=True,
            )
            result = "accepted"
        except RuntimeError as exc:
            result = "refused" if "capture" in str(exc) else f"other: {exc}"
        with guard:
            outcomes.append(result)

    threads = [threading.Thread(target=_reuse) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes == ["refused"] * threads_count, outcomes
    assert resources.state_scratch is scratch, "a refused call replaced the scratch"


def test_sm120_cute_dsl_request_does_not_intercept_a_non_prefill_call():
    """Declining a call may not change where that call goes.

    This backend records why it declined so that an explicit ``"cute-dsl"``
    request can be answered accurately further down, and recording is all it
    may do: decode reaches the same dispatcher, and answering there would take
    a call the decode path owns. Compared against ``auto`` rather than asserted
    absolutely, because what decode makes of these tensors is not the subject --
    only that naming a backend did not reroute it.
    """
    _skip_if_not_sm120()
    inputs = _make_inputs_sm120(seq_lens=[1], num_heads=4, packed=False, seed=SEED)

    def outcome(backend):
        try:
            _call(inputs, backend=backend)
        except Exception as exc:  # noqa: BLE001 -- the type and text are the assertion
            return type(exc).__name__, str(exc)
        return "ok", ""

    assert outcome("cute-dsl") == outcome("auto")


def test_sm120_cute_dsl_request_names_the_reason_this_backend_refused():
    """``backend="cute-dsl"`` on CC 12.0 must say what it could not support.

    The CC 10.0/10.3 block below this backend answers such a request with a
    message about the *contract*, which on this architecture names neither the
    argument at fault nor the architecture that refused it.
    """
    _skip_if_not_sm120()

    inputs = _make_inputs_sm120(seq_lens=[64], num_heads=4, packed=False, seed=SEED)
    seq_order = torch.zeros(1, dtype=torch.int32, device=inputs["q"].device)
    with pytest.raises(ValueError, match="seq_order"):
        _call(inputs, backend="cute-dsl", seq_order=seq_order)


# ===========================================================================
# Correctness through the public API.
# ===========================================================================


@torch.inference_mode()
@pytest.mark.parametrize("seq_lens", [[16], [17], [15], [64], [65], [63], [128]])
@pytest.mark.parametrize("num_heads", [2, 8])
@pytest.mark.parametrize("has_initial_state", [False, True])
def test_recurrent_kda_prefill_sm120_fixed_matches_reference(
    seq_lens, num_heads, has_initial_state
):
    """Fixed-length input, every chunk-boundary case around 16 and 64."""
    _skip_if_not_sm120()
    inputs = _make_inputs_sm120(
        seq_lens=seq_lens,
        num_heads=num_heads,
        packed=False,
        initial_state=has_initial_state,
        seed=SEED,
    )
    expected_out, expected_state = _reference_kda_prefill(inputs)

    out, state = _call(inputs, output_final_state=True)

    _assert_elementwise(out, expected_out, "output", rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
    assert state is not None
    _assert_elementwise(
        state, expected_state, "final_state", rtol=STATE_RTOL, atol=STATE_ATOL
    )


@torch.inference_mode()
@pytest.mark.parametrize(
    "seq_lens", [[16], [15, 32, 65], [1, 16, 129], [17, 17], [64, 1, 33]]
)
@pytest.mark.parametrize("offsets_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("has_initial_state", [False, True])
def test_recurrent_kda_prefill_sm120_packed_matches_reference(
    seq_lens, offsets_dtype, has_initial_state
):
    """Packed varlen, mixed lengths, and both offsets dtypes the API accepts."""
    _skip_if_not_sm120()
    inputs = _make_inputs_sm120(
        seq_lens=seq_lens,
        num_heads=4,
        packed=True,
        initial_state=has_initial_state,
        seed=SEED,
        offsets_dtype=offsets_dtype,
    )
    expected_out, expected_state = _reference_kda_prefill(inputs)

    out, state = _call(inputs, output_final_state=True)

    _assert_elementwise(out, expected_out, "output", rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
    assert state is not None
    _assert_elementwise(
        state, expected_state, "final_state", rtol=STATE_RTOL, atol=STATE_ATOL
    )


@torch.inference_mode()
@pytest.mark.parametrize("variant", ["decomp", "fused"])
@pytest.mark.parametrize("packed", [False, True])
def test_recurrent_kda_prefill_sm120_variants_match_reference(variant, packed):
    """Each variant on its own, so neither is left untested by ``auto``.

    ``auto`` picks one per shape, and which one depends on the runner's SM
    count -- so a suite that only entered through ``auto`` would silently test
    one variant on one machine and the other on another.
    """
    _skip_if_not_sm120()
    sm120_prefill = _sm120_prefill()

    seq_lens = [17, 33] if packed else [64, 64]
    inputs = _make_inputs_sm120(
        seq_lens=seq_lens, num_heads=4, packed=packed, initial_state=True, seed=SEED
    )
    expected_out, expected_state = _reference_kda_prefill(inputs)

    out = torch.empty_like(inputs["v"])
    state = inputs["initial_state"]
    sm120_prefill.run_kda_prefill_sm120(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        lower_bound=-5.0,
        initial_state=state,
        final_state=state,
        cu_seqlens=inputs["cu_seqlens"],
        output=out,
        variant=variant,
    )

    _assert_elementwise(out, expected_out, "output", rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
    _assert_elementwise(
        state, expected_state, "final_state", rtol=STATE_RTOL, atol=STATE_ATOL
    )


@torch.inference_mode()
@pytest.mark.parametrize("variant", ["decomp", "fused"])
@pytest.mark.parametrize("exact_alias", [False, True])
def test_recurrent_kda_prefill_sm120_zero_tokens_preserves_fp32_state(
    variant, exact_alias
):
    """Both backend variants implement the same state-only ABI."""
    _skip_if_not_sm120()
    heads = 2
    shape = (1, 0, heads, HEAD_DIM)
    q = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    initial = torch.randn(
        (1, heads, HEAD_DIM, HEAD_DIM), dtype=torch.float32, device="cuda"
    )
    expected = initial.clone()
    final = initial if exact_alias else torch.empty_like(initial)

    _sm120_prefill().run_kda_prefill_sm120(
        q=q,
        k=torch.empty_like(q),
        v=torch.empty_like(q),
        g=torch.empty_like(q),
        beta=torch.empty((1, 0, heads), dtype=torch.bfloat16, device="cuda"),
        A_log=torch.zeros(heads, dtype=torch.float32, device="cuda"),
        dt_bias=torch.zeros((heads, HEAD_DIM), dtype=torch.float32, device="cuda"),
        lower_bound=-5.0,
        initial_state=initial,
        final_state=final,
        output=torch.empty_like(q),
        variant=variant,
    )

    torch.testing.assert_close(final, expected, rtol=0.0, atol=0.0)


@torch.inference_mode()
def test_recurrent_kda_prefill_sm120_auto_enters_the_sm120_backend(monkeypatch):
    """``auto`` through the public API really reaches this backend.

    Asserted with a spy rather than inferred from a timing or from the answer
    being right: both would also be true if the call had quietly fallen through
    to another implementation, which is the failure worth catching.
    """
    _skip_if_not_sm120()
    sm120_prefill = _sm120_prefill()

    seen = []
    original = sm120_prefill.choose_variant

    def spy(*args, **kwargs):
        chosen = original(*args, **kwargs)
        seen.append(chosen)
        return chosen

    monkeypatch.setattr(sm120_prefill, "choose_variant", spy)

    inputs = _make_inputs_sm120(seq_lens=[64], num_heads=4, packed=False, seed=SEED)
    out, _ = _call(inputs, output_final_state=False)

    assert seen, "the SM120 auto selector was never consulted"
    assert seen[0] in ("decomp", "fused")
    assert out.shape == inputs["v"].shape


@torch.inference_mode()
def test_recurrent_kda_prefill_sm120_state_return_follows_the_public_contract():
    """State semantics: updated in place always, returned only when asked."""
    _skip_if_not_sm120()
    inputs = _make_inputs_sm120(
        seq_lens=[48], num_heads=4, packed=False, initial_state=True, seed=SEED
    )
    expected_out, expected_state = _reference_kda_prefill(inputs)

    initial = inputs["initial_state"]
    before = initial.clone()

    out, returned = _call(inputs, output_final_state=False)

    # Updated in place: that is what the caller's buffer is for.
    assert not torch.equal(initial, before)
    _assert_elementwise(
        initial, expected_state, "in-place state", rtol=STATE_RTOL, atol=STATE_ATOL
    )
    # ...and still not returned, because the caller did not ask for it.
    assert returned is None
    _assert_elementwise(out, expected_out, "output", rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)


@torch.inference_mode()
def test_recurrent_kda_prefill_sm120_allocates_a_final_state_when_asked():
    """No initial state, ``output_final_state=True`` -> a fresh bfloat16 state."""
    _skip_if_not_sm120()
    inputs = _make_inputs_sm120(seq_lens=[32, 48], num_heads=4, packed=True, seed=SEED)
    expected_out, expected_state = _reference_kda_prefill(inputs)

    out, state = _call(inputs, output_final_state=True)

    assert state is not None
    assert state.dtype is torch.bfloat16
    assert state.shape == (2, 4, HEAD_DIM, HEAD_DIM)
    _assert_elementwise(out, expected_out, "output", rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
    _assert_elementwise(
        state, expected_state, "final_state", rtol=STATE_RTOL, atol=STATE_ATOL
    )


@torch.inference_mode()
def test_recurrent_kda_prefill_sm120_writes_the_caller_output_buffer():
    """A caller-provided ``output`` is written in place and returned as-is."""
    _skip_if_not_sm120()
    inputs = _make_inputs_sm120(seq_lens=[64], num_heads=4, packed=False, seed=SEED)
    expected_out, _ = _reference_kda_prefill(inputs)

    provided = torch.empty_like(inputs["v"])
    out, _ = _call(inputs, output=provided, output_final_state=False)

    assert out is provided
    _assert_elementwise(out, expected_out, "output", rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)


@torch.inference_mode()
def test_recurrent_kda_prefill_sm120_does_not_modify_read_only_inputs():
    """Nothing the caller passed as an input comes back changed."""
    _skip_if_not_sm120()
    inputs = _make_inputs_sm120(seq_lens=[33, 64], num_heads=4, packed=True, seed=SEED)
    snapshots = {
        name: inputs[name].clone()
        for name in ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "cu_seqlens")
    }

    _call(inputs, output_final_state=True)

    for name, before in snapshots.items():
        assert torch.equal(inputs[name], before), f"{name} was modified"


@torch.inference_mode()
def test_recurrent_kda_prefill_sm120_chained_state_stays_bounded():
    """Feeding each call's final state into the next must not blow up.

    Long-chain drift is the failure a single-call comparison cannot see: a
    small per-token state error that a chunked schedule accumulates differently
    than a serial one shows up only after many chunks, and by then it looks
    like a tolerance problem rather than a schedule one.
    """
    _skip_if_not_sm120()
    inputs = _make_inputs_sm120(
        seq_lens=[128], num_heads=4, packed=False, initial_state=True, seed=SEED
    )
    state = inputs["initial_state"]
    reference_state = state.clone()

    errors = []
    for step in range(6):
        step_inputs = _make_inputs_sm120(
            seq_lens=[128], num_heads=4, packed=False, seed=SEED + step
        )
        step_inputs["initial_state"] = state
        expected_out, expected_state = _reference_kda_prefill(
            {**step_inputs, "initial_state": reference_state}
        )
        out, _ = _call(step_inputs, output_final_state=False)
        reference_state = expected_state
        errors.append((out.float() - expected_out.float()).abs().max().item())

    # Not a fixed threshold on the last step: the question is whether the error
    # *grows with chain length*, which a per-step bound cannot answer.
    assert errors[-1] <= max(errors[0], OUTPUT_ATOL) * 4.0, (
        f"state error grew with recurrence length: {errors}"
    )
    for step, error in enumerate(errors):
        assert error < 0.5, f"step {step} diverged: {errors}"


@torch.inference_mode()
def test_recurrent_kda_prefill_sm120_on_a_non_default_stream():
    """A non-default stream must be used, not merely tolerated.

    The plan caches key on the stream because a compiled entry bakes its
    ``CUstream`` into its argument tuple; a plan replayed from another stream
    would launch work correctly ordered against the wrong stream, and the
    numbers would still be right most of the time.
    """
    _skip_if_not_sm120()
    inputs = _make_inputs_sm120(seq_lens=[64], num_heads=4, packed=False, seed=SEED)
    expected_out, expected_state = _reference_kda_prefill(inputs)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out, state = _call(inputs, output_final_state=True)
    stream.synchronize()

    _assert_elementwise(out, expected_out, "output", rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
    _assert_elementwise(
        state, expected_state, "final_state", rtol=STATE_RTOL, atol=STATE_ATOL
    )


@torch.inference_mode()
def test_recurrent_kda_prefill_sm120_repeated_calls_agree():
    """The same inputs twice give the same answer, cold cache and warm.

    The second call takes the plan cache's fast path and the first does not, so
    a disagreement here means the cached plan and the freshly built one are not
    the same launch.
    """
    _skip_if_not_sm120()
    inputs = _make_inputs_sm120(seq_lens=[17, 48], num_heads=4, packed=True, seed=SEED)

    first_out, first_state = _call(inputs, output_final_state=True)
    first_out = first_out.clone()
    first_state = first_state.clone()

    second_out, second_state = _call(inputs, output_final_state=True)

    assert torch.equal(first_out, second_out)
    assert torch.equal(first_state, second_state)


@torch.inference_mode()
def test_recurrent_kda_prefill_sm120_concurrent_calls_are_serialized():
    """Two host threads on one stream must not interleave a decomp launch pair.

    The decomposed variant enqueues prepare and then recurrence against one
    scratch arena. CUDA orders launches within a stream, but nothing orders two
    Python threads submitting those pairs, so without a lock the second
    prepare can overwrite factors the first recurrence has not read.
    """
    _skip_if_not_sm120()
    sm120_prefill = _sm120_prefill()

    inputs = _make_inputs_sm120(seq_lens=[64], num_heads=4, packed=False, seed=SEED)
    expected_out, _ = _reference_kda_prefill(inputs)

    results = []
    errors = []

    def worker():
        # ``inference_mode`` is thread-local. The tensors above were created
        # inside it on this test's thread, and writing to one from a thread
        # that is not in inference mode raises -- so each worker enters it,
        # which is also what a real serving thread does.
        try:
            with torch.inference_mode():
                out = torch.empty_like(inputs["v"])
                sm120_prefill.run_kda_prefill_sm120(
                    q=inputs["q"],
                    k=inputs["k"],
                    v=inputs["v"],
                    g=inputs["g"],
                    beta=inputs["beta"],
                    A_log=inputs["A_log"],
                    dt_bias=inputs["dt_bias"],
                    lower_bound=-5.0,
                    cu_seqlens=None,
                    output=out,
                    variant="decomp",
                )
            results.append(out)
        except Exception as exc:  # noqa: BLE001 -- reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    torch.cuda.synchronize()

    assert not errors, errors
    for out in results:
        _assert_elementwise(
            out, expected_out, "concurrent output", rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL
        )


# ===========================================================================
# CUDA graph capture and replay.
# ===========================================================================


def _warm_and_capture(inputs, *, workspace, output, state, variant=None):
    """Eager warmup on the capture stream, then capture. Returns the graph.

    The order is the contract: warm with the exact tensors and stream, sync,
    then capture. Anything the capture would have had to build -- a compile, a
    descriptor, an allocation, a host read of the offsets -- has to have
    happened during the warmup, because none of it is legal inside.
    """
    sm120_prefill = _sm120_prefill()
    stream = torch.cuda.Stream()

    def call():
        kwargs = dict(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g=inputs["g"],
            beta=inputs["beta"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            lower_bound=-5.0,
            initial_state=inputs["initial_state"],
            final_state=state,
            cu_seqlens=inputs["cu_seqlens"],
            output=output,
            resources=workspace,
        )
        if variant is not None:
            kwargs["variant"] = variant
        return sm120_prefill.run_kda_prefill_sm120(**kwargs)

    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    return graph


@torch.inference_mode()
@pytest.mark.parametrize("variant", ["decomp", "fused", None])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("offsets_dtype", [torch.int32, torch.int64])
def test_recurrent_kda_prefill_sm120_graph_replay_matches_eager(
    variant, packed, offsets_dtype
):
    """Every published combination captures, replays and agrees with eager.

    ``variant=None`` is ``auto``, which must resolve once during warmup and
    keep its answer: re-deciding between warmup and capture would record a
    different kernel than the one the warmup proved.
    """
    _skip_if_not_sm120()
    if not packed and offsets_dtype is torch.int64:
        pytest.skip("fixed input has no caller offsets tensor")

    from flashinfer.kda_kernels.sm120_prefill.runtime import SM120PrefillResources

    seq_lens = [17, 47] if packed else [64, 64]
    inputs = _make_inputs_sm120(
        seq_lens=seq_lens,
        num_heads=4,
        packed=packed,
        initial_state=True,
        seed=SEED,
        offsets_dtype=offsets_dtype,
    )
    expected_out, expected_state = _reference_kda_prefill(inputs)

    eager_state = inputs["initial_state"].clone()
    eager_out = torch.empty_like(inputs["v"])
    _sm120_prefill().run_kda_prefill_sm120(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        lower_bound=-5.0,
        initial_state=eager_state,
        final_state=eager_state,
        cu_seqlens=inputs["cu_seqlens"],
        output=eager_out,
        **({} if variant is None else {"variant": variant}),
    )
    torch.cuda.synchronize()

    workspace = SM120PrefillResources(device=inputs["q"].device)
    graph_out = torch.empty_like(inputs["v"])
    graph = _warm_and_capture(
        inputs,
        workspace=workspace,
        output=graph_out,
        state=inputs["initial_state"],
        variant=variant,
    )

    # Reset the state so replay does the same work the eager call did.
    inputs["initial_state"].copy_(
        _make_inputs_sm120(
            seq_lens=seq_lens,
            num_heads=4,
            packed=packed,
            initial_state=True,
            seed=SEED,
            offsets_dtype=offsets_dtype,
        )["initial_state"]
    )
    graph.replay()
    torch.cuda.synchronize()

    _assert_elementwise(
        graph_out, eager_out, "graph vs eager output", rtol=0.0, atol=0.0
    )
    _assert_elementwise(
        graph_out, expected_out, "graph output", rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL
    )
    _assert_elementwise(
        inputs["initial_state"],
        expected_state,
        "graph final_state",
        rtol=STATE_RTOL,
        atol=STATE_ATOL,
    )


@torch.inference_mode()
@pytest.mark.parametrize("variant", ["decomp", "fused"])
def test_recurrent_kda_prefill_sm120_plan_is_isolated_per_workspace(variant):
    """A plan built for one workspace must never populate another one.

    Plans embed workspace-owned offsets, descriptors and (for decomp) writable
    scratch.  Reusing the first workspace's plan for a second workspace leaves
    the second empty and can make two captured graphs share mutable storage.
    """
    _skip_if_not_sm120()
    from flashinfer.kda_kernels.sm120_prefill.runtime import SM120PrefillResources

    sm120_prefill = _sm120_prefill()
    sm120_prefill.clear_kda_prefill_sm120_caches()
    inputs = _make_inputs_sm120(seq_lens=[17, 33], num_heads=4, packed=True, seed=SEED)
    output = torch.empty_like(inputs["v"])
    call = dict(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        lower_bound=-5.0,
        cu_seqlens=inputs["cu_seqlens"],
        output=output,
        variant=variant,
    )
    first = SM120PrefillResources(device=inputs["q"].device)
    second = SM120PrefillResources(device=inputs["q"].device)

    sm120_prefill.run_kda_prefill_sm120(**call, resources=first)
    sm120_prefill.run_kda_prefill_sm120(**call, resources=second)

    assert first.cu_seqlens_i32 is not None
    assert second.cu_seqlens_i32 is not None
    assert first.cu_seqlens_i32.data_ptr() != second.cu_seqlens_i32.data_ptr()
    assert first.pins
    assert second.pins
    if variant == "decomp":
        assert first._arena is not None
        assert second._arena is not None
        assert first._arena.storage.data_ptr() != second._arena.storage.data_ptr()

    # A workspace freezes addresses and layouts, not merely logical shapes.
    changed = dict(call)
    changed["q"] = inputs["q"].clone()
    with pytest.raises(RuntimeError, match="call signature"):
        sm120_prefill.run_kda_prefill_sm120(**changed, resources=second)


@torch.inference_mode()
def test_recurrent_kda_prefill_sm120_cold_capture_is_refused():
    """Capturing without a warmup fails loudly rather than silently degrading.

    A cold cache inside a capture would have to compile, allocate or
    synchronize; every one of those turns a capture-time mistake into a
    replay-time corruption that surfaces far from its cause.
    """
    _skip_if_not_sm120()
    from flashinfer.kda_kernels.sm120_prefill.runtime import SM120PrefillResources

    sm120_prefill = _sm120_prefill()
    sm120_prefill.clear_kda_prefill_sm120_caches()

    inputs = _make_inputs_sm120(seq_lens=[33, 31], num_heads=4, packed=True, seed=SEED)
    workspace = SM120PrefillResources(device=inputs["q"].device)
    out = torch.empty_like(inputs["v"])

    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(
            RuntimeError, match=r"CUDA graph capture cannot validate cu_seqlens"
        ),
        torch.cuda.graph(graph, stream=stream),
    ):
        sm120_prefill.run_kda_prefill_sm120(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g=inputs["g"],
            beta=inputs["beta"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            lower_bound=-5.0,
            cu_seqlens=inputs["cu_seqlens"],
            output=out,
            resources=workspace,
        )
    torch.cuda.synchronize()


@torch.inference_mode()
def test_recurrent_kda_prefill_sm120_workspace_refuses_a_second_binding():
    """One workspace, one variant, one stream, one signature.

    A workspace that has been captured holds addresses a live graph reads.
    Letting it serve another call -- another variant, another stream, or plain
    eager Python -- would let those addresses move underneath the graph.
    """
    _skip_if_not_sm120()
    from flashinfer.kda_kernels.sm120_prefill.runtime import SM120PrefillResources

    workspace = SM120PrefillResources(device=torch.device("cuda", 0))
    workspace.bind(variant="decomp", stream_ptr=1, signature=("a",))
    workspace.bind(variant="decomp", stream_ptr=1, signature=("a",))  # idempotent

    with pytest.raises(RuntimeError, match="variant"):
        workspace.bind(variant="fused", stream_ptr=1, signature=("a",))
    with pytest.raises(RuntimeError, match="stream"):
        workspace.bind(variant="decomp", stream_ptr=2, signature=("a",))
    with pytest.raises(RuntimeError, match="call signature"):
        workspace.bind(variant="decomp", stream_ptr=1, signature=("b",))

    workspace.captured = True
    with pytest.raises(RuntimeError, match="already participated"):
        workspace.bind(variant="decomp", stream_ptr=1, signature=("a",))


def test_sm120_spent_workspace_is_rejected_before_scratch_mutation(monkeypatch):
    """A rejected reuse cannot release storage still read by a live graph."""
    from flashinfer import kda_kernels, kda_prefill

    scratch = SimpleNamespace(shape=(1,))
    resources = SimpleNamespace(
        captured=True,
        state_scratch=scratch,
        lock=threading.Lock(),
    )
    workspace = SimpleNamespace(_captured=False, _sm120_state=resources)

    def reject_late(**kwargs):
        raise RuntimeError("workspace participated in capture and cannot be reused")

    monkeypatch.setattr(kda_kernels, "run_kda_prefill_sm120", reject_late)
    shape = (2, 2, 2, HEAD_DIM)
    q = torch.zeros(shape, dtype=torch.bfloat16)

    def run(candidate):
        return kda_prefill._run_sm120_kda_prefill(
            q=q,
            k=q.clone(),
            v=q.clone(),
            g=q.clone(),
            beta=torch.zeros(shape[:-1], dtype=torch.bfloat16),
            A_log=torch.zeros(shape[2], dtype=torch.float32),
            dt_bias=torch.zeros((shape[2], HEAD_DIM), dtype=torch.float32),
            scale=None,
            initial_state=None,
            output_final_state=True,
            lower_bound=-5.0,
            cu_seqlens=None,
            output=None,
            prefill_workspace=candidate,
        )

    with pytest.raises(RuntimeError, match=r"capture.*cannot be reused"):
        run(workspace)
    assert resources.state_scratch is scratch

    # The shared workspace may instead have been spent by Cake, before it ever
    # acquired SM120 resources. Reject before even creating that state.
    cake_workspace = SimpleNamespace(_captured=True, _sm120_state=None)
    with pytest.raises(RuntimeError, match=r"capture.*cannot be reused"):
        run(cake_workspace)
    assert cake_workspace._sm120_state is None


@torch.inference_mode()
def test_recurrent_kda_prefill_sm120_workspace_capacity_only_grows():
    """Workspace buffers grow eagerly and never shrink.

    Shrinking would move an address a captured graph already recorded, so the
    monotonicity is a correctness property rather than an allocator nicety.
    """
    _skip_if_not_sm120()
    from flashinfer.kda_kernels.sm120_prefill.runtime import SM120PrefillResources

    workspace = SM120PrefillResources(device=torch.device("cuda", 0))
    big = workspace.ensure_capacity("cu_seqlens_i32", 64, torch.int32)
    base = big.data_ptr()
    small = workspace.ensure_capacity("cu_seqlens_i32", 8, torch.int32)
    assert small.numel() == 8
    assert small.data_ptr() == base
    grown = workspace.ensure_capacity("cu_seqlens_i32", 256, torch.int32)
    assert grown.numel() == 256


# ===========================================================================
# Runtime unit tests that need no device.
# ===========================================================================


@pytest.mark.parametrize(
    "class_name",
    ["TVMFFIJitCompiledFunction", "TVMFFIJitCompiledFunctionWithKwargs"],
)
def test_sm120_tvm_ffi_dispatch_accepts_known_types_after_module_move(class_name):
    """Known provider types remain valid if a newer DSL moves their module."""
    from flashinfer.kda_kernels.sm120_prefill import runtime

    compiled_type = type(class_name, (), {"__module__": "renamed.provider"})
    compiled = compiled_type()
    assert runtime.assert_tvm_ffi_dispatched(compiled, "test-kernel") is compiled


def test_sm120_tvm_ffi_dispatch_rejects_the_ctypes_fallback():
    """The compatibility fallback must not admit the slow ctypes callable."""
    from flashinfer.kda_kernels.sm120_prefill import runtime

    compiled_type = type(
        "CudaDialectJitCompiledFunction", (), {"__module__": "renamed.provider"}
    )
    with pytest.raises(RuntimeError, match=r"not a TVM-FFI callable"):
        runtime.assert_tvm_ffi_dispatched(compiled_type(), "test-kernel")


def test_sm120_flat_view_rejects_noncontiguous_tensors_before_conversion():
    """A reshape copy must not silently replace the address being described."""
    from flashinfer.kda_kernels.sm120_prefill import runtime

    tensor = torch.zeros((2, 3)).T
    assert not tensor.is_contiguous()
    with pytest.raises(runtime.KDAPrefillValidationError, match=r"contiguous"):
        runtime.flat_view(tensor)


def test_sm120_runtime_version_is_readable_under_inference_mode():
    """Cache keys must not read ``_version`` unguarded.

    Every tensor created under ``torch.inference_mode()`` refuses that read --
    ``RuntimeError: Inference tensors do not track version counter`` -- and
    inference mode is how serving calls this backend. A call-plan cache that
    reads it unguarded therefore works in every benchmark and fails in the
    deployment it exists for. This is a regression test for exactly that: the
    whole correctness half of this file runs under ``inference_mode``, but a
    failure there reads as a numerical problem rather than as this.
    """
    from flashinfer.kda_kernels.sm120_prefill import runtime

    normal = torch.zeros(4)
    assert runtime.tensor_version(normal) == normal._version
    assert runtime.tensor_identity(normal) is not None

    with torch.inference_mode():
        inference = torch.zeros(4)
        assert runtime.tensor_version(inference) is runtime.NO_VERSION
        # ...and the identity a cache key is built from still works.
        identity = runtime.tensor_identity(inference)
        assert identity[-1] is runtime.NO_VERSION
        # A sentinel that compares equal to itself keeps the plan cache usable;
        # what it cannot do is notice an in-place edit, which is why the
        # offsets values are a documented caller contract in this mode.
        assert runtime.tensor_identity(inference) == identity


def test_sm120_runtime_alias_rules():
    """Exact aliases are recognized; partial overlaps are not aliases."""
    from flashinfer.kda_kernels.sm120_prefill import runtime

    base = torch.zeros(64, dtype=torch.bfloat16)
    a = base[:32]
    b = base[:32]
    assert runtime.is_exact_alias(a, b)
    assert runtime.intervals_overlap(
        runtime.storage_interval(a), runtime.storage_interval(base[16:48])
    )
    assert not runtime.is_exact_alias(a, base[16:48])
    assert not runtime.intervals_overlap(
        runtime.storage_interval(base[:16]), runtime.storage_interval(base[16:32])
    )
    # A zero-element tensor owns no bytes, so it cannot overlap anything...
    assert runtime.storage_interval(base[:0]) is None
    assert not runtime.intervals_overlap(
        runtime.storage_interval(base[:0]), runtime.storage_interval(base)
    )
    # ...and two of them are trivially the same (empty) range. That reads as an
    # exact alias, which is the harmless answer: the only thing the caller does
    # with it is skip an overwrite check on a buffer with nothing to overwrite.
    assert runtime.is_exact_alias(base[:0], base[:0])


def test_sm120_runtime_bounded_cache_evicts_from_the_lru_tail():
    """The cache is bounded; the oldest entry is the one that goes."""
    from flashinfer.kda_kernels.sm120_prefill import runtime

    cache = runtime.BoundedDeviceCache("test", max_entries=2)
    device = torch.device("cpu")
    cache.put(device, "a", 1)
    cache.put(device, "b", 2)
    assert cache.get(device, "a") == 1  # refreshes a's position
    cache.put(device, "c", 3)
    assert cache.get(device, "b") is None
    assert cache.get(device, "a") == 1
    assert cache.get(device, "c") == 3
    assert cache.stats(device).evictions == 1


def test_sm120_runtime_refuses_a_mismatched_persistent_cache_target(monkeypatch):
    """An artifact must not be named for a target it was not built for.

    This is the failure mode the explicit compile option exists to prevent:
    ``JitSpecCuteDsl`` names its module directory from ``CUTE_DSL_ARCH`` or the
    device, while the kernel is compiled for whatever option we passed. If
    those disagree, the on-disk artifact claims one target and contains
    another, and the next process loads it believing the name.
    """
    from flashinfer.kda_kernels.sm120_prefill import runtime

    monkeypatch.setattr(
        "flashinfer.jit.cute_dsl_core._get_compile_arch", lambda: "sm100a"
    )
    with pytest.raises(runtime.UnsupportedArchitectureError, match="sm120a"):
        runtime._assert_cache_target_matches()


def test_sm120_runtime_build_kernel_uses_the_input_device(monkeypatch):
    """Cold compile and persistent-cache lookup run under the tensor's device."""
    from flashinfer.jit import cute_dsl_core
    from flashinfer.kda_kernels.sm120_prefill import runtime

    active_devices = []
    entered_devices = []

    class DeviceGuard:
        def __init__(self, device):
            self.device = torch.device(device)

        def __enter__(self):
            active_devices.append(self.device)
            entered_devices.append(self.device)

        def __exit__(self, exc_type, exc, traceback):
            active_devices.pop()

    monkeypatch.setattr(runtime.torch.cuda, "device", DeviceGuard)
    monkeypatch.setattr(
        cute_dsl_core,
        "_get_compile_arch",
        lambda: "sm120a" if active_devices else "sm100a",
    )
    monkeypatch.setattr(
        cute_dsl_core,
        "build_and_load_cute_dsl_kernel",
        lambda _module, _kernel, compile_fn, extra_key_files=(): compile_fn(),
    )

    compiled = object()
    assert (
        runtime.build_kernel(
            "device-guard-test", lambda: compiled, device=torch.device("cuda", 1)
        )
        is compiled
    )
    assert entered_devices == [torch.device("cuda", 1)]


def test_sm120_runtime_offsets_reject_malformed_metadata():
    """Offsets content is validated, with an error that says what was wrong."""
    from flashinfer.kda_kernels.sm120_prefill import runtime

    if not torch.cuda.is_available():
        pytest.skip("offsets validation needs a CUDA tensor")

    good = torch.tensor([0, 4, 10], dtype=torch.int32, device="cuda")
    try:
        record = runtime.validate_packed_offsets(good, 10)
        assert record.sequences == 2
        assert record.lengths == (4, 6)
        assert record.longest_sequence == 6

        with pytest.raises(runtime.KDAPrefillValidationError, match="start at 0"):
            runtime.validate_packed_offsets(
                torch.tensor([1, 4, 10], dtype=torch.int32, device="cuda"), 10
            )
        with pytest.raises(runtime.KDAPrefillValidationError, match="end at"):
            runtime.validate_packed_offsets(good, 11)
        with pytest.raises(runtime.KDAPrefillValidationError, match="non-decreasing"):
            runtime.validate_packed_offsets(
                torch.tensor([0, 8, 4], dtype=torch.int32, device="cuda"), 4
            )
    finally:
        runtime.clear_offsets_caches()


def test_sm120_runtime_packed_offsets_hits_the_ordered_lru(monkeypatch):
    """An identity hit must not bypass the cache event or LRU accounting."""
    from flashinfer.kda_kernels.sm120_prefill import runtime

    if not torch.cuda.is_available():
        pytest.skip("offsets validation needs a CUDA tensor")

    runtime.clear_offsets_caches()
    offsets = torch.tensor([0, 4, 10], dtype=torch.int64, device="cuda")
    record = runtime.validate_packed_offsets(offsets, 10)
    calls = []
    original_get = runtime._PACKED_OFFSETS.get

    def tracked_get(device, key):
        calls.append((device, key))
        return original_get(device, key)

    monkeypatch.setattr(runtime._PACKED_OFFSETS, "get", tracked_get)
    try:
        assert runtime.validate_packed_offsets(offsets, 10) is record
        assert len(calls) == 1
    finally:
        runtime.clear_offsets_caches()


def test_sm120_runtime_packed_offsets_eviction_releases_the_source():
    """The identity memo must not keep bounded-cache payloads alive forever."""
    from flashinfer.kda_kernels.sm120_prefill import runtime

    if not torch.cuda.is_available():
        pytest.skip("offsets validation needs a CUDA tensor")

    runtime.clear_offsets_caches()
    sources = []
    try:
        for _ in range(runtime._PACKED_OFFSETS.max_entries + 1):
            offsets = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
            sources.append(weakref.ref(offsets))
            record = runtime.validate_packed_offsets(offsets, 1)
        del offsets, record
        torch.cuda.synchronize()
        gc.collect()
        assert sources[0]() is None
    finally:
        runtime.clear_offsets_caches()
