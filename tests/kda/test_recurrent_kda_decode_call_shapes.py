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

"""Call-shape pins for #6560 (H=64 single-token KDA decode).

Pins the three single-token decode call forms of ``recurrent_kda`` at the
H=64, K=V=128 geometry:

* packed   ``[1, B]`` + ``cu_seqlens`` + ``ssm_state_indices``
           (the serving frameworks' call form; the fast path)
* batched  ``[B, 1]`` + ``ssm_state_indices``, no ``cu_seqlens``
* batched  ``[B, 1]`` with a contiguous state slice and no indices

All three must produce bit-identical outputs and final states, and each call
must be exactly one device launch. The full-census assertion (every kernel,
memcpy, and memset per call, not just the decode kernel's name) is
intentional: the batched + ``ssm_state_indices`` form currently wraps the
same decode kernel in a host-side gather/scatter of the full state pool
(5 launches per call), which a name-filtered census does not see. That case
is marked xfail until the batched indexed form is routed onto the packed
kernel path (#6560); when the routing lands it flips to a passing census.

The state-dtype contract is pinned as well: the serving frameworks hold
float32 state pools, and the CuTe decode path must refuse them with a
``TypeError`` (#6893) rather than crash.
"""

import pytest
import torch

from flashinfer.utils import is_sm100a_supported

try:
    from flashinfer import recurrent_kda
    from flashinfer.kda_decode import _RECURRENT_KDA_AVAILABLE

    _has_recurrent_kda = _RECURRENT_KDA_AVAILABLE
except ImportError:
    recurrent_kda = None
    _has_recurrent_kda = False


H = HV = 64
K = V = 128
LOWER_BOUND = -5.0
SCALE = K**-0.5
BATCHES = (1, 8)
PROFILE_ITERS = 4


@pytest.fixture(autouse=True)
def _require_recurrent_kda():
    if not torch.cuda.is_available():
        pytest.skip("Recurrent KDA requires CUDA")
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Recurrent KDA requires SM100a (Blackwell)")
    if not _has_recurrent_kda:
        pytest.skip("recurrent_kda kernel not available (missing cutlass DSL deps)")


def _inputs(B, layout, state_dtype, with_indices=True, seed=0):
    """Build one call form's inputs; identical values across layouts/seeds."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dev = "cuda"

    def r(*shape):
        return torch.randn(*shape, generator=gen).to(torch.bfloat16).to(dev)

    shape4 = (1, B) if layout == "packed" else (B, 1)
    p = dict(
        q=r(*shape4, H, K),
        k=r(*shape4, H, K),
        v=r(*shape4, HV, V),
        g=r(*shape4, HV, K),
        beta=r(*shape4, HV),
        A_log=(torch.rand(HV, generator=gen) * 3.0 + 0.1).log().float().to(dev),
        dt_bias=(torch.randn(HV * K, generator=gen) * 0.5).float().to(dev),
        state=(torch.randn(B, HV, V, K, generator=gen) * 0.1)
        .to(state_dtype)
        .to(dev)
        .contiguous(),
        out=torch.empty(*shape4, HV, V, dtype=torch.bfloat16, device=dev),
    )
    if with_indices:
        p["idx"] = torch.arange(B, dtype=torch.int32, device=dev)
    if layout == "packed":
        p["cu"] = torch.arange(0, B + 1, dtype=torch.int32, device=dev)
    return p


def _call(p):
    kw = dict(
        q=p["q"],
        k=p["k"],
        v=p["v"],
        g=p["g"],
        beta=p["beta"],
        A_log=p["A_log"],
        dt_bias=p["dt_bias"],
        scale=SCALE,
        initial_state=p["state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        beta_is_logit=True,
        lower_bound=LOWER_BOUND,
        output=p["out"],
        backend="auto",
    )
    if "idx" in p:
        kw["ssm_state_indices"] = p["idx"]
    if "cu" in p:
        kw["cu_seqlens"] = p["cu"]
    return recurrent_kda(**kw)


def _device_launch_census(run):
    """Count EVERY device-side launch per call: kernels, memcpys, memsets.

    A census filtered on the decode kernel's name misses the host-side
    gather/scatter copies around the batched indexed form, so everything
    with device time is counted (#6560).
    """
    from torch.profiler import ProfilerActivity, profile

    run()  # warmup: JIT compile / caches outside the profiled window
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(PROFILE_ITERS):
            run()
        torch.cuda.synchronize()
    census = {}
    for ev in prof.key_averages():
        if getattr(ev, "self_device_time_total", 0) > 0:
            census[ev.key] = ev.count / PROFILE_ITERS
    return census


def _run_form(B, layout, with_indices):
    p = _inputs(B, layout, torch.bfloat16, with_indices=with_indices)
    out, final_state = _call(p)
    torch.cuda.synchronize()
    return p, out, final_state


@pytest.mark.parametrize("B", BATCHES)
def test_call_forms_bit_identical(B):
    """The three single-token call forms are the same computation: outputs and
    final states must be bit-identical across them."""
    forms = {
        "packed": _run_form(B, "packed", with_indices=True),
        "batched_indexed": _run_form(B, "batched", with_indices=True),
        "batched_contiguous": _run_form(B, "batched", with_indices=False),
    }
    _, ref_out, _ = forms["packed"]
    ref_out = ref_out.reshape(B, HV, V)
    ref_state = forms["packed"][0]["state"]
    for name, (p, out, final_state) in forms.items():
        out = out.reshape(B, HV, V)
        assert torch.equal(ref_out, out), f"{name} output differs from packed"
        # Indexed forms update the pool in place; compare pool contents too.
        assert torch.equal(ref_state, p["state"]), (
            f"{name} final state pool differs from packed"
        )
        if final_state is not None:
            assert torch.equal(
                ref_state, final_state.reshape(B, HV, V, K).to(ref_state.dtype)
            ), f"{name} returned final state differs from packed"


def _assert_single_launch(B, layout, with_indices):
    p = _inputs(B, layout, torch.bfloat16, with_indices=with_indices)
    census = _device_launch_census(lambda: _call(p))
    total = sum(census.values())
    assert total == pytest.approx(1.0), (
        f"single-token decode ({layout}, indices={with_indices}, B={B}) should be "
        f"exactly one device launch per call, got {total}: {census}"
    )


@pytest.mark.parametrize("B", BATCHES)
def test_packed_decode_is_one_launch(B):
    """Packed [1, B] + cu_seqlens + ssm_state_indices: one launch per call."""
    _assert_single_launch(B, "packed", with_indices=True)


@pytest.mark.parametrize("B", BATCHES)
def test_batched_contiguous_decode_is_one_launch(B):
    """Batched [B, 1] with a contiguous state slice and no indices runs at the
    packed speed: one launch per call."""
    _assert_single_launch(B, "batched", with_indices=False)


@pytest.mark.parametrize("B", BATCHES)
@pytest.mark.xfail(
    reason="batched [B,1] + ssm_state_indices pays a host-side gather/scatter "
    "of the full state pool (5 launches per call) until it is routed onto the "
    "packed kernel path; see #6560",
    strict=False,
)
def test_batched_indexed_decode_is_one_launch(B):
    """Batched [B, 1] + ssm_state_indices is the issue's 4-6x call form: the
    full census fails it until the routing lands, then flips to passing."""
    _assert_single_launch(B, "batched", with_indices=True)


@pytest.mark.parametrize("layout", ("packed", "batched"))
def test_fp32_state_pool_raises_type_error(layout):
    """The serving frameworks hold float32 state pools; the CuTe decode path
    requires bfloat16 state and must refuse fp32 with TypeError (#6893). A
    future fp32-state decode variant flips this to a passing path."""
    p = _inputs(8, layout, torch.float32)
    with pytest.raises(TypeError):
        _call(p)
        torch.cuda.synchronize()
