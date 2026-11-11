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

The state-pool forms use permuted indices into a pool larger than the batch,
so an implementation that reads ``state[row]`` instead of ``state[idx[row]]``
fails, and slots outside the indices must come through untouched. All three
forms must produce bit-identical outputs and final states, and each call must
be exactly one device launch. The full-census assertion (every kernel and
memcpy per call, via ``tests.cake.test_helpers.cake_profiler``, not a
name-filtered count) is intentional: the batched + ``ssm_state_indices`` form
currently wraps the same decode kernel in a gather and scatter of the B
selected state rows (5 launches per call: two int32->int64 index casts, the
gather, the decode kernel, and the scatter), which a name-filtered census
does not see. That case is a strict xfail: the PR that routes the batched
indexed form onto the packed kernel path (#6560) must remove the mark. On a
part where the profiler raises or returns no device records, the census
helper skips, so the census tests (the strict xfail included) report skipped
there rather than a false pass or XPASS.

Float32 state pools are covered by ``test_recurrent_kda_fp32_state.py``.
"""

import pytest
import torch

from flashinfer.utils import is_sm100a_supported
from tests.cake.test_helpers.cake_profiler import cuda_kernel_names

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


@pytest.fixture(autouse=True)
def _require_recurrent_kda():
    if not torch.cuda.is_available():
        pytest.skip("Recurrent KDA requires CUDA")
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Recurrent KDA requires SM100a (Blackwell)")
    if not _has_recurrent_kda:
        pytest.skip("recurrent_kda kernel not available (missing cutlass DSL deps)")


def _inputs(B, layout, state_dtype, indexed=True, seed=0):
    """Build one call form's inputs; identical values across layouts.

    The state pool holds ``2 * B + 3`` slots and the B sequences select
    permuted slots, so slot i does not hold sequence i's state: a route that
    ignores ``ssm_state_indices`` reads the wrong rows and fails the
    bit-identical checks. The contiguous (``indexed=False``) form gets the
    same selected rows as its ``[B, ...]`` ``initial_state``.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dev = "cuda"

    def r(*shape):
        return torch.randn(*shape, generator=gen).to(torch.bfloat16).to(dev)

    pool_slots = 2 * B + 3
    shape4 = (1, B) if layout == "packed" else (B, 1)
    p = dict(
        q=r(*shape4, H, K),
        k=r(*shape4, H, K),
        v=r(*shape4, HV, V),
        g=r(*shape4, HV, K),
        beta=r(*shape4, HV),
        A_log=(torch.rand(HV, generator=gen) * 3.0 + 0.1).log().float().to(dev),
        dt_bias=(torch.randn(HV * K, generator=gen) * 0.5).float().to(dev),
        out=torch.empty(*shape4, HV, V, dtype=torch.bfloat16, device=dev),
    )
    pool = (
        (torch.randn(pool_slots, HV, V, K, generator=gen) * 0.1)
        .to(state_dtype)
        .to(dev)
        .contiguous()
    )
    idx = torch.randperm(pool_slots, generator=gen)[:B].to(torch.int32).to(dev)
    p["base_pool"] = pool.clone()
    p["idx"] = idx
    if indexed:
        p["state"] = pool
        p["pass_idx"] = True
    else:
        p["state"] = pool[idx.long()].contiguous()
        p["pass_idx"] = False
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
    if p["pass_idx"]:
        kw["ssm_state_indices"] = p["idx"]
    if "cu" in p:
        kw["cu_seqlens"] = p["cu"]
    return recurrent_kda(**kw)


def _selected_final_state(p, final_state, B):
    """The B sequences' final states of one call form, in sequence order.

    The returned-state shape is per form: the packed state-pool call returns
    the whole pool (indexed by slot), while the batched indexed call returns
    its gathered per-sequence copy and the contiguous-slice call returns the
    B rows, both already in sequence order. Falls back to the
    in-place-updated ``initial_state`` when the call returned no state
    tensor.
    """
    idx = p["idx"].long()
    state = final_state if final_state is not None else p["state"]
    if state.shape[0] == B:
        return state
    assert state.shape[0] == p["base_pool"].shape[0], (
        f"unexpected final-state shape {list(state.shape)}"
    )
    return state[idx]


def _run_form(B, layout, indexed):
    p = _inputs(B, layout, torch.bfloat16, indexed=indexed)
    out, final_state = _call(p)
    torch.cuda.synchronize()
    return p, out, final_state


@pytest.mark.parametrize("B", BATCHES)
def test_call_forms_bit_identical(B):
    """The three single-token call forms are the same computation: outputs and
    final states must be bit-identical across them, and pool slots outside
    ``ssm_state_indices`` must come through untouched."""
    forms = {
        "packed": _run_form(B, "packed", indexed=True),
        "batched_indexed": _run_form(B, "batched", indexed=True),
        "batched_contiguous": _run_form(B, "batched", indexed=False),
    }
    p_ref, out_ref, state_ref = forms["packed"]
    out_ref = out_ref.reshape(B, HV, V)
    sel_ref = _selected_final_state(p_ref, state_ref, B)
    for name, (p, out, final_state) in forms.items():
        out = out.reshape(B, HV, V)
        assert torch.equal(out_ref, out), f"{name} output differs from packed"
        sel = _selected_final_state(p, final_state, B)
        assert torch.equal(sel_ref, sel), f"{name} final state differs from packed"
        if p["pass_idx"]:
            untouched = torch.ones(
                p["base_pool"].shape[0], dtype=torch.bool, device=p["idx"].device
            )
            untouched[p["idx"].long()] = False
            assert torch.equal(p["state"][untouched], p["base_pool"][untouched]), (
                f"{name} modified pool slots outside ssm_state_indices"
            )


def _census(B, layout, indexed):
    """Every device launch of one call (kernels and copies), or a skip when
    the profiler loses its records (``cake_profiler``'s sentinel protocol)."""
    p = _inputs(B, layout, torch.bfloat16, indexed=indexed)
    _call(p)  # warmup: JIT compile / caches outside the profiled window
    torch.cuda.synchronize()
    try:
        return cuda_kernel_names(lambda: _call(p), exclude=())
    except RuntimeError as ex:  # profiler unavailable on this host
        pytest.skip(f"torch.profiler unavailable: {ex}")


def _assert_single_launch(B, layout, indexed):
    names = _census(B, layout, indexed)
    assert len(names) == 1, (
        f"single-token decode ({layout}, indexed={indexed}, B={B}) should be "
        f"exactly one device launch per call, got {len(names)}: {names}"
    )


@pytest.mark.parametrize("B", BATCHES)
def test_packed_decode_is_one_launch(B):
    """Packed [1, B] + cu_seqlens + ssm_state_indices: one launch per call."""
    _assert_single_launch(B, "packed", indexed=True)


@pytest.mark.parametrize("B", BATCHES)
def test_batched_contiguous_decode_is_one_launch(B):
    """Batched [B, 1] with a contiguous state slice and no indices runs at the
    packed speed: one launch per call."""
    _assert_single_launch(B, "batched", indexed=False)


@pytest.mark.parametrize("B", BATCHES)
@pytest.mark.xfail(
    reason="batched [B,1] + ssm_state_indices gathers and scatters the B "
    "selected state rows around the same decode kernel (5 launches per call: "
    "two int32->int64 index casts, the gather, the decode kernel, the "
    "scatter); remove this mark when the batched indexed form is routed onto "
    "the packed kernel path, see #6560",
    strict=True,
)
def test_batched_indexed_decode_is_one_launch(B):
    """Batched [B, 1] + ssm_state_indices is the issue's 4-6x call form: the
    full census fails it (strictly) until the routing lands; the routing PR
    removes the xfail mark. Where the profiler cannot read launches, the
    census helper skips, so this never XPASSes for lack of records."""
    _assert_single_launch(B, "batched", indexed=True)
