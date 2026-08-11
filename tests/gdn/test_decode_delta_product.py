"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Phase 1, step 3: Gated DeltaProduct decode (MTP).

Decode diverges from prefill in two ways that need their own tests, not just a
port of the prefill suite:

  * the gate is computed INSIDE the kernel from A_log/a/dt_bias, so the
    neutral value for the non-first micro-steps is a sentinel in `a`, not a 1.0
    in `g`. `test_gate_sentinel_is_exactly_neutral` pins that.
  * the per-token state scatter is unguarded, so intermediate micro-steps write
    to throwaway pool rows. `test_scratch_slots_are_inert` pins that.

Layout note: decode is DENSE [B, T, ...], not varlen. The reference is still
`delta_product`, reached by flattening to [B*T, ...] with seq_lens = [T]*B.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch

from flashinfer.gdn_product import GATE_NEUTRAL_A_SENTINEL, gated_delta_product_mtp

from .reference_delta_product import delta_product
from .test_prefill_delta_product import _skip_if_unsupported

# Matches gdn_decode_mtp.py's constexpr softplus params.
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0


def gates_from_logits(A_log, a, dt_bias, b):
    """Host-side twin of the kernel's fused gating (gdn_decode_mtp.py:415-436).

        alpha = exp(-exp(A_log) * softplus(a + dt_bias))
        beta  = sigmoid(b)

    The reference takes alpha/beta directly, so every decode test has to cross
    this bridge. Keep it in lockstep with the kernel or the tests measure the
    wrong thing.
    """
    # a/b arrive in the MODEL dtype (fp16/bf16); the kernel promotes internally,
    # so the host-side twin has to as well or it measures a different function.
    a, b = a.float(), b.float()
    x = a + dt_bias
    bx = SOFTPLUS_BETA * x
    softplus_x = torch.where(
        bx <= SOFTPLUS_THRESHOLD, (1.0 / SOFTPLUS_BETA) * torch.log1p(torch.exp(bx)), x
    )
    alpha = torch.exp(-torch.exp(A_log) * softplus_x)
    return alpha, torch.sigmoid(b)


class DecodeInputs(NamedTuple):
    q: torch.Tensor  # [B, T,      Hq, K]
    k: torch.Tensor  # [B, T, n_h, Hq, K]
    v: torch.Tensor  # [B, T, n_h, HV, V]
    A_log: torch.Tensor  # [HV]
    a: torch.Tensor  # [B, T,      HV]
    dt_bias: torch.Tensor  # [HV]
    b: torch.Tensor  # [B, T, n_h, HV]
    pool: torch.Tensor  # [pool_size, HV, V, K]
    initial_state_indices: torch.Tensor  # [B]
    ssm_state_indices: torch.Tensor  # [B, T] one snapshot slot per REAL token
    scratch_state_indices: torch.Tensor  # [B]


def _gen_decode_inputs(B, T, n_h, num_q_heads, num_v_heads, K, V, dtype, device, seed):
    """Dense decode inputs plus a state pool partitioned into disjoint regions.

    Pool layout -- every region must be disjoint or the tests measure nothing:

        row 0                       unused (0 is a sentinel elsewhere in flashinfer)
        [1, 1+B)                    initial states, one per batch row
        [1+B, 1+B+B*T)              per-REAL-token snapshots, ssm_state_indices
        [1+B+B*T, 1+B+B*T+B)        scratch, absorbs the intermediate micro-steps
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    HV = num_v_heads
    with device:
        # Magnitudes and dtypes mirror the house MTP test
        # (test_decode_delta_rule.py::test_gated_delta_rule_mtp). Two of these
        # are load-bearing, not cosmetic:
        #
        #  * `a` and `b` are the MODEL dtype, not fp32. They are raw logits the
        #    kernel converts internally. Only A_log and dt_bias are fp32. This
        #    is the opposite of prefill, where `g`/`beta` are consumed directly
        #    and must be fp32 -- generalising the prefill rule to decode makes
        #    the gate wrong from the very first token.
        #  * everything is scaled to ~0.1 (state ~0.01). Unscaled randn drives
        #    softplus(a + dt_bias) into a range where alpha swings hard, which
        #    is numerically far nastier than anything the kernel is tuned for.
        q = torch.randn(B, T, num_q_heads, K, dtype=dtype) * 0.1
        k = torch.randn(B, T, n_h, num_q_heads, K, dtype=dtype) * 0.1
        v = torch.randn(B, T, n_h, HV, V, dtype=dtype) * 0.1

        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        a = torch.randn(B, T, HV, dtype=dtype) * 0.1
        b = torch.randn(B, T, n_h, HV, dtype=dtype) * 0.1

        pool = torch.randn(1 + B + B * T + B, HV, V, K, dtype=torch.float32) * 0.01
        initial = torch.arange(1, 1 + B, dtype=torch.int32)
        ssm = torch.arange(1 + B, 1 + B + B * T, dtype=torch.int32).reshape(B, T)
        scratch = torch.arange(1 + B + B * T, 1 + B + B * T + B, dtype=torch.int32)
    return DecodeInputs(q, k, v, A_log, a, dt_bias, b, pool, initial, ssm, scratch)


def _reference(q, k, v, A_log, a, dt_bias, b, pool, idx, scale=1.0, use_l2_norm=True):
    """delta_product over the dense batch, seeded from the pool rows.

    ``use_l2_norm`` must track the wrapper's ``use_qk_l2norm``: the decode
    kernel normalises **both q and k** internally (see
    ``reference_delta_rule.decode_delta_rule``, which does the same under
    ``use_l2_norm``), while ``delta_product`` normalises nothing. Forgetting q
    here leaves the two sides differing by a per-row factor of ||q||.
    """
    B, T, n_h = k.shape[0], k.shape[1], k.shape[2]
    alpha, beta = gates_from_logits(A_log, a, dt_bias, b)
    if use_l2_norm:
        q = torch.nn.functional.normalize(q.float(), p=2.0, dim=-1)
        k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1)
    # pool is [.., V, K]; the reference wants [.., K, V]
    init = pool[idx.long()].transpose(-1, -2).contiguous()
    o, state = delta_product(
        q.reshape(B * T, *q.shape[2:]).float(),
        k.reshape(B * T, *k.shape[2:]).float(),
        v.reshape(B * T, *v.shape[2:]).float(),
        [T] * B,
        alpha=alpha.reshape(B * T, -1),
        beta=beta.reshape(B * T, n_h, -1),
        scale_factor=scale,
        initial_state=init,
    )
    return o.reshape(B, T, *o.shape[1:]), state


# --------------------------------------------------------------------------
# 1. The sentinel. Pure arithmetic -- no kernel, no GPU arch requirement.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("A_log_val", [-2.0, 0.0, 3.0], ids=lambda x: f"A_log={x}")
@pytest.mark.parametrize("dt_bias_val", [-5.0, 0.0, 10.0], ids=lambda x: f"dt_bias={x}")
def test_gate_sentinel_is_exactly_neutral(A_log_val, dt_bias_val):
    """alpha must be EXACTLY 1.0 at the sentinel, for any A_log / dt_bias.

    Not approximately: a micro-step that decays by even one ULP compounds over
    n_h steps per token and over the whole sequence. -30 (the value the plan
    originally suggested) fails this by one ULP once exp(A_log)*softplus()
    exceeds 2^-24.
    """
    A_log = torch.tensor([A_log_val], dtype=torch.float32)
    dt_bias = torch.tensor([dt_bias_val], dtype=torch.float32)
    a = torch.tensor([[[GATE_NEUTRAL_A_SENTINEL]]], dtype=torch.float32)
    b = torch.zeros_like(a)

    alpha, _ = gates_from_logits(A_log, a, dt_bias, b)
    assert (alpha == 1.0).all(), (
        f"sentinel {GATE_NEUTRAL_A_SENTINEL} gave alpha={alpha.item():.10f} "
        f"at A_log={A_log_val}, dt_bias={dt_bias_val}; must be exactly 1.0"
    )
    assert torch.isfinite(alpha).all(), "sentinel produced a non-finite gate"


# --------------------------------------------------------------------------
# 2. n_h == 1 must be the GDN MTP kernel, untouched.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("T", [2, 4], ids=lambda t: f"T={t}")
def test_decode_nh1_matches_gdn_mtp(T):
    _skip_if_unsupported()
    from flashinfer.gdn_decode import gated_delta_rule_mtp

    B, n_h, H, K, V = 2, 1, 4, 128, 128
    device, dtype = torch.device("cuda"), torch.float16
    q, k, v, A_log, a, dt_bias, b, pool, idx, ssm, scratch = _gen_decode_inputs(
        B, T, n_h, H, H, K, V, dtype, device, seed=0
    )
    pool_ref = pool.clone()

    got_o, _ = gated_delta_product_mtp(
        q,
        k,
        v,
        pool,
        idx,
        A_log,
        a,
        dt_bias,
        b,
        scale=1.0,
        disable_state_update=False,
    )
    ref_o, _ = gated_delta_rule_mtp(
        q,
        k.squeeze(2),
        v.squeeze(2),
        pool_ref,
        idx,
        A_log,
        a,
        dt_bias,
        b.squeeze(2),
        scale=1.0,
        disable_state_update=False,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(got_o, ref_o, atol=0, rtol=0)
    torch.testing.assert_close(pool, pool_ref, atol=0, rtol=0)


# --------------------------------------------------------------------------
# 3. n_h > 1 against the reference.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_householder", [1, 2, 3], ids=lambda nh: f"num_householder={nh}"
)
@pytest.mark.parametrize("T", [1, 2, 4], ids=lambda t: f"T={t}")
@pytest.mark.parametrize(
    "num_heads",
    [(4, 4), (4, 8)],
    ids=lambda qkv: "num_heads={0}/{1}".format(*qkv),  # (q, v) -- (4,8) is GVA
)
def test_decode_matches_reference(num_householder, T, num_heads):
    _skip_if_unsupported()
    if num_householder == 1 and T == 1:
        # The expanded axis is n_h*T == 1, and gated_delta_rule_mtp is only
        # valid for T > 1 -- its sibling gated_delta_rule_decode_pretranspose
        # asserts `T == 1` and owns that case, while MTP has NO guard and
        # silently returns garbage. Every other (n_h, T) here is fine: n_h >= 2
        # makes the expanded axis >= 2 even when T == 1.
        pytest.skip("n_h=1, T=1 expands to T=1, which is outside MTP's contract")
    num_q_heads, num_v_heads = num_heads
    B, K, V = 3, 128, 128
    n_h = num_householder
    device, dtype = torch.device("cuda"), torch.float16

    q, k, v, A_log, a, dt_bias, b, pool, idx, ssm, scratch = _gen_decode_inputs(
        B, T, n_h, num_q_heads, num_v_heads, K, V, dtype, device, seed=1
    )
    ref_o, ref_state = _reference(q, k, v, A_log, a, dt_bias, b, pool, idx)

    got_o, got_pool = gated_delta_product_mtp(
        q,
        k,
        v,
        pool,
        idx,
        A_log,
        a,
        dt_bias,
        b,
        scale=1.0,
        # no ssm_state_indices: this is the PLAIN continuous-batching path.
        # State moves via initial_state_indices (read) and output_state_indices
        # (write, defaulting to the read slot); no per-token scatter, hence no
        # scratch. Snapshots are test 7's job.
        disable_state_update=False,
    )
    torch.cuda.synchronize()

    assert got_o.shape == (B, T, num_v_heads, V), "one output row per REAL token"
    torch.testing.assert_close(got_o, ref_o.to(dtype), atol=2e-3, rtol=1e-3)
    # the live rows must hold each sequence's final state; pool is [.., V, K]
    torch.testing.assert_close(
        got_pool[idx.long()].transpose(-1, -2), ref_state, atol=1e-3, rtol=1e-4
    )


# --------------------------------------------------------------------------
# 4. Scratch rows are written but never read back into the answer.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_householder", [2, 3], ids=lambda nh: f"num_householder={nh}"
)
def test_scratch_slots_are_inert(num_householder):
    """Poisoning the scratch rows beforehand must not change the result.

    This is the check on the workaround for gdn_decode_mtp.py's unguarded
    per-token scatter: the intermediate micro-steps DO write, we just need
    their landing site to be irrelevant.
    """
    _skip_if_unsupported()
    B, T, n_h, H, K, V = 3, 2, num_householder, 4, 128, 128
    device, dtype = torch.device("cuda"), torch.float16

    q, k, v, A_log, a, dt_bias, b, pool, idx, ssm, scratch = _gen_decode_inputs(
        B, T, n_h, H, H, K, V, dtype, device, seed=2
    )
    clean_o, _ = gated_delta_product_mtp(
        q,
        k,
        v,
        pool.clone(),
        idx,
        A_log,
        a,
        dt_bias,
        b,
        scale=1.0,
        ssm_state_indices=ssm,
        scratch_state_indices=scratch,
        disable_state_update=False,
    )

    poisoned = pool.clone()
    poisoned[scratch.long()] = 1e30
    poisoned_o, poisoned_pool = gated_delta_product_mtp(
        q,
        k,
        v,
        poisoned,
        idx,
        A_log,
        a,
        dt_bias,
        b,
        scale=1.0,
        ssm_state_indices=ssm,
        scratch_state_indices=scratch,
        disable_state_update=False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(clean_o, poisoned_o, atol=0, rtol=0)
    assert torch.isfinite(poisoned_pool[idx.long()]).all(), (
        "poison leaked from a scratch row into a live slot -- scratch and live "
        "slots are not disjoint, or an intermediate micro-step wrote a live row"
    )


# --------------------------------------------------------------------------
# 5. Batch rows must not contaminate one another.
#
# NOT a scratch-race test: sharing one scratch row across the batch is legal.
# The kernel reads the pool once, before the recurrence, via
# initial_state_indices; the per-token scatter is write-only after that, so
# concurrent stores to a row nobody reads are benign. This checks the broader
# invariant -- a row's result must not depend on who it was batched with --
# and pins that BOTH scratch layouts satisfy it.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_householder", [2, 3], ids=lambda nh: f"num_householder={nh}"
)
@pytest.mark.parametrize("shared_scratch", [True, False], ids=["shared", "per_row"])
def test_batch_rows_are_independent(num_householder, shared_scratch):
    """Running a row alone must match running it in a batch."""
    _skip_if_unsupported()
    B, T, n_h, H, K, V = 4, 2, num_householder, 4, 128, 128
    device, dtype = torch.device("cuda"), torch.float16

    q, k, v, A_log, a, dt_bias, b, pool, idx, ssm, scratch = _gen_decode_inputs(
        B, T, n_h, H, H, K, V, dtype, device, seed=3
    )
    if shared_scratch:
        # every row funnels its intermediates into ONE throwaway pool row
        scratch = scratch[:1].expand(B).contiguous()

    batched_o, _ = gated_delta_product_mtp(
        q,
        k,
        v,
        pool.clone(),
        idx,
        A_log,
        a,
        dt_bias,
        b,
        scale=1.0,
        ssm_state_indices=ssm,
        scratch_state_indices=scratch,
        disable_state_update=False,
    )
    torch.cuda.synchronize()

    for i in range(B):
        sl = slice(i, i + 1)
        solo_o, _ = gated_delta_product_mtp(
            q[sl],
            k[sl],
            v[sl],
            pool.clone(),
            idx[sl],
            A_log,
            a[sl],
            dt_bias,
            b[sl],
            scale=1.0,
            ssm_state_indices=ssm[sl],
            scratch_state_indices=scratch[sl],
            disable_state_update=False,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            batched_o[sl],
            solo_o,
            atol=0,
            rtol=0,
            msg=lambda m: f"row {i} differs when run alone -- cross-batch leak\n{m}",
        )


# --------------------------------------------------------------------------
# 6. The reference BRIDGE itself, against flashinfer's own decode reference.
#
# _reference reaches delta_product through a dense->varlen reshape, a gate
# computation and a state transpose. Any of those can be wrong independently of
# the wrapper, and a wrong reference makes every other test in this file
# meaningless. This pins it against decode_delta_rule -- an implementation
# neither we nor the wrapper touch.
#
# Pure torch: no kernel, so this runs on any GPU, not just SM90+.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("B", [1, 3], ids=lambda b: f"B={b}")
@pytest.mark.parametrize(
    "num_heads", [(4, 4), (4, 8)], ids=lambda qkv: "num_heads={0}/{1}".format(*qkv)
)
def test_reference_bridge_matches_decode_delta_rule(B, num_heads):
    from .reference_delta_rule import decode_delta_rule

    num_q_heads, num_v_heads = num_heads
    T, n_h, K, V = 1, 1, 128, 128  # decode_delta_rule is single-step, n_h-free
    device = torch.device("cuda")

    q, k, v, A_log, a, dt_bias, b, pool, idx, ssm, scratch = _gen_decode_inputs(
        B, T, n_h, num_q_heads, num_v_heads, K, V, torch.float32, device, seed=0
    )
    mine_o, mine_state = _reference(q, k, v, A_log, a, dt_bias, b, pool, idx)

    ref_o, ref_state = decode_delta_rule(
        q.squeeze(1).float(),
        k.squeeze(1).squeeze(1).float(),
        v.squeeze(1).squeeze(1).float(),
        pool[idx.long()].transpose(-1, -2).contiguous(),  # [B, H, K, V]
        A_log=A_log,
        a=a.squeeze(1),
        dt_bias=dt_bias,
        b=b.squeeze(1).squeeze(1),
        scale_factor=1.0,
        use_l2_norm=True,
    )

    torch.testing.assert_close(mine_o.squeeze(1), ref_o, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mine_state, ref_state, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# 7. Per-token state snapshots -- the property speculative decoding needs.
#
# ssm_state_indices[i, t] must end up holding the state AS OF real token t, so
# that a rejected draft can roll the sequence back to any accepted prefix. This
# is the only test that pins the scatter remap: it fails if the LAST micro-step
# of each token is not the one routed to the caller's slot (e.g. an off-by-one
# leaving the state after householder 0 there instead), and it fails if the
# intermediates land on live rows rather than scratch.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_householder", [1, 2, 3], ids=lambda nh: f"num_householder={nh}"
)
@pytest.mark.parametrize("T", [2, 3], ids=lambda t: f"T={t}")
def test_per_token_state_snapshots(num_householder, T):
    _skip_if_unsupported()
    B, n_h, H, K, V = 3, num_householder, 4, 128, 128
    device, dtype = torch.device("cuda"), torch.float16

    q, k, v, A_log, a, dt_bias, b, pool, idx, ssm, scratch = _gen_decode_inputs(
        B, T, n_h, H, H, K, V, dtype, device, seed=5
    )

    # state after each REAL token: rerun the reference over growing prefixes
    want = []
    for t in range(T):
        _, st = _reference(
            q[:, : t + 1],
            k[:, : t + 1],
            v[:, : t + 1],
            A_log,
            a[:, : t + 1],
            dt_bias,
            b[:, : t + 1],
            pool,
            idx,
        )
        want.append(st)

    _, got_pool = gated_delta_product_mtp(
        q,
        k,
        v,
        pool,
        idx,
        A_log,
        a,
        dt_bias,
        b,
        scale=1.0,
        ssm_state_indices=ssm,
        scratch_state_indices=scratch,
        disable_state_update=False,
    )
    torch.cuda.synchronize()

    for t in range(T):
        got = got_pool[ssm[:, t].long()].transpose(-1, -2)  # [.., V, K] -> [.., K, V]
        torch.testing.assert_close(
            got,
            want[t],
            atol=1e-3,
            rtol=1e-4,
            msg=lambda m: (
                f"snapshot for real token {t} is wrong. A state after only the "
                f"FIRST householder here means the scatter remap targets "
                f"[0::n_h] instead of [n_h-1::n_h].\n{m}"
            ),
        )
