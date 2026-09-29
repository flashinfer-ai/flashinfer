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

Unlike prefill, the gate is computed INSIDE the kernel from A_log/a/dt_bias, so the
neutral value for the non-first micro-steps is a sentinel in `a`, not a 1.0
in `g`. `test_gate_sentinel_is_exactly_neutral` pins that.

Layout note: decode is DENSE [B, T, ...], not varlen. The reference is still
`delta_product`, reached by flattening to [B*T, ...] with seq_lens = [T]*B.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch

from flashinfer.gdn_product import GATE_NEUTRAL_A_SENTINEL, gated_delta_product_mtp

from .reference_delta_product import delta_product


def _skip_if_head_size_unsupported(K: int) -> None:
    if K not in (64, 128):
        pytest.skip(f"GDP decode for head_size={K} is unavailable")


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


def _gen_decode_inputs(B, T, n_h, num_q_heads, num_v_heads, K, V, dtype, device, seed):
    """Dense decode inputs plus a state pool partitioned into disjoint regions.

    Pool layout -- every region must be disjoint or the tests measure nothing:

        row 0                       unused (0 is a sentinel elsewhere in flashinfer)
        [1, 1+B)                    initial states, one per batch row
        [1+B, 1+B+B*T)              per-REAL-token snapshots, ssm_state_indices
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
    return DecodeInputs(q, k, v, A_log, a, dt_bias, b, pool, initial, ssm)


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

    B, n_h, HQ, HV, K, V = 2, 1, 16, 32, 128, 128
    device, dtype = torch.device("cuda"), torch.bfloat16
    q, k, v, A_log, a, dt_bias, b, pool, idx, ssm = _gen_decode_inputs(
        B, T, n_h, HQ, HV, K, V, dtype, device, seed=0
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
@pytest.mark.parametrize("T", [1, 2, 4, 8], ids=lambda t: f"T={t}")
@pytest.mark.parametrize(
    "head_size",
    [(64, 64), (128, 128), (128, 64)],
    ids=lambda kv: f"K={kv[0]}_V={kv[1]}",
)
@pytest.mark.parametrize(
    "num_heads",
    [(16, 32)],
    ids=lambda qkv: "num_heads={0}/{1}".format(*qkv),
)
def test_decode_matches_reference(num_householder, T, head_size, num_heads):
    K, V = head_size
    _skip_if_unsupported()
    # n_h=1, T=1 is included deliberately. gated_delta_rule_mtp documents itself
    # as T > 1 in seven places and enforces it nowhere, but T=1 is computed
    # correctly -- verified against the first token of a T=2 run (max|d| = 0).
    # The docs are over-restrictive; the code is not.
    _skip_if_head_size_unsupported(K)
    _skip_if_head_size_unsupported(V)
    num_q_heads, num_v_heads = num_heads
    B = 3
    n_h = num_householder
    device, dtype = torch.device("cuda"), torch.bfloat16

    q, k, v, A_log, a, dt_bias, b, pool, idx, ssm = _gen_decode_inputs(
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
        # (write, defaulting to the read slot).
        disable_state_update=False,
    )
    torch.cuda.synchronize()

    assert got_o.shape == (B, T, num_v_heads, V), "one output row per REAL token"
    # tolerances from test_decode_delta_rule.py's MTP test (bf16)
    torch.testing.assert_close(got_o, ref_o.to(dtype), atol=1e-2, rtol=5e-3)
    # the live rows must hold each sequence's final state; pool is [.., V, K]
    torch.testing.assert_close(
        got_pool[idx.long()].transpose(-1, -2), ref_state, atol=1e-2, rtol=5e-3
    )


@pytest.mark.parametrize("batch_size", [1, 16])
def test_decode_strided_pool_resize_matches_reference(batch_size):
    """Profiling and serving pools can share strides but have different sizes."""
    _skip_if_unsupported()
    device, dtype = torch.device("cuda"), torch.bfloat16
    q, k, v, A_log, a, dt_bias, b, pool, idx, _ = _gen_decode_inputs(
        batch_size, 1, 3, 12, 12, 128, 64, dtype, device, seed=1
    )
    for pool_size in (pool.shape[0], pool.shape[0] + 7):
        backing = torch.randn(pool_size, 13, 64, 128, device=device)
        state = backing[:, :12]
        state[: pool.shape[0]].copy_(pool)
        expected = backing.clone()
        ref_o, ref_state = _reference(q, k, v, A_log, a, dt_bias, b, state, idx)
        expected[idx.long(), :12] = ref_state.transpose(-1, -2)

        out, _ = gated_delta_product_mtp(
            q,
            k,
            v,
            state,
            idx,
            A_log,
            a,
            dt_bias,
            b,
            scale=1.0,
            disable_state_update=False,
        )
        torch.testing.assert_close(out, ref_o.to(dtype), atol=1e-2, rtol=5e-3)
        torch.testing.assert_close(backing, expected, atol=1e-2, rtol=5e-3)


# --------------------------------------------------------------------------
# 4. Batch rows must not contaminate one another.
#
# The kernel reads the pool once, before the recurrence, via
# initial_state_indices; the per-token scatter is write-only after that, so
# concurrent stores to a row nobody reads are benign.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_householder", [2, 3], ids=lambda nh: f"num_householder={nh}"
)
def test_batch_rows_are_independent(num_householder):
    """Running a row alone must match running it in a batch."""
    _skip_if_unsupported()
    B, T, n_h, HQ, HV, K, V = 4, 2, num_householder, 16, 32, 128, 128
    device, dtype = torch.device("cuda"), torch.bfloat16

    q, k, v, A_log, a, dt_bias, b, pool, idx, ssm = _gen_decode_inputs(
        B, T, n_h, HQ, HV, K, V, dtype, device, seed=3
    )

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
            # .clone() the INDEX slices: a 1-element int32 slice is contiguous
            # but carries a 4-byte-granular offset, and the kernel demands
            # 16-byte data alignment. The q/k/v/a/b slices are safe -- their
            # per-row strides are large enough to stay aligned.
            idx[sl].clone(),
            A_log,
            a[sl],
            dt_bias,
            b[sl],
            scale=1.0,
            ssm_state_indices=ssm[sl].clone(),
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
# 5. The reference BRIDGE itself, against flashinfer's own decode reference.
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
    "num_heads", [(16, 32)], ids=lambda qkv: "num_heads={0}/{1}".format(*qkv)
)
def test_reference_bridge_matches_decode_delta_rule(B, num_heads):
    from .reference_delta_rule import decode_delta_rule

    num_q_heads, num_v_heads = num_heads
    T, n_h, K, V = 1, 1, 128, 128  # decode_delta_rule is single-step, n_h-free
    device = torch.device("cuda")

    q, k, v, A_log, a, dt_bias, b, pool, idx, ssm = _gen_decode_inputs(
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
# 6. Per-token state snapshots -- the property speculative decoding needs.
#
# ssm_state_indices[i, t] must end up holding the state AS OF real token t, so
# that a rejected draft can roll the sequence back to any accepted prefix. This
# is the only test that pins the scatter remap: it fails if the LAST micro-step
# of each token is not the one routed to the caller's slot (e.g. an off-by-one
# leaving the state after householder 0 there instead).
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_householder", [1, 2, 3], ids=lambda nh: f"num_householder={nh}"
)
@pytest.mark.parametrize("T", [2, 3], ids=lambda t: f"T={t}")
def test_per_token_state_snapshots(num_householder, T):
    _skip_if_unsupported()
    B, n_h, HQ, HV, K, V = 3, num_householder, 16, 32, 128, 128
    device, dtype = torch.device("cuda"), torch.bfloat16

    q, k, v, A_log, a, dt_bias, b, pool, idx, ssm = _gen_decode_inputs(
        B, T, n_h, HQ, HV, K, V, dtype, device, seed=5
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
        disable_state_update=False,
    )
    torch.cuda.synchronize()

    for t in range(T):
        got = got_pool[ssm[:, t].long()].transpose(-1, -2)  # [.., V, K] -> [.., K, V]
        torch.testing.assert_close(
            got,
            want[t],
            atol=1e-2,
            rtol=5e-3,
            msg=lambda m: (
                f"snapshot for real token {t} is wrong. A state after only the "
                f"FIRST householder here means the scatter remap targets "
                f"[0::n_h] instead of [n_h-1::n_h].\n{m}"
            ),
        )


# --------------------------------------------------------------------------
# 8. A negative ssm_state_indices entry must SKIP the snapshot write.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_householder", [2, 3], ids=lambda nh: f"num_householder={nh}"
)
@pytest.mark.parametrize(
    "batch_size,dispatch",
    [(4, "inline"), (8, "warp")],
    ids=["inline_kernel", "warp_kernel"],
)
def test_negative_ssm_state_index_skips_write(batch_size, dispatch, num_householder):
    """The per-token scatter must treat a negative slot as "do not write".

    This is what makes the wrapper's expansion cheap: micro-steps 1..n_h-1 get
    -1 and cost no state traffic, instead of being funnelled into throwaway pool
    rows. At n_h=3 that removes two thirds of the writes, which measured ~2.1x
    end-to-end on this kernel -- state writes dominate it.

    Three properties, and no other test in this file covers any of them,
    because they all assert on rows that ARE written:

      * rows a sentinel would have hit stay untouched -- the guard fires and is
        not compiled out. Writing ``if cutlass.const_expr(pool_slot_t >= 0)``
        instead of a plain ``if`` would resolve at trace time, emit the store
        unconditionally, and still pass every other test here.
      * rows named by a real slot are still written -- the guard suppresses the
        write, not the computation.
      * nothing lands before the pool. ``fla_idx = Int64(pool_slot_t) * HV +
        i_hv`` is ``-HV + i_hv`` at slot -1, i.e. an address BELOW the pool
        base: silent out-of-bounds corruption, not a wasted store. The pool is
        therefore a view into a larger tensor so such a write lands in a canary
        region we own.

    Both dispatch arms run because ``use_inline_kernel = (B * HV) <= 128``
    selects between two kernels with five scatter sites between them; guarding
    only one set would leave the other corrupting memory.
    """
    _skip_if_unsupported()
    from flashinfer.gdn_decode import gated_delta_rule_mtp

    n_h = num_householder
    B, T, HQ, HV, K = batch_size, 2, 16, 32, 128
    TN = T * n_h
    assert ((B * HV) <= 128) == (dispatch == "inline"), "dispatch arm mismatch"
    device, dtype = torch.device("cuda"), torch.bfloat16

    torch.manual_seed(11)
    with device:
        q = torch.randn(B, TN, HQ, K, dtype=dtype) * 0.1
        k = torch.randn(B, TN, HQ, K, dtype=dtype) * 0.1
        v = torch.randn(B, TN, HV, K, dtype=dtype) * 0.1
        a = torch.randn(B, TN, HV, dtype=dtype) * 0.1
        b = torch.randn(B, TN, HV, dtype=dtype) * 0.1
        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        # backing = [canary | pool]; `pool` is an offset view, so a write at a
        # negative slot lands in the canary instead of outside our allocation.
        canary_rows = 2
        backing = torch.zeros(
            canary_rows + 1 + B + B * TN, HV, K, K, dtype=torch.float32
        )
        pool = backing[canary_rows:]
        initial_idx = torch.arange(1, 1 + B, dtype=torch.int32)
        # every micro-step gets a distinct row, then all but the last of each
        # token is replaced by the sentinel -- exactly the wrapper's expansion
        idx_all = torch.arange(1 + B, 1 + B + B * TN, dtype=torch.int32).reshape(B, TN)
        keep = torch.zeros(TN, dtype=torch.bool)
        keep[n_h - 1 :: n_h] = True
        idx_sentinel = idx_all.clone()
        idx_sentinel[:, ~keep] = -1

    gated_delta_rule_mtp(
        q,
        k,
        v,
        pool,
        initial_idx,
        A_log,
        a,
        dt_bias,
        b,
        scale=K**-0.5,
        ssm_state_indices=idx_sentinel,
        disable_state_update=False,
    )
    torch.cuda.synchronize()

    assert torch.equal(
        backing[:canary_rows], torch.zeros_like(backing[:canary_rows])
    ), (
        f"{int((backing[:canary_rows] != 0).sum())} elements were written BEFORE the "
        "pool base -- a negative slot reached the store as fla_idx = -HV + i_hv. "
        "The scatter needs `if pool_slot_t >= 0` at all five sites."
    )
    skipped = idx_all[:, ~keep].flatten().long()
    written = pool[skipped].reshape(len(skipped), -1).any(dim=1).sum()
    assert written == 0, (
        f"{int(written)}/{len(skipped)} sentinel micro-steps were written anyway. "
        "A `const_expr` guard resolves at trace time and stores unconditionally; "
        "the guard must be a plain runtime `if`."
    )
    kept = idx_all[:, keep].flatten().long()
    unwritten = (pool[kept].reshape(len(kept), -1) == 0).all(dim=1).sum()
    assert unwritten == 0, (
        f"{int(unwritten)}/{len(kept)} real-token snapshots are missing; the guard "
        "is suppressing the computation, not just the redundant write"
    )


# --------------------------------------------------------------------------
# 11. SMEM chunking must not change results, and must bound the footprint.
# --------------------------------------------------------------------------
def _smem_bytes(chunk: int, k_dim: int, tile_v: int) -> int:
    """Mirror of run_gdn_verify_kernel_mtp's smem_bytes formula."""
    return (
        4 * chunk * (k_dim + 8)  # sQ
        + 4 * chunk * (k_dim + 8)  # sK
        + 4 * chunk  # sG
        + 4 * chunk  # sBeta
        + 4 * chunk * tile_v  # sVdata
        + 2 * chunk * tile_v  # sOutput
        + 128
    )


def test_choose_stage_rows_leaves_gdn_unchunked():
    """n_h=1 GDN must get CHUNK == T, i.e. the pre-chunking kernel verbatim.

    The chunked path is only reachable when the staged range is long enough to
    cost occupancy, which under expansion means n_h > 1. Any GDN shape returning
    CHUNK < T would be a behaviour change on a path this work is not meant to
    touch.
    """
    from flashinfer.gdn_kernels.gdn_decode_mtp import choose_stage_rows, get_mtp_config

    for batch in (1, 2, 8, 32, 128, 512):
        for seq_len in (1, 2, 4, 8):  # n_h=1: seq_len is the draft length
            tile_v, _, ilp_rows, _ = get_mtp_config(
                batch, seq_len, num_v_heads=32, v_dim=128, k_dim=128
            )
            assert choose_stage_rows(seq_len, 128, tile_v, ilp_rows) == seq_len, (
                f"B={batch} T={seq_len} would be chunked (tile_v={tile_v}, "
                f"ilp={ilp_rows}); GDN must keep CHUNK == T"
            )


@pytest.mark.parametrize("seq_len", [24, 30, 48], ids=lambda t: f"seq_len={t}")
def test_choose_stage_rows_bounds_smem(seq_len):
    """Above the budget, CHUNK must cap SMEM and divide T evenly.

    A non-divisor would leave a short final chunk, which the kernel's
    range_constexpr(CHUNK) loop cannot express -- it would read past the end of
    the staged range.
    """
    from flashinfer.gdn_kernels.gdn_decode_mtp import choose_stage_rows

    k_dim, tile_v, ilp_rows = 128, 16, 4  # tile_v // 4 == ilp_rows -> single sweep
    chunk = choose_stage_rows(seq_len, k_dim, tile_v, ilp_rows)
    assert chunk < seq_len, f"seq_len={seq_len} should be chunked"
    assert seq_len % chunk == 0, f"CHUNK={chunk} does not divide T={seq_len}"
    assert _smem_bytes(chunk, k_dim, tile_v) <= 16384


def test_choose_stage_rows_requires_single_sweep():
    """Chunking is unsafe when the consumer re-sweeps [0, T) per row tile.

    Each warp sweeps the token range rows_per_group // ilp_rows times. With more
    than one sweep the producer would refill a chunk before the second sweep read
    it, so the helper must decline rather than corrupt.
    """
    from flashinfer.gdn_kernels.gdn_decode_mtp import choose_stage_rows

    # tile_v // 4 == 8 != ilp_rows == 4  -> two sweeps
    assert choose_stage_rows(48, 128, 32, 4) == 48


@pytest.mark.parametrize(
    "num_householder,T,chunk",
    [(3, 8, 12), (2, 8, 8)],
    ids=["nh3_T8_chunk12", "nh2_T8_chunk8"],
)
def test_chunked_smem_is_bit_identical(num_householder, T, chunk):
    """Chunking must be exactly invisible in the results.

    It splits the token loop but reorders no arithmetic: every token still sees
    the same state in the same sequence. So this is exact equality, not a
    tolerance -- any difference means state or staging leaked across a chunk
    boundary. The regression it guards against is real: seeding the recurrent
    state r_h inside the chunk loop instead of once silently resets it every
    chunk, which no shape or tolerance check would catch.

    Note this does NOT use _skip_if_unsupported() -- that guard states the
    *prefill* arch requirement. gated_delta_rule_mtp runs more widely, and this
    test is worth running wherever it does.

    One alternative chunk per config: each distinct CHUNK is a separate JIT
    compilation, so the matrix is kept deliberately small.
    """
    import flashinfer.gdn_kernels.gdn_decode_mtp as mtp

    n_h, B, HQ, HV, K, V = num_householder, 3, 16, 32, 128, 128
    TN = T * n_h
    assert chunk < TN and TN % chunk == 0, "chunk must be a proper divisor of T*n_h"
    device, dtype = torch.device("cuda"), torch.bfloat16
    q, k, v, A_log, a, dt_bias, b, pool, idx, ssm = _gen_decode_inputs(
        B, T, n_h, HQ, HV, K, V, dtype, device, seed=17
    )

    def run(staged_rows):
        original = mtp.choose_stage_rows
        mtp.choose_stage_rows = lambda *args, **kwargs: staged_rows
        try:
            pool_copy = pool.clone()
            out, _ = gated_delta_product_mtp(
                q,
                k,
                v,
                pool_copy,
                idx,
                A_log,
                a,
                dt_bias,
                b,
                scale=K**-0.5,
                ssm_state_indices=ssm,
                disable_state_update=False,
            )
            torch.cuda.synchronize()
            return out.clone(), pool_copy.clone()
        finally:
            mtp.choose_stage_rows = original

    ref_out, ref_pool = run(TN)  # CHUNK == T: the pre-chunking code path
    out, pool_after = run(chunk)
    assert torch.equal(out, ref_out), (
        f"CHUNK={chunk} changed the output vs CHUNK={TN}; "
        "chunking must not reorder arithmetic"
    )
    assert torch.equal(pool_after, ref_pool), (
        f"CHUNK={chunk} changed the final state vs CHUNK={TN}; "
        "r_h must be seeded once and written back on the last chunk only"
    )


# --------------------------------------------------------------------------
# 9. q and a are read at the real token index; nothing is expanded.
#
# The kernel synthesizes the micro-steps it does not have rows for, so the
# wrapper hands it q/a untouched. These two tests pin the contract from both
# sides: no scratch is allocated, and a k/q token-count mismatch is rejected
# rather than read out of bounds.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_householder", [2, 3], ids=lambda nh: f"num_householder={nh}"
)
def test_decode_allocates_no_expansion_scratch(num_householder):
    """A GDP call must not allocate anything that scales with n_h.

    The expanded q/a/output buffers were 362 MiB at the target model's shape.
    Peak-allocation delta is the only observable that catches their return: the
    results are identical either way.
    """
    n_h, B, T, HQ, HV, K, V = num_householder, 3, 4, 16, 32, 128, 128
    device, dtype = torch.device("cuda"), torch.bfloat16
    q, k, v, A_log, a, dt_bias, b, pool, idx, ssm = _gen_decode_inputs(
        B, T, n_h, HQ, HV, K, V, dtype, device, seed=23
    )
    out = torch.empty(B, T, HV, V, dtype=dtype, device=device)

    def call():
        gated_delta_product_mtp(
            q,
            k,
            v,
            pool,
            idx,
            A_log,
            a,
            dt_bias,
            b,
            scale=K**-0.5,
            output=out,
            ssm_state_indices=ssm,
            disable_state_update=False,
        )
        torch.cuda.synchronize()

    call()  # JIT compile and warm the caching allocator first
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    call()
    grew = torch.cuda.max_memory_allocated() - before

    # One expanded q alone would be B*T*n_h*HQ*K*2 bytes.
    one_expanded_q = B * T * n_h * HQ * K * 2
    assert grew < one_expanded_q, (
        f"call allocated {grew} B; a single expanded q is {one_expanded_q} B, "
        "so the expansion is back"
    )


def test_decode_rejects_token_count_mismatch():
    """k carries n_h rows per real token; q carries one. Enforce the ratio."""
    from flashinfer.gdn_decode import gated_delta_rule_mtp

    B, T, n_h, HQ, HV, K, V = 2, 2, 3, 16, 32, 128, 128
    device, dtype = torch.device("cuda"), torch.bfloat16
    q, k, v, A_log, a, dt_bias, b, pool, idx, _ = _gen_decode_inputs(
        B, T, n_h, HQ, HV, K, V, dtype, device, seed=29
    )
    with pytest.raises(AssertionError, match="num_householder"):
        gated_delta_rule_mtp(
            q,
            k.flatten(1, 2),
            v.flatten(1, 2),
            pool,
            idx,
            A_log,
            a,
            dt_bias,
            b.flatten(1, 2),
            scale=K**-0.5,
            disable_state_update=False,
            num_householder=n_h + 1,
        )
