# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Stage gates, exact selection membership, and exact TP8 communication math."""

import torch

from . import attention_reference as ref
from .moe_reference import bitwise
from ._records import attention_rank0_partials


def check_indexer(control, plan, weights, caches, x, rope, past, *, indexer, selection):
    q_len, diag = x.shape[0], plan._diagnostics
    freqs = rope[past : past + q_len]

    def indexer_checks():
        if control.rank != 0:
            return None
        if indexer != "full":
            return selection.cpu().clone()
        assert q_len == 4
        w = weights
        xn = ref.fp32_indexer_rmsnorm(x, w["input_layernorm.weight"])
        ref.assert_close(
            diag["qa"],
            ref.ref_gemv_w8a16(xn, *ref.projection(w, "q_a_proj")),
            2,
            2**-14,
        )
        ref.assert_close(
            diag["ki"],
            ref.ref_gemv_w8a16(xn, *ref.projection(w, "indexer.wk")),
            2,
            2**-14,
        )
        ref.assert_close(
            diag["headw"],
            ref.ref_gemv_bf16(xn, w["self_attn.indexer.weights_proj.weight"]),
            2,
            2**-14,
        )
        qn = ref.fp32_indexer_rmsnorm(diag["qa"], w["self_attn.q_a_layernorm.weight"])
        iq = ref.ref_gemv_w8a16(qn, *ref.projection(w, "indexer.wq_b")).reshape(
            4, 32, 128
        )
        ref.assert_close(diag["iq"], iq, 2, 2**-14)
        ref.assert_close(diag["iqrt"], ref.ref_rotate_q(diag["iq"], freqs), 1, 2**-14)
        new_k = ref.ref_k_ln_rope_had(
            diag["ki"],
            w["self_attn.indexer.k_norm.weight"],
            w["self_attn.indexer.k_norm.bias"],
            ref.LN_EPS,
            freqs,
        )
        ref.assert_close(caches["indexer_k_cache"][past : past + 4], new_k, 1, 2**-14)
        # Only causal logits are defined, and dense rows do not publish IDs.
        logits = diag["logits"].reshape(-1)[: 4 * (past + 4)].reshape(4, past + 4)
        if past >= 2044:
            expected = ref.ref_index_score(
                caches["indexer_k_cache"], past + 4, diag["iqrt"], diag["headw"]
            )
            causal = (
                torch.arange(past + 4, device=x.device)[None, :]
                <= torch.arange(past, past + 4, device=x.device)[:, None]
            )
            actual, expected = logits[causal], expected[causal]
            assert bool(torch.isfinite(actual).all())
            relative = float((actual - expected).abs().max()) / max(
                float(expected.abs().max()), 1e-30
            )
            assert relative <= 1e-5, {
                "causal_logit_relative_error": relative,
                "budget": 1e-5,
            }
        return ref.check_exact_topk(logits, diag["ids"], past, q_len).cpu()

    ids = control.checked(indexer_checks)
    return control.broadcast(ids)


def check_attention(
    control, plan, weights, caches, initial, x, out, rope, past, *, indexer, selection
):
    q_len, diag = x.shape[0], plan._diagnostics
    freqs = rope[past : past + q_len]
    ids = check_indexer(
        control,
        plan,
        weights,
        caches,
        x,
        rope,
        past,
        indexer=indexer,
        selection=selection,
    ).to(control.device)

    def records():
        if control.rank == 0:
            return attention_rank0_partials(
                diag["records"], q_len, int(diag["generation"]())
            )
        return None

    partials = control.broadcast(control.checked(records))
    outputs = control.gather(out.cpu())
    expected_exit = ref.ref_tp_exit_sum(list(partials.unbind(0)), x.cpu())
    assert all(bitwise(value, expected_exit) for value in outputs), (
        "TP8 wire sum + residual must be bit exact on all ranks"
    )

    def local_checks():
        for name, cache in caches.items():
            before = initial[name]
            if control.rank == 0 and indexer != "full":
                assert bitwise(cache, before), (
                    "shared indexer changed its caller-owned indexer key cache"
                )
            else:
                assert bitwise(cache[:past], before[:past]), (
                    f"{name} wrote before cur_pos"
                )
                assert bitwise(cache[past + q_len :], before[past + q_len :]), (
                    f"{name} wrote beyond the new query rows"
                )
        if control.rank == 0:
            return None
        pipeline = ref.attention_pipeline(weights, initial, x, past, freqs, ids)
        ref.assert_close(diag["post_kv"][:q_len], pipeline["kv"], 2, 2**-14)
        kv = ref.ref_rmsnorm(
            diag["post_kv"][:q_len],
            weights["self_attn.kv_a_layernorm.weight"],
            ref.KV_EPS,
        )
        ref.assert_close(caches["kv_cache"][past : past + q_len], kv, 1, 2**-14)
        ref.assert_rope_close(
            caches["pe_cache"][past : past + q_len], pipeline["pe_row"]
        )
        actual_latent = diag["post_o"][:q_len]
        ref.assert_close(actual_latent, pipeline["latent"], 2, 1, per_row=True)
        expected_latent = ref.ref_sparse_attention(
            diag["post_qnope"][:q_len],
            diag["post_qpe"][:q_len],
            caches["kv_cache"],
            caches["pe_cache"],
            ids,
            past,
        ).reshape_as(actual_latent)
        ref.assert_close(actual_latent, expected_latent, 1, 1, per_row=True)
        zero = expected_latent.abs().amax(-1) == 0
        assert bitwise(actual_latent[zero], expected_latent[zero]), (
            "zero latent rows must be exact"
        )
        uv = (
            ref.ref_gemv_uv(actual_latent.permute(1, 0, 2).contiguous(), weights)
            .permute(1, 0, 2)
            .contiguous()
        )
        ref.assert_close(diag["post_uv"][:q_len], uv, 2, 2**-14)
        actual_uv = diag["post_uv"][:q_len].reshape(q_len, 2560)
        o_proj = ref.projection(weights, "o_proj")
        expected_partial = ref.ref_gemv_w8a16(actual_uv, *o_proj)
        ref.assert_close(
            partials[control.rank - 1].to(x.device), expected_partial, 2, 2**-14
        )
        return ref.fp32_o_proj(actual_uv, *o_proj).cpu()

    predicted = control.gather(control.checked(local_checks))
    if q_len == 4:
        # q4 also checks the TP output against the rank-ordered sum of the
        # FP32-order o_proj references. q1 is covered by its stage checks.
        ordered_exit = ref.ref_tp_exit_sum(predicted[1:], x.cpu())
        ref.assert_close(outputs[0], ordered_exit, 2, 1 / 16)
    return outputs[0].clone()
