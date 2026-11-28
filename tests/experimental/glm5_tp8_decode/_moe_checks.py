# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Distributed MoE checks, with routing ties bounded by the sigmoid error."""

import torch

from . import moe_reference as ref
from ._records import moe_previous_sender


def _ulp(value):
    return torch.exp2(torch.floor(torch.log2(value.abs().clamp_min(2.0**-126))) - 7)


def check_moe(control, plan, weights, x, out, *, cancellation=False):
    q_len = x.shape[0]
    diagnostic = plan._diagnostics
    x_cpu = x.cpu().clone()
    generation = int(diagnostic["generation"]())
    received = control.checked(
        lambda: moe_previous_sender(
            diagnostic["records"], q_len, control.rank, generation
        )
    )
    wire = control.gather(received)
    assert sorted(sender for sender, _ in wire) == list(range(8))
    partials = [dict(wire)[rank] for rank in range(8)]
    outputs = control.gather(out.cpu().clone())
    expected_exit = ref.reduce_ranks(partials, x_cpu)
    assert all(ref.bitwise(value, expected_exit) for value in outputs), (
        "TP exit must sum current wire partials and residual once"
    )

    norm = ref.rmsnorm(x, weights["post_attention_layernorm.weight"]).to(control.device)
    if q_len == 1:

        def local_stages():
            actual_norm = diagnostic["norm"][:1]
            ids, probs = diagnostic["ids"][:1], diagnostic["probs"][:1]
            ref.floored_ulp(actual_norm.cpu(), norm.cpu())
            sigma, choice = ref.route_reference(
                actual_norm,
                weights["mlp.gate.weight"],
                weights["mlp.gate.e_score_correction_bias"],
            )
            ref.check_routing(ids, probs, sigma, choice)
            ref.floored_ulp(
                diagnostic["mid"][:1], ref.reference_mid(norm, ids, weights)
            )
            expected_partial = ref.reference_partial(
                diagnostic["mid"][:1], ids, probs, weights
            )
            ref.floored_ulp(partials[control.rank], expected_partial.cpu())
            return ids.cpu().clone()

        ids = control.checked(local_stages)
        all_ids = control.gather(ids)
        assert all(ref.bitwise(value, all_ids[0]) for value in all_ids), (
            "routed expert identities differ across ranks"
        )
    else:

        def local_logits():
            logits = ref.kernel_logits(diagnostic["scores"], q_len).double().cpu()
            ref.floored_ulp(diagnostic["norm"][:q_len].cpu(), norm.cpu())
            ref.check_router_logits(
                logits,
                x_cpu,
                weights["post_attention_layernorm.weight"],
                weights["mlp.gate.weight"],
            )
            return logits

        logits = control.checked(local_logits)
        all_logits = control.gather(logits)
        assert all(torch.equal(value, all_logits[0]) for value in all_logits), (
            "replicated router logits differ across ranks"
        )
        sigma = torch.sigmoid(logits)
        choice = (
            sigma + weights["mlp.gate.e_score_correction_bias"].double().cpu()[None, :]
        )
        options = [
            ref.candidate_selections(choice[token], ref.SIGMOID_ABS_ERR + 1e-6)
            for token in range(q_len)
        ]
        ids = torch.tensor([values[0] for values in options], dtype=torch.int32)

        def reference_output(selection):
            probabilities, probability_bound = ref.routing_weights(sigma, selection)

            def local_reference():
                local_ids = selection.to(control.device)
                mid = ref.reference_mid(norm, local_ids, weights)
                partial = ref.reference_partial(
                    mid, local_ids, probabilities.float().to(control.device), weights
                ).cpu()
                terms = (
                    ref.reference_partial_terms(mid, local_ids, weights).double().cpu()
                )
                bound = (probability_bound[:, :, None] * terms[:, 1:].abs()).sum(1)
                bound += _ulp(partial.double().abs() + bound)
                return partial, bound

            local = control.checked(local_reference)
            values = control.gather(local)
            expected = ref.reduce_ranks([value[0] for value in values], x_cpu)
            bound = sum((value[1] for value in values), torch.zeros_like(values[0][1]))
            return expected, bound

        expected, bound = reference_output(ids)
        budget = (
            2
            * torch.exp2(
                torch.floor(torch.log2(expected.double().abs().clamp_min(1 / 16))) - 7
            )
            + bound
        )
        error = (outputs[0].double() - expected.double()).abs()
        for token in range(q_len):
            candidate = 0
            while not bool((error[token] <= budget[token]).all()):
                candidate += 1
                assert candidate < len(options[token]), {
                    "token": token,
                    "max_error_over_budget": float(
                        (error[token] / budget[token]).max()
                    ),
                    "legal_routing_sets": len(options[token]),
                }
                alternate = ids.clone()
                alternate[token] = torch.tensor(
                    options[token][candidate], dtype=torch.int32
                )
                revised, revised_bound = reference_output(alternate)
                budget[token] = (
                    2
                    * torch.exp2(
                        torch.floor(
                            torch.log2(revised[token].double().abs().clamp_min(1 / 16))
                        )
                        - 7
                    )
                    + revised_bound[token]
                )
                error[token] = (
                    outputs[0][token].double() - revised[token].double()
                ).abs()

    if cancellation:
        expected = torch.full((q_len,), 1 / 256, dtype=torch.bfloat16)
        assert all(ref.bitwise(value[:, 0], expected) for value in outputs), (
            "extra BF16 rounding destroyed the cancellation result"
        )
    return outputs[0]
