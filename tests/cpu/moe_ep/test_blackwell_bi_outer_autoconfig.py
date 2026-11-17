# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Public admission contracts for batch-invariant outer selection."""

from dataclasses import replace

import pytest
import torch

from flashinfer import moe_ep

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("other_mode", ("ordinary", "situ"))
def test_bi_outer_candidates_require_the_same_computation(other_mode):
    first = moe_ep.Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=256, top_k=3, batch_invariant=True
    )
    other = (
        replace(first, batch_invariant=False)
        if other_mode == "ordinary"
        else replace(first, activation="situ", situ_beta=1.0)
    )
    with pytest.raises(ValueError, match="cannot mix|same calculation"):
        moe_ep.AutoConfig(
            candidates=(moe_ep.MegaConfig(first), moe_ep.MegaConfig(other))
        )


def test_bi_outer_candidates_require_the_same_tensor_source():
    first = moe_ep.Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=256,
        top_k=3,
        batch_invariant=True,
        fc1_alpha=torch.ones(4),
    )
    shared = replace(first, knobs={"load_balance_mode": "atomic_counter"})
    moe_ep.AutoConfig(candidates=(moe_ep.MegaConfig(first), moe_ep.MegaConfig(shared)))
    with pytest.raises(ValueError, match="same runtime"):
        moe_ep.AutoConfig(
            candidates=(
                moe_ep.MegaConfig(first),
                moe_ep.MegaConfig(replace(shared, fc1_alpha=first.fc1_alpha.clone())),
            )
        )
