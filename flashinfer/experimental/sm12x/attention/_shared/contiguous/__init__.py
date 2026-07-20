# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/contiguous/__init__.py @ 149d6bb2 (2026-06-12) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Public contiguous-attention API."""

from .api import (
    AttentionBinding,
    AttentionPlan,
    AttentionPlanKey,
    AttentionScratchPlan,
    VarlenAttentionBinding,
    VarlenAttentionPlan,
    VarlenAttentionPlanKey,
    VarlenAttentionScratchPlan,
    sm12x_attention_forward,
    sm12x_varlen_attention_forward,
    clear_attention_caches,
    create_attention_plan,
    create_varlen_attention_plan,
    plan_attention_scratch,
    plan_varlen_attention_scratch,
)

__all__ = [
    "AttentionBinding",
    "AttentionPlan",
    "AttentionPlanKey",
    "AttentionScratchPlan",
    "VarlenAttentionBinding",
    "VarlenAttentionPlan",
    "VarlenAttentionPlanKey",
    "VarlenAttentionScratchPlan",
    "sm12x_attention_forward",
    "sm12x_varlen_attention_forward",
    "clear_attention_caches",
    "create_attention_plan",
    "create_varlen_attention_plan",
    "plan_attention_scratch",
    "plan_varlen_attention_scratch",
]
