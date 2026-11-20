"""Rubin (SM107) generation-phase tactics: our tuned tables (``<model>_genphase``) and the TensorRT-LLM table (``trtllm``).

``--tuned <model>_genphase`` (``dsv4_pro_genphase``, ``dsv4_1_flash_genphase``, ``kimi_k3_genphase``,
``qwen3_8_a95b_genphase``) uses our own tune results in ``genphase_tuned_tactics.py`` (hand-maintained, see its header),
keyed on (EP, routing, tokens); see ``--tuned_ep``. A fused shared-expert run takes its entry from that file's
``SHARED_TUNED_TACTICS`` when the key has one.

Both kinds of preset are for NVFP4 only (tile K 256): the runner rejects ``--tuned`` with any other ``--quant_kind``; other
precisions are not tuned yet.

``--tuned trtllm`` uses the TRT-LLM token-bucket table below and is only used when asked for: it is the reference the tuned
tables are compared against. Source: ``tensorrt_llm/_torch/custom_ops/cute_dsl_megamoe_custom_op.py``
(``_SM107_GENPHASE_TACTICS``). Only the knobs the standalone runner exposes are kept. The TRT-LLM tactic fields that are fixed
by the generation-phase kernel here (cluster shape [4,1,1], token_back_mode ``epi_warps``, bulk FC2 store, flag_batch 1) are
not represented. It is presumed tuned on DSv4 Pro (hidden 7168, intermediate 3072, topk 6, DEP4/DEP8); the TRT-LLM commit does
not record the tuning model, so that attribution is an inference; other shapes may want their own sweep. Keyed on tokens only;
the bucket is the smallest key >= max_tokens_per_rank. TRT-LLM's DSv4 Pro shape-tuned table is intentionally not ported: for
<=1024 tokens/rank the gen-phase table takes precedence over it there.
"""

from typing import Dict, Optional, Tuple

from .genphase_tuned_tactics import (
    CLAIM_GATE_OFF,
    SF_PLAIN_TRANSPORT,
    SHARED_TUNED_TACTICS,
    TUNED_TACTICS,
)

# bucket -> (mma_tiler_mnk, schedule_policy, work_id_mode, fc2_tma_stages, epi_flag_batches)
Tactic = Tuple[Tuple[int, int, int], Tuple[str, Optional[int]], str, Optional[int], Tuple[int, int]]

_ATOMIC = "atomic_counter"
_N128 = (256, 128, 256)
_N256 = (256, 256, 256)

DSV4_PRO_GENPHASE_TACTICS: Dict[int, Tactic] = {
    32: (_N128, ("phase_interleave", 3), _ATOMIC, 1, (2, 4)),
    64: (_N128, ("phase_interleave", 3), _ATOMIC, 1, (1, 4)),
    128: (_N256, ("phase_interleave", 3), _ATOMIC, 1, (1, 4)),
    192: (_N256, ("phase_interleave", 3), _ATOMIC, 1, (1, 4)),
    256: (_N256, ("phase_interleave", 3), _ATOMIC, 1, (1, 1)),
    384: (_N256, ("phase_interleave", 3), _ATOMIC, 2, (1, 1)),
    512: (_N256, ("phase_interleave", 3), _ATOMIC, 2, (1, 4)),
    1024: (_N256, ("phase_interleave", None), _ATOMIC, 2, (1, 1)),
}
# TRT-LLM runs the FC2 top-k reduce in the kernel for these buckets (requires pre_fc1_quant).
TRTLLM_IN_KERNEL_FC2_REDUCE_BUCKETS = frozenset({32})


TRTLLM_PRESET = "trtllm"


def tuned_presets() -> Tuple[str, ...]:
    """Every ``--tuned`` preset: one ``<model>_genphase`` per tuned model, then the explicit ``trtllm`` table."""
    return (*sorted(TUNED_TACTICS), TRTLLM_PRESET)


def resolve_tactic(preset: str, *, max_tokens_per_rank: int) -> Tuple[Tactic, bool]:
    """TRT-LLM table lookup: ``(tactic, reduce_topk_in_kernel)``; raises ValueError without an entry."""
    if preset != TRTLLM_PRESET:
        raise ValueError(f"Unknown preset {preset!r}: resolve_tactic serves the TRT-LLM table, preset {TRTLLM_PRESET!r}.")
    bucket = min((b for b in DSV4_PRO_GENPHASE_TACTICS if max_tokens_per_rank <= b), default=None)
    if bucket is None:
        raise ValueError(f"No tuned gen-phase tactic for max_tokens_per_rank={max_tokens_per_rank} (max 1024).")
    return DSV4_PRO_GENPHASE_TACTICS[bucket], bucket in TRTLLM_IN_KERNEL_FC2_REDUCE_BUCKETS


def _shared_tuned_entry(preset: str, *, max_tokens_per_rank: int, ep: Optional[int], route: str):
    """The fused shared-expert entry of this key's routed bucket, or None when the fused kernel runs the routed tactic."""
    routed = TUNED_TACTICS[preset].get((ep, route), {})
    bucket = min((b for b in routed if max_tokens_per_rank <= b), default=None)
    return SHARED_TUNED_TACTICS.get(preset, {}).get((ep, route), {}).get(bucket)


def resolve_tuned_tactic(
    preset: str, *, max_tokens_per_rank: int, ep: Optional[int], route: str, shared: bool = False
) -> Tuple[Tactic, bool, str]:
    """``<model>_genphase`` lookup keyed on (ep, route, tokens): ``(tactic, reduce_topk_in_kernel, topk_position)``.

    ``shared`` selects the fused shared-expert entry when the key has one. EP16/EP32 entries were tuned with EP
    emulated on 4 ranks (expert count scaled, communication fan-out still 4), not on a real 16/32-rank launch; see
    ``genphase_tuned_tactics.py``.
    """
    if ep is None:
        raise ValueError(f"Preset {preset!r} is keyed on EP: pass --tuned_ep.")
    table = TUNED_TACTICS[preset].get((ep, route))
    if table is None:
        known = ", ".join(f"EP{e}/{r}" for e, r in sorted(TUNED_TACTICS[preset]))
        raise ValueError(f"Preset {preset!r} has no entry for EP{ep}/{route}; tuned: {known}.")
    bucket = min((b for b in table if max_tokens_per_rank <= b), default=None)
    if bucket is None:
        raise ValueError(f"No tuned gen-phase tactic for max_tokens_per_rank={max_tokens_per_rank} (max {max(table)}).")
    entry = table[bucket]
    if shared:
        shared_entry = _shared_tuned_entry(preset, max_tokens_per_rank=max_tokens_per_rank, ep=ep, route=route)
        if shared_entry is not None:
            entry = shared_entry[:-1]
    *tactic, reduce_in_kernel, position = entry
    return tuple(tactic), reduce_in_kernel, position


def resolve_shared_knobs(preset: str, *, max_tokens_per_rank: int, ep: Optional[int], route: str) -> Dict[str, object]:
    """The fused entry's impl knobs that differ from the kernel's defaults (empty without a fused entry)."""
    entry = _shared_tuned_entry(preset, max_tokens_per_rank=max_tokens_per_rank, ep=ep, route=route)
    return {} if entry is None else dict(entry[-1])


def resolve_claim_gate(
    preset: str, *, max_tokens_per_rank: int, ep: Optional[int], route: str, shared: bool = False
) -> bool:
    """Whether the scheduler gates tile claims on TMA-A progress for this tuned entry."""
    if shared:
        entry = _shared_tuned_entry(preset, max_tokens_per_rank=max_tokens_per_rank, ep=ep, route=route)
        if entry is not None:
            return entry[-1].get("claim_gate_on_tma_a", True)
    table = TUNED_TACTICS[preset].get((ep, route), {})
    bucket = min((b for b in table if max_tokens_per_rank <= b), default=None)
    return (preset, ep, route, bucket) not in CLAIM_GATE_OFF


def resolve_sf_plain_transport(
    preset: str, *, max_tokens_per_rank: int, ep: Optional[int], route: str, shared: bool = False
) -> bool:
    """Whether scale factors go as a flag-gated plane (rather than LL words) for this tuned entry."""
    if shared:
        entry = _shared_tuned_entry(preset, max_tokens_per_rank=max_tokens_per_rank, ep=ep, route=route)
        if entry is not None:
            return entry[-1].get("sf_plain_transport", False)
    table = TUNED_TACTICS[preset].get((ep, route), {})
    bucket = min((b for b in table if max_tokens_per_rank <= b), default=None)
    return (preset, ep, route, bucket) in SF_PLAIN_TRANSPORT
