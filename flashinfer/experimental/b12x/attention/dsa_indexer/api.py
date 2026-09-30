"""Prepared public surface for native paged DSA indexing."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from ..._lib.gating import default_is_supported
from ...preparation import FrozenMapping, Plan, require_prepared
from . import META
from ._impl import clear_indexer_caches as clear_caches
from ._preparation import invocation_from_descriptors, invocation_from_tensors, plan
from .paged import PAGED_INDEX_PAGE_SIZE
from .mxfp4 import MXFP4_INDEX_PAGE_BYTES, MXFP4PreparedState, index_mxfp4_page_bytes

INDEX_HEAD_DIM = 128


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Capacity semantics for one immutable prepared paged DSA plan."""
    device: torch.device | str
    num_q_heads: int
    max_q_rows: int
    max_page_table_width: int
    topk: int
    mode: Literal["decode", "prefill"] = "decode"
    max_batch: int | None = None
    output_index_space: Literal["logical", "physical"] = "logical"
    route: str = "auto"
    page_size: int = PAGED_INDEX_PAGE_SIZE
    supertile_k: int = 0
    prefill_block_k: int = 256
    reserve_paged_logits: bool = False
    paged_logits_k_rows: int = 0
    score_mode: str = "dsa"
    num_idx_heads: int = 1
    cache_format: Literal["fp8", "mxfp4"] = "fp8"
    max_candidates: int = 0
    candidate_topk_blocks: int = 0

    def __post_init__(self):
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        for name in ("num_q_heads", "max_q_rows", "max_page_table_width", "topk", "page_size", "num_idx_heads"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            object.__setattr__(self, name, value)
        if self.mode not in ("decode", "prefill"):
            raise ValueError("mode must be decode or prefill")
        if self.output_index_space not in ("logical", "physical"):
            raise ValueError("output_index_space must be logical or physical")
        if self.route not in ("auto", "paged_fused", "paged_tiled", "packed_contiguous"):
            raise ValueError("unsupported DSA route")
        object.__setattr__(self, "max_batch", self.max_q_rows if self.max_batch is None else max(int(self.max_batch), 1))
        if self.cache_format not in ("fp8", "mxfp4"):
            raise ValueError("cache_format must be fp8 or mxfp4")
        object.__setattr__(self, "max_candidates", int(self.max_candidates))
        object.__setattr__(self, "candidate_topk_blocks", int(self.candidate_topk_blocks))
        if self.cache_format == "fp8":
            if self.page_size != PAGED_INDEX_PAGE_SIZE:
                raise ValueError(
                    f"FP8 indexer requires page_size={PAGED_INDEX_PAGE_SIZE}"
                )
            if self.max_candidates or self.candidate_topk_blocks:
                raise ValueError("FP8 indexer does not support MXFP4 candidate routes")
            return
        index_mxfp4_page_bytes(self.page_size)
        if self.output_index_space != "logical" or self.topk != 512:
            raise ValueError("MXFP4 requires logical topk=512 output")
        if self.num_q_heads > 32 or 32 % self.num_q_heads:
            raise ValueError("MXFP4 index heads must divide 32")
        if not 0 <= self.max_candidates <= 16384:
            raise ValueError("MXFP4 max_candidates must be in [0, 16384]")
        if self.candidate_topk_blocks not in (0, 2048):
            raise ValueError("MXFP4 candidate_topk_blocks must be zero or 2048")
        if self.max_candidates and self.candidate_topk_blocks:
            raise ValueError("MXFP4 source and reindex candidates are exclusive")
@dataclass(frozen=True, kw_only=True)
class Binding:
    plan: Plan
    runtime: object
    q_fp8: torch.Tensor | None
    query_weights: torch.Tensor
    index_k_cache: torch.Tensor
    output_indices: torch.Tensor
    output_scores: torch.Tensor | None = None
    q_mxfp4: torch.Tensor | None = None
    q_scales: torch.Tensor | None = None


def bind(plan: Plan, *, scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
         q_fp8: torch.Tensor | None = None, q_mxfp4: torch.Tensor | None = None,
         q_scales: torch.Tensor | None = None, query_weights: torch.Tensor,
         index_k_cache: torch.Tensor, page_table: torch.Tensor,
         cache_lengths: torch.Tensor, active_width: torch.Tensor,
         output_indices: torch.Tensor, output_scores: torch.Tensor | None = None,
         candidate_indices: torch.Tensor | None = None,
         candidate_lengths: torch.Tensor | None = None,
         candidate_output: torch.Tensor | None = None,
         candidate_output_lengths: torch.Tensor | None = None,
         score_width: int | None = None) -> Binding:
    """Bind real tensors to a prepared FP8 or MXFP4 plan."""
    if not isinstance(plan, Plan):
        raise TypeError("bind requires a session-prepared Plan")
    device_tensor = q_mxfp4 if q_mxfp4 is not None else q_fp8
    if device_tensor is None:
        raise ValueError("bind requires q_fp8 or q_mxfp4")
    state = require_prepared(plan, "attention.dsa_indexer", device_tensor.device)
    if isinstance(state, MXFP4PreparedState):
        if q_fp8 is not None or q_mxfp4 is None or q_scales is None:
            raise ValueError("MXFP4 prepared plan requires q_mxfp4 and q_scales only")
        runtime_binding = state.bind(
            scratch=scratch, q_mxfp4=q_mxfp4, q_scales=q_scales,
            query_weights=query_weights, index_k_cache=index_k_cache,
            page_table=page_table, cache_lengths=cache_lengths,
            active_width=active_width, output_indices=output_indices,
            output_scores=output_scores, candidate_indices=candidate_indices,
            candidate_lengths=candidate_lengths, candidate_output=candidate_output,
            candidate_output_lengths=candidate_output_lengths, score_width=score_width)
        return Binding(plan=plan, runtime=runtime_binding, q_fp8=None,
                       q_mxfp4=q_mxfp4, q_scales=q_scales,
                       query_weights=query_weights, index_k_cache=index_k_cache,
                       output_indices=output_indices, output_scores=output_scores)
    if q_mxfp4 is not None or q_scales is not None:
        raise ValueError("FP8 prepared plan requires q_fp8 only")
    caps = state.layout.caps
    if q_fp8 is None or q_fp8.ndim != 3 or tuple(q_fp8.shape[1:]) != (caps.num_q_heads, INDEX_HEAD_DIM) or q_fp8.dtype != torch.float8_e4m3fn or not q_fp8.is_contiguous():
        raise ValueError("q_fp8 must be contiguous (rows, heads, 128) torch.float8_e4m3fn")
    if int(q_fp8.shape[0]) > caps.max_q_rows or q_fp8.device != caps.device:
        raise ValueError("q_fp8 exceeds prepared DSA capacity or device")
    runtime = state.bind(scratch=scratch, real_page_table=page_table,
                         cache_seqlens_int32=cache_lengths, active_width=active_width,
                         expected_num_q_heads=caps.num_q_heads,
                         shared_page_table=caps.shared_page_table,
                         output_physical_slots=caps.output_physical_slots)
    return Binding(plan=plan, runtime=runtime, q_fp8=q_fp8,
                   query_weights=query_weights, index_k_cache=index_k_cache,
                   output_indices=output_indices, output_scores=output_scores)


def _mxfp4_state(plan: Plan, device: torch.device) -> MXFP4PreparedState:
    if not isinstance(plan, Plan):
        raise TypeError("MXFP4 quantization requires a session-prepared Plan")
    state = require_prepared(plan, "attention.dsa_indexer", device)
    if not isinstance(state, MXFP4PreparedState):
        raise ValueError("MXFP4 quantization requires an MXFP4 prepared plan")
    return state


def quantize_q_mxfp4(
    plan: Plan,
    query: torch.Tensor,
    *,
    q_mxfp4: torch.Tensor,
    q_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize Q with the callable retained by an MXFP4 prepared plan."""
    return _mxfp4_state(plan, query.device).quantize_query(
        query, q_mxfp4=q_mxfp4, q_scales=q_scales
    )


def quantize_write_index_k_mxfp4(
    plan: Plan,
    keys: torch.Tensor,
    *,
    index_k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    """Write K through the prepared MXFP4 paged-writer callable."""
    return _mxfp4_state(plan, keys.device).write_index_keys(
        keys, index_k_cache=index_k_cache, slot_mapping=slot_mapping
    )


def scratch_specs(plan: Plan, *, device: torch.device):
    """Return scratch requirements for an already prepared DSA plan."""
    if not isinstance(plan, Plan):
        raise TypeError("scratch_specs requires a session-prepared Plan")
    state = require_prepared(plan, "attention.dsa_indexer", device)
    return state.layout.scratch_specs()


def run(binding: Binding) -> torch.Tensor:
    if not isinstance(binding, Binding):
        raise TypeError("binding must be dsa_indexer.Binding")
    state = require_prepared(
        binding.plan, "attention.dsa_indexer",
        (binding.q_mxfp4 if binding.q_mxfp4 is not None else binding.q_fp8).device)
    if isinstance(state, MXFP4PreparedState):
        return state.run(binding.runtime)
    return state.run(binding.runtime, q_fp8=binding.q_fp8,
                     query_weights=binding.query_weights,
                     index_k_cache=binding.index_k_cache,
                     output_indices=binding.output_indices,
                     output_scores=binding.output_scores)


def score(binding: Binding) -> torch.Tensor:
    """Score an MXFP4 binding; TP callers reduce this BF16 matrix before select."""
    from .mxfp4 import score_mxfp4
    if not isinstance(binding, Binding):
        raise TypeError("binding must be dsa_indexer.Binding")
    state = require_prepared(
        binding.plan, "attention.dsa_indexer",
        (binding.q_mxfp4 if binding.q_mxfp4 is not None else binding.q_fp8).device)
    if not isinstance(state, MXFP4PreparedState):
        raise TypeError("score is only the staged MXFP4 DSA API")
    return score_mxfp4(binding.runtime, launchers=state._launchers)


def select(binding: Binding) -> torch.Tensor:
    """Select logical top-k indices after MXFP4 score reduction."""
    from .mxfp4 import select_mxfp4
    if not isinstance(binding, Binding):
        raise TypeError("binding must be dsa_indexer.Binding")
    state = require_prepared(
        binding.plan, "attention.dsa_indexer",
        (binding.q_mxfp4 if binding.q_mxfp4 is not None else binding.q_fp8).device)
    if not isinstance(state, MXFP4PreparedState):
        raise TypeError("select is only the staged MXFP4 DSA API")
    return select_mxfp4(binding.runtime, launchers=state._launchers)


def is_supported(device=None) -> bool:
    return default_is_supported(device, requires=META.requires)


__all__ = ["Caps", "Plan", "Binding", "plan", "bind", "run", "score", "select", "scratch_specs", "invocation_from_descriptors", "invocation_from_tensors", "quantize_q_mxfp4", "quantize_write_index_k_mxfp4", "index_mxfp4_page_bytes", "MXFP4_INDEX_PAGE_BYTES", "INDEX_HEAD_DIM", "PAGED_INDEX_PAGE_SIZE", "is_supported", "clear_caches"]
