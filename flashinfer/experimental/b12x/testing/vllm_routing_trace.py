"""Opt-in vLLM worker extension for graph-replayed MoE routing observations.

Launch with ``--worker-extension-cls
b12x.testing.vllm_routing_trace.RoutingTraceWorker``. The development RPCs
``begin_moe_routing_trace`` and ``end_moe_routing_trace`` delimit one request.
Only TP rank zero records; instrumented runs are not throughput measurements.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from types import MethodType

import torch
import triton
import triton.language as tl


_RECORDS = 512
_MAX_TOKENS = 8
_owners = {}
_enabled = None


@triton.jit(do_not_specialize=["rows"])
def _record_routes(
    ids, weights, enabled, counter, counts, recorded_ids, recorded_weights,
    rows, TOPK: tl.constexpr, CAPACITY: tl.constexpr, BLOCK: tl.constexpr,
):
    if tl.load(enabled) != 0:
        slot = tl.atomic_add(counter, 1)
        if slot < CAPACITY:
            tl.store(counts + slot, rows)
            index = tl.arange(0, BLOCK)
            live = index < rows * TOPK
            expert = tl.load(ids + index, live, other=-1)
            weight = tl.load(weights + index, live, other=0)
            destination = slot * (8 * TOPK) + index
            tl.store(recorded_ids + destination, expert, index < 8 * TOPK)
            tl.store(recorded_weights + destination, weight, index < 8 * TOPK)


def _observe(owner, ids, weights):
    global _enabled
    from vllm.distributed import get_tensor_model_parallel_rank

    if ids.shape[0] > _MAX_TOKENS or get_tensor_model_parallel_rank() != 0:
        return
    if _enabled is None:
        _enabled = torch.zeros(1, device=ids.device, dtype=torch.int32)
    key = id(owner)
    if key not in _owners:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("routing trace must be warmed before graph capture")
        topk = ids.shape[1]
        _owners[key] = (
            owner,
            torch.zeros(1, device=ids.device, dtype=torch.int32),
            torch.empty(_RECORDS, device=ids.device, dtype=torch.int32),
            torch.empty((_RECORDS, _MAX_TOKENS, topk), device=ids.device, dtype=torch.int32),
            torch.empty((_RECORDS, _MAX_TOKENS, topk), device=ids.device, dtype=torch.float32),
        )
    _, counter, counts, recorded_ids, recorded_weights = _owners[key]
    _record_routes[(1,)](
        ids, weights, _enabled, counter, counts, recorded_ids, recorded_weights,
        ids.shape[0], TOPK=ids.shape[1], CAPACITY=_RECORDS,
        BLOCK=triton.next_power_of_2(_MAX_TOKENS * ids.shape[1]), num_warps=1,
    )


def _install():
    from vllm.model_executor.layers.fused_moe.b12x import B12xExperts

    original_apply = B12xExperts.apply
    original_units = B12xExperts.get_b12x_preparation_units

    @functools.wraps(original_apply)
    def apply(self, *args, **kwargs):
        ids = kwargs.get("topk_ids", args[5] if len(args) > 5 else None)
        weights = kwargs.get("topk_weights", args[4] if len(args) > 4 else None)
        _observe(self, ids, weights)
        return original_apply(self, *args, **kwargs)

    @functools.wraps(original_units)
    def units(self, layer, workload):
        self._routing_trace_layer_name = layer.layer_name
        return original_units(self, layer, workload)

    B12xExperts.apply = apply
    B12xExperts.get_b12x_preparation_units = units


class RoutingTraceWorker:
    def set_moe_trace_verifier_rows(self, rows: int | None):
        """Control single-request verification only in this diagnostic worker."""
        manager = self.model_runner.adaptive_verification
        if manager is None:
            raise RuntimeError("fixed-row tracing requires adaptive verification")
        if rows is not None and not 1 <= rows <= _MAX_TOKENS:
            raise ValueError("trace verifier rows must be between one and eight")
        if not hasattr(self, "_trace_original_get_num_tokens"):
            self._trace_original_get_num_tokens = manager.get_num_tokens
        original = self._trace_original_get_num_tokens
        if rows is None:
            manager.get_num_tokens = original
        else:
            def fixed_rows(manager, num_tokens_per_req, draft_tokens):
                result = original(num_tokens_per_req, draft_tokens)
                drafts, non_drafts, _ = manager._batch_budget
                if len(drafts) == 1 and sum(non_drafts.values()) == 1:
                    budget = min(rows - 1, sum(drafts.values()))
                    manager._batch_budget = drafts, non_drafts, budget
                    return 1 + budget
                return result

            manager.get_num_tokens = MethodType(fixed_rows, manager)
        return {"rank": self.rank, "verifier_rows": rows}

    @torch.inference_mode()
    def begin_moe_routing_trace(self):
        if self.rank != 0:
            return {"rank": self.rank, "recording": False}
        if _enabled is None or not _owners:
            raise RuntimeError("no graph-ready MoE routing observers were installed")
        torch.cuda.synchronize()
        for _, counter, *_ in _owners.values():
            counter.zero_()
        _enabled.fill_(1)
        torch.cuda.synchronize()
        return {"rank": self.rank, "layers": len(_owners), "capacity": _RECORDS}

    @torch.inference_mode()
    def end_moe_routing_trace(self, destination: str):
        if self.rank != 0:
            return {"rank": self.rank}
        torch.cuda.synchronize()
        _enabled.zero_()
        torch.cuda.synchronize()
        records = []
        for owner, counter, counts, ids, weights in _owners.values():
            observed = int(counter.item())
            count = min(observed, _RECORDS)
            lengths = counts[:count].cpu().tolist()
            captured_ids = ids[:count].cpu().tolist()
            captured_weights = weights[:count].cpu().tolist()
            prepared = owner._prepared()
            records.append({
                "layer": owner._routing_trace_layer_name,
                "num_experts": prepared.num_experts,
                "hidden_size": prepared.hidden_size,
                "intermediate_size": prepared.intermediate_size,
                "total_calls": observed,
                "truncated": observed > _RECORDS,
                "calls": [
                    {"tokens": rows, "ids": call_ids[:rows], "weights": call_weights[:rows]}
                    for rows, call_ids, call_weights in zip(lengths, captured_ids, captured_weights, strict=True)
                ],
            })
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records))
        return {"rank": self.rank, "path": str(path), "layers": len(records)}


_install()
