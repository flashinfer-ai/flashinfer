"""Metadata and chunk-contraction helpers for independent prefill oracles."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal
import torch


def _scalar(value: torch.Tensor | int) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("device count tensors must contain one element")
        return int(value.item())
    return int(value)


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()



def _validate_packed(
    *,
    cu_seqlens: torch.Tensor,
    initial_state_indices: torch.Tensor,
    final_state_indices: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    checkpoint_offsets: torch.Tensor,
    num_seqs: int,
    num_tokens: int,
    token_capacity: int,
    seq_capacity: int,
    state_slots: int,
    chunk: int,
    null_state_index: int | None,
) -> list[tuple[int, int]]:
    """Raise on packed metadata outside the kernels' contract; return spans."""
    if num_seqs < 0 or num_seqs > seq_capacity:
        raise ValueError(f"num_seqs={num_seqs} exceeds capacity {seq_capacity}")
    if num_tokens < 0 or num_tokens > token_capacity:
        raise ValueError(f"num_tokens={num_tokens} exceeds capacity {token_capacity}")
    if int(cu_seqlens[0]) != 0:
        raise ValueError("cu_seqlens[0] must be zero")
    if int(cu_seqlens[num_seqs]) != num_tokens:
        raise ValueError("cu_seqlens[num_seqs] must equal num_tokens")
    spans: list[tuple[int, int]] = []
    write_slots: set[int] = set()
    read_slots: set[int] = set()
    for request in range(num_seqs):
        start = int(cu_seqlens[request])
        end = int(cu_seqlens[request + 1])
        if start < 0 or end < start or end > num_tokens:
            raise ValueError(f"invalid query interval [{start}, {end})")
        spans.append((start, end))

    def is_null(slot: int) -> bool:
        return null_state_index is not None and slot == null_state_index

    for request, (start, end) in enumerate(spans):
        initial = int(initial_state_indices[request])
        final = int(final_state_indices[request])
        checkpoint = int(checkpoint_state_indices[request])
        offset = int(checkpoint_offsets[request])
        for slot, role in ((initial, "initial"), (final, "final"), (checkpoint, "checkpoint")):
            if is_null(slot):
                continue
            if slot < 0 or slot >= state_slots:
                raise IndexError(f"{role} state index {slot} is out of range")
        if not is_null(initial):
            read_slots.add(initial)
        if offset > end - start:
            raise ValueError("checkpoint offset exceeds the sequence length")
        if offset > 0 and offset % chunk != 0:
            raise ValueError(f"checkpoint offset {offset} is not a multiple of {chunk}")
        for slot in (final, checkpoint if offset > 0 else None):
            if slot is None or is_null(slot):
                continue
            if slot in write_slots:
                raise ValueError(f"duplicate write state index {slot}")
            write_slots.add(slot)
    for request in range(len(spans)):
        initial = int(initial_state_indices[request])
        final = int(final_state_indices[request])
        if is_null(initial):
            continue
        conflicting = write_slots - ({final} if not is_null(final) else set())
        if initial in conflicting:
            raise ValueError(f"initial state index {initial} is written by another sequence")
    return spans



@dataclass(frozen=True)
class MirrorPolicy:
    """Rounding points of the chunk mirror; the default is the kernel's policy."""

    state_master: Literal["fp32", "bf16"] = "fp32"
    shadow: bool = True
    inv_operand: Literal["bf16", "fp32"] = "bf16"
    u_operand: Literal["bf16", "fp32"] = "bf16"
    single_rounding: bool = True
    scale_dtype: Literal["fp32", "bf16"] = "fp32"
    operands: Literal["bf16", "fp32"] = "bf16"



def _neumann_inverse(lower: torch.Tensor, chunk: int) -> torch.Tensor:
    """Return ``(I + L)^{-1}`` for strictly lower-triangular ``L`` in fp32.

    ``-L`` is nilpotent, so the product ``(I - L)(I + L^2)(I + L^4)...`` over
    ``log2(chunk)`` factors is the exact inverse.
    """
    eye = torch.eye(chunk, dtype=torch.float32, device=lower.device)
    inverse = eye - lower
    power = lower
    steps = int(math.log2(chunk))
    for _ in range(1, steps):
        power = power @ power
        inverse = inverse + inverse @ power
    return inverse



def _recur_tile(
    state: torch.Tensor,
    v: torch.Tensor,
    prep: dict[str, torch.Tensor],
    *,
    rows: int,
    chunk: int,
    policy: MirrorPolicy,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Chunk recurrence mirror with an FP32 master state in [heads, V, K] order."""
    mask = torch.zeros((chunk, 1, 1), dtype=torch.bool, device=v.device)
    mask[:rows] = True
    v_h = torch.where(mask, v.float(), torch.zeros_like(v.float())).transpose(0, 1)
    shadow = _bf16(state) if policy.shadow else state
    v_prime = (v_h - prep["k_tilde"] @ shadow.transpose(-1, -2)) * prep["beta"][:, :, None]
    v_prime_op = _bf16(v_prime) if policy.operands == "bf16" else v_prime
    u = prep["inv_op"] @ v_prime_op
    u_op = _bf16(u) if policy.u_operand == "bf16" else u
    out = prep["q_tilde"] @ shadow.transpose(-1, -2) + prep["mqk"] @ u_op
    delta_t = u_op.transpose(-1, -2) @ prep["k_r"]
    new_state = state * prep["lambda_c"][:, None, :] + delta_t
    if policy.state_master == "bf16":
        new_state = _bf16(new_state)
    trace = {
        "v_prime": v_prime,
        "u": u,
        "out_tile": out,
        "delta_state": delta_t,
        "state_after": new_state,
    }
    return out.transpose(0, 1).to(torch.bfloat16), new_state, trace
