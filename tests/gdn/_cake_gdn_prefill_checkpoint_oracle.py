#!/usr/bin/env python3
"""Materialize isolated semantic and checkpoint oracles in a fresh process."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def main(input_path: Path, output_path: Path) -> None:
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    q, k, v, alpha, beta = (
        payload[name].to(device="cuda") for name in ("q", "k", "v", "alpha", "beta")
    )
    seq_lens = tuple(int(value) for value in payload["seq_lens"])
    interval = int(payload["interval"])
    mode = str(payload["mode"])
    state_heads = max(q.shape[1], v.shape[1])
    total = sum(seq_lens)
    scale = float(payload.get("scale", 1.0 / q.shape[-1] ** 0.5))
    expected_output = torch.full(
        (total, state_heads, q.shape[-1]),
        float("nan"),
        dtype=q.dtype,
        device="cuda",
    )
    cu_seqlens = payload.get("cu_seqlens")
    if cu_seqlens is None:
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
            dtype=torch.int64,
        )
    cu_seqlens = cu_seqlens.to(device="cuda")
    if mode == "batched":
        expected_initial = payload["initial_state"].to(device="cuda")
        expected_initial_before = expected_initial.clone()
        output_state = payload.get("output_state")
        if output_state is None:
            output_state = expected_initial
        else:
            output_state = output_state.to(device="cuda")
        expected_state = torch.empty_strided(
            output_state.shape,
            output_state.stride(),
            dtype=output_state.dtype,
            device="cuda",
        )
        expected_state.copy_(output_state)
        state_indices = payload.get("state_indices")
        if state_indices is None:
            state_indices = torch.arange(len(seq_lens), device="cuda")
        else:
            state_indices = state_indices.to(device="cuda", dtype=torch.int64)
        q_expanded = q.repeat_interleave(state_heads // q.shape[1], dim=1)
        k_expanded = k.repeat_interleave(state_heads // k.shape[1], dim=1)
        v_expanded = v.repeat_interleave(state_heads // v.shape[1], dim=1)
        token_start = 0
        for seq_index, seq_len in enumerate(seq_lens):
            state_row = int(state_indices[seq_index])
            state = expected_initial[state_row].transpose(-1, -2).float().clone()
            for local_index in range(seq_len):
                token = token_start + local_index
                old_state = alpha[token].reshape(-1, 1, 1) * state
                old_value = torch.einsum(
                    "hd,hdv->hv", k_expanded[token].float(), old_state
                )
                new_value = beta[token].reshape(-1, 1) * v_expanded[token].float()
                new_value += (1.0 - beta[token].reshape(-1, 1)) * old_value
                state = old_state - k_expanded[token].float().unsqueeze(
                    -1
                ) * old_value.unsqueeze(-2)
                state += k_expanded[token].float().unsqueeze(-1) * new_value.unsqueeze(
                    -2
                )
                expected_output[token] = scale * torch.einsum(
                    "hd,hdv->hv", q_expanded[token].float(), state
                )
            expected_state[state_row] = state.transpose(-1, -2)
            token_start += seq_len
        if not torch.equal(expected_initial, expected_initial_before):
            raise RuntimeError("fresh semantic oracle mutated initial state")
        expected_checkpoints = torch.empty(
            (0, state_heads, q.shape[-1], q.shape[-1]),
            dtype=expected_state.dtype,
            device="cuda",
        )
    elif mode == "checkpoint_per_sequence":
        from flashinfer.gdn_kernels.blackwell.gdn_cp_prefill import (
            cp_delta_rule_dsl_sm100,
        )

        source_state = torch.empty(
            (len(seq_lens), state_heads, q.shape[-1], q.shape[-1]),
            dtype=torch.float32,
            device="cuda",
        )
        cp_delta_rule_dsl_sm100(
            expected_output,
            source_state,
            q,
            k,
            v,
            alpha,
            beta,
            cu_seqlens,
            scale,
            max_seqlen=total,
        )
        expected_state = torch.empty(
            (len(seq_lens), state_heads, q.shape[-1], q.shape[-1]),
            dtype=torch.float32,
            device="cuda",
        )
        checkpoints = []
        token_start = 0
        recurrent_q = q.float().repeat_interleave(
            state_heads // q.shape[1], dim=1
        )
        recurrent_k = k.float().repeat_interleave(state_heads // k.shape[1], dim=1)
        recurrent_v = v.float().repeat_interleave(state_heads // v.shape[1], dim=1)
        for seq_index, seq_len in enumerate(seq_lens):
            state = torch.zeros(
                (state_heads, q.shape[-1], q.shape[-1]),
                dtype=torch.float32,
                device="cuda",
            )
            for local_index in range(seq_len):
                token = token_start + local_index
                old_state = alpha[token].reshape(-1, 1, 1) * state
                old_value = torch.einsum("hd,hdv->hv", recurrent_k[token], old_state)
                new_value = beta[token].reshape(-1, 1) * recurrent_v[token]
                new_value += (1.0 - beta[token].reshape(-1, 1)) * old_value
                state = old_state - recurrent_k[token].unsqueeze(
                    -1
                ) * old_value.unsqueeze(-2)
                state += recurrent_k[token].unsqueeze(-1) * new_value.unsqueeze(-2)
                expected_output[token] = scale * torch.einsum(
                    "hd,hdv->hv", recurrent_q[token], state
                )
                if (local_index + 1) % interval == 0:
                    checkpoints.append(state.transpose(-1, -2).clone())
            expected_state[seq_index] = state.transpose(-1, -2)
            token_start += seq_len
        expected_checkpoints = torch.stack(checkpoints)
    else:
        raise ValueError(f"unknown oracle mode: {mode}")

    torch.cuda.synchronize()
    if not bool(torch.isfinite(expected_output).all().item()):
        raise RuntimeError("fresh semantic oracle left output storage unwritten")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "output": expected_output.cpu(),
            "final_state": expected_state.cpu(),
            "checkpoints": expected_checkpoints.cpu(),
        },
        output_path,
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: oracle.py INPUT OUTPUT")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
