"""BF16 / page-16 DCP speculative decode through the static split-KV route."""

from __future__ import annotations

import math

import pytest
import torch

import flashinfer.cake_dcp as cake_dcp
from flashinfer.cake_dcp import (
    get_dcp_spec_counter_bytes,
    get_dcp_spec_workspace_size_bytes,
    run_dcp_spec_decode,
)
from flashinfer.utils import get_compute_capability

_HEAD_DIM = 128
_PAGE_SIZE = 16
_BLOCK_N = 128


def _require_blackwell_dcp() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3)):
        pytest.skip("Cake FMHA DCP requires SM100 or SM103")


def _local_len(prefix: int, q_len: int, cp_world: int, cp_rank: int) -> int:
    """Rank-local keys visible to the last speculative row: ``|arange(rank, prefix + q_len, W)|``."""
    last = prefix + q_len - 1 - cp_rank
    return 0 if last < 0 else last // cp_world + 1


def _visible(prefix: int, row: int, cp_world: int, cp_rank: int) -> int:
    last = prefix + row - cp_rank
    return 0 if last < 0 else last // cp_world + 1


class _Problem:
    """One rank-local BF16 / page-16 problem with a shuffled page table and caller-owned scratch."""

    def __init__(
        self,
        prefixes,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        cp_rank,
        *,
        seed,
        device,
    ):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.q_len, self.cp_world, self.cp_rank = q_len, cp_world, cp_rank
        self.num_q_heads, self.num_kv_heads = num_q_heads, num_kv_heads
        self.prefixes = [int(p) for p in prefixes]
        batch = len(self.prefixes)
        self.local_lens = [
            _local_len(p, q_len, cp_world, cp_rank) for p in self.prefixes
        ]
        self.max_local = max(self.local_lens)
        pages_per_seq = max(1, -(-self.max_local // _PAGE_SIZE))
        # Round the per-request page count up to whole 128-key blocks, as the dispatcher does.
        blocks = -(-self.max_local // _BLOCK_N)
        self.max_pages_per_seq = max(pages_per_seq, blocks * _BLOCK_N // _PAGE_SIZE)
        total_pages = batch * self.max_pages_per_seq + 1
        perm = (
            torch.randperm(total_pages - 1, generator=generator) + 1
        )  # page 0 stays unused
        self.block_tables = (
            perm.view(batch, self.max_pages_per_seq).to(torch.int32).to(device)
        )
        self.k_cache = (
            torch.randn(
                (total_pages, num_kv_heads, _PAGE_SIZE, _HEAD_DIM),
                generator=generator,
                dtype=torch.float32,
            )
            .to(torch.bfloat16)
            .to(device)
        )
        self.v_cache = (
            torch.randn(
                (total_pages, num_kv_heads, _PAGE_SIZE, _HEAD_DIM),
                generator=generator,
                dtype=torch.float32,
            )
            .to(torch.bfloat16)
            .to(device)
        )
        self.query = (
            torch.randn(
                (batch * q_len, num_q_heads, _HEAD_DIM),
                generator=generator,
                dtype=torch.float32,
            )
            .to(torch.bfloat16)
            .to(device)
        )
        self.seq_lens = torch.tensor(self.local_lens, dtype=torch.int32, device=device)
        self.causal_seqlens_kv_global = torch.tensor(
            self.prefixes, dtype=torch.int32, device=device
        )
        self.scale = _HEAD_DIM**-0.5
        self.out = torch.empty_like(self.query)
        self.lse = torch.empty(
            (batch * q_len, num_q_heads), dtype=torch.float32, device=device
        )
        self.workspace = torch.empty(
            get_dcp_spec_workspace_size_bytes(batch, q_len, num_q_heads),
            dtype=torch.uint8,
            device=device,
        )
        self.counters = torch.zeros(
            get_dcp_spec_counter_bytes(batch, q_len, num_kv_heads),
            dtype=torch.uint8,
            device=device,
        )

    def selected_num_split(self, device) -> int:
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        local_blocks = max(1, -(-self.max_local // _BLOCK_N))
        return cake_dcp._select_num_split(
            logical_tiles=len(self.prefixes) * self.q_len * self.num_kv_heads,
            sm_count=sm_count,
            local_blocks=local_blocks,
        )

    def run(self, route: str) -> None:
        self.out.fill_(float("nan"))
        self.lse.fill_(float("nan"))
        run_dcp_spec_decode(
            query=self.query,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            workspace_buffer=self.workspace,
            block_tables=self.block_tables,
            seq_lens=self.seq_lens,
            causal_seqlens_kv_global=self.causal_seqlens_kv_global,
            max_local_seq_len=self.max_local,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            cp_world=self.cp_world,
            cp_rank=self.cp_rank,
            q_len_per_req=self.q_len,
            out=self.out,
            lse=self.lse,
            completion_buffer=self.counters,
            route=route,
        )

    def reference(self):
        """Rank-local normalized output and base-2 LSE over the keys the rank owns and the row sees."""
        batch = len(self.prefixes)
        group = self.num_q_heads // self.num_kv_heads
        out = torch.zeros_like(self.out, dtype=torch.float32)
        lse = torch.full_like(self.lse, float("-inf"))
        for b in range(batch):
            pages = self.block_tables[b].long()
            k = (
                self.k_cache[pages]
                .permute(1, 0, 2, 3)
                .reshape(self.num_kv_heads, -1, _HEAD_DIM)
                .float()
            )
            v = (
                self.v_cache[pages]
                .permute(1, 0, 2, 3)
                .reshape(self.num_kv_heads, -1, _HEAD_DIM)
                .float()
            )
            for row in range(self.q_len):
                visible = _visible(self.prefixes[b], row, self.cp_world, self.cp_rank)
                if visible == 0:
                    continue
                q = self.query[b * self.q_len + row].float()  # [num_q_heads, D]
                for kv_head in range(self.num_kv_heads):
                    qh = q[kv_head * group : (kv_head + 1) * group]
                    s = (qh @ k[kv_head, :visible].T) * self.scale  # [group, visible]
                    m = s.max(dim=-1, keepdim=True).values
                    p = torch.exp(s - m)
                    denom = p.sum(dim=-1, keepdim=True)
                    out[
                        b * self.q_len + row, kv_head * group : (kv_head + 1) * group
                    ] = (p / denom) @ v[kv_head, :visible]
                    lse[
                        b * self.q_len + row, kv_head * group : (kv_head + 1) * group
                    ] = (m.squeeze(-1) + torch.log(denom.squeeze(-1))) / math.log(2.0)
        return out, lse


# (prefixes, q_len, num_q_heads, num_kv_heads, cp_world, cp_rank, expected static num_split on 148 SMs)
_CASES = {
    "split16_b1_q1_s49152_cp4_r0": ([49152], 1, 64, 8, 4, 0, 16),
    "split4_b1_q4_s16384_cp4_r3": ([16384], 4, 64, 8, 4, 3, 4),
    "split2_b2_q4_s8192_cp4_r1": ([8192, 8191], 4, 64, 8, 4, 1, 2),
    "split3_b1_q2_s4096_cp2_r1_mha8": ([4096], 2, 8, 8, 2, 1, 3),
    "split1_b8_q4_s4096_cp4_r0": ([4096] * 8, 4, 64, 8, 4, 0, 1),
    "split2_ragged_empty_rows": ([8192, 0, 3, 7000], 2, 64, 8, 4, 2, 2),
}


@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_bf16_page16_static_split_kv_matches_reference(case) -> None:
    _require_blackwell_dcp()
    device = torch.device("cuda")
    prefixes, q_len, hq, hkv, cp_world, cp_rank, expected_split = _CASES[case]
    problem = _Problem(
        prefixes,
        q_len,
        hq,
        hkv,
        cp_world,
        cp_rank,
        seed=4323000 + len(case),
        device=device,
    )
    if torch.cuda.get_device_properties(device).multi_processor_count == 148:
        assert problem.selected_num_split(device) == expected_split, case
    problem.run("static")
    torch.cuda.synchronize()
    ref_out, ref_lse = problem.reference()
    got_out, got_lse = problem.out.float(), problem.lse

    empty = torch.isneginf(ref_lse)
    assert torch.equal(torch.isneginf(got_lse), empty), (
        "LSE -inf positions must match the empty rows"
    )
    assert bool((got_out[empty] == 0).all()), (
        "empty rows must produce exact zero output"
    )
    torch.testing.assert_close(got_out[~empty], ref_out[~empty], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(got_lse[~empty], ref_lse[~empty], atol=1e-2, rtol=1e-2)
    # The split counters self-reset so the same scratch serves the next launch.
    assert int(problem.counters.view(torch.int32).abs().sum()) == 0


@pytest.mark.parametrize(
    "case", ["split16_b1_q1_s49152_cp4_r0", "split2_ragged_empty_rows"]
)
def test_bf16_page16_static_split_kv_agrees_with_balanced_route(case) -> None:
    _require_blackwell_dcp()
    device = torch.device("cuda")
    prefixes, q_len, hq, hkv, cp_world, cp_rank, _ = _CASES[case]
    problem = _Problem(
        prefixes,
        q_len,
        hq,
        hkv,
        cp_world,
        cp_rank,
        seed=4324000 + len(case),
        device=device,
    )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    balanced_workspace = torch.empty(
        max(
            problem.workspace.numel(),
            cake_dcp.get_dcp_spec_balanced_workspace_bytes(sm_count),
        ),
        dtype=torch.uint8,
        device=device,
    )
    balanced_counters = torch.zeros(
        max(
            problem.counters.numel(),
            cake_dcp.get_dcp_spec_balanced_counter_bytes(sm_count),
        ),
        dtype=torch.uint8,
        device=device,
    )
    problem.run("static")
    torch.cuda.synchronize()
    static_out, static_lse = problem.out.clone(), problem.lse.clone()
    problem.workspace, problem.counters = balanced_workspace, balanced_counters
    problem.run("balanced")
    torch.cuda.synchronize()
    finite = torch.isfinite(static_lse)
    assert torch.equal(torch.isfinite(problem.lse), finite)
    torch.testing.assert_close(
        problem.out.float()[finite], static_out.float()[finite], atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        problem.lse[finite], static_lse[finite], atol=1e-2, rtol=1e-2
    )
