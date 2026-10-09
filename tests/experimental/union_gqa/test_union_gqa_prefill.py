"""
Copyright (c) 2026 by NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from flashinfer import union_gqa_prefill

BLOCK_SIZE = 64


def test_public_entry_point_is_experimental() -> None:
    assert union_gqa_prefill.is_experimental


@pytest.mark.parametrize("scale", [1e100, 1e-100, 1e-40, 2.4e38])
@pytest.mark.parametrize("direct_binding", [False, True])
def test_union_gqa_rejects_unsupported_softmax_scale(
    scale: float, direct_binding: bool
) -> None:
    _require_sm100()
    q = torch.zeros(1, 16, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(1, 1, 128, dtype=torch.bfloat16, device="cuda")
    indices = torch.zeros(1, 1, 1, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="softmax_scale"):
        if direct_binding:
            from flashinfer.experimental.union_gqa.jit import get_union_gqa_module

            get_union_gqa_module().run(
                q, k, k, indices, cu, torch.empty_like(q), scale, 1
            )
        else:
            union_gqa_prefill(q, k, k, indices, cu, softmax_scale=scale, max_seqlen_q=1)


def test_union_gqa_row_max_has_no_finite_floor() -> None:
    _require_sm100()
    seq_len = 65
    q = torch.full((seq_len, 16, 128), 1e15, dtype=torch.bfloat16, device="cuda")
    k = torch.full((seq_len, 1, 128), -1e15, dtype=torch.bfloat16, device="cuda")
    values = torch.arange(seq_len, device="cuda").clamp(max=63)
    v = values.view(seq_len, 1, 1).expand(seq_len, 1, 128).to(torch.bfloat16)
    indices = torch.zeros(seq_len, 1, 1, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    out = torch.full_like(q, torch.nan)

    # This scale makes the kernel's log2 scale exactly one in float32, so equal
    # logits cancel in the existing FMA and isolate the row-max floor.
    union_gqa_prefill(
        q,
        k,
        v,
        indices,
        cu,
        softmax_scale=math.log(2),
        max_seqlen_q=seq_len,
        out=out,
    )
    expected = (values.float() / 2).view(seq_len, 1, 1).expand_as(out)
    torch.testing.assert_close(out.float(), expected, atol=0.125, rtol=1e-2)


def test_union_gqa_row_max_first_valid_tile_after_empty_tile() -> None:
    _require_sm100()
    seq_len = 130
    q = torch.full((seq_len, 16, 128), 1e15, dtype=torch.bfloat16, device="cuda")
    k = torch.full((seq_len, 1, 128), -1e15, dtype=torch.bfloat16, device="cuda")
    values = torch.arange(seq_len, device="cuda").clamp(max=127)
    v = values.view(seq_len, 1, 1).expand(seq_len, 1, 128).to(torch.bfloat16)
    indices = torch.zeros(seq_len, 1, 1, dtype=torch.int32, device="cuda")
    indices[128, 0, 0] = 2
    indices[129, 0, 0] = 1
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    out = torch.full_like(q, torch.nan)

    union_gqa_prefill(
        q,
        k,
        v,
        indices,
        cu,
        softmax_scale=math.log(2),
        max_seqlen_q=seq_len,
        out=out,
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[128].float(), v[128].float().expand_as(out[128]))
    expected_129 = v[64:128].float().mean(dim=0).expand_as(out[129])
    torch.testing.assert_close(out[129].float(), expected_129, atol=0.125, rtol=1e-2)


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("union_gqa correctness requires a CUDA GPU")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("union_gqa correctness requires exact SM100")
    from flashinfer.jit.cpp_ext import is_cuda_version_at_least

    if not is_cuda_version_at_least("12.8"):
        pytest.skip("union_gqa correctness requires CUDA 12.8 or newer")


def _run_persistent_case_in_child(test_name: str) -> bool:
    """Bound a synchronization hang without relying on pytest-timeout."""

    child_device = os.environ.get("FLASHINFER_UNION_GQA_TEST_DEVICE")
    if child_device is not None:
        torch.cuda.set_device(int(child_device))
        return False

    _require_sm100()
    env = os.environ.copy()
    env["FLASHINFER_UNION_GQA_TEST_DEVICE"] = str(torch.cuda.current_device())
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", f"{__file__}::{test_name}"],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"{test_name} timed out: {exc}")
    assert result.returncode == 0, result.stdout + result.stderr
    return True


def _build_indices(
    lengths: tuple[int, ...],
    num_kv_heads: int,
    topk_blocks: int,
    pattern: str,
) -> torch.Tensor:
    rows: list[list[list[int]]] = []
    for seq_len in lengths:
        num_blocks = max(1, math.ceil(seq_len / BLOCK_SIZE))
        for q_local in range(seq_len):
            token_rows = []
            for kv_head in range(num_kv_heads):
                if pattern == "same":
                    chosen = list(range(min(num_blocks, topk_blocks)))
                elif pattern == "unordered":
                    chosen = list(reversed(range(min(num_blocks, topk_blocks))))
                elif pattern == "sliding":
                    count = min(num_blocks, topk_blocks)
                    start = (q_local // 8 + kv_head) % num_blocks
                    chosen = [(start + j) % num_blocks for j in range(count)]
                elif pattern == "future":
                    count = min(num_blocks, topk_blocks)
                    chosen = list(range(num_blocks - count, num_blocks))
                elif pattern == "mixed_order":
                    count = min(num_blocks, topk_blocks)
                    chosen = list(range(count))
                    if q_local % 2:
                        chosen.reverse()
                elif pattern == "none":
                    chosen = []
                else:  # pragma: no cover
                    raise ValueError(f"unknown pattern: {pattern}")
                token_rows.append(chosen + [-1] * (topk_blocks - len(chosen)))
            rows.append(token_rows)
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    total_q, num_qo_heads, _ = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_qo_heads // num_kv_heads
    mask = torch.zeros(
        (num_kv_heads, total_q, total_q), dtype=torch.bool, device=q.device
    )
    cu = cu_seqlens_q.tolist()
    for batch_idx in range(len(cu) - 1):
        seq_start, seq_end = cu[batch_idx], cu[batch_idx + 1]
        seq_len = seq_end - seq_start
        for q_local in range(seq_len):
            q_global = seq_start + q_local
            for kv_head in range(num_kv_heads):
                for block_id in block_indices[q_global, kv_head].tolist():
                    if block_id < 0:
                        continue
                    begin = block_id * BLOCK_SIZE
                    end = min(begin + BLOCK_SIZE, seq_len, q_local + 1)
                    if begin < end:
                        mask[kv_head, q_global, seq_start + begin : seq_start + end] = (
                            True
                        )

    k_hq = k.repeat_interleave(group_size, dim=1).float()
    v_hq = v.repeat_interleave(group_size, dim=1).float()
    mask_hq = mask.repeat_interleave(group_size, dim=0).transpose(0, 1)
    scores = torch.einsum("qhd,khd->qhk", q.float(), k_hq) * softmax_scale
    scores = scores.masked_fill(~mask_hq, -torch.inf)
    has_key = mask_hq.any(dim=-1, keepdim=True)
    scores = torch.where(has_key, scores, torch.zeros_like(scores))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("qhk,khd->qhd", probs, v_hq)
    return torch.where(has_key, out, torch.zeros_like(out)).to(q.dtype)


def _reference_single_kv_head_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    query_positions: list[int],
    softmax_scale: float,
) -> torch.Tensor:
    rows = []
    seq_len = q.shape[0]
    for query_pos in query_positions:
        key_positions = []
        for block_id in block_indices[query_pos, 0].tolist():
            if block_id < 0:
                continue
            begin = block_id * BLOCK_SIZE
            key_positions.extend(
                range(begin, min(begin + BLOCK_SIZE, query_pos + 1, seq_len))
            )
        if not key_positions:
            rows.append(torch.zeros_like(q[query_pos]))
            continue
        key_pos = torch.tensor(key_positions, dtype=torch.int64, device="cuda")
        selected_k = k[key_pos, 0].float()
        selected_v = v[key_pos, 0].float()
        scores = (
            torch.einsum("hd,kd->hk", q[query_pos].float(), selected_k) * softmax_scale
        )
        rows.append((torch.softmax(scores, dim=-1) @ selected_v).to(q.dtype))
    return torch.stack(rows)


def _build_token_similarity_indices(
    seq_len: int,
    num_kv_heads: int,
    topk_blocks: int,
    similarity_ratio: float,
    device: str,
) -> tuple[torch.Tensor, int, float]:
    """Build a 16-token pattern with a controlled adjacent-token overlap."""

    if not 0.0 <= similarity_ratio <= 1.0:
        raise ValueError("similarity_ratio must be in [0, 1]")
    stride = round(topk_blocks * (1.0 - similarity_ratio))
    query_in_cta = torch.arange(seq_len, device=device) % 16
    block_offset = torch.arange(topk_blocks, device=device)
    rows = query_in_cta[:, None] * stride + block_offset[None, :]
    indices = rows[:, None, :].expand(seq_len, num_kv_heads, topk_blocks)
    effective_ratio = 1.0 - stride / topk_blocks
    return indices.to(torch.int32).contiguous(), stride, effective_ratio


@pytest.mark.parametrize(
    ("lengths", "num_kv_heads", "topk_blocks", "pattern", "scale"),
    [
        ((1,), 1, 1, "same", 128**-0.5),
        ((33,), 1, 1, "none", 128**-0.5),
        ((65,), 1, 2, "same", 0.2),
        ((17, 70), 1, 3, "unordered", 128**-0.5),
        ((64, 65, 129), 1, 3, "sliding", 128**-0.5),
        ((193,), 1, 4, "mixed_order", 128**-0.5),
        ((13, 70, 131, 257), 4, 4, "future", 128**-0.5),
        ((1025,), 1, 16, "sliding", 128**-0.5),
    ],
)
def test_union_gqa_prefill_matches_reference(
    lengths: tuple[int, ...],
    num_kv_heads: int,
    topk_blocks: int,
    pattern: str,
    scale: float,
) -> None:
    _require_sm100()
    torch.manual_seed(7)
    total_q = sum(lengths)
    num_qo_heads = 16 * num_kv_heads
    q = (
        torch.randn(total_q, num_qo_heads, 128, dtype=torch.bfloat16, device="cuda")
        * 0.3
    )
    k = (
        torch.randn(total_q, num_kv_heads, 128, dtype=torch.bfloat16, device="cuda")
        * 0.3
    )
    v = torch.randn_like(k) * 0.3
    indices = _build_indices(lengths, num_kv_heads, topk_blocks, pattern)
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    actual = union_gqa_prefill(
        q,
        k,
        v,
        indices,
        cu_seqlens_q,
        softmax_scale=scale,
        max_seqlen_q=max(lengths),
    )
    expected = _reference(q, k, v, indices, cu_seqlens_q, scale)
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)


def test_union_gqa_max_union_capacity_matches_reference() -> None:
    _require_sm100()
    torch.manual_seed(11)
    seq_len = 8192
    q = torch.randn(seq_len, 16, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    k = torch.randn(seq_len, 1, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    v = torch.randn_like(k) * 0.3
    indices = torch.full((seq_len, 1, 16), -1, dtype=torch.int32, device="cuda")
    for query_in_stage in range(8):
        block_start = query_in_stage * 16
        indices[seq_len - 8 + query_in_stage, 0] = torch.arange(
            block_start, block_start + 16, dtype=torch.int32, device="cuda"
        )
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    scale = 128**-0.5

    actual = torch.full_like(q, torch.nan)
    union_gqa_prefill(q, k, v, indices, cu, max_seqlen_q=seq_len, out=actual)
    query_positions = list(range(seq_len - 8, seq_len))
    expected = _reference_single_kv_head_rows(q, k, v, indices, query_positions, scale)

    torch.testing.assert_close(actual[-8:], expected, atol=5e-2, rtol=2e-2)


def test_union_gqa_persistent_cta_phase_reuse_matches_reference() -> None:
    if _run_persistent_case_in_child(
        "test_union_gqa_persistent_cta_phase_reuse_matches_reference"
    ):
        return
    _require_sm100()
    torch.manual_seed(23)
    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    seq_len = 16 * (4 * sm_count + 1)
    q = torch.randn(seq_len, 16, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    k = torch.randn(seq_len, 1, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    v = torch.randn_like(k) * 0.3
    indices = torch.zeros(seq_len, 1, 1, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    actual = torch.full_like(q, torch.nan)
    union_gqa_prefill(q, k, v, indices, cu, max_seqlen_q=seq_len, out=actual)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    query_positions = [
        0,
        15,
        16 * sm_count,
        32 * sm_count,
        48 * sm_count,
        64 * sm_count,
        seq_len - 1,
    ]
    expected = _reference_single_kv_head_rows(
        q, k, v, indices, query_positions, 128**-0.5
    )
    torch.testing.assert_close(actual[query_positions], expected, atol=5e-2, rtol=2e-2)


def test_union_gqa_persistent_cta_crosses_kv_head_and_partial_stage() -> None:
    if _run_persistent_case_in_child(
        "test_union_gqa_persistent_cta_crosses_kv_head_and_partial_stage"
    ):
        return
    _require_sm100()
    torch.manual_seed(31)
    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    seq_len = 16 * (2 * sm_count + 1) - 3
    q = torch.randn(seq_len, 32, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    k = torch.randn(seq_len, 2, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    v = torch.randn_like(k) * 0.3
    indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    indices = indices.view(1, 2, 1).expand(seq_len, 2, 1).contiguous()
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    out = torch.full_like(q, torch.nan)
    union_gqa_prefill(q, k, v, indices, cu, max_seqlen_q=seq_len, out=out)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    for pos in (
        0,
        16,
        63,
        64,
        65,
        16 * (sm_count - 1),
        16 * (2 * sm_count - 1),
        seq_len - 1,
    ):
        for kv_head in range(2):
            start = kv_head * BLOCK_SIZE
            count = min(BLOCK_SIZE, max(0, pos - start + 1))
            if count == 0:
                expected = torch.zeros_like(out[pos, kv_head * 16 : (kv_head + 1) * 16])
            else:
                scores = torch.einsum(
                    "hd,kd->hk",
                    q[pos, kv_head * 16 : (kv_head + 1) * 16].float(),
                    k[start : start + count, kv_head].float(),
                ) * (128**-0.5)
                expected = (
                    torch.softmax(scores, dim=-1)
                    @ v[start : start + count, kv_head].float()
                ).to(torch.bfloat16)
            torch.testing.assert_close(
                out[pos, kv_head * 16 : (kv_head + 1) * 16],
                expected,
                atol=5e-2,
                rtol=2e-2,
            )


def test_union_gqa_persistent_cta_skips_invalid_work() -> None:
    if _run_persistent_case_in_child(
        "test_union_gqa_persistent_cta_skips_invalid_work"
    ):
        return
    _require_sm100()
    torch.manual_seed(37)
    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    num_seq = 8 * sm_count + 1
    seq_stride = sm_count // math.gcd(sm_count, 2)
    lengths = [
        17 if seq_idx % seq_stride == 0 and (seq_idx // seq_stride) % 3 == 0 else 1
        for seq_idx in range(num_seq)
    ]
    offsets = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    total_q = offsets[-1]
    q = torch.randn(total_q, 16, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    k = torch.randn(total_q, 1, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    v = torch.randn_like(k) * 0.3
    indices = torch.zeros(total_q, 1, 1, dtype=torch.int32, device="cuda")
    cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    # CTA 1 visits q-tile 1 of sequences 0, seq_stride, 2*seq_stride, ... .
    # Among those visits, every third work is valid.
    out = torch.full_like(q, torch.nan)
    union_gqa_prefill(q, k, v, indices, cu, max_seqlen_q=17, out=out)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    singleton_positions = torch.tensor(
        [offsets[i] for i, length in enumerate(lengths) if length == 1],
        dtype=torch.int64,
        device="cuda",
    )
    torch.testing.assert_close(
        out[singleton_positions],
        v[singleton_positions, 0].unsqueeze(1).expand(-1, 16, -1),
        atol=5e-2,
        rtol=2e-2,
    )
    for seq_idx in (
        0,
        1,
        seq_stride - 1,
        seq_stride,
        2 * seq_stride,
        3 * seq_stride,
        num_seq - 1,
    ):
        start = offsets[seq_idx]
        for local_pos in {0, lengths[seq_idx] - 1}:
            pos = start + local_pos
            scores = torch.einsum(
                "hd,kd->hk",
                q[pos].float(),
                k[start : pos + 1, 0].float(),
            ) * (128**-0.5)
            expected = torch.softmax(scores, dim=-1) @ v[start : pos + 1, 0].float()
            torch.testing.assert_close(
                out[pos], expected.to(torch.bfloat16), atol=5e-2, rtol=2e-2
            )


def test_union_gqa_metadata_wait_survives_late_warp(tmp_path: Path) -> None:
    """Force the metadata warp to wait until two epilogues have completed."""

    _require_sm100()
    source_dir = (
        Path(__file__).resolve().parents[3]
        / "flashinfer/experimental/union_gqa/csrc/sm100a"
    )
    script = textwrap.dedent(
        """
        from pathlib import Path
        import shutil
        import sys

        import torch
        import flashinfer.experimental.union_gqa.jit as union_jit
        from flashinfer import union_gqa_prefill

        source_dir, temp_dir = map(Path, sys.argv[1:3])
        torch.cuda.set_device(int(sys.argv[3]))
        copy_root = temp_dir / "csrc"
        target_dir = copy_root / "sm100a"
        shutil.copytree(source_dir, target_dir)
        body = target_dir / "union_gqa_prefill.cu"
        source = body.read_text()
        def replace_exact(source, before, after):
            assert source.count(before) == 1, (before, source.count(before))
            return source.replace(before, after)

        source = replace_exact(
            source,
            "  transac_bar_t bar_o_smem_free[Q_STAGE];",
            "  transac_bar_t bar_o_smem_free[Q_STAGE];\\n"
            "  int review_epilogue_gate;",
        )
        source = replace_exact(
            source,
            "  if (threadIdx.x == 0) {\\n    plan->bar_metadata_ready[0].init(1);",
            "  if (threadIdx.x == 0) {\\n    plan->review_epilogue_gate = 0;\\n"
            "    plan->bar_metadata_ready[0].init(1);",
        )
        marker = (
            "          for (int qstage = 0; qstage < Q_STAGE; ++qstage) {\\n"
            "            plan->bar_o_smem_free[qstage].arrive();\\n"
            "          }\\n"
            "        }\\n"
            "      } else if (warp_idx == 14"
        )
        source = replace_exact(
            source,
            marker,
            marker.replace(
                "        }\\n      } else if (warp_idx == 14",
                "          if (work_iter == 1 || work_iter == 3) {\\n"
                "            atomicExch(&plan->review_epilogue_gate, work_iter + 1);\\n"
                "          }\\n"
                "        }\\n      } else if (warp_idx == 14",
            ),
        )
        marker = (
            "        generate_union_metadata_warp(plan, params, work.offset_q, work.cur_seq_len, work.q_pos_base,\\n"
            "                                     work.h_kv_idx, input_kv_blocks, meta_phase);\\n"
            "        __syncwarp();\\n"
            "      }\\n"
            "      ++work_iter;"
        )
        # Wait on an independent signal after work 1/3 epilogues finish, then
        # exercise the metadata barrier wait for work 2/4.
        source = replace_exact(
            source,
            marker,
            marker.replace(
                "        __syncwarp();",
                "        __syncwarp();\\n"
                "        if (work_iter == 1 || work_iter == 3) {\\n"
                "          while (atomicAdd(&plan->review_epilogue_gate, 0) < work_iter + 1) {\\n"
                "            __nanosleep(100);\\n"
                "          }\\n"
                "        }",
            ),
        )
        body.write_text(source)
        union_jit._get_union_gqa_csrc_dir = lambda: copy_root
        union_jit._UNION_GQA_URI = "union_gqa_prefill_sm100a_late_metadata_test"
        union_jit.gen_union_gqa_module.cache_clear()
        union_jit.load_union_gqa_module.cache_clear()

        torch.manual_seed(29)
        sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        seq_len = 16 * (4 * sm_count + 1) - 3
        q = torch.randn(seq_len, 16, 128, dtype=torch.bfloat16, device="cuda") * 0.3
        k = torch.randn(seq_len, 1, 128, dtype=torch.bfloat16, device="cuda") * 0.3
        v = torch.randn_like(k) * 0.3
        indices = torch.zeros(seq_len, 1, 1, dtype=torch.int32, device="cuda")
        cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
        out = torch.full_like(q, torch.nan)
        union_gqa_prefill(q, k, v, indices, cu, max_seqlen_q=seq_len, out=out)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out).all())
        for pos in (0, 15, 16 * sm_count, 32 * sm_count, 48 * sm_count,
                    64 * sm_count, seq_len - 1):
            count = min(pos + 1, 64)
            scores = torch.einsum(
                "hd,kd->hk", q[pos].float(), k[:count, 0].float()
            ) * (128**-0.5)
            expected = torch.softmax(scores, dim=-1) @ v[:count, 0].float()
            torch.testing.assert_close(
                out[pos], expected.to(torch.bfloat16), atol=5e-2, rtol=2e-2
            )
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[3])
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    env["FLASHINFER_WORKSPACE_BASE"] = str(tmp_path / "workspace")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                "-c",
                script,
                str(source_dir),
                str(tmp_path),
                str(torch.cuda.current_device()),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"late metadata warp reproduction timed out: {exc}")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.solo
def test_union_gqa_partial_epilogue_uses_64_bit_output_offset() -> None:
    """Exercise the first output token whose flat offset exceeds int32."""

    _require_sm100()
    seq_len = 131073
    num_kv_heads = 8
    num_qo_heads = 16 * num_kv_heads
    q = torch.zeros(seq_len, num_qo_heads, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(seq_len, num_kv_heads, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros_like(k)
    indices = torch.full(
        (seq_len, num_kv_heads, 1), -1, dtype=torch.int32, device="cuda"
    )
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    out = torch.full_like(q, torch.nan)

    union_gqa_prefill(q, k, v, indices, cu, max_seqlen_q=seq_len, out=out)
    torch.cuda.synchronize()

    torch.testing.assert_close(out[-1], torch.zeros_like(out[-1]))


def test_union_gqa_accepts_cpu_cu_seqlens() -> None:
    _require_sm100()
    torch.manual_seed(13)
    seq_len = 65
    q = torch.randn(seq_len, 16, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    k = torch.randn(seq_len, 1, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    v = torch.randn_like(k) * 0.3
    indices = _build_indices((seq_len,), 1, 2, "same")
    cu_cpu = torch.tensor([0, seq_len], dtype=torch.int32)
    cu_cuda = cu_cpu.cuda()

    expected = union_gqa_prefill(q, k, v, indices, cu_cuda, max_seqlen_q=seq_len)
    actual = union_gqa_prefill(q, k, v, indices, cu_cpu, max_seqlen_q=seq_len)

    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("offsets", ([0, 2], [1, 1], [0, -1, 1], [0, 1, 0]))
def test_union_gqa_rejects_invalid_cpu_cu_seqlens(offsets: list[int]) -> None:
    _require_sm100()
    q = torch.zeros(1, 16, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(1, 1, 128, dtype=torch.bfloat16, device="cuda")
    indices = torch.zeros(1, 1, 1, dtype=torch.int32, device="cuda")
    cu = torch.tensor(offsets, dtype=torch.int32)

    with pytest.raises(ValueError, match="cu_seqlens_q must be nondecreasing"):
        union_gqa_prefill(q, k, k, indices, cu, max_seqlen_q=1)


def test_union_gqa_rejects_partially_overlapping_output() -> None:
    _require_sm100()
    q_storage = torch.zeros(16 * 128 + 1, dtype=torch.bfloat16, device="cuda")
    q = q_storage[:-1].view(1, 16, 128)
    out = q_storage[1:].view(1, 16, 128)
    k = torch.zeros(1, 1, 128, dtype=torch.bfloat16, device="cuda")
    indices = torch.zeros(1, 1, 1, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="out must not overlap"):
        union_gqa_prefill(q, k, k, indices, cu, max_seqlen_q=1, out=out)


def test_union_gqa_rejects_misaligned_tma_tensors() -> None:
    child_device = os.environ.get("FLASHINFER_UNION_GQA_ALIGNMENT_CHILD")
    if child_device is None:
        _require_sm100()
        env = os.environ.copy()
        env["FLASHINFER_UNION_GQA_ALIGNMENT_CHILD"] = str(torch.cuda.current_device())
        # A CUTLASS assertion in the unfixed binding must not terminate pytest.
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_union_gqa_rejects_misaligned_tma_tensors",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return

    torch.cuda.set_device(int(child_device))
    _require_sm100()

    def misaligned_bf16(shape: tuple[int, ...]) -> torch.Tensor:
        storage = torch.zeros(math.prod(shape) + 1, dtype=torch.bfloat16, device="cuda")
        tensor = storage[1:].view(shape)
        assert tensor.is_contiguous()
        assert tensor.data_ptr() % 16 == 2
        return tensor

    indices = torch.zeros(1, 1, 1, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    for direct_ffi in (False, True):
        if direct_ffi:
            from flashinfer.experimental.union_gqa.jit import get_union_gqa_module

            module = get_union_gqa_module()
        for name in ("q", "k", "v", "out"):
            tensors = {
                "q": torch.zeros(1, 16, 128, dtype=torch.bfloat16, device="cuda"),
                "k": torch.zeros(1, 1, 128, dtype=torch.bfloat16, device="cuda"),
                "v": torch.zeros(1, 1, 128, dtype=torch.bfloat16, device="cuda"),
                "out": torch.empty(1, 16, 128, dtype=torch.bfloat16, device="cuda"),
            }
            tensors[name] = misaligned_bf16(tuple(tensors[name].shape))
            with pytest.raises(ValueError, match=f"{name} must be 16-byte aligned"):
                if direct_ffi:
                    module.run(
                        tensors["q"],
                        tensors["k"],
                        tensors["v"],
                        indices,
                        cu,
                        tensors["out"],
                        128**-0.5,
                        1,
                    )
                else:
                    union_gqa_prefill(
                        tensors["q"],
                        tensors["k"],
                        tensors["v"],
                        indices,
                        cu,
                        max_seqlen_q=1,
                        out=tensors["out"],
                    )


@pytest.mark.parametrize("aliased_input", ["block_indices", "cu_seqlens_q"])
@pytest.mark.parametrize("direct_ffi", [False, True])
def test_union_gqa_rejects_output_overlapping_index_inputs(
    aliased_input: str, direct_ffi: bool
) -> None:
    _require_sm100()
    q = torch.zeros(1, 16, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(1, 1, 128, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros_like(q)
    indices = torch.zeros(1, 1, 1, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    out_as_int32 = out.flatten().view(torch.int32)
    if aliased_input == "block_indices":
        indices = out_as_int32[:1].view(1, 1, 1)
    else:
        cu = out_as_int32[:2]
        cu.copy_(torch.tensor([0, 1], dtype=torch.int32, device="cuda"))

    with pytest.raises(ValueError, match="out must not overlap"):
        if direct_ffi:
            from flashinfer.experimental.union_gqa.jit import get_union_gqa_module

            get_union_gqa_module().run(q, k, k, indices, cu, out, 128**-0.5, 1)
        else:
            union_gqa_prefill(q, k, k, indices, cu, max_seqlen_q=1, out=out)


def test_union_gqa_rejects_underestimated_cpu_max_seqlen() -> None:
    _require_sm100()
    q = torch.zeros(2, 16, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(2, 1, 128, dtype=torch.bfloat16, device="cuda")
    indices = torch.zeros(2, 1, 1, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 2], dtype=torch.int32)

    with pytest.raises(ValueError, match="max_seqlen_q is smaller"):
        union_gqa_prefill(q, k, k, indices, cu, max_seqlen_q=1)


def test_union_gqa_invalid_cuda_cu_seqlens_fails_before_output_store() -> None:
    _require_sm100()
    # A device trap poisons the CUDA context, so exercise it in another process.
    script = textwrap.dedent(
        """
        import sys
        import torch
        from flashinfer import union_gqa_prefill

        torch.cuda.set_device(int(sys.argv[1]))
        q = torch.zeros(1, 16, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(1, 1, 128, dtype=torch.bfloat16, device="cuda")
        indices = torch.zeros(1, 1, 1, dtype=torch.int32, device="cuda")
        cu = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
        try:
            union_gqa_prefill(q, k, k, indices, cu, max_seqlen_q=1)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            assert "CUDA error" in str(exc), str(exc)
        else:
            raise AssertionError("invalid CUDA cu_seqlens_q was accepted")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(torch.cuda.current_device())],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("similarity_ratio", [1.0, 0.875, 0.75, 0.5, 0.0])
def test_token_similarity_ratio_pattern(similarity_ratio: float) -> None:
    topk_blocks = 8
    indices, stride, effective_ratio = _build_token_similarity_indices(
        32, 1, topk_blocks, similarity_ratio, "cpu"
    )

    assert effective_ratio == similarity_ratio
    assert torch.unique(indices[:16, 0]).numel() == topk_blocks + 15 * stride
    torch.testing.assert_close(indices[:16], indices[16:])
    expected_shared_blocks = topk_blocks - stride
    for query_in_cta in range(15):
        lhs = set(indices[query_in_cta, 0].tolist())
        rhs = set(indices[query_in_cta + 1, 0].tolist())
        assert len(lhs & rhs) == expected_shared_blocks


@pytest.mark.parametrize("similarity_ratio", [1.0, 0.875, 0.75, 0.5, 0.0])
def test_union_gqa_token_similarity_ratio_matches_reference(
    similarity_ratio: float,
) -> None:
    """Check output across adjacent-token block-overlap ratios."""

    _require_sm100()
    seq_len = 8192
    topk_blocks = 8
    indices, _, effective_ratio = _build_token_similarity_indices(
        seq_len, 1, topk_blocks, similarity_ratio, "cuda"
    )
    assert effective_ratio == similarity_ratio
    q = torch.randn(seq_len, 16, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    k = torch.randn(seq_len, 1, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    v = torch.randn_like(k) * 0.3
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    out = union_gqa_prefill(q, k, v, indices, cu, max_seqlen_q=seq_len)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    query_positions = list(range(seq_len - 16, seq_len))
    expected = _reference_single_kv_head_rows(
        q, k, v, indices, query_positions, 128**-0.5
    )
    torch.testing.assert_close(out[query_positions], expected, atol=5e-2, rtol=2e-2)


def test_union_gqa_cuda_graph_replay() -> None:
    _require_sm100()
    seq_len = 129
    q = torch.randn(seq_len, 16, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    k = torch.randn(seq_len, 1, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    v = torch.randn_like(k) * 0.3
    indices = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    indices = indices.view(1, 1, 3).expand(seq_len, 1, 3).contiguous()
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    # Warm the JIT module and CUDA function attributes outside capture.
    expected = union_gqa_prefill(q, k, v, indices, cu, max_seqlen_q=seq_len).clone()
    torch.cuda.synchronize()
    out = torch.empty_like(q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = union_gqa_prefill(
            q,
            k,
            v,
            indices,
            cu,
            max_seqlen_q=seq_len,
            out=out,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert captured.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, expected, atol=5e-2, rtol=2e-2)
