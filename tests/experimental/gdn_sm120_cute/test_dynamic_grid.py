# SPDX-License-Identifier: Apache-2.0
"""Launch-grid changes must reuse kernels without retaining the first grid."""

import pytest
import torch

from flashinfer.utils import get_compute_capability
from .test_cute import _assert_relative, _reference


@pytest.mark.parametrize("cp", [False, True])
@pytest.mark.parametrize("consumer_v_tile", [64, 128])
@torch.inference_mode()
def test_dynamic_grid_reuses_compilation(cp, consumer_v_tile, monkeypatch):
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (12, 0):
        pytest.skip("dynamic CuTe GDN launch grids require SM120")

    from flashinfer.experimental.gdn_sm120_cute._api import (
        build_plan,
        chunk_gated_delta_rule,
    )
    from flashinfer.gdn_kernels.delta_rule_dsl import custom_compile_cache

    cache = {}
    monkeypatch.setattr(custom_compile_cache, "_in_mem_compile_cache", cache)
    torch.manual_seed(634233)
    hq, hv = 2, 6
    dtype, index_dtype = torch.bfloat16, torch.int32
    first_keys = None
    grids = set()
    # Grow and shrink token, sequence, and CP-segment counts. Keep tail handling
    # enabled throughout so V128's full_chunks specialization stays unchanged.
    for lengths in ((33,), (65, 97), (95, 33), (67,), (33,)):
        total = sum(lengths)
        q, k = [
            torch.randn(1, total, hq, 128, device="cuda", dtype=dtype) for _ in range(2)
        ]
        v = torch.randn(1, total, hv, 128, device="cuda", dtype=dtype)
        g = (
            torch.nn.functional.logsigmoid(torch.randn(1, total, hv, device="cuda"))
            / 16
        )
        g[..., -1] = 0
        beta = torch.rand_like(g)
        initial = torch.randn(len(lengths), hv, 128, 128, device="cuda")
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu = torch.tensor(offsets, device="cuda", dtype=index_dtype)
        plan = build_plan(lengths, hv, q.device, index_dtype, False)
        if cp:
            # Force multiple short CP segments independently of the production
            # heuristic, so every boundary kernel's grid changes in this test.
            segments, ranges = [], []
            for seq, (start, end) in enumerate(
                zip(offsets[:-1], offsets[1:], strict=True)
            ):
                first = len(segments)
                for token in range(start, end, 32):
                    segments.append(
                        (token, min(token + 32, end), seq, int(token + 32 >= end))
                    )
                ranges.append((first, len(segments)))
            plan.segments = torch.tensor(segments, device=q.device, dtype=index_dtype)
            plan.ranges = torch.tensor(ranges, device=q.device, dtype=index_dtype)
            plan.use_cp = True
        assert not plan.full_chunks
        grids.add((total, len(lengths), plan.segments.shape[0]))
        actual, final = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
            state_v_first=True,
            plan=plan,
            consumer_v_tile=consumer_v_tile,
        )
        torch.cuda.synchronize()
        if first_keys is None:
            first_keys = set(cache)
            expected_classes = {
                "Normalize128",
                "Prefix32",
                "Inverse32",
                "FusedForward32" if consumer_v_tile == 64 else "FusedForward128",
            }
            if cp:
                expected_classes.update({"Warmup", "PrepareH", "CorrectH"})
            assert {key[0][0][0].__name__ for key in cache} == expected_classes
            assert len(cache) == len(expected_classes)
        else:
            assert set(cache) == first_keys, f"recompiled for lengths={lengths}"

        q_ref, k_ref = [
            (
                x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)
            ).to(dtype)
            for x in (q, k)
        ]
        expected, expected_final = _reference(
            q_ref[0],
            k_ref[0],
            v[0],
            g[0].double().exp(),
            beta[0],
            initial,
            lengths,
            128**-0.5,
        )
        _assert_relative(actual[0], expected)
        _assert_relative(final, expected_final)
    assert len(grids) == 4
