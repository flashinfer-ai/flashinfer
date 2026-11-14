"""attention.sparse_mla: decode and extend over top-k-selected tokens vs the
in-tree pure-torch reference (packed NSA MLA cache layout), through the public
plan -> bind -> run lifecycle, including -1-padded selections. Also checks that
run() returns only the output when return_lse=False and (output, lse) when
return_lse=True, in both modes.
"""

from __future__ import annotations

import torch
import pytest
from b12x.preparation import PreparedCall, PreparationSession

from b12x.attention import sparse_mla
from b12x.attention._shared.mla.reference import (
    pack_mla_kv_cache_reference,
    sparse_mla_reference,
)

from ..conftest import require_b12x

NOPE_DIM = 512
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
V_HEAD_DIM = NOPE_DIM


def _make_case(*, rows: int, heads: int, cache_tokens: int, width: int):
    torch.manual_seed(20260716)
    k_nope = (
        torch.randn(cache_tokens, NOPE_DIM, device="cuda", dtype=torch.bfloat16) / 4
    )
    k_rope = (
        torch.randn(cache_tokens, ROPE_DIM, device="cuda", dtype=torch.bfloat16) / 4
    )
    kv_cache = pack_mla_kv_cache_reference(k_nope, k_rope)
    q_all = torch.randn(rows, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16) / 4

    selected = torch.stack(
        [
            torch.randperm(cache_tokens, device="cuda")[:width].sort().values
            for _ in range(rows)
        ]
    ).to(torch.int32)
    cache_seqlens = torch.full((rows,), cache_tokens, dtype=torch.int32, device="cuda")
    active = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    return q_all, kv_cache, selected, cache_seqlens, active


def _run_public(
    q_all,
    kv_cache,
    selected,
    cache_seqlens,
    active,
    *,
    width,
    mode: str = "decode",
    return_lse: bool = False,
):
    rows, heads, _ = q_all.shape
    sm_scale = HEAD_DIM**-0.5
    declaration = sparse_mla.plan(
        sparse_mla.Caps(
            device=q_all.device,
            num_q_heads=heads,
            max_q_rows=rows,
            max_width=width,
            softmax_scale=sm_scale,
            kv_dtype=torch.uint8,
            mode=mode,
            return_lse=return_lse,
        )
    )
    owned = {}

    def prepare_call(state):
        spec = state.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=q_all.device)
        runtime = state.bind(
            scratch=scratch,
            q=q_all,
            kv_cache=kv_cache,
            selected_indices=selected,
            cache_seqlens_int32=cache_seqlens,
            nsa_cache_seqlens_int32=active,
        )
        owned["scratch"] = scratch
        state.prime(runtime, kv_cache=kv_cache)
        return PreparedCall(run=lambda: state.run(runtime, kv_cache=kv_cache))

    session = PreparationSession(device=q_all.device)
    result = session.prepare(
        (
            declaration.request(
                name="sparse-mla-test",
                prepare_call=prepare_call,
            ),
        )
    )
    plan = result.plans["sparse-mla-test"]
    binding = sparse_mla.bind(
        plan,
        scratch=owned["scratch"],
        q=q_all,
        kv_cache=kv_cache,
        selected_indices=selected,
        cache_lengths=cache_seqlens,
        selected_lengths=active,
    )
    out = sparse_mla.run(binding)
    ref = sparse_mla_reference(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=sm_scale,
        v_head_dim=V_HEAD_DIM,
    )
    return out, ref


def _assert_matches(out: torch.Tensor, ref: torch.Tensor) -> None:
    torch.cuda.synchronize()
    assert bool(torch.isfinite(out).all().item())
    assert int(torch.count_nonzero(out).item()) > 0
    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    )
    assert float(cosine.item()) > 0.99, f"cosine {float(cosine.item()):.5f}"
    torch.testing.assert_close(
        out.float(), ref.to(out.dtype).float(), rtol=5e-2, atol=5e-2
    )


def test_run_decode_matches_reference() -> None:
    require_b12x()
    q, kv, sel, lens, active = _make_case(rows=4, heads=16, cache_tokens=512, width=128)
    out, ref = _run_public(q, kv, sel, lens, active, width=128)
    _assert_matches(out, ref)


def test_run_decode_masks_padded_selection() -> None:
    require_b12x()
    width = 128
    q, kv, sel, lens, active = _make_case(
        rows=4, heads=16, cache_tokens=512, width=width
    )
    # Invalidate the back half of every row's selection: pad with -1 and
    # shrink the active counts to match; the kernel and reference must both
    # attend only to the front half.
    sel[:, width // 2 :] = -1
    active.fill_(width // 2)
    out, ref = _run_public(q, kv, sel, lens, active, width=width)
    _assert_matches(out, ref)


@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_run_return_format_follows_return_lse(mode: str) -> None:
    require_b12x()
    q, kv, sel, lens, active = _make_case(
        rows=4, heads=16, cache_tokens=1024, width=512
    )

    without_lse, _ = _run_public(
        q, kv, sel, lens, active, width=512, mode=mode, return_lse=False
    )
    assert isinstance(without_lse, torch.Tensor)

    with_lse, _ = _run_public(
        q, kv, sel, lens, active, width=512, mode=mode, return_lse=True
    )
    assert isinstance(with_lse, tuple)
    assert len(with_lse) == 2


def test_run_extend_output_unchanged_by_return_lse() -> None:
    require_b12x()
    q, kv, sel, lens, active = _make_case(
        rows=4, heads=16, cache_tokens=1024, width=512
    )

    output_only, ref = _run_public(
        q, kv, sel, lens, active, width=512, mode="extend", return_lse=False
    )
    (output_with_lse, _), _ = _run_public(
        q, kv, sel, lens, active, width=512, mode="extend", return_lse=True
    )

    torch.testing.assert_close(output_only, output_with_lse, rtol=0, atol=0)
    _assert_matches(output_only, ref)
