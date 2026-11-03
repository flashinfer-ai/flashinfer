"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Route delivery of the Cake Hopper (SM90) sparse-decode planner, on the CPU.

``plan_sparse_decode`` ports the Cake latency model and emits one logical
route per decode geometry; ``_decode_template`` resolves it through
``route_program``, which raises ``KeyError`` for a route the export did not
deliver.  The model's own 128-key coordinate went unchecked (#7222): on 132-SM
parts the model picks three ring stages for nc16 bands the export delivered at
two stages only, so ``msa_sparse_decode_attention`` crashed for GQA groups
above 8 at those batch sizes.  These checks hold every route the planner emits
over the serving grid to the delivered set, pin the three crashing coordinates
to a delivered route of the same grid, and the no-route error to a
``ValueError`` that names both routes.  No CUDA device and no build; the
registry is the input, so a regenerated export is checked without an edit.
"""

import itertools

import pytest

from flashinfer.jit import cake_hopper_msa as jm
from flashinfer.msa_ops import cake_hopper_sm90 as ch

# The sweep's serving grid.  SM counts of the Hopper parts (H100 SXM / H200, H100 PCIe, H20); GQA groups
# covering both head folds at chunk counts 1, 2 and 4 (nc8 below group 9, else nc16 with
# ``ceil(group / 16)`` chunks: the plan depends on the group only through them); KV head counts and
# batch sizes whose products cover the model's item-count bands (66 items wide, repeating every
# ``2 * num_sms`` items) densely to 512 items and at a stride of at most 8 to 4096.
_NUM_SMS = (132, 114, 78)
_GROUPS = (4, 16, 32, 64)
_KV_HEADS = (1, 4, 8)
_SEQLEN_Q = (1, 2, 4, 8)
_MAX_BATCH = 512


def _page_counts(seqlen_q: int) -> list[int]:
    """``max_pages`` on both sides of the three L2-policy thresholds of the page re-selection ratio
    ``seqlen_q * 16 / max_pages`` (1.5, 1.0, 0.5), plus one page and a long context: the plan depends
    on the page count only through those comparisons."""
    pages = {1, 1024}
    for threshold in (1.5, 1.0, 0.5):
        edge = int(seqlen_q * 16 / threshold)
        pages.update({edge, edge + 1})
    return sorted(pages)


def _grid():
    for num_sms, group, kv_heads, seqlen_q in itertools.product(
        _NUM_SMS, _GROUPS, _KV_HEADS, _SEQLEN_Q
    ):
        for max_pages in _page_counts(seqlen_q):
            for batch in range(1, _MAX_BATCH + 1):
                yield dict(
                    total_q=batch * seqlen_q,
                    num_q_heads=group * kv_heads,
                    num_kv_heads=kv_heads,
                    num_sms=num_sms,
                    seqlen_q=seqlen_q,
                    max_pages=max_pages,
                )


def test_every_emitted_decode_route_is_delivered():
    """The crash of #7222 is a planner route with no program behind it."""
    undelivered: dict[str, list[dict]] = {}
    for point in _grid():
        route = ch.plan_sparse_decode(**point).route
        if route not in jm.ROUTES:
            undelivered.setdefault(route, []).append(point)
    assert not undelivered, {
        route: (len(points), points[:3]) for route, points in undelivered.items()
    }


@pytest.mark.parametrize(
    "total_q,num_q_heads,num_kv_heads,seqlen_q,max_pages,model_route,served_route",
    [
        # q-len 1, 11+ pages, 265-330 items: the issue's hand case (batch 67, 4 KV heads).
        pytest.param(
            67,
            64,
            4,
            1,
            16,
            "decode_v5:nc16:s2:st3:evict_first:main",
            "decode_v5:nc16:s2:st2:evict_first:main",
            id="s2-evict_first",
        ),
        # q-len 1, at most 10 pages, 331-396 items: rule C's lean HALF form exists for ``evict_first`` only.
        pytest.param(
            83,
            64,
            4,
            1,
            8,
            "decode_v5:nc16:s1:st3:none:main",
            "decode_v5:nc16:s1:st2:none:main",
            id="s1-none",
        ),
        # q-len 2, 65+ pages, 332-396 items: the persistent form exists for the default policy only.
        pytest.param(
            84,
            64,
            4,
            2,
            65,
            "decode_v5:nc16:s1:st3:evict_first:main",
            "decode_v5:nc16:s1:st2:evict_first:main",
            id="s1-evict_first",
        ),
    ],
)
def test_the_crashing_coordinates_are_served_by_the_two_stage_ring(
    total_q, num_q_heads, num_kv_heads, seqlen_q, max_pages, model_route, served_route
):
    """The three coordinates of #7222 on a 132-SM part: the model's three-stage pick was not delivered,
    the two-stage ring of the same split count and L2 policy is.  A regenerated export that delivers
    the model's route serves it instead, so the pin applies only while that route is absent."""
    plan = ch.plan_sparse_decode(
        total_q=total_q,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_sms=132,
        seqlen_q=seqlen_q,
        max_pages=max_pages,
    )
    assert plan.route in jm.ROUTES, plan.route
    if model_route not in jm.ROUTES:
        assert plan.route == served_route
        assert (plan.half, plan.persist) == (False, False)


def test_a_coordinate_with_no_delivered_program_raises_a_named_error(monkeypatch):
    """A ``ValueError`` naming the model's route and its two-stage form, from the planner, instead of
    ``route_program``'s ``KeyError`` from inside the launch template."""
    monkeypatch.setattr(ch, "_route_available", lambda route: False)
    ch.plan_sparse_decode.cache_clear()
    try:
        with pytest.raises(
            ValueError,
            match=r"'decode_v5:nc16:s2:st3:evict_first:main'.*'decode_v5:nc16:s2:st2:evict_first:main'",
        ):
            ch.plan_sparse_decode(
                total_q=67,
                num_q_heads=64,
                num_kv_heads=4,
                num_sms=132,
                seqlen_q=1,
                max_pages=16,
            )
    finally:
        ch.plan_sparse_decode.cache_clear()
