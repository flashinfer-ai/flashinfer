"""Deferred routing-counter initialisation of the SM120 static MoE kernel.

The static kernel used to clear its routing counters (row counts, global->local map, row allocators, chunk map, work
claim counter, active-expert count) in every launch's prologue behind a resident-grid barrier.  With deferred
initialisation the finalize kernel restores the clean state after the launch (only the touched entries) and publishes a
marker in ``route_state[0]``; the next launch's prologue finds the marker and starts routing at once.  A workspace
without the marker (fresh, or cleared by hand) takes the full clear.  The tiny-decode micro kernel keeps private copies
of the two counters it shares with the static kernel, so it can never disturb the clean state.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available

from .utils import check_accuracy, compute_reference_moe_fp4, create_moe_tensors

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_cute_dsl_available()),
    reason="CUDA + CuTe-DSL required",
)

HIDDEN, INTERMEDIATE, EXPERTS, TOPK = 256, 320, 64, 2
CAPACITY = 1024


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


sm120_only = pytest.mark.skipif(not _is_sm120(), reason="SM120 static kernel only")


def _wrapper():
    from flashinfer import B12xMoEWrapper

    return B12xMoEWrapper(
        num_experts=EXPERTS,
        top_k=TOPK,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        use_cuda_graph=True,
        max_num_tokens=CAPACITY,
    )


def _tensors(num_tokens: int, seed: int):
    return create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=EXPERTS,
        num_local_experts=EXPERTS,
        top_k=TOPK,
        seed=seed,
        interleave_gated_weights=False,
        use_nontrivial_alphas=False,
    )


def _kwargs(t):
    return {
        "x": t["x_bf16"],
        "w1_weight": t["w1_weight"],
        "w1_weight_sf": t["w1_weight_sf"],
        "w1_alpha": t["w1_alpha"],
        "fc2_input_scale": t["fc2_input_scale"],
        "w2_weight": t["w2_weight"],
        "w2_weight_sf": t["w2_weight_sf"],
        "w2_alpha": t["w2_alpha"],
        "token_selected_experts": t["token_selected_experts"],
        "token_final_scales": t["token_final_scales"],
    }


def _reference(t, num_tokens: int):
    return compute_reference_moe_fp4(
        hidden_states=t["x_bf16"].float().cuda(),
        gemm1_weights=t["w1_weight_bf16"].float().cuda(),
        gemm2_weights=t["w2_weight_bf16"].float().cuda(),
        token_selected_experts=t["token_selected_experts"],
        token_final_scales=t["token_final_scales"],
        num_tokens=num_tokens,
        num_experts=EXPERTS,
        top_k=TOPK,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        fc2_input_scale=t["fc2_input_scale"],
    )


def _clean_marker() -> int:
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_static_kernel import (
        _ROUTE_STATE_CLEAN,
    )

    return int(_ROUTE_STATE_CLEAN)


def _assert_counters_clean(ws, *, marker: bool):
    """The exact state the static prologue's full clear produces (marker aside)."""
    torch.cuda.synchronize()
    weight_e = int(ws.global_to_local_expert.shape[0])
    scratch = ws.virt_route_scratch
    claim_slot = int(scratch.numel()) - 8
    assert int(ws.active_expert_count.item()) == 0
    assert int(ws.row_counts.abs().sum().item()) == 0
    assert int(scratch[:weight_e].abs().sum().item()) == 0, "row allocators"
    assert bool((scratch[weight_e:claim_slot] == -1).all().item()), "chunk map"
    assert bool((scratch[claim_slot:] == 0).all().item()), "work-claim counter + pad"
    assert bool((ws.global_to_local_expert == -1).all().item())
    assert int(ws.route_state[1:].abs().sum().item()) == 0, "reserved slots stay zero"
    assert (int(ws.route_state[0].item()) == _clean_marker()) is marker


def _run_static(moe, num_tokens: int, seed: int):
    t = _tensors(num_tokens, seed)
    out = moe.run(**_kwargs(t))
    torch.cuda.synchronize()
    passed, pct, atol = check_accuracy(out, _reference(t, num_tokens))
    assert passed, (num_tokens, seed, pct, atol)
    return out


@sm120_only
class TestDeferredInit:
    def test_cache_key_and_artifact_name_carry_the_mode(self):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

        common = dict(
            activation_precision="fp4",
            quant_mode="nvfp4",
            state_E=EXPERTS,
            weight_E=EXPERTS,
            m=64,
            k=HIDDEN,
            n=384,
            weight_n=INTERMEDIATE,
            route_rows=CAPACITY * TOPK,
            num_topk=TOPK,
            max_rows=32,
            mac=110,
            mma_tiler_mn=(64, 128),
            topk_ids_dtype=torch.int32,
            input_scales_are_reciprocal=False,
            fast_math=True,
            activation="silu",
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
            swiglu_limit=None,
        )
        on = md._static_kernel_cache_key(deferred_init=True, **common)
        off = md._static_kernel_cache_key(deferred_init=False, **common)
        import inspect

        pos = (
            list(inspect.signature(md._static_kernel_cache_key).parameters).index(
                "deferred_init"
            )
            + 1
        )  # "static" tag first
        assert on != off and on[pos] is True and off[pos] is False
        floor = md._STATIC_DEFERRED_INIT_MIN_PAIRS
        assert md._static_deferred_init(floor) is True
        assert (
            md._static_deferred_init(floor - 1) is False
        )  # tiny launches keep the prologue clear

    def test_same_workspace_launch_sequence_matches_the_reference(self):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

        moe = _wrapper()
        # varying M and routes on one workspace: deferred launches (>= 256 routed pairs) restore the
        # counters and publish the marker; smaller static launches clear in their prologue and withdraw
        # the marker; tiny-decode launches (<= 8 tokens, micro path) leave the static state untouched
        state = None
        for num_tokens, seed in (
            (200, 1),
            (16, 2),
            (64, 3),
            (200, 4),
            (8, 5),
            (300, 6),
            (4, 7),
            (150, 8),
            (200, 9),
        ):
            _run_static(moe, num_tokens, seed)
            ws = moe._static_workspace
            torch.cuda.synchronize()
            if num_tokens <= md._MICRO_MAX_TOKENS:
                assert state is not None
                assert int(ws.route_state[0].item()) == state
            elif md._static_deferred_init(num_tokens * TOPK):
                _assert_counters_clean(ws, marker=True)
                state = _clean_marker()
            else:
                assert int(ws.route_state[0].item()) == 0
                state = 0

    def test_tiny_decode_launch_between_static_launches_keeps_the_clean_state(self):
        moe = _wrapper()
        _run_static(moe, 200, 11)
        ws = moe._static_workspace
        _assert_counters_clean(ws, marker=True)
        # tiny decode (micro / direct micro path) on the same wrapper
        for num_tokens in (1, 2, 4, 8):
            _run_static(moe, num_tokens, 20 + num_tokens)
            _assert_counters_clean(ws, marker=True)
        _run_static(moe, 200, 12)
        _assert_counters_clean(ws, marker=True)

    def test_forced_full_clear_and_deferred_off_agree_bitwise(self, monkeypatch):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

        t = _tensors(200, 31)
        moe = _wrapper()
        out_deferred = moe.run(**_kwargs(t)).clone()
        ws = moe._static_workspace
        _assert_counters_clean(ws, marker=True)
        # a workspace without the marker takes the full clear and produces the same output
        ws.route_state.zero_()
        _assert_counters_clean(ws, marker=False)
        out_cleared = moe.run(**_kwargs(t)).clone()
        _assert_counters_clean(ws, marker=True)
        assert torch.equal(out_deferred, out_cleared)
        # the previous schedule (clear in every prologue) on a fresh wrapper
        monkeypatch.setattr(md, "_static_deferred_init", lambda routed_pairs: False)
        assert md._static_deferred_init(200 * TOPK) is False
        moe_off = _wrapper()
        out_off = moe_off.run(**_kwargs(t)).clone()
        torch.cuda.synchronize()
        assert int(moe_off._static_workspace.route_state[0].item()) == 0, (
            "the non-deferred kernel never publishes the marker"
        )
        assert torch.equal(out_deferred, out_off)
        assert check_accuracy(out_deferred, _reference(t, 200))[0]

    def test_cuda_graph_replay_and_alternating_graphs(self):
        moe = _wrapper()
        cases = {}
        for num_tokens, seed in ((150, 41), (300, 42)):
            t = _tensors(num_tokens, seed)
            kwargs = _kwargs(t)
            eager = moe.run(
                **kwargs
            ).clone()  # eager warm-up of this M (compile / load)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = moe.run(**kwargs)
            # the captured launch reads these input tensors on every replay: keep them alive
            cases[num_tokens] = (
                graph,
                captured,
                eager,
                _reference(t, num_tokens),
                t,
                kwargs,
            )
        ws = moe._static_workspace
        for _ in range(3):
            for num_tokens in (150, 300):
                graph, captured, eager, ref, _keep_t, _keep_kwargs = cases[num_tokens]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(captured, eager), num_tokens
                assert check_accuracy(captured, ref)[0]
                _assert_counters_clean(ws, marker=True)


@pytest.mark.parametrize("routed_pairs", [0, 1, 255, 256, 257, 1920, 81920])
def test_deferred_init_routing_density(routed_pairs):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

    assert md._static_deferred_init(routed_pairs) is (routed_pairs >= 256)
