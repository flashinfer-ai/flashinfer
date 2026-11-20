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
"""

"""
Expert-parallelism tests for the b12x W4A16 fused MoE on SM120/SM121.

The EP contract has no collectives. Every rank receives the same bf16
activations and global top-k routes, expert_map (int32 [num_experts], -1
for non-local) maps global expert ids to rank-local weight ids, each rank
produces a zero-filled partial output, and the caller sums the partials
across the EP group.
"""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from .utils import create_b12x_moe_tensors as create_moe_tensors


def _is_sm12x_supported():
    from flashinfer.utils import is_sm120a_supported, is_sm121a_supported

    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    return is_sm120a_supported(device) or is_sm121a_supported(device)


def _cuda_13_or_newer():
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        return get_cuda_version().major >= 13
    except Exception:
        return False


cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="CuteDSL not available"
)
sm120_required = pytest.mark.skipif(
    not _is_sm12x_supported(),
    reason="Requires SM120/SM121 GPU with CUDA 12.8+",
)
cuda_13_required = pytest.mark.skipif(
    not _cuda_13_or_newer(),
    reason="b12x fused MoE requires CUDA 13 or later",
)


def _quantize_expert_weights(w_bf16: torch.Tensor, m: int, k: int):
    """FP4-quantize per-expert weights the same way as create_moe_tensors.

    Rows are quantized independently with a unit global scale, so quantizing
    a slice of experts is bit-identical to slicing the quantized global
    weights. Each fake EP rank therefore holds exactly the experts the
    full-model reference run used.
    """
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp4_quantization import fp4_quantize

    num_experts = w_bf16.shape[0]
    global_scale = torch.tensor([1.0], device=w_bf16.device, dtype=torch.float32)
    q_flat, sf_flat = fp4_quantize(
        w_bf16.reshape(num_experts * m, k),
        global_scale=global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    q = q_flat.view(num_experts, m, k // 2)
    sf = convert_sf_to_mma_layout(
        sf_flat, m=m, k=k, num_groups=num_experts, sf_vec_size=16
    )
    return q, sf


def _make_rank_shard(tensors: dict, global_ids: torch.Tensor):
    """Build one EP rank's local weights + expert_map from global tensors."""
    device = tensors["x_bf16"].device
    global_ids = global_ids.to(device=device, dtype=torch.long)
    w1_bf16 = tensors["w1_weight_bf16"].index_select(0, global_ids).contiguous()
    w2_bf16 = tensors["w2_weight_bf16"].index_select(0, global_ids).contiguous()
    w1_weight, w1_weight_sf = _quantize_expert_weights(
        w1_bf16, m=w1_bf16.shape[1], k=w1_bf16.shape[2]
    )
    w2_weight, w2_weight_sf = _quantize_expert_weights(
        w2_bf16, m=w2_bf16.shape[1], k=w2_bf16.shape[2]
    )
    num_experts = tensors["w1_weight_bf16"].shape[0]
    local_e = int(global_ids.numel())
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    expert_map[global_ids] = torch.arange(local_e, dtype=torch.int32, device=device)
    return {
        "w1_weight": w1_weight,
        "w1_weight_sf": w1_weight_sf,
        "w1_alpha": torch.ones(local_e, dtype=torch.float32, device=device),
        "w2_weight": w2_weight,
        "w2_weight_sf": w2_weight_sf,
        "w2_alpha": torch.ones(local_e, dtype=torch.float32, device=device),
        "expert_map": expert_map,
        "num_local_experts": local_e,
    }


def _run_ep_rank(tensors: dict, shard: dict, num_experts: int, top_k: int):
    from flashinfer import b12x_fused_moe

    return b12x_fused_moe(
        x=tensors["x_bf16"],
        w1_weight=shard["w1_weight"],
        w1_weight_sf=shard["w1_weight_sf"],
        w1_alpha=shard["w1_alpha"],
        w2_weight=shard["w2_weight"],
        w2_weight_sf=shard["w2_weight_sf"],
        w2_alpha=shard["w2_alpha"],
        token_selected_experts=tensors["token_selected_experts"],
        token_final_scales=tensors["token_final_scales"],
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=shard["num_local_experts"],
        expert_map=shard["expert_map"],
        quant_mode="w4a16",
    )


# =============================================================================
# expert_map contract validation (host-side, no GPU required)
# =============================================================================


def test_prepare_ep_expert_map_accepts_linear_and_round_robin_placement():
    from flashinfer.fused_moe.cute_dsl.b12x_moe import _prepare_ep_expert_map

    linear = torch.tensor([-1, -1, 0, 1, -1], dtype=torch.int32)
    round_robin = torch.tensor([0, -1, 1, -1, 2], dtype=torch.int32)
    assert _prepare_ep_expert_map(linear, num_local_experts=2, num_experts=5) is linear
    assert (
        _prepare_ep_expert_map(round_robin, num_local_experts=3, num_experts=5)
        is round_robin
    )


@pytest.mark.parametrize(
    ("values", "local_experts", "match"),
    [
        ([-1, 0, 0], 2, "exactly once"),
        ([-1, 0, 2], 2, "valid local expert ids"),
        ([-2, 0, 1], 2, "valid local expert ids"),
    ],
)
def test_prepare_ep_expert_map_rejects_unsafe_values(values, local_experts, match):
    from flashinfer.fused_moe.cute_dsl.b12x_moe import _prepare_ep_expert_map

    with pytest.raises(ValueError, match=match):
        _prepare_ep_expert_map(
            torch.tensor(values, dtype=torch.int32),
            num_local_experts=local_experts,
            num_experts=len(values),
        )


def test_prepare_ep_expert_map_rejects_bad_tensor_forms():
    from flashinfer.fused_moe.cute_dsl.b12x_moe import _prepare_ep_expert_map

    with pytest.raises(TypeError, match=r"torch\.int32"):
        _prepare_ep_expert_map(
            torch.tensor([0, -1], dtype=torch.int64),
            num_local_experts=1,
            num_experts=2,
        )
    with pytest.raises(ValueError, match="num_experts"):
        _prepare_ep_expert_map(
            torch.tensor([0, -1], dtype=torch.int32),
            num_local_experts=1,
            num_experts=4,
        )
    with pytest.raises(ValueError, match="rank-1"):
        _prepare_ep_expert_map(
            torch.tensor([[0, -1]], dtype=torch.int32),
            num_local_experts=1,
            num_experts=2,
        )


# =============================================================================
# EP guard rails
# =============================================================================


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestEPGuards:
    def _tensors(self):
        return create_moe_tensors(
            num_tokens=4,
            hidden_size=256,
            intermediate_size=512,
            num_experts=8,
            num_local_experts=8,
            top_k=2,
            seed=7,
        )

    def _call(self, tensors, **overrides):
        from flashinfer import b12x_fused_moe

        kwargs = dict(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=8,
            top_k=2,
        )
        kwargs.update(overrides)
        return b12x_fused_moe(**kwargs)

    def test_nvfp4_rejects_ep_counts(self):
        tensors = self._tensors()
        with pytest.raises(NotImplementedError, match="w4a16"):
            self._call(tensors, quant_mode="nvfp4", num_local_experts=4)

    def test_nvfp4_rejects_expert_map(self):
        tensors = self._tensors()
        expert_map = torch.arange(8, dtype=torch.int32, device="cuda")
        with pytest.raises(NotImplementedError, match="w4a16"):
            self._call(tensors, quant_mode="nvfp4", expert_map=expert_map)

    def test_w4a16_ep_requires_expert_map(self):
        tensors = self._tensors()
        with pytest.raises(ValueError, match="requires expert_map"):
            self._call(tensors, quant_mode="w4a16", num_local_experts=4)

    def test_w4a16_ep_rejects_wrong_length_map(self):
        tensors = self._tensors()
        expert_map = torch.tensor([0, 1, -1], dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="num_experts"):
            self._call(
                tensors,
                quant_mode="w4a16",
                num_local_experts=2,
                expert_map=expert_map,
            )

    def test_w4a16_ep_rejects_bad_map_forms(self):
        tensors = self._tensors()
        with pytest.raises(TypeError, match=r"torch\.int32"):
            self._call(
                tensors,
                quant_mode="w4a16",
                expert_map=torch.arange(8, dtype=torch.int64, device="cuda"),
            )
        with pytest.raises(ValueError, match="contiguous rank-1"):
            self._call(
                tensors,
                quant_mode="w4a16",
                expert_map=torch.arange(16, dtype=torch.int32, device="cuda")[::2],
            )
        with pytest.raises(ValueError, match="must be on"):
            self._call(
                tensors,
                quant_mode="w4a16",
                expert_map=torch.arange(8, dtype=torch.int32),
            )

    def test_wrapper_nvfp4_rejects_ep(self):
        from flashinfer import B12xMoEWrapper

        with pytest.raises(NotImplementedError, match="w4a16"):
            B12xMoEWrapper(
                num_experts=8,
                top_k=2,
                hidden_size=256,
                intermediate_size=512,
                num_local_experts=4,
                quant_mode="nvfp4",
            )

    def test_wrapper_w4a16_ep_requires_expert_map(self):
        from flashinfer import B12xMoEWrapper

        with pytest.raises(ValueError, match="requires expert_map"):
            B12xMoEWrapper(
                num_experts=8,
                top_k=2,
                hidden_size=256,
                intermediate_size=512,
                num_local_experts=4,
                quant_mode="w4a16",
            )

    def test_wrapper_w4a16_ep_rejects_wrong_device_map(self):
        from flashinfer import B12xMoEWrapper

        expert_map = torch.tensor([0, -1, 1, -1, -1, -1, -1, -1], dtype=torch.int32)
        with pytest.raises(ValueError, match="must be on"):
            B12xMoEWrapper(
                num_experts=8,
                top_k=2,
                hidden_size=256,
                intermediate_size=512,
                num_local_experts=2,
                expert_map=expert_map,
                quant_mode="w4a16",
            )

    def test_wrapper_w4a16_ep_validates_map_contract(self):
        from flashinfer import B12xMoEWrapper

        expert_map = torch.tensor(
            [0, 0, -1, -1, 1, -1, -1, -1], dtype=torch.int32, device="cuda"
        )
        with pytest.raises(ValueError, match="exactly once"):
            B12xMoEWrapper(
                num_experts=8,
                top_k=2,
                hidden_size=256,
                intermediate_size=512,
                num_local_experts=2,
                expert_map=expert_map,
                quant_mode="w4a16",
            )


# =============================================================================
# EP execution: partials sum to the full fused MoE output
# =============================================================================


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestEPExecution:
    @pytest.mark.parametrize("num_tokens", [4, 24, 256])
    @pytest.mark.parametrize("placement", ["contiguous", "round_robin"])
    def test_ep_rank_partials_sum_to_full_moe(self, num_tokens, placement):
        from flashinfer import b12x_fused_moe

        torch.manual_seed(20260730)
        hidden_size, intermediate_size = 256, 512
        num_experts, top_k = 7, 3
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=20260730,
        )

        expected = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            quant_mode="w4a16",
        )

        partials = []
        for rank in range(2):
            if placement == "round_robin":
                global_ids = torch.arange(rank, num_experts, 2)
            else:
                split = (num_experts + 1) // 2
                global_ids = (
                    torch.arange(0, split)
                    if rank == 0
                    else torch.arange(split, num_experts)
                )
            shard = _make_rank_shard(tensors, global_ids)
            partials.append(_run_ep_rank(tensors, shard, num_experts, top_k).clone())

        actual = partials[0] + partials[1]
        torch.cuda.synchronize()
        assert int(torch.count_nonzero(expected).item()) > 0
        assert int(torch.count_nonzero(actual).item()) > 0
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
        assert float(cosine.item()) > 0.999
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)

    def test_ep_zero_partial_when_no_local_expert_routed(self):
        torch.manual_seed(20260731)
        hidden_size, intermediate_size = 256, 512
        num_experts, top_k = 4, 2
        num_tokens = 8
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=20260731,
        )
        # This rank holds globals {0, 2}, and every route goes to {1, 3}.
        shard = _make_rank_shard(tensors, torch.tensor([0, 2]))
        tensors["token_selected_experts"] = (
            torch.tensor([[1, 3]], dtype=torch.int32, device="cuda")
            .expand(num_tokens, -1)
            .contiguous()
        )
        tensors["token_final_scales"] = torch.full(
            (num_tokens, top_k), 0.5, dtype=torch.float32, device="cuda"
        )

        from flashinfer import b12x_fused_moe

        # Pre-fill the output with a sentinel to prove the kernel zero-fills.
        output = torch.full(
            (num_tokens, hidden_size), 1e3, dtype=torch.bfloat16, device="cuda"
        )
        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=shard["w1_weight"],
            w1_weight_sf=shard["w1_weight_sf"],
            w1_alpha=shard["w1_alpha"],
            w2_weight=shard["w2_weight"],
            w2_weight_sf=shard["w2_weight_sf"],
            w2_alpha=shard["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=2,
            expert_map=shard["expert_map"],
            output=output,
            quant_mode="w4a16",
        )
        torch.cuda.synchronize()
        assert int(torch.count_nonzero(result).item()) == 0
        assert int(torch.count_nonzero(output).item()) == 0

    def test_ep_wrapper_cuda_graph_replays_changed_routes(self):
        from flashinfer import B12xMoEWrapper

        torch.manual_seed(20260732)
        hidden_size, intermediate_size = 256, 512
        num_experts, top_k = 4, 2
        num_tokens = 8
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=20260732,
        )
        shard = _make_rank_shard(tensors, torch.tensor([0, 2]))
        topk_ids = (
            torch.tensor([[1, 3]], dtype=torch.int32, device="cuda")
            .expand(num_tokens, -1)
            .contiguous()
        )
        topk_weights = torch.full(
            (num_tokens, top_k), 0.5, dtype=torch.float32, device="cuda"
        )

        wrapper = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=True,
            max_num_tokens=num_tokens,
            num_local_experts=2,
            expert_map=shard["expert_map"],
            quant_mode="w4a16",
        )

        def run():
            return wrapper.run(
                x=tensors["x_bf16"],
                w1_weight=shard["w1_weight"],
                w1_weight_sf=shard["w1_weight_sf"],
                w1_alpha=shard["w1_alpha"],
                w2_weight=shard["w2_weight"],
                w2_weight_sf=shard["w2_weight_sf"],
                w2_alpha=shard["w2_alpha"],
                token_selected_experts=topk_ids,
                token_final_scales=topk_weights,
            )

        # No local expert routed -> partial must be exactly zero.
        nonlocal_output = run()
        torch.cuda.synchronize()
        assert int(torch.count_nonzero(nonlocal_output).item()) == 0

        topk_ids.copy_(
            torch.tensor([[0, 1]], dtype=torch.int32, device="cuda").expand(
                num_tokens, -1
            )
        )
        run()  # resolve all route-pack/GEMM variants before capture
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            captured = run()

        # Replay must observe route changes staged after capture.
        topk_ids.copy_(
            torch.tensor([[2, 3]], dtype=torch.int32, device="cuda").expand(
                num_tokens, -1
            )
        )
        graph.replay()
        torch.cuda.synchronize()
        replayed = captured.clone()
        eager = run()
        torch.cuda.synchronize()
        torch.testing.assert_close(replayed, eager, rtol=0, atol=0)

        # Ground truth: the wrapper partial must match the functional API.
        reference = _run_ep_rank(
            {
                "x_bf16": tensors["x_bf16"],
                "token_selected_experts": topk_ids,
                "token_final_scales": topk_weights,
            },
            shard,
            num_experts,
            top_k,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(eager, reference, rtol=0, atol=0)

    def test_ep_permutation_map_matches_non_ep_run(self):
        """num_local == num_experts with a permutation map: permuted weights
        plus the inverse map must reproduce the non-EP run exactly."""
        from flashinfer import b12x_fused_moe

        torch.manual_seed(20260733)
        hidden_size, intermediate_size = 256, 512
        num_experts, top_k = 6, 2
        num_tokens = 16
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=20260733,
        )

        expected = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            quant_mode="w4a16",
        )

        # Local weight slot i holds global expert perm[i].
        perm = torch.tensor([3, 0, 5, 1, 4, 2])
        shard = _make_rank_shard(tensors, perm)
        actual = _run_ep_rank(tensors, shard, num_experts, top_k)
        torch.cuda.synchronize()
        assert int(torch.count_nonzero(expected).item()) > 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_wrapper_rejects_rebound_expert_map(self):
        from flashinfer import B12xMoEWrapper

        num_experts, top_k = 4, 2
        tensors = create_moe_tensors(
            num_tokens=4,
            hidden_size=256,
            intermediate_size=512,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=20260734,
        )
        shard = _make_rank_shard(tensors, torch.tensor([0, 2]))
        wrapper = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=256,
            intermediate_size=512,
            num_local_experts=2,
            expert_map=shard["expert_map"],
            quant_mode="w4a16",
        )
        wrapper.expert_map = shard["expert_map"].clone()
        with pytest.raises(RuntimeError, match="storage changed"):
            wrapper.run(
                x=tensors["x_bf16"],
                w1_weight=shard["w1_weight"],
                w1_weight_sf=shard["w1_weight_sf"],
                w1_alpha=shard["w1_alpha"],
                w2_weight=shard["w2_weight"],
                w2_weight_sf=shard["w2_weight_sf"],
                w2_alpha=shard["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
            )

    def test_wrapper_isolated_from_caller_map_mutation(self):
        from flashinfer import B12xMoEWrapper

        num_experts, top_k = 4, 2
        tensors = create_moe_tensors(
            num_tokens=4,
            hidden_size=256,
            intermediate_size=512,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=20260735,
        )
        shard = _make_rank_shard(tensors, torch.tensor([0, 2]))
        caller_map = shard["expert_map"].clone()
        wrapper = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=256,
            intermediate_size=512,
            num_local_experts=2,
            expert_map=caller_map,
            quant_mode="w4a16",
        )
        kwargs = dict(
            x=tensors["x_bf16"],
            w1_weight=shard["w1_weight"],
            w1_weight_sf=shard["w1_weight_sf"],
            w1_alpha=shard["w1_alpha"],
            w2_weight=shard["w2_weight"],
            w2_weight_sf=shard["w2_weight_sf"],
            w2_alpha=shard["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )
        expected = wrapper.run(**kwargs).clone()
        torch.cuda.synchronize()
        assert int(torch.count_nonzero(expected).item()) > 0
        caller_map.fill_(-1)
        actual = wrapper.run(**kwargs)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
