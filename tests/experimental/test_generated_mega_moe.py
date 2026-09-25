"""Public packed-input MoE behavior, metadata and replay tests on SM100a and SM103a."""

import importlib.util
from pathlib import Path

import pytest
import torch

from flashinfer.experimental.mega_moe_v3 import runtime as _v3_runtime
from flashinfer.experimental.source_mega_moe import runtime as _source_runtime

_HELPER = (
    Path(__file__).resolve().parents[2] / "examples/experimental/mega_moe_inputs.py"
)
_spec = importlib.util.spec_from_file_location("mega_moe_example_inputs", _HELPER)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


@pytest.fixture(autouse=True)
def supported_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for runtime in (_source_runtime, _v3_runtime):
        try:
            arch = runtime.device_arch(torch.device("cuda"))
        except RuntimeError as error:
            pytest.skip(str(error))
        if sms not in runtime.supported_num_sms(arch):
            pytest.skip(
                f"The exported {arch} schedules cover {runtime.supported_num_sms(arch)} SMs, "
                f"this device has {sms}"
            )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def output_of(plan, family):
    return plan.output if family == "source" else plan.outputs


def check_v3_metadata(plan, inputs):
    bindings = plan.stages[0].bindings
    routes = inputs["topk_idx"]
    count = torch.bincount(routes.reshape(-1), minlength=8).to(torch.int32)
    padded = ((count + 255) // 256) * 256
    offsets = padded.cumsum(0).to(torch.int32) - padded
    torch.testing.assert_close(bindings["expert_counts"], count, atol=0, rtol=0)
    torch.testing.assert_close(bindings["expert_row_offsets"], offsets, atol=0, rtol=0)
    assert bindings["total_m_tiles_out"].item() == (padded.sum() // 128).item()
    permuted = bindings["token_to_permuted"].reshape_as(routes).long()
    assert torch.unique(permuted).numel() == routes.numel()
    tokens = torch.arange(inputs["num_tokens"], device=routes.device)[
        :, None
    ].expand_as(routes)
    slots = torch.arange(inputs["top_k"], device=routes.device)[None, :].expand_as(
        routes
    )
    assert torch.all(permuted >= offsets[routes]).item()
    assert torch.all(permuted < (offsets + count)[routes]).item()
    torch.testing.assert_close(
        bindings["meta_token"][permuted].long(), tokens, atol=0, rtol=0
    )
    torch.testing.assert_close(
        bindings["meta_slot"][permuted].long(), slots, atol=0, rtol=0
    )
    torch.testing.assert_close(
        bindings["routing_weight_pool"][permuted],
        inputs["topk_weights"],
        atol=0,
        rtol=0,
    )
    expected_x = (
        inputs["x_fp8_packed"]
        .view(torch.int32)[:, None, :]
        .expand(-1, inputs["top_k"], -1)
    )
    torch.testing.assert_close(
        bindings["pool_fp8"][permuted], expected_x, atol=0, rtol=0
    )
    expected_sf = inputs["x_sf_packed"][:, None, :].expand(-1, inputs["top_k"], -1)
    torch.testing.assert_close(
        bindings["pool_sf"].view(torch.int32)[permuted], expected_sf, atol=0, rtol=0
    )


def check_source_metadata(plan, inputs):
    views = plan.views
    count = torch.bincount(inputs["topk_idx"].reshape(-1), minlength=8).cpu().tolist()
    metadata = views["token_src_metadata"].reshape(-1, 3).cpu().long()
    block = plan.config.block_m
    offset = 0
    seen = []
    for expert, amount in enumerate(count):
        actual = metadata[offset : offset + amount]
        expected = (inputs["topk_idx"].cpu() == expert).nonzero().tolist()
        assert torch.all(actual[:, 0] == 0).item()
        pairs = actual[:, 1:].tolist()
        assert sorted(pairs) == sorted(expected)
        seen.extend(pairs)
        offset += ((amount + block - 1) // block) * block
    assert len(seen) == inputs["topk_idx"].numel()
    # These arrays are explicitly cleaned by the kernel, not by plan.run().
    for name in ("expert_send_count", "expert_recv_count", "expert_recv_count_sum"):
        assert torch.count_nonzero(views[name][:8].view(torch.int64)).item() == 0
    peer = views["peer_grid_idx"].view(torch.int64)[0].item()
    ready = views["combine_ready_grid_idx"].view(torch.int64)[0].item()
    assert peer == ready and peer > 0


@pytest.mark.parametrize("family", ["source", "v3"])
@pytest.mark.parametrize("precision", ["fp4", "fp8"])
@pytest.mark.parametrize("seed", [0, 1])
def test_pipeline_reference_and_replay(family, precision, seed):
    plan, inputs, xs, shared = fixtures.make_smoke(family, precision, seed)
    expected = fixtures.reference(inputs, xs, shared)
    output = output_of(plan, family)
    atol = 1.0 if precision == "fp4" else 0.1
    check_metadata = check_source_metadata if family == "source" else check_v3_metadata
    # First launch and direct replay preserve the same workspace/counter owners.
    first = None
    for _ in range(2):
        output.fill_(float("nan"))
        plan.run()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, atol=atol, rtol=0.1)
        check_metadata(plan, inputs)
        if first is None:
            first = output.clone()
        else:
            torch.testing.assert_close(
                output,
                first,
                atol=0 if family == "source" else atol,
                rtol=0 if family == "source" else 0.1,
            )
    # Capture contains plan.run(): v3 counter resets remain in the graph;
    # the source plan relies on device cleanup and performs no host reset.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run()
    for _ in range(2):
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output,
            first,
            atol=0 if family == "source" else atol,
            rtol=0 if family == "source" else 0.1,
        )
        check_metadata(plan, inputs)


def test_source_update_inputs_reuses_workspace():
    from flashinfer.source_mega_moe import prepare_mega_moe

    plan, inputs, xs, shared = fixtures.make_smoke("source", "fp4")
    plan.run()
    original = plan.output.clone()
    pointer = plan.workspace.data_ptr()
    changed = dict(inputs)
    changed["x_fp8_packed"] = (-inputs["x_fp8_packed"].float()).to(torch.float8_e4m3fn)
    changed["topk_weights"] = inputs["topk_weights"].neg().contiguous()
    plan.update_inputs(
        changed["x_fp8_packed"],
        changed["x_sf_packed"],
        changed["topk_idx"],
        changed["topk_weights"],
    )
    plan.output.fill_(float("nan"))
    plan.run()
    torch.cuda.synchronize()
    assert plan.workspace.data_ptr() == pointer
    assert not torch.equal(plan.output, original)
    torch.testing.assert_close(
        plan.output, fixtures.reference(changed, xs, shared), atol=1.0, rtol=0.1
    )
    # A fresh independent workspace must agree exactly with the reused one.
    fresh = prepare_mega_moe(
        changed["x_fp8_packed"],
        changed["x_sf_packed"],
        changed["topk_idx"],
        changed["topk_weights"],
        weights=fixtures.source_weights(inputs, shared),
        num_experts=8,
        intermediate=256,
        routed_weight_dtype="fp4",
        num_shared_experts=1,
        activation_clamp=10.0,
        fast_math=True,
        num_sms=2,
    )
    fresh.run()
    torch.cuda.synchronize()
    torch.testing.assert_close(plan.output, fresh.output, atol=0, rtol=0)


def test_grouped_l2_repack_and_graph_replay():
    plan, a, b, sa, sb, words_a, words_b = fixtures.make_grouped_l2()
    for changed in (None, "activation", "weight"):
        if changed == "activation":
            sa.mul_(2.0)
            words_a.copy_(fixtures.scale_words(sa))
        if changed == "weight":
            sb.mul_(0.5)
            words_b.copy_(fixtures.scale_words(sb))
        expected = fixtures.grouped_reference(a, b, sa, sb)
        plan.outputs.fill_(float("nan"))
        plan.run()
        torch.cuda.synchronize()
        actual = plan.outputs.float()
        target = expected.float()
        difference = 1 - 2 * (actual * target).sum() / (
            actual.square().sum() + target.square().sum()
        )
        assert difference.item() < 1e-2
        first = plan.outputs.clone()
        plan.outputs.fill_(float("nan"))
        plan.run()
        torch.cuda.synchronize()
        torch.testing.assert_close(plan.outputs, first, atol=0, rtol=0)
        # Exact natural->group-folded scale words prove both repacks occurred.
        repack = plan.stages[0].bindings
        permuted_a = (
            words_a.reshape(4096 // 128, 4, 32, -1)
            .transpose(1, 2)
            .contiguous()
            .reshape(4096, -1)
        )
        torch.testing.assert_close(
            repack["dst_a"].view(torch.int32), permuted_a.T.contiguous(), atol=0, rtol=0
        )
        permuted_b = (
            words_b.reshape(4, 7168 // 128, 4, 32, -1)
            .transpose(2, 3)
            .contiguous()
            .reshape(4, 7168, -1)
        )
        expected_b = permuted_b.permute(0, 2, 1).contiguous().reshape(-1, 7168)
        torch.testing.assert_close(
            repack["dst_b"].view(torch.int32), expected_b, atol=0, rtol=0
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.run()
        plan.outputs.fill_(float("nan"))
        repack["dst_a"].zero_()
        repack["dst_b"].zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(plan.outputs, first, atol=0, rtol=0)
