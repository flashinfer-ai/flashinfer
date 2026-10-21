"""Public packed-input MoE behavior, metadata and replay tests on SM100a and SM103a."""

import importlib.util
from pathlib import Path

import pytest
import torch

from flashinfer.experimental.mega_moe_v3 import runtime as _v3_runtime

_HELPER = (
    Path(__file__).resolve().parents[2] / "examples/experimental/mega_moe_inputs.py"
)
_spec = importlib.util.spec_from_file_location("mega_moe_example_inputs", _HELPER)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)

# The packed 384-expert weights, workspaces and the dequantized reference of the
# routed experts need this much free device memory.
MODEL_MEMORY_BYTES = 48 * 2**30


@pytest.fixture(autouse=True)
def supported_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    try:
        arch = _v3_runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))
    if sms not in _v3_runtime.supported_num_sms(arch):
        pytest.skip(
            f"The exported {arch} schedules cover {_v3_runtime.supported_num_sms(arch)} SMs, "
            f"this device has {sms}"
        )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def require_model_memory():
    if torch.cuda.mem_get_info()[0] < MODEL_MEMORY_BYTES:
        pytest.skip(
            "the 384-expert model inputs need about 48 GiB of free device memory"
        )


def make_model(precision, num_tokens=16, seed=0, l2_scale_shift=0):
    require_model_memory()
    return fixtures.make_model(
        precision,
        num_tokens=num_tokens,
        seed=seed,
        l2_scale_shift=l2_scale_shift,
    )


def model_reference(inputs, x_scales):
    return fixtures.model_reference(inputs, x_scales)


def graph_node_names(graph):
    """Names of the nodes captured in ``graph`` (kept, not yet instantiated):
    kernel function names, or ``<memset>`` for memset nodes."""
    driver = pytest.importorskip("cuda.bindings.driver")

    def checked(result):
        assert int(result[0]) == 0, repr(result)
        return result[1] if len(result) == 2 else result[1:]

    raw = driver.CUgraph(graph.raw_cuda_graph())
    _, count = checked(driver.cuGraphGetNodes(raw, 0))
    nodes, count = checked(driver.cuGraphGetNodes(raw, count))
    names = []
    for node in nodes[:count]:
        kind = checked(driver.cuGraphNodeGetType(node))
        if kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            params = checked(driver.cuGraphKernelNodeGetParams(node))
            names.append(checked(driver.cuFuncGetName(params.func)).decode())
        elif kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET:
            names.append("<memset>")
        else:
            raise AssertionError(f"unexpected graph node type {kind}")
    return names


def check_single_kernel_graph(names):
    """Every exported pipeline route and the prepared grouped L2 route capture
    exactly one kernel node: no host reset, copy or repack inside run()."""
    kernels = [name for name in names if name.startswith("kernel_")]
    assert len(kernels) == 1, names
    assert len(names) == 1, names


def check_v3_workspace(plan, *, scratch_cleared=False):
    """After a self-cleaning run every per-launch counter word is zero again
    and each grid gate holds only its phase bit (bits 0..30 clear).

    ``expert_scatter_offsets`` is a scratch table: every launch re-claims its
    slots ``[0, expert_counts[e])`` before reading them, so the multi-token
    kernels leave the previous launch's indices in place. Only the single-token
    kernel zeroes it on exit; pass ``scratch_cleared=True`` for that route."""
    assert plan.self_cleaning
    bindings = plan.stages[0].bindings
    counters = ["expert_counts", "l1_arrival"]
    if scratch_cleared:
        counters.append("expert_scatter_offsets")
    for name in counters:
        assert bindings[name].view(torch.int32).count_nonzero().item() == 0, name
    for name in ("histogram_done", "prefix_done", "dispatch_done", "l2_done"):
        word = bindings[name].view(torch.int32) & 0x7FFFFFFF
        assert word.count_nonzero().item() == 0, name


@pytest.mark.parametrize("precision", ["fp4", "fp8"])
@pytest.mark.parametrize("seed", [0, 1])
def test_pipeline_reference_and_replay(precision, seed):
    """16-token model route of both routed precisions: reference match, direct
    and graph replay, one kernel node per run() and a clean self-cleaning
    workspace. Row assignment inside an expert follows the atomic claim order,
    so replays agree to the reference tolerance."""
    plan, inputs, xs = make_model(precision, seed=seed)
    expected = model_reference(inputs, xs)
    atol = 1.0 if precision == "fp4" else 0.1
    # First launch and direct replay preserve the same workspace/counter owners.
    first = None
    for _ in range(2):
        plan.outputs.fill_(float("nan"))
        plan.run()
        torch.cuda.synchronize()
        torch.testing.assert_close(plan.outputs, expected, atol=atol, rtol=0.1)
        check_v3_workspace(plan)
        if first is None:
            first = plan.outputs.clone()
        else:
            torch.testing.assert_close(plan.outputs, first, atol=atol, rtol=0.1)
    # Capture contains plan.run(): one kernel node for every exported route.
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        plan.run()
    check_single_kernel_graph(graph_node_names(graph))
    graph.instantiate()
    for _ in range(2):
        plan.outputs.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(plan.outputs, first, atol=atol, rtol=0.1)
        check_v3_workspace(plan)


def test_v3_single_token_route_workspace_lifecycle():
    """FP4 single-token model route: one kernel per run(), host reset rejected,
    workspace words clean after every direct and replayed launch."""
    plan, inputs, xs = make_model("fp4", num_tokens=1)
    assert plan.self_cleaning
    with pytest.raises(RuntimeError):
        plan.reset()
    expected = model_reference(inputs, xs)
    first = None
    for _ in range(2):
        plan.outputs.fill_(float("nan"))
        plan.run()
        torch.cuda.synchronize()
        torch.testing.assert_close(plan.outputs, expected, atol=1.0, rtol=0.1)
        check_v3_workspace(plan, scratch_cleared=True)
        if first is None:
            first = plan.outputs.clone()
        else:
            torch.testing.assert_close(plan.outputs, first, atol=1.0, rtol=0.1)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        plan.run()
    check_single_kernel_graph(graph_node_names(graph))
    graph.instantiate()
    for _ in range(2):
        plan.outputs.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(plan.outputs, first, atol=1.0, rtol=0.1)
        check_v3_workspace(plan, scratch_cleared=True)


@pytest.mark.parametrize("num_tokens", [1024, 4096])
def test_v3_long_token_routes(num_tokens):
    """FP4 model routes at 1024 and 4096 tokens: reference match, direct and
    graph replay, one kernel node per run() and a clean self-cleaning workspace.
    These routes use the 32- and 128-row source tile heights. The down-projection
    weight scales are lowered by five powers of two so that every bf16-rounded
    weighted expert output stays below 128 in magnitude: with the fixture's
    default scales a single accumulation-order flip in one expert output moves
    a small element by one bf16 ulp of that output (2.0 near 256-512), outside
    the elementwise tolerance for any implementation of this arithmetic.
    Tolerances are unchanged."""
    plan, inputs, xs = make_model("fp4", num_tokens=num_tokens, l2_scale_shift=5)
    assert plan.self_cleaning
    expected = model_reference(inputs, xs)
    first = None
    for _ in range(2):
        plan.outputs.fill_(float("nan"))
        plan.run()
        torch.cuda.synchronize()
        torch.testing.assert_close(plan.outputs, expected, atol=1.0, rtol=0.1)
        check_v3_workspace(plan)
        if first is None:
            first = plan.outputs.clone()
        else:
            torch.testing.assert_close(plan.outputs, first, atol=1.0, rtol=0.1)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        plan.run()
    check_single_kernel_graph(graph_node_names(graph))
    graph.instantiate()
    for _ in range(2):
        plan.outputs.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(plan.outputs, first, atol=1.0, rtol=0.1)
        check_v3_workspace(plan)


@pytest.mark.parametrize(
    ("precision", "num_tokens"),
    [
        ("fp4", 1),
        ("fp4", 16),
        ("fp8", 16),
        ("fp4", 128),
        ("fp4", 512),
        ("fp4", 1024),
        ("fp4", 4096),
    ],
)
def test_pipeline_negative_topk_idx_routes_to_no_expert(precision, num_tokens):
    """A topk_idx of -1 routes its slot to no expert, on every pipeline route.

    Every fourth token is masked the way a routing gate writes it (all slots -1,
    weight 0); another quarter of the tokens drop one slot whose weight stays
    nonzero, which must not count. Such a slot contributes exactly what the same
    route at weight 0 contributes, nothing: the output equals, bit for bit, a
    plan whose slots keep their experts at weight 0, and it is close to the
    PyTorch reference that skips -1. Masked rows are zero. This holds on the
    first launch, a replay and a captured graph; each launch marks a -1 slot's
    token_to_permuted entry -1 (the entries start at a real row), and the
    workspace stays clean."""
    require_model_memory()
    inputs, xs = fixtures.model_inputs(
        precision,
        num_tokens=num_tokens,
        l2_scale_shift=5 if num_tokens >= 1024 else 0,
    )
    token = torch.arange(num_tokens, device="cuda")
    masked = token % 4 == 1
    dropped = token % 4 == 0
    inputs["topk_weights"][masked] = 0.0
    weight_zero = dict(inputs, topk_weights=inputs["topk_weights"].clone())
    weight_zero["topk_weights"][dropped, 3] = 0.0
    zero_plan = _v3_runtime.prepare_pipeline(weight_zero)
    zero_plan.run()
    torch.cuda.synchronize()
    unrouted = zero_plan.outputs.clone()
    del zero_plan
    inputs["topk_idx"] = inputs["topk_idx"].clone()
    inputs["topk_idx"][masked] = -1
    inputs["topk_idx"][dropped, 3] = -1
    plan = _v3_runtime.prepare_pipeline(inputs)
    expected = model_reference(inputs, xs).float()
    permuted = plan.stages[0].bindings["token_to_permuted"]
    unrouted_slots = (inputs["topk_idx"] < 0).reshape(-1)
    for launch in range(3):
        plan.outputs.fill_(float("nan"))
        # Row 0 is a real row: a -1 slot must not be read from a stale entry.
        permuted.fill_(0)
        if launch < 2:
            plan.run()
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                plan.run()
            plan.outputs.fill_(float("nan"))
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(plan.outputs, unrouted, atol=0, rtol=0)
        assert plan.outputs[masked].count_nonzero().item() == 0
        assert (permuted[unrouted_slots] == -1).all().item()
        actual = plan.outputs.float()
        difference = 1 - 2 * (actual * expected).sum() / (
            actual.square().sum() + expected.square().sum()
        )
        assert difference.item() < 1e-3
        check_v3_workspace(plan, scratch_cleared=num_tokens == 1)


@pytest.mark.parametrize("num_tokens", [1, 16])
def test_pipeline_fully_masked_batch_is_zero(num_tokens):
    """A batch whose every slot is -1 routes no token: it binds to the route's
    grid, every launch writes zeros and the workspace stays clean."""
    require_model_memory()
    inputs, _ = fixtures.model_inputs("fp4", num_tokens=num_tokens)
    inputs["topk_idx"][:] = -1
    inputs["topk_weights"][:] = 0.0
    plan = _v3_runtime.prepare_pipeline(inputs)
    for _ in range(2):
        plan.outputs.fill_(float("nan"))
        plan.run()
        torch.cuda.synchronize()
        assert plan.outputs.count_nonzero().item() == 0
        check_v3_workspace(plan, scratch_cleared=num_tokens == 1)


def test_grouped_l2_update_scales_and_graph_replay():
    plan, a, b, sa, sb, words_a, words_b = fixtures.make_grouped_l2()
    for changed in (None, "activation", "weight"):
        if changed == "activation":
            sa.mul_(2.0)
            words_a.copy_(fixtures.scale_words(sa))
        if changed == "weight":
            sb.mul_(0.5)
            words_b.copy_(fixtures.scale_words(sb))
        if changed is not None:
            # Scale words changed in place: the plan repacks them on request,
            # never inside run().
            plan.update_scales()
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
        repack = plan.preparation_stages[0].bindings
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
        # A captured run() is the GEMM alone and replays bit-exactly.
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            plan.run()
        check_single_kernel_graph(graph_node_names(graph))
        graph.instantiate()
        plan.outputs.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(plan.outputs, first, atol=0, rtol=0)
