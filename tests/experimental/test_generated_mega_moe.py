"""Public packed-input MoE behavior, metadata and replay tests on SM100a and SM103a."""

import importlib.util
from pathlib import Path

import pytest
import torch

from flashinfer.experimental.mega_moe_v3 import preparation as _v3_preparation
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

# The tensor-parallel and expert-parallel shapes of the 384-expert, top-6,
# hidden-5120 model (FI #6587): experts held per rank, the logical intermediate
# width (576 and 288 route to the 640- and 384-wide programs) and the rank
# count the route ids spread over.
TP_EP_CASES = {
    "tp4": dict(num_experts=384, intermediate=576, ep=1),
    "tp8": dict(num_experts=384, intermediate=288, ep=1),
    "tp4ep4": dict(num_experts=96, intermediate=2304, ep=4),
    "tp8ep8": dict(num_experts=48, intermediate=2304, ep=8),
}
EP_CASES = [case for case, geometry in TP_EP_CASES.items() if geometry["ep"] > 1]
TP_CASES = [case for case, geometry in TP_EP_CASES.items() if geometry["ep"] == 1]
TP_EP_TOKENS = (1, 4, 16, 32, 128, 512, 1024, 4096)
# The token counts the single-rank model route (384 experts, 2304 wide) shares
# with the TP/EP routes.
SHARED_TOKENS = (1, 16, 128, 512, 1024, 4096)
TOP_K, HIDDEN = 6, 5120
# The TP/EP routes of up to 32 tokens, whose dispatch marks a foreign route's
# token_to_permuted entry -1 from its pair loop, and the rank each case runs
# as: rank 0, and rank 1 of the expert-parallel cases.
ROUTING_CHANGE_TOKENS = (1, 4, 16, 32)
ROUTING_CHANGE_RANKS = [
    (case, rank)
    for case, geometry in TP_EP_CASES.items()
    for rank in ((0, 1) if geometry["ep"] > 1 else (0,))
]


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


def require_model_memory(num_experts=384, intermediate=2304):
    """Skip below the free memory the inputs of a shape need: the 48 GiB of the
    384-expert, 2304-wide model scaled by experts held times physical width,
    with an 8 GiB floor for activations, pools and workspaces."""
    physical = _v3_preparation.physical_intermediate(intermediate)
    needed = max(8 * 2**30, MODEL_MEMORY_BYTES * num_experts * physical // (384 * 2304))
    if torch.cuda.mem_get_info()[0] < needed:
        pytest.skip(
            f"the {num_experts}-expert, {intermediate}-wide model inputs need "
            f"about {needed / 2**30:.0f} GiB of free device memory"
        )


def route_available(num_experts, intermediate, num_tokens, precision="fp4"):
    """Whether this architecture's catalog has the pipeline route of the shape."""
    arch = _v3_runtime.device_arch(torch.device("cuda"))
    args = _v3_preparation.pipeline_route_args(
        dict(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=TOP_K,
            hidden=HIDDEN,
            intermediate=intermediate,
            routed_weight_dtype=precision,
            activation_clamp=10.0,
        )
    )
    return _v3_runtime.key("pipeline", args) in _v3_runtime.catalog().routes(arch)


def require_route(num_experts, intermediate, num_tokens):
    if not route_available(num_experts, intermediate, num_tokens):
        pytest.skip(
            f"no catalogued FP4 route for {num_experts} experts, intermediate "
            f"{intermediate}, {num_tokens} tokens on this architecture"
        )


def case_inputs(case, num_tokens, *, rank=0, **overrides):
    """One rank's inputs of a TP/EP case at ``num_tokens`` (skips without the
    route or the memory). The down-projection scales are lowered at every token
    count, as the model-route tests lower them at the long ones: with the
    default scales a weighted expert output of these geometries reaches ~500,
    where one accumulation-order flip moves an element by one bf16 ulp (2.0),
    outside the elementwise tolerance for any implementation of this
    arithmetic. Tolerances are unchanged."""
    geometry = dict(TP_EP_CASES[case])
    ep = geometry.pop("ep")
    require_route(geometry["num_experts"], geometry["intermediate"], num_tokens)
    require_model_memory(**geometry)
    return fixtures.model_inputs(
        "fp4",
        num_tokens=num_tokens,
        l2_scale_shift=5,
        ep=ep,
        rank=rank,
        **geometry,
        **overrides,
    )


def scratch_cleared(plan):
    """Whether the plan's program zeroes ``expert_scatter_offsets`` on exit: the
    histogram dispatch form keeps one cursor per expert and cleans it, the
    slot-list form keeps ``num_experts * num_tokens`` entries and rewrites them."""
    bindings = plan.stages[0].bindings
    return (
        bindings["expert_scatter_offsets"].numel() == bindings["expert_counts"].numel()
    )


def run_once(plan):
    plan.outputs.fill_(float("nan"))
    plan.run()
    torch.cuda.synchronize()
    return plan.outputs.clone()


def check_route_contract(plan, expected):
    """Reference match at the FP4 tolerance on the first launch and a direct
    replay, bitwise agreement between the launches, one kernel node per
    captured run(), bitwise graph replays and a clean self-cleaning workspace;
    then three consecutive run() in one captured graph, chained by the
    programmatic dependencies their launch attribute records, replaying bit
    for bit with a clean workspace. Returns the first launch's output."""
    cleared = scratch_cleared(plan)
    first = None
    for _ in range(2):
        actual = run_once(plan)
        torch.testing.assert_close(actual, expected, atol=1.0, rtol=0.1)
        check_v3_workspace(plan, scratch_cleared=cleared)
        if first is None:
            first = actual
        else:
            torch.testing.assert_close(actual, first, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        plan.run()
    check_single_kernel_graph(graph_node_names(graph))
    graph.instantiate()
    for _ in range(2):
        plan.outputs.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(plan.outputs, first, atol=0, rtol=0)
        check_v3_workspace(plan, scratch_cleared=cleared)
    check_programmatic_dependent_graph(plan, first, scratch_cleared=cleared)
    return first


def local_slots(inputs):
    """Boolean ``[T, top_k]`` mask of the slots this rank computes."""
    local = inputs["topk_idx"] - inputs["local_expert_offset"]
    return (local >= 0) & (local < inputs["num_experts"])


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


def checked(result):
    """Payload of a ``cuda.bindings.driver`` call: its value after the
    success status, or the tuple of values after it."""
    assert int(result[0]) == 0, repr(result)
    return result[1] if len(result) == 2 else result[1:]


def graph_node_names(graph):
    """Names of the nodes captured in ``graph`` (kept, not yet instantiated):
    kernel function names, or ``<memset>`` for memset nodes."""
    driver = pytest.importorskip("cuda.bindings.driver")
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


def graph_edges(graph):
    """``(from, to, dependency type)`` of every edge of ``graph`` (kept, not
    yet instantiated) between kernel nodes, the ends named as in
    ``graph_node_names`` and the type the edge data's ``CUgraphDependencyType``
    name: a launch captured with the programmatic stream serialization
    attribute records a ``CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC`` edge from
    the kernel node before it on the stream; a plain launch records the
    default full dependency."""
    driver = pytest.importorskip("cuda.bindings.driver")
    raw = driver.CUgraph(graph.raw_cuda_graph())
    # The edge-data form of the query: suffixed in cuda-bindings 12, the plain
    # name in 13.
    get_edges = getattr(driver, "cuGraphGetEdges_v2", None) or driver.cuGraphGetEdges
    *_, count = checked(get_edges(raw, 0))
    edges = checked(get_edges(raw, count))
    assert len(edges) == 4, f"{get_edges.__name__} exposes no edge data: {edges!r}"
    sources, targets, data, count = edges

    def kernel_name(node):
        params = checked(driver.cuGraphKernelNodeGetParams(node))
        return checked(driver.cuFuncGetName(params.func)).decode()

    return [
        (
            kernel_name(source),
            kernel_name(target),
            driver.CUgraphDependencyType(edge.type).name,
        )
        for source, target, edge in zip(
            sources[:count], targets[:count], data[:count], strict=True
        )
    ]


def check_programmatic_dependent_graph(
    plan, expected, *, scratch_cleared=False, calls=3
):
    """``calls`` consecutive run() captured in one graph: one kernel node per
    call and, between consecutive calls, the programmatic dependency the TP/EP
    launch attribute records, so a call's CTAs may become resident while its
    predecessor still runs and ``griddepcontrol.wait`` orders every global
    access after the predecessor's exit; two replays write ``expected`` bit
    for bit and leave the self-cleaning workspace clean."""
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        for _ in range(calls):
            plan.run()
    names = graph_node_names(graph)
    assert len(names) == calls and len(set(names)) == 1, names
    assert names[0].startswith("kernel_"), names
    edges = graph_edges(graph)
    assert len(edges) == calls - 1, edges
    for source, target, kind in edges:
        assert (source, target) == (names[0], names[0]), edges
        assert kind == "CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC", edges
    graph.instantiate()
    for _ in range(2):
        plan.outputs.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(plan.outputs, expected, atol=0, rtol=0)
        check_v3_workspace(plan, scratch_cleared=scratch_cleared)


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


@pytest.mark.parametrize("num_tokens", TP_EP_TOKENS)
@pytest.mark.parametrize("case", list(TP_EP_CASES))
def test_tp_ep_route_contract(case, num_tokens):
    """Rank 0 of every tensor-parallel and expert-parallel route: reference
    match at the FP4 tolerance, bitwise replay, one kernel node per captured
    run(), bitwise graph replay, a clean workspace, three consecutive run()
    captured as programmatic dependents replaying bit for bit, and a second
    preparation of the same inputs that reproduces the output bit for bit. On the
    expert-parallel cases rank 0 holds a quarter or an eighth of the experts
    and the global ids spread over every rank, so the output is the rank's
    partial sum: a slot routed to another rank takes no row and a token
    without a local route is zero."""
    inputs, xs = case_inputs(case, num_tokens)
    plan = _v3_runtime.prepare_pipeline(inputs)
    assert plan.self_cleaning
    assert plan.stages[0].bindings["local_expert_offset"] == 0
    expected = model_reference(inputs, xs)
    first = check_route_contract(plan, expected)
    local = local_slots(inputs)
    permuted = plan.stages[0].bindings["token_to_permuted"].reshape(local.shape)
    assert (permuted[~local] == -1).all().item()
    assert (permuted[local] >= 0).all().item()
    assert first[~local.any(dim=1)].count_nonzero().item() == 0
    if TP_EP_CASES[case]["ep"] > 1:
        assert not local.all().item() and local.any().item()
    del plan
    again = _v3_runtime.prepare_pipeline(inputs)
    torch.testing.assert_close(run_once(again), first, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", TP_EP_TOKENS)
@pytest.mark.parametrize("case", TP_CASES)
def test_tp_logical_padding_invariance(case, num_tokens):
    """The logical tensor-parallel widths (576, 288) route to the 128-wide
    programs (640, 384). The output of the logical call, padded at
    preparation, equals bit for bit the output of the same call with weights
    pre-padded by the caller (zero rows and columns, scale 1.0), and so do
    the two PyTorch references."""
    inputs, xs = case_inputs(case, num_tokens)
    physical = _v3_preparation.physical_intermediate(inputs["intermediate"])
    assert physical != inputs["intermediate"]
    padded = fixtures.pad_intermediate(inputs, physical)
    torch.testing.assert_close(
        model_reference(padded, xs), model_reference(inputs, xs), atol=0, rtol=0
    )
    plan = _v3_runtime.prepare_pipeline(inputs)
    assert plan.stages[0].bindings["N1"] == 2 * physical
    assert plan.stages[0].bindings["K2"] == physical
    logical_output = run_once(plan)
    del plan
    plan = _v3_runtime.prepare_pipeline(padded)
    torch.testing.assert_close(run_once(plan), logical_output, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", TP_EP_TOKENS)
@pytest.mark.parametrize("case", EP_CASES)
def test_ep_nonzero_rank_matches_reference_and_local_ids(case, num_tokens):
    """Rank 1 of an expert-parallel case (nonzero offset) with -1 slots mixed
    in: tokens 1 mod 4 are masked (all slots -1, weight 0) and tokens 2 mod 4
    drop slot 5, whose weight stays nonzero and which rank 1 holds on those
    tokens in both cases. Token 0 keeps every slot, so rank 1 computes a real
    row at every token count (slot 1 of token 0 is its own). The output
    matches the reference over the local routes, masked rows are zero, and
    it equals bit for bit the output of the same call with the ids remapped
    on the host (local ids, -1 for every remote or masked slot, offset 0): the
    device filter and the -1 path are one path."""
    inputs, xs = case_inputs(case, num_tokens, rank=1)
    assert inputs["local_expert_offset"] == inputs["num_experts"]
    token = torch.arange(num_tokens, device="cuda")
    masked = token % 4 == 1
    dropped = token % 4 == 2
    # Before the edits: slot 1 of token 0 and slot 5 of every dropped token
    # are this rank's, so the dropped -1 (nonzero weight) is a local slot
    # that must not count, and token 0 keeps a real local row.
    before = local_slots(inputs)
    assert before[0, 1].item() and before[dropped, 5].all().item()
    inputs["topk_idx"] = inputs["topk_idx"].clone()
    inputs["topk_idx"][masked] = -1
    inputs["topk_idx"][dropped, 5] = -1
    inputs["topk_weights"][masked] = 0.0
    local = local_slots(inputs)
    assert local.any().item() and not local.all().item()
    plan = _v3_runtime.prepare_pipeline(inputs)
    assert (
        plan.stages[0].bindings["local_expert_offset"] == inputs["local_expert_offset"]
    )
    expected = model_reference(inputs, xs)
    first = check_route_contract(plan, expected)
    assert first[masked].count_nonzero().item() == 0
    permuted = plan.stages[0].bindings["token_to_permuted"].reshape(local.shape)
    assert (permuted[~local] == -1).all().item()
    del plan
    remapped = dict(
        inputs,
        local_expert_offset=0,
        num_global_experts=inputs["num_experts"],
        topk_idx=torch.where(
            local, inputs["topk_idx"] - inputs["local_expert_offset"], -1
        ),
    )
    plan = _v3_runtime.prepare_pipeline(remapped)
    torch.testing.assert_close(run_once(plan), first, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", TP_EP_TOKENS)
@pytest.mark.parametrize("case", EP_CASES)
def test_ep_all_remote_batch_is_zero(case, num_tokens):
    """A rank none of whose experts is routed (the ids spread over the other
    ranks of a wider global space): the plan binds to the route's grid, every
    slot takes no row, every launch writes zeros, a captured run() is one
    kernel node and the workspace stays clean."""
    geometry = TP_EP_CASES[case]
    inputs, _ = case_inputs(
        case,
        num_tokens,
        local_expert_offset=geometry["ep"] * geometry["num_experts"],
        num_global_experts=(geometry["ep"] + 1) * geometry["num_experts"],
    )
    assert not local_slots(inputs).any().item()
    plan = _v3_runtime.prepare_pipeline(inputs)
    # No local row, yet the pools keep one 256-row aligned block for the
    # program's tensor maps.
    data = plan.owners[1]
    bindings = plan.stages[0].bindings
    assert data["M_total"] == _v3_preparation.ALIGN_M and data["total_m_tiles"] == 0
    for name in (
        "pool_fp8",
        "pool_sf",
        "routing_weight_pool",
        "meta_token",
        "A2",
        "expert_output",
    ):
        assert bindings[name].shape[0] == _v3_preparation.ALIGN_M, name
    assert bindings["tile_expert"].numel() == 0
    expected = torch.zeros_like(plan.outputs)
    check_route_contract(plan, expected)
    assert (bindings["token_to_permuted"] == -1).all().item()


@pytest.mark.parametrize("num_tokens", SHARED_TOKENS)
@pytest.mark.parametrize("case", EP_CASES)
def test_ep_rank_matches_the_global_program(case, num_tokens):
    """Rank 1 of an expert-parallel case against the single-rank model
    program at the same token count (the six counts both routes share): the
    rank's program, given its slice of the global weights, its offset and the
    global ids, writes bit for bit what the 384-expert program writes for the
    global weight set when every slot this rank does not hold is set to -1.
    Both compute the same per-route arithmetic and combine the slots in
    ascending order; a mismatch is a bug, not noise."""
    local_experts = TP_EP_CASES[case]["num_experts"]
    require_route(local_experts, 2304, num_tokens)
    require_route(384, 2304, num_tokens)
    require_model_memory()
    inputs, _ = fixtures.model_inputs(
        "fp4", num_tokens=num_tokens, l2_scale_shift=5 if num_tokens >= 1024 else 0
    )
    # Move the even slots one rank window up: every token then routes to both
    # rank 1 and other ranks at every token count (the ids stay distinct).
    ids = inputs["topk_idx"].clone()
    ids[:, 0::2] = (ids[:, 0::2] + local_experts) % inputs["num_experts"]
    inputs["topk_idx"] = ids
    rank = fixtures.expert_parallel_rank(inputs, local_experts, rank=1)
    assert rank["local_expert_offset"] == local_experts
    local = local_slots(rank)
    assert local.any().item() and not local.all().item()
    plan = _v3_runtime.prepare_pipeline(rank)
    partial = run_once(plan)
    del plan
    masked = dict(inputs, topk_idx=torch.where(local, inputs["topk_idx"], -1))
    plan = _v3_runtime.prepare_pipeline(masked)
    torch.testing.assert_close(run_once(plan), partial, atol=0, rtol=0)


def fresh_plan_output(inputs):
    """Output of one launch of a plan prepared for ``inputs`` on a workspace of
    its own; the plan is released before the caller prepares another."""
    plan = _v3_runtime.prepare_pipeline(inputs)
    output = run_once(plan)
    del plan
    return output


def masked_routing(inputs):
    """The routing of ``inputs`` with -1 slots added and nothing else: tokens
    1 mod 4 are masked the way a routing gate writes them (every slot -1,
    weight 0) and tokens 0 mod 4 with a local slot drop their first local
    slot, whose weight stays nonzero and which must not count. Returns the
    ids, the weights and the ``[T, top_k]`` mask of the slots this rank
    computes under the routing of ``inputs`` and routes nowhere under the
    masked one."""
    local = local_slots(inputs)
    token = torch.arange(inputs["num_tokens"], device="cuda")
    masked = token % 4 == 1
    dropped = ((token % 4 == 0) & local.any(dim=1)).nonzero().squeeze(1)
    ids = inputs["topk_idx"].clone()
    weights = inputs["topk_weights"].clone()
    ids[masked] = -1
    weights[masked] = 0.0
    ids[dropped, local.to(torch.int8).argmax(dim=1)[dropped]] = -1
    return ids, weights, local & (ids < 0)


@pytest.mark.parametrize("num_tokens", ROUTING_CHANGE_TOKENS)
@pytest.mark.parametrize(
    ("case", "rank"),
    ROUTING_CHANGE_RANKS,
    ids=[f"{case}-rank{rank}" for case, rank in ROUTING_CHANGE_RANKS],
)
def test_tp_ep_routing_change_on_one_workspace(case, rank, num_tokens):
    """One plan, one workspace, two routings of a TP/EP route of up to 32
    tokens (the dispatch form that marks a foreign route's token_to_permuted
    entry -1 from its pair loop). Routing B is routing A with -1 slots added
    and nothing else (tokens 1 mod 4 masked at weight 0, tokens 0 mod 4
    dropping their first local slot at nonzero weight), so B's rows and tiles
    fit the pools prepared for A, and it is written in place into the route
    tensors the plan binds. A launch of B after a launch of A marks -1 an
    entry the launch before wrote as a real row: the one order in which an
    entry read before the dispatch's store would add a row to the combine.
    Three alternations, three more with every entry pre-filled to row 0, and
    the alternation captured as one graph reproduce the outputs of fresh plans
    of both routings bit for bit, leave every -1 slot's entry -1 and the
    workspace clean."""
    inputs, xs = case_inputs(case, num_tokens, rank=rank)
    bound_ids, bound_weights = inputs["topk_idx"], inputs["topk_weights"]
    ids_a, weights_a = bound_ids.clone(), bound_weights.clone()
    local_a = local_slots(inputs)
    ids_b, weights_b, flipped = masked_routing(inputs)
    # Token 0 holds a local slot on every rank, so every token count flips one.
    assert flipped.any().item()
    unrouted_b = ids_b < 0
    inputs_b = dict(inputs, topk_idx=ids_b, topk_weights=weights_b)
    ref_a = fresh_plan_output(inputs)
    ref_b = fresh_plan_output(inputs_b)
    torch.testing.assert_close(ref_a, model_reference(inputs, xs), atol=1.0, rtol=0.1)
    torch.testing.assert_close(ref_b, model_reference(inputs_b, xs), atol=1.0, rtol=0.1)
    plan = _v3_runtime.prepare_pipeline(inputs)
    bindings = plan.stages[0].bindings
    # The plan binds int32 and flat views of the caller's route tensors: a
    # routing copied into those tensors is the routing of the next launch.
    assert bindings["topk_idx_i32"].data_ptr() == bound_ids.data_ptr()
    assert bindings["topk_weights"].data_ptr() == bound_weights.data_ptr()
    cleared = scratch_cleared(plan)
    permuted = bindings["token_to_permuted"].reshape(ids_a.shape)

    def launch(ids, weights):
        bound_ids.copy_(ids)
        bound_weights.copy_(weights)
        return run_once(plan)

    def check_b(output, entries):
        torch.testing.assert_close(output, ref_b, atol=0, rtol=0)
        assert (entries[unrouted_b] == -1).all().item()
        assert (entries[local_a & ~unrouted_b] >= 0).all().item()
        check_v3_workspace(plan, scratch_cleared=cleared)

    def check_a(output, entries):
        torch.testing.assert_close(output, ref_a, atol=0, rtol=0)
        assert (entries[~local_a] == -1).all().item()
        assert (entries[local_a] >= 0).all().item()
        check_v3_workspace(plan, scratch_cleared=cleared)

    check_a(launch(ids_a, weights_a), permuted)
    for _ in range(3):
        check_b(launch(ids_b, weights_b), permuted)
        check_a(launch(ids_a, weights_a), permuted)
    # Row 0 of the pools is a real row: an entry read as 0 would add
    # expert_output[0] to the combine.
    for _ in range(3):
        permuted.fill_(0)
        check_b(launch(ids_b, weights_b), permuted)
        check_a(launch(ids_a, weights_a), permuted)
    # The alternation captured once: the route copies are memcpy nodes, which
    # the kernel nodes depend on fully, and the launch before every B launch
    # is an A launch, whose entries a read before the dispatch's store returns.
    outputs_b = [torch.empty_like(plan.outputs) for _ in range(3)]
    entries_b = [torch.empty_like(permuted) for _ in range(3)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for output, entries in zip(outputs_b, entries_b, strict=True):
            bound_ids.copy_(ids_b)
            bound_weights.copy_(weights_b)
            plan.run()
            output.copy_(plan.outputs)
            entries.copy_(permuted)
            bound_ids.copy_(ids_a)
            bound_weights.copy_(weights_a)
            plan.run()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        for output, entries in zip(outputs_b, entries_b, strict=True):
            check_b(output, entries)
        check_a(plan.outputs, permuted)
