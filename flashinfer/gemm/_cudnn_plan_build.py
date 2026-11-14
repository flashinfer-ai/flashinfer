"""Torch-free construction of the cuDNN BF16 GEMM override-shape graph.

``build_cudnn_gemm_bf16_graph_override_shape`` builds its graph through this
module, and plan-build worker processes load this file directly (without
importing the ``flashinfer`` package or torch) to build the same graph. Keep it
free of torch and flashinfer imports so both paths describe one graph.
"""

from dataclasses import dataclass

Tactic = tuple  # (engine_id, ((knob_type, value), ...))


@dataclass(frozen=True)
class Bf16OverrideGraphSpec:
    """Everything that determines the BF16 override-shape graph.

    ``o_type`` is ``int(cudnn.data_type.X)`` so the spec stays picklable
    without cudnn types.
    """

    batch: int
    n: int
    k: int
    o_type: int
    bias: bool
    cache_m: int
    is_a_k_major: bool
    is_b_k_major: bool
    uid_a: int
    uid_b: int
    uid_bias: int
    uid_o: int


def build_bf16_override_graph(spec: Bf16OverrideGraphSpec, handle):
    """Return the graph's tensors and operations; no plan work is done."""
    import cudnn

    batch, n, k, cache_m = spec.batch, spec.n, spec.k, spec.cache_m
    a_shape = (batch, cache_m, k)
    a_stride = (
        (cache_m * k, k, 1) if spec.is_a_k_major else (cache_m * k, 1, cache_m)
    )
    b_shape = (batch, k, n)
    b_stride = (k * n, 1, k) if spec.is_b_k_major else (k * n, n, 1)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
        is_override_shape_enabled=True,
    )
    a = graph.tensor(
        name="a",
        dim=list(a_shape),
        stride=list(a_stride),
        data_type=cudnn.data_type.BFLOAT16,
    )
    b = graph.tensor(
        name="b",
        dim=list(b_shape),
        stride=list(b_stride),
        data_type=cudnn.data_type.BFLOAT16,
    )
    c = graph.matmul(
        name="matmul", A=a, B=b, compute_data_type=cudnn.data_type.FLOAT
    )
    c.set_data_type(cudnn.data_type.FLOAT)

    if spec.bias:
        bias = graph.tensor(
            name="bias",
            dim=(1, 1, n),
            stride=(n, n, 1),
            data_type=cudnn.data_type.BFLOAT16,
        )
        c_final = graph.add(name="bias_add", a=c, b=bias)
        bias.set_uid(spec.uid_bias)
    else:
        c_final = c

    c_final.set_name("c_final").set_output(True).set_data_type(
        cudnn.data_type(spec.o_type)
    )
    a.set_uid(spec.uid_a)
    b.set_uid(spec.uid_b)
    c_final.set_uid(spec.uid_o)
    return graph


def normalize_tactic(engine_id, knobs) -> Tactic:
    return (
        int(engine_id),
        tuple(sorted((int(kind), int(value)) for kind, value in (knobs or {}).items())),
    )


def enumerate_tactics(graph, heur_modes: list[int]) -> list[Tactic]:
    """Candidate engine/knob tactics in plan-index order, without building plans."""
    import cudnn

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode(m) for m in heur_modes])
    graph.check_support()
    tactics = []
    for index in range(graph.get_execution_plan_count()):
        try:
            engine_id, knobs = graph.get_engine_and_knobs_at_index(index)
        except (AttributeError, RuntimeError):
            continue
        tactics.append(normalize_tactic(engine_id, knobs))
    return tactics


def build_single_plan(graph, tactic: Tactic) -> None:
    """Build exactly the plan for one engine/knob tactic."""
    import cudnn

    engine_id, knob_items = tactic
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plan(
        int(engine_id),
        {cudnn.knob_type(int(kind)): int(value) for kind, value in knob_items},
    )
    graph.check_support()
    graph.build_plans()
