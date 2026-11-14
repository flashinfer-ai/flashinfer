"""Torch-free construction of the cuDNN BF16 GEMM override-shape graph.

``build_cudnn_gemm_bf16_graph_override_shape`` builds its graph through this
module, and plan-build worker processes load this file directly (without
importing the ``flashinfer`` package or torch) to build the same graph. Keep it
free of torch and flashinfer imports so both paths describe one graph.
"""

import os
import pickle
import struct
import sys
from dataclasses import asdict, dataclass

Tactic = tuple  # (engine_id, ((knob_type, value), ...))

# Worker wire format: 8-byte little-endian length, then a pickle payload.
_FRAME_HEADER = struct.Struct("<Q")


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
    a_stride = (cache_m * k, k, 1) if spec.is_a_k_major else (cache_m * k, 1, cache_m)
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
    c = graph.matmul(name="matmul", A=a, B=b, compute_data_type=cudnn.data_type.FLOAT)
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


def write_frame(stream, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_FRAME_HEADER.pack(len(data)) + data)
    stream.flush()


def _read_exact(stream, size: int):
    buf = b""
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            if buf:
                raise EOFError("truncated frame")
            return None
        buf += chunk
    return buf


def read_frame(stream):
    """Read one frame; ``None`` at a clean end of stream."""
    header = _read_exact(stream, _FRAME_HEADER.size)
    if header is None:
        return None
    (size,) = _FRAME_HEADER.unpack(header)
    payload = _read_exact(stream, size)
    if payload is None:
        raise EOFError("truncated frame")
    return pickle.loads(payload)


def spec_to_wire(spec: Bf16OverrideGraphSpec) -> dict:
    # A plain dict: unpickling the dataclass by its package path would import
    # flashinfer (and torch) in the worker.
    return asdict(spec)


def build_shard(spec: Bf16OverrideGraphSpec, handle, heur_modes, shard):
    """Enumerate all candidates; build and serialize this shard's plans.

    Returns ``(candidates, [(index, tactic, blob), ...])``. A candidate whose
    plan fails to build is left out, as the all-plans build leaves it out.
    """
    shard_index, shard_count = shard
    # One graph per shard: building plans by index on it costs what
    # build_plans(ALL) does, while a fresh graph per tactic repeats validation
    # and support checks (~25% more on BF16).
    graph = build_bf16_override_graph(spec, handle)
    candidates = enumerate_tactics(graph, heur_modes)
    plans = []
    for index, tactic in enumerate(candidates):
        if index % shard_count != shard_index:
            continue
        try:
            # Select, then build: serialize() exports the selected, built plan.
            graph.select_plan(index)
            graph.build_plan_at_index(index)
            blob = graph.serialize(as_bytes=True)
        except Exception:  # noqa: BLE001 - an unbuildable plan is skipped
            continue
        plans.append((index, tactic, bytes(blob)))
    return candidates, plans


def _worker_main() -> None:
    # cuDNN may print to fd 1; keep the protocol on a private copy of it.
    protocol_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    import cudnn

    handle = None
    stdin = sys.stdin.buffer
    while True:
        request = read_frame(stdin)
        if request is None:
            return
        response = {
            "device_index": request["device_index"],
            "candidates": [],
            "plans": [],
            "error": None,
        }
        try:
            if handle is None:
                handle = cudnn.create_handle()
            candidates, plans = build_shard(
                Bf16OverrideGraphSpec(**request["spec"]),
                handle,
                request["heur_modes"],
                tuple(request["shard"]),
            )
            response["candidates"] = candidates
            response["plans"] = plans
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            response["error"] = f"{type(exc).__name__}: {exc}"
        write_frame(protocol_out, response)


if __name__ == "__main__":
    _worker_main()
