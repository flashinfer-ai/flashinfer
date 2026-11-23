"""CPU coverage for per-invocation DA graph resource ownership."""

import gc
import weakref
from types import SimpleNamespace

import torch

from flashinfer.fused_moe.da_moe import (
    DABindingRecord,
    DACaptureOutcome,
    DAMoEDispatcher,
    DAPlanMode,
    DAWorkspaceLane,
    tensor_binding_signature,
)


def test_invocation_buffers_survive_until_graph_lease_release(monkeypatch):
    dispatcher = DAMoEDispatcher(8)
    dispatcher._plan = SimpleNamespace(generation=1, mode=DAPlanMode.DA_SWITCH)
    lane = DAWorkspaceLane(lane_id=0, generation=1, workspace=object())
    dispatcher._workspace_lanes[0] = lane
    bindings = (torch.empty(1), torch.empty(1))
    for binding in bindings:
        signature = tensor_binding_signature((binding,))
        dispatcher._bindings[signature] = DABindingRecord(signature)

    capturing = True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    references = []
    predecessors = []

    class Buffer:
        pass

    def capture(_plan, _workspace, expected_capture_id, previous_node):
        predecessors.append((expected_capture_id, previous_node))
        outputs = (Buffer(), Buffer())
        references.extend(weakref.ref(output) for output in outputs)
        return DACaptureOutcome(
            result=None,
            topology=SimpleNamespace(is_workspace_lane_serialized=True, capture_id=7),
            conditional_node_handle=100 + len(predecessors),
            retained_resources=outputs,
        )

    for binding in bindings:
        dispatcher.dispatch(
            (binding,),
            run_fallback=lambda: None,
            run_body=lambda _body: None,
            capture_switch=capture,
        )
    assert predecessors == [(0, 0), (7, 101)]
    assert len(lane.capture_resources) == 4
    gc.collect()
    assert all(reference() is not None for reference in references)

    capturing = False
    graph = SimpleNamespace(raw_cuda_graph_exec=lambda: 1)
    lease = dispatcher.acquire_graph_lease(graph)
    assert lease.resource_count == 1
    gc.collect()
    assert all(reference() is not None for reference in references)
    lease.release()
    gc.collect()
    assert all(reference() is None for reference in references)
    assert lane.capture_resources == []
