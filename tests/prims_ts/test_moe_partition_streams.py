# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
from flashinfer.prims_ts.moe.partition import MoePartitionResources


def _exercise_fork_join(streams, *, use_graph: bool) -> None:
    resources = MoePartitionResources(streams)
    fork_join = resources.fork_join
    assert fork_join is not None
    source = torch.arange(256, device="cuda", dtype=torch.float32)
    outputs = [torch.empty_like(source), torch.empty_like(source)]

    def enqueue() -> None:
        # torch.cuda.graph() captures on a dedicated stream, not necessarily
        # the stream that entered this helper. Resolve it inside the capture.
        caller = torch.cuda.current_stream()
        fork_join.fork(caller)
        for partition_id, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                torch.mul(source, partition_id + 2, out=outputs[partition_id])
        fork_join.join(caller)

    enqueue()
    torch.cuda.current_stream().synchronize()
    torch.testing.assert_close(outputs[0], source * 2)
    torch.testing.assert_close(outputs[1], source * 3)
    if not use_graph:
        return

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        enqueue()
    source.add_(1)
    graph.replay()
    torch.cuda.current_stream().synchronize()
    torch.testing.assert_close(outputs[0], source * 2)
    torch.testing.assert_close(outputs[1], source * 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_partition_resources_rejects_ordinary_streams():
    try:
        import cuda.bindings.driver as cuda_drv
    except ImportError as exc:
        pytest.skip(f"cuda-python is unavailable: {exc}")
    if not hasattr(cuda_drv, "cuStreamGetGreenCtx"):
        pytest.skip("cuda-python does not expose green-context stream queries")

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    with pytest.raises(ValueError, match="must belong to a green context"):
        MoePartitionResources(streams)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fork_join_on_green_context_streams_with_graph_replay():
    properties = torch.cuda.get_device_properties(0)
    try:
        streams, resources = split_device_green_ctx_by_sm_count(
            torch.device("cuda:0"), [properties.multi_processor_count // 2]
        )
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"green contexts are unavailable: {exc}")
    assert resources  # Keep resources alive through graph replay.
    _exercise_fork_join(streams[:2], use_graph=True)
