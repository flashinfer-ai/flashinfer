# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GDP/GDN2 selected-state updates use native pool addressing without copies."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import chunk_gated_delta_product, chunk_gated_delta_rule2
from tests.test_helpers.cudnn_linear_attention import (
    assert_rel_close,
    requires_cudnn_linear_attention,
    serial_delta_product,
    serial_delta_rule2,
)

pytestmark = requires_cudnn_linear_attention
D, H = 128, 4


@pytest.fixture(params=["gdn2", "gdp"])
def family(request):
    import cudnn

    if tuple(map(int, cudnn.__version__.split(".")[:2])) < (1, 31):
        pytest.skip("state pools require cuDNN frontend 1.31+")
    return request.param


def _inputs(family, lengths):
    torch.manual_seed(503)
    total, expanded = sum(lengths), sum(lengths) * (2 if family == "gdp" else 1)
    q = F.normalize(torch.randn(total, H, D, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(expanded, H, D, device="cuda"), dim=-1).bfloat16()
    args = dict(q=q, k=k, v=torch.randn_like(k))
    if family == "gdn2":
        args.update(
            g=-torch.rand_like(q).float(), beta=torch.rand_like(q), w=torch.rand_like(q)
        )
    else:
        args.update(
            g=torch.rand(total, H, device="cuda") * 0.8 + 0.1,
            beta=torch.rand(expanded, H, device="cuda"),
            num_householder=2,
        )
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    args["cu_seqlens"] = torch.tensor(cu, device="cuda", dtype=torch.int32)
    return args


def _pool(dtype):
    slot_stride = H * D * D + 96
    storage = torch.randn(7 * slot_stride, device="cuda", dtype=dtype) * 0.01
    return storage.as_strided((7, H, D, D), (slot_stride, D * D, D, 1))


def _run(family, args, **kwargs):
    fn = chunk_gated_delta_rule2 if family == "gdn2" else chunk_gated_delta_product
    return fn(**args, **kwargs)


def _reference(family, args, state):
    common = dict(initial_state=state, scale=D**-0.5, beta=args["beta"])
    if family == "gdn2":
        return serial_delta_rule2(
            args["q"],
            args["k"],
            args["v"],
            args["cu_seqlens"],
            alpha=args["g"].exp(),
            w=args["w"],
            **common,
        )
    return serial_delta_product(
        args["q"],
        args["k"],
        args["v"],
        args["cu_seqlens"],
        alpha=args["g"],
        num_householder=args["num_householder"],
        **common,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["inplace", "separate", "output_only"])
def test_family_pool_fresh_slots_and_untouched_rows(family, dtype, mode):
    args = _inputs(family, [33, 0, 65])
    pool = _pool(dtype)
    destination = pool if mode == "inplace" else torch.full_like(pool, 17)
    for selected in ([5, 2, 4], [1, 6, 3]):
        slots = torch.tensor(selected, device="cuda", dtype=torch.int32)
        before, before_dest = pool.clone(), destination.clone()
        expected, state = _reference(family, args, pool.index_select(0, slots))
        version = pool._version
        result = _run(
            family,
            args,
            initial_state=pool,
            output_state=destination,
            state_indices=slots,
            output_final_state=mode != "output_only",
        )
        if mode == "output_only":
            actual = result
            assert torch.equal(pool, before)
            assert torch.equal(destination, before_dest)
            assert pool._version == version
        else:
            actual, returned = result
            assert returned is destination
            assert_rel_close("selected state", destination[slots], state, 5e-2)
            untouched = [i for i in range(7) if i not in selected]
            assert torch.equal(destination[untouched], before_dest[untouched])
            if mode == "inplace":
                assert pool._version > version
            else:
                assert torch.equal(pool, before)
        assert_rel_close("pool output", actual, expected, 5e-2)


def test_family_pool_long_prefill(family):
    args = _inputs(family, [2049, 0, 1023])
    pool = _pool(torch.float32)
    slots = torch.tensor([5, 2, 4], device="cuda", dtype=torch.int32)
    before = pool.clone()
    expected, state = _reference(family, args, pool.index_select(0, slots))
    actual, returned = _run(
        family,
        args,
        initial_state=pool,
        output_state=pool,
        state_indices=slots,
        output_final_state=True,
    )
    assert returned is pool
    assert_rel_close("long pool output", actual, expected, 5e-2)
    assert_rel_close("long pool state", pool[slots], state, 5e-2)
    assert torch.equal(pool[[0, 1, 3, 6]], before[[0, 1, 3, 6]])


def test_family_pool_capture_reads_changed_slots_and_contents(family):
    args = _inputs(family, [33, 65])
    pool = _pool(torch.bfloat16)
    seed = pool.clone()
    slots = torch.tensor([5, 2], device="cuda", dtype=torch.int32)
    output = torch.empty_like(args["q"])

    def run():
        return _run(
            family,
            args,
            initial_state=pool,
            output_state=pool,
            state_indices=slots,
            output_final_state=True,
            output=output,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    pool.copy_(seed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured, returned = run()
    assert returned is pool and captured.data_ptr() == output.data_ptr()
    for selected in ([5, 2], [1, 6]):
        slots.copy_(torch.tensor(selected, device="cuda", dtype=torch.int32))
        pool.copy_(seed)
        args["v"].mul_(0.9)
        expected, state = _reference(family, args, pool.index_select(0, slots))
        graph.replay()
        torch.cuda.synchronize()
        assert_rel_close("captured output", output, expected, 5e-2)
        assert_rel_close("captured state", pool[slots], state, 5e-2)
        untouched = [i for i in range(7) if i not in selected]
        assert torch.equal(pool[untouched], seed[untouched])
    old_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        run()
    finally:
        torch.cuda.set_sync_debug_mode(old_mode)
