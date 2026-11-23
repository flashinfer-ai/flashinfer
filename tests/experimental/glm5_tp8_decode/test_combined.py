# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Attention -> MoE on the public default workspace, in one stream and graph.

The same chain launched with ``enable_pdl=False`` must produce identical bits.
"""

import pytest
import torch

from flashinfer import glm5_tp8_decode as api

from . import _api, data
from ._attention_case import AttentionCase
from ._attention_checks import check_attention
from ._moe_case import MoECase
from ._moe_checks import check_moe
from .moe_reference import bitwise

pytestmark = pytest.mark.node_exclusive


def test_default_workspace_attention_then_moe_eager_and_graph(
    collective, collective_watchdog
):
    attention = moe = workspace = None
    attention_closed = moe_closed = False
    graphs = {}
    try:
        # Reuse the independently checked synthetic data/packing fixtures, then
        # retire their family-only workspaces before binding the combined one.
        attention = AttentionCase(collective, capacity=4104, seed=1)
        attention.close()
        attention_closed = True
        moe = MoECase(collective, weight_seed=847223)
        moe.close()
        moe_closed = True
        workspace = api.create_glm5_tp8_decode_workspace(
            rank=collective.rank,
            group=collective.group,
            device=collective.device,
            max_seq_len=attention.capacity,
        )  # Exercise the public default operator set.
        stream = torch.cuda.Stream(device=collective.device)
        attention_plans, moe_plans, outputs, output_storage = {}, {}, {}, {}
        counts = {1: 0, 4: 0}
        keys = ((4, "full"), (1, "shared"), (4, "shared"))
        for key in keys:
            q_len, mode = key
            attention_plans[key] = _api.prepare_attention(
                attention.x[key],
                attention.packed,
                attention.caches,
                attention.rope,
                attention.position,
                attention.out[key],
                workspace,
                indexer=mode,
                selection=attention.selection[key],
            )
            output_storage[key] = torch.full(
                (q_len + 1, 6144),
                17,
                dtype=torch.bfloat16,
                device=collective.device,
            )
            outputs[key] = output_storage[key][:q_len]
            # Direct binding: there is no copy or host synchronization between
            # attention producing this tensor and MoE consuming it.
            moe_plans[key] = _api.prepare_moe(
                attention.out[key], moe.packed, outputs[key], workspace
            )
        # This happens once, before the first combined call. Every later MoE
        # call must initialize its own sentinels without help from attention.
        moe_plans[(4, "full")]._diagnostics["scores"].fill_(-13.125)
        moe_plans[(4, "full")]._diagnostics["mid"].fill_(17)
        collective.sync()

        def generations():
            return {
                (family, key): int(plan._diagnostics["generation"]())
                for family, plans in (
                    ("attention", attention_plans),
                    ("moe", moe_plans),
                )
                for key, plan in plans.items()
            }

        def run(key, past, seed, *, graph=False):
            q_len, mode = key
            assert past >= 0 and past + q_len <= attention.capacity
            expected_input = data.make_attention_input(q_len, seed, collective.device)
            selected = data.make_selection(q_len, past, seed)
            initial_cache = {
                name: value.clone() for name, value in attention.caches.items()
            }
            stream.wait_stream(torch.cuda.current_stream(collective.device))
            with torch.cuda.stream(stream):
                attention.x[key].copy_(expected_input)
                attention.position.fill_(past)
                attention.out_storage[key].fill_(17)
                output_storage[key].fill_(17)
                if attention.selection[key] is not None:
                    attention.selection[key].copy_(selected)
                if graph:
                    graphs[key].replay()
                else:
                    attention_plans[key]()
                    result = moe_plans[key]()
                    assert result.data_ptr() == outputs[key].data_ptr()
            counts[q_len] += 1
            collective.sync()

            def guards():
                assert bitwise(attention.x[key], expected_input), "input was modified"
                assert bool((attention.out_storage[key][q_len:] == 17).all()), (
                    "attention output guard was modified"
                )
                assert bool((output_storage[key][q_len:] == 17).all()), (
                    "MoE output guard was modified"
                )
                assert attention.position.item() == past
                assert bitwise(attention.rope, attention.rope_snapshot)
                if attention.selection[key] is not None:
                    assert bitwise(attention.selection[key].cpu(), selected)
                for plan in (attention_plans[key], moe_plans[key]):
                    assert int(plan._diagnostics["generation"]()) == counts[q_len]
                return generations()

            local_generations = collective.checked(guards)
            assert collective.gather(local_generations) == [local_generations] * 8
            # Both families' completed diagnostics must survive the combined
            # launch. This also checks that their scratch/record regions do not
            # overlap and that MoE left its attention-produced input untouched.
            check_attention(
                collective,
                attention_plans[key],
                attention.weights,
                attention.caches,
                initial_cache,
                attention.x[key],
                attention.out[key],
                attention.rope,
                past,
                indexer=mode,
                selection=attention.selection[key],
            )
            check_moe(
                collective,
                moe_plans[key],
                moe.weights,
                attention.out[key],
                outputs[key],
            )

        run((4, "full"), 2048, 1)
        run((1, "shared"), 31, 7)
        run((4, "shared"), 2049, 23)
        before_capture = collective.checked(generations)
        for key in keys:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                attention_plans[key]()
                moe_plans[key]()
            graphs[key] = graph
            del graph
        collective.sync()
        assert collective.checked(generations) == before_capture
        for key, past, seed in (
            ((4, "full"), 2047, 31),
            ((1, "shared"), 4096, 41),
            ((4, "shared"), 4096, 43),
            ((1, "shared"), 2048, 47),
            ((4, "full"), 4096, 53),
            ((4, "shared"), 31, 59),
        ):
            run(key, past, seed, graph=True)

        def immutable_weights():
            for name, tensor in attention.layer.items():
                assert bitwise(tensor, attention.layer_snapshots[name])
            for name, tensor in attention.packed.tensors.items():
                assert bitwise(tensor, attention.packed_snapshots[name])
            for name, snapshot in moe.weight_snapshots.items():
                assert bitwise(getattr(moe.packed, name), snapshot)

        collective.checked(immutable_weights)
    finally:
        if workspace is not None:
            collective.sync()
            graphs.clear()
            workspace.destroy()
        if moe is not None and not moe_closed:
            moe.close()
        if attention is not None and not attention_closed:
            attention.close()


def test_attention_then_moe_without_pdl_matches_default(
    collective, collective_watchdog
):
    # enable_pdl only selects the programmatic stream serialization launch
    # attribute; the kernels are the same. With fixed shared-mode selections,
    # both launch modes must pass the stage checks with identical bits.
    attention = moe = workspace = None
    attention_closed = moe_closed = False
    try:
        attention = AttentionCase(collective, capacity=4104, seed=1)
        attention.close()
        attention_closed = True
        moe = MoECase(collective, weight_seed=847223)
        moe.close()
        moe_closed = True
        workspace = _api.workspace(collective, capacity=attention.capacity)
        stream = torch.cuda.Stream(device=collective.device)
        for key, past, seed in (((4, "shared"), 2049, 23), ((1, "shared"), 4096, 41)):
            q_len, mode = key
            x, out = attention.x[key], attention.out[key]
            selection = attention.selection[key]
            moe_out = torch.empty_like(out)
            plans = {
                enable_pdl: (
                    _api.prepare_attention(
                        x,
                        attention.packed,
                        attention.caches,
                        attention.rope,
                        attention.position,
                        out,
                        workspace,
                        indexer=mode,
                        selection=selection,
                        enable_pdl=enable_pdl,
                    ),
                    _api.prepare_moe(
                        out, moe.packed, moe_out, workspace, enable_pdl=enable_pdl
                    ),
                )
                for enable_pdl in (True, False)
            }
            results = {}
            for enable_pdl, (attention_plan, moe_plan) in plans.items():
                expected_input = data.make_attention_input(
                    q_len, seed, collective.device
                )
                selected = data.make_selection(q_len, past, seed)
                initial_cache = {
                    name: value.clone() for name, value in attention.caches.items()
                }
                stream.wait_stream(torch.cuda.current_stream(collective.device))
                with torch.cuda.stream(stream):
                    x.copy_(expected_input)
                    attention.position.fill_(past)
                    if selection is not None:
                        selection.copy_(selected)
                    attention_plan()
                    moe_plan()
                collective.sync()
                results[enable_pdl] = (
                    check_attention(
                        collective,
                        attention_plan,
                        attention.weights,
                        attention.caches,
                        initial_cache,
                        x,
                        out,
                        attention.rope,
                        past,
                        indexer=mode,
                        selection=selection,
                    ),
                    check_moe(collective, moe_plan, moe.weights, out, moe_out),
                )
            for name, actual, expected in zip(
                ("attention", "MoE"), results[False], results[True], strict=True
            ):
                assert bitwise(actual, expected), (
                    f"{key}: enable_pdl=False changed the {name} output"
                )
    finally:
        if workspace is not None:
            collective.sync()
            workspace.destroy()
        if moe is not None and not moe_closed:
            moe.close()
        if attention is not None and not attention_closed:
            attention.close()
