# SPDX-License-Identifier: Apache-2.0
"""Real dispatcher/Inductor qualification of the public bucket API; no model claims."""

import json
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from test_xqa_decode_buckets import _assert_output, _case, _reference, _require_sm120


def _fixture(*, direct=False, mode="persistent", planner="gpu", layers=1):
    _require_sm120()
    from flashinfer.xqa import create_xqa_decode_bucket_plan, xqa_decode_with_buckets

    torch.manual_seed(0)
    lengths = [257] if direct else [257, 512, 3, 0]
    tokens, heads = (1, 1) if direct else (3, 4)
    banks = []
    for layer in range(layers):
        queries, data, scales, table, lens = _case(
            lengths,
            capacity_length=512 if direct else 8193,
            interleaved=bool(layer),
            num_kv_heads=heads,
            q_len_per_req=tokens,
        )
        banks.append((queries[0], *data, *scales, torch.empty_like(queries[0])))
        if layer == 0:
            page_table, device_lengths = table, lens
    host_lengths = torch.tensor(lengths, dtype=torch.int32)
    plan = create_xqa_decode_bucket_plan(
        device_lengths,
        page_table_width=page_table.shape[1],
        mode=mode,
        planner=planner,
        host_seq_lens=host_lengths if planner == "host" else None,
        num_kv_heads=heads,
        q_len_per_req=tokens,
        query_tile=16 if direct else 32,
    )
    assert plan.direct == direct
    workspaces = [plan.create_workspace() for _ in banks]

    def forward(table, *operands):
        if planner == "gpu":
            plan.prepare()
        outputs = []
        for layer, workspace in enumerate(workspaces):
            q, k, v, ks, vs, out = operands[layer * 6 : (layer + 1) * 6]
            outputs.append(
                xqa_decode_with_buckets(
                    q,
                    k,
                    v,
                    ks,
                    vs,
                    table,
                    plan,
                    workspace,
                    bmm1_scale=0.7 / 16,
                    bmm2_scale=1.3,
                    out=out,
                )
            )
        return tuple(outputs)

    args = (page_table, *(tensor for bank in banks for tensor in bank))
    forward(*args)  # Legitimate eager warmup also emits the one-time public warning.
    return plan, workspaces, banks, args, forward, host_lengths, lengths


def _compile_check(
    *, direct=False, mode="persistent", planner="gpu", layers=1, opcheck=False
):
    plan, workspaces, banks, args, forward, host_lengths, initial = _fixture(
        direct=direct, mode=mode, planner=planner, layers=layers
    )
    if opcheck:
        from flashinfer.experimental.xqa_decode_buckets import backend

        checks = ("test_schema", "test_faketensor")
        plan_args = (
            plan.lengths,
            plan.tasks,
            plan.requests,
            plan.spans,
            plan.header,
            plan.ctas,
            plan.mode == "aligned",
            plan.page_size,
            plan.head_group_ratio,
            plan.q_len_per_req,
            plan.query_tile,
        )
        if not plan.direct:
            report = torch.library.opcheck(
                backend._prepare_op, plan_args, test_utils=checks
            )
            print(dict(opcheck="plan", subtests=report), flush=True)
        q, k, v, ks, vs, out = banks[0]
        workspace = workspaces[0]
        compute_args = (
            q,
            k,
            v,
            ks,
            vs,
            args[0],
            plan.lengths,
            plan.tasks,
            plan.requests,
            plan.spans,
            plan.header,
            plan.causal_mask,
            workspace.semaphores,
            workspace.scratch,
            out,
            plan.ctas,
            plan.mode == "aligned",
            0.7 / 16,
            1.3,
            plan.enable_pdl,
            plan.page_size,
            plan.head_group_ratio,
            plan.q_len_per_req,
            plan.query_tile,
        )
        readonly = compute_args[:12]
        before = [value.clone().view(torch.uint8) for value in readonly]
        report = torch.library.opcheck(
            backend._compute_op, compute_args, test_utils=checks, raise_exception=False
        )
        schema = report["test_schema"]
        if schema != "SUCCESS":
            # This checker uses allclose on read-only FP8 scales, unsupported
            # by the qualified PyTorch build. Do not change the operator dtype.
            assert isinstance(schema, NotImplementedError)
            assert "mul_cuda" in str(schema) and "Float8_e4m3fn" in str(schema)
        assert report["test_faketensor"] == "SUCCESS"
        assert all(
            torch.equal(value.view(torch.uint8), snapshot)
            for value, snapshot in zip(readonly, before, strict=True)
        )
        print(
            dict(
                opcheck="compute",
                subtests={k: str(v) for k, v in report.items()},
                readonly_input_bytes_unchanged=True,
            ),
            flush=True,
        )
    assert not torch._dynamo.config.suppress_errors
    compiled = torch.compile(forward, fullgraph=True)
    returned = compiled(*args)
    assert all(
        output.data_ptr() == bank[-1].data_ptr()
        for output, bank in zip(returned, banks, strict=True)
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args)
    changed = [512] if direct else [513, 0, 257, 3]
    for step, lengths in enumerate((initial, changed, [0] * len(initial), initial)):
        plan.lengths.copy_(
            torch.tensor(lengths, dtype=torch.int32, device=plan.lengths.device)
        )
        host_lengths.copy_(torch.tensor(lengths, dtype=torch.int32))
        for q, k, v, _, _, _ in banks:
            q.add_(0.01 * (step + 1))
            k.bitwise_xor_(0x11)
            v.bitwise_xor_(0x22)
        if planner == "host":
            plan.prepare()  # CPU fill/H2D stays outside the compiled consumer/Graph.
        graph.replay()
        torch.cuda.synchronize()
        for bank, workspace in zip(banks, workspaces, strict=True):
            q, k, v, ks, vs, out = bank
            _assert_output(
                out, _reference(q, [k, v], [ks, vs], args[0], lengths, 0.7 / 16, 1.3)
            )
            assert not workspace.semaphores.any()
    explanation = torch._dynamo.explain(forward)(*args)
    targets = [
        str(node.target)
        for g in explanation.graphs
        for node in g.graph.nodes
        if node.op == "call_function"
    ]
    counts = dict(
        plan=sum("experimental_xqa_bucket_plan" in name for name in targets),
        compute=sum("experimental_xqa_bucket_compute" in name for name in targets),
    )
    assert explanation.graph_count == 1 and explanation.graph_break_count == 0
    assert counts == dict(plan=int(not direct and planner == "gpu"), compute=layers)
    print(
        json.dumps(
            dict(
                inductor="default fullgraph",
                direct=direct,
                mode=mode,
                planner=planner,
                layers=layers,
                captured_ops=counts,
                resources=plan.resources,
                changed_inputs="Q/K/V and lengths; zero/restore",
                counters_reset=True,
            )
        ),
        flush=True,
    )
    print(explanation.graphs[0].code, flush=True)


class BucketCompileTests(unittest.TestCase):
    def test_direct_public_inductor(self):
        _compile_check(direct=True, opcheck=True)

    def test_prepared_public_inductor(self):
        for mode in ("persistent", "aligned"):
            _compile_check(mode=mode, opcheck=mode == "aligned")

    def test_host_prepared_consumer(self):
        _compile_check(mode="aligned", planner="host")

    def test_shared_plan_two_layers(self):
        _compile_check(mode="aligned", layers=2)

    def test_storage_guards(self):
        plan, _, banks, args, forward, _, lengths = _fixture(direct=True)
        compiled = torch.compile(forward, fullgraph=True)
        compiled(*args)
        q, k, v, ks, vs, out = banks[0]
        bad = torch.empty(q.numel() + 1, dtype=q.dtype, device=q.device)[1:].view(
            q.shape
        )
        bad.copy_(q)
        with self.assertRaisesRegex(Exception, "alias"):
            forward(args[0], q, k, v, ks, vs, q)
        with self.assertRaisesRegex(Exception, "query.*16-byte"):
            forward(args[0], bad, k, v, ks, vs, out)
        with self.assertRaisesRegex(Exception, "out.*16-byte"):
            forward(args[0], q, k, v, ks, vs, bad)
        original = q.clone()
        behavior = []
        for name, query, output in (
            ("alias", q, q),
            ("query", bad, out),
            ("out", q, bad),
        ):
            q.copy_(original)
            if name == "query":
                bad.copy_(-original)
            if name != "alias":
                output.fill_(float("nan"))
            expected = _reference(
                query.clone(), [k, v], [ks, vs], args[0], lengths, 0.7 / 16, 1.3
            )
            try:
                compiled(args[0], query, k, v, ks, vs, output)
                torch.cuda.synchronize()
            except Exception as error:
                behavior.append(dict(input=name, behavior="rejected", error=str(error)))
            else:
                _assert_output(output, expected)
                behavior.append(
                    dict(input=name, behavior="accepted; numerical check passed")
                )
        print(
            json.dumps(
                dict(
                    eager_storage_guards="all rejected",
                    compiled_storage_behavior=behavior,
                )
            ),
            flush=True,
        )

    def test_output_metadata_and_workspace_aliases(self):
        from flashinfer.experimental.xqa_decode_buckets import backend
        from flashinfer.xqa import xqa_decode_with_buckets

        plan, workspaces, banks, args, _, _, _ = _fixture(mode="aligned")
        workspace = workspaces[0]
        q, k, v, ks, vs, output = banks[0]
        for name in (
            "lengths",
            "tasks",
            "requests",
            "spans",
            "header",
            "causal_mask",
            "scratch",
            "semaphores",
        ):
            with self.subTest(buffer=name):
                owner = workspace if name in ("scratch", "semaphores") else plan
                tensor = getattr(owner, name)
                backing = torch.empty(
                    max(
                        output.numel() * output.element_size(),
                        tensor.numel() * tensor.element_size(),
                    ),
                    dtype=torch.uint8,
                    device=q.device,
                )
                aliased_tensor = (
                    backing[: tensor.numel() * tensor.element_size()]
                    .view(tensor.dtype)
                    .reshape(tensor.shape)
                )
                aliased_tensor.copy_(tensor)
                out = (
                    backing[: output.numel() * output.element_size()]
                    .view(output.dtype)
                    .reshape(output.shape)
                )
                aliased_owner = replace(owner, **{name: aliased_tensor})
                # Rejection must precede JIT lookup or a native kernel launch.
                with (
                    patch.object(
                        backend,
                        "get_module",
                        side_effect=AssertionError(
                            "Aliased output reached kernel lookup"
                        ),
                    ),
                    self.assertRaisesRegex(ValueError, "out.*alias"),
                ):
                    xqa_decode_with_buckets(
                        q,
                        k,
                        v,
                        ks,
                        vs,
                        args[0],
                        aliased_owner if owner is plan else plan,
                        aliased_owner if owner is workspace else workspace,
                        bmm1_scale=0.7 / 16,
                        bmm2_scale=1.3,
                        out=out,
                    )


if __name__ == "__main__":
    unittest.main()

