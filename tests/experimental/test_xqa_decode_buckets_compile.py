# SPDX-License-Identifier: Apache-2.0
"""Real dispatcher/Inductor qualification of the public bucket API; no model claims."""

import json
from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_xqa_decode_buckets import (
    _assert_output,
    _case,
    _check_changed_graph,
    _reference,
    _require_sm120,
)


def _fixture(*, direct=False, mode="persistent", planner="gpu", layers=1):
    """Return a warmed public consumer with fixed plan/workspace addresses.

    Layer banks share compatible metadata but retain independent writable
    buffers. Keep host lengths available for later CPU refreshes outside Graphs.
    """
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
        """Refresh GPU descriptors once, then consume them across the layer banks."""
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
    """Qualify schema/FakeTensor and real Inductor fullgraph Graph replay.

    Mutate Q/K/V and lengths through zero/restore generations, compare every
    output with reference, and require one graph with the expected plan/compute
    operator counts. This checks integration behavior, not model performance.
    """
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


class BucketReviewCPUTests(unittest.TestCase):
    """CPU adversarial checks for storage ownership, scalar bounds and reporting."""

    @staticmethod
    def _operands():
        """Build disjoint aligned byte buffers for address checks without GPU work."""
        return {
            name: torch.empty(64, dtype=torch.uint8, device="cpu")
            for name in (
                "q",
                "k",
                "v",
                "ks",
                "vs",
                "table",
                "lengths",
                "tasks",
                "requests",
                "spans",
                "header",
                "causal_mask",
                "output",
                "scratch",
                "semaphores",
            )
        }

    @staticmethod
    def _example():
        """Load the actual CLI module for cheap report and numerical-check tests."""
        path = Path(__file__).resolve().parents[2] / "examples/xqa_decode_buckets.py"
        spec = importlib.util.spec_from_file_location("xqa_bucket_example", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_external_storage_overlap(self):
        """Catch byte overlap between BF16 wrappers with different storage bases."""
        from flashinfer.experimental.xqa_decode_buckets import backend

        operands = self._operands()
        backing = torch.empty(256, dtype=torch.uint8, device="cpu").numpy()
        operands["q"] = torch.from_numpy(backing[32:128]).view(torch.bfloat16)
        operands["output"] = torch.from_numpy(backing[96:192]).view(torch.bfloat16)
        self.assertNotEqual(
            operands["q"].untyped_storage().data_ptr(),
            operands["output"].untyped_storage().data_ptr(),
        )
        with self.assertRaisesRegex(ValueError, "out.*alias"):
            backend._storage_checks(**operands)

    def test_writable_buffer_overlaps(self):
        """Reject every write/read pairing and all three writable-buffer pairings."""
        from flashinfer.experimental.xqa_decode_buckets import backend

        readonly = (
            "q",
            "k",
            "v",
            "ks",
            "vs",
            "table",
            "lengths",
            "tasks",
            "requests",
            "spans",
            "header",
            "causal_mask",
        )
        for writable in ("output", "scratch", "semaphores"):
            for source in readonly:
                with self.subTest(writable=writable, source=source):
                    operands = self._operands()
                    backing = torch.empty(128, dtype=torch.uint8, device="cpu")
                    operands[source] = backing[:64]
                    operands[writable] = backing[32:96]
                    with self.assertRaisesRegex(ValueError, "alias"):
                        backend._storage_checks(**operands)
        for first, second in (
            ("output", "scratch"),
            ("output", "semaphores"),
            ("scratch", "semaphores"),
        ):
            with self.subTest(first=first, second=second):
                operands = self._operands()
                backing = torch.empty(128, dtype=torch.uint8, device="cpu")
                operands[first], operands[second] = backing[:64], backing[32:96]
                with self.assertRaisesRegex(ValueError, "alias"):
                    backend._storage_checks(**operands)

    def test_disjoint_subregions_and_empty_descriptors(self):
        """Allow disjoint suballocations, empty descriptors and read-only interleaving."""
        from flashinfer.experimental.xqa_decode_buckets import backend

        operands = self._operands()
        backing = torch.empty(256, dtype=torch.uint8, device="cpu")
        operands.update(
            q=backing[:64],
            output=backing[64:128],
            scratch=backing[128:192],
            semaphores=backing[192:],
        )
        for name in ("tasks", "requests", "spans", "header", "causal_mask"):
            operands[name] = backing[:0]
        cache = torch.empty((4, 2, 16), dtype=torch.uint8, device="cpu")
        operands.update(k=cache[:, 0], v=cache[:, 1], ks=cache[:, 0], vs=cache[:, 1])
        backend._storage_checks(**operands)

    def test_strided_input_extent(self):
        """Catch overlap at a strided input's tail beyond its logical element count."""
        from flashinfer.experimental.xqa_decode_buckets import backend

        operands = self._operands()
        backing = torch.empty(256, dtype=torch.uint8, device="cpu")
        operands["k"] = backing.as_strided((4, 16), (64, 1))
        operands["output"] = torch.from_numpy(backing.numpy()[192:256])
        with self.assertRaisesRegex(ValueError, "out.*alias"):
            backend._storage_checks(**operands)

    def test_scale_validation(self):
        """Accept float32 endpoints and reject overflow/nonfinite scales before dispatch."""
        from flashinfer.experimental.xqa_decode_buckets import backend

        maximum = torch.finfo(torch.float32).max
        backend._check_scales(maximum, -maximum)
        backend._check_scales(0.0, 1.3)
        for invalid in (1e300, -1e300, float("inf"), -float("inf"), float("nan")):
            for scales in ((invalid, 1.0), (1.0, invalid)):
                with (
                    self.subTest(scales=scales),
                    self.assertRaisesRegex(ValueError, "finite float32"),
                ):
                    backend._compute(
                        **self._operands(),
                        ctas=1,
                        aligned=False,
                        bmm1_scale=scales[0],
                        bmm2_scale=scales[1],
                        enable_pdl=False,
                        page_size=32,
                        head_group_ratio=6,
                        q_len_per_req=1,
                        query_tile=16,
                    )

    def test_single_case_json(self):
        """Persist ordinary CLI JSON atomically while mocking only expensive attention."""
        example = self._example()
        result = dict(relative_error=[0.01])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "single.json"
            with patch.object(example, "run_case", return_value=result) as run:
                self.assertEqual(
                    example.main(["--lengths", "32", "--output-json", str(target)]),
                    result,
                )
            run.assert_called_once()
            self.assertEqual(
                json.loads(target.read_text()), dict(complete=True, result=result)
            )
            self.assertFalse(target.with_suffix(".json.partial").exists())

    def test_nonfinite_output_and_reference(self):
        """Reject invalid references and NaN errors, including finite-input norm overflow."""
        example = self._example()
        reference = torch.ones(4, device="cpu")
        self.assertLess(
            example.checked_relative_error(reference * 1.02, reference), 0.03
        )
        for invalid in (float("nan"), float("inf"), -float("inf")):
            value = torch.full_like(reference, invalid)
            for output, expected in ((value, reference), (reference, value)):
                with (
                    self.subTest(invalid=invalid),
                    self.assertRaisesRegex(RuntimeError, "finite"),
                ):
                    example.checked_relative_error(output, expected)
        # Finite inputs can still produce a NaN error through overflowing norms.
        huge = torch.full_like(reference, 1e38)
        with self.assertRaisesRegex(RuntimeError, "relative error"):
            example.checked_relative_error(-huge, huge)
        with self.assertRaisesRegex(RuntimeError, "relative error"):
            example.checked_relative_error(reference * 1.04, reference)

    def test_small_sm_direct_fixture_skips_before_allocation(self):
        """Skip an ineligible direct geometry before allocating any CUDA fixture."""
        from flashinfer.experimental.xqa_decode_buckets import backend

        with (
            patch("test_xqa_decode_buckets._require_sm120"),
            patch(
                "test_xqa_decode_buckets._case",
                side_effect=AssertionError("fixture allocated"),
            ),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(backend, "get_device_sm_count", return_value=1),
            self.assertRaisesRegex(
                unittest.SkipTest, "Direct fixture exceeds SM budget"
            ),
        ):
            _check_changed_graph(
                "aligned",
                False,
                direct=True,
                direct_batch=4,
                capacity_length=512,
                num_kv_heads=4,
                q_len_per_req=3,
                query_tile=32,
            )


class BucketReviewSM120Tests(unittest.TestCase):
    """Real-device regressions for default-device isolation and native scalar checks."""

    def test_default_cuda_device_host_plan_and_interleaved_graph(self):
        """Keep cold host allocations on CPU under a scoped CUDA default device.

        Exercise native resource configuration after clearing the Python module
        cache, pinned host planning and suffix-mask construction, then validate
        an interleaved public Graph.
        """
        _require_sm120()
        from flashinfer.experimental.xqa_decode_buckets import jit
        from flashinfer.xqa import (
            create_xqa_decode_bucket_plan,
            xqa_decode_with_buckets,
        )

        lengths = [257, 512, 3, 0]
        queries, data, scales, table, lens = _case(
            lengths,
            capacity_length=8193,
            interleaved=True,
            num_kv_heads=4,
            q_len_per_req=3,
        )
        host_lengths = torch.tensor(lengths, dtype=torch.int32, device="cpu")
        with torch.device(lens.device):
            jit.get_module.cache_clear()  # Exercise cold configure's CPU resources.
            with patch.object(torch, "tensor", wraps=torch.tensor) as tensor_factory:
                plan = create_xqa_decode_bucket_plan(
                    lens,
                    page_table_width=table.shape[1],
                    planner="host",
                    host_seq_lens=host_lengths,
                    mode="aligned",
                    num_kv_heads=4,
                    q_len_per_req=3,
                    query_tile=32,
                )
            tensor_factory.assert_any_call((1, 3, 7), dtype=torch.int32, device="cpu")
            self.assertFalse(plan.direct)
            self.assertEqual(plan._host_descriptors.device.type, "cpu")
            self.assertTrue(plan._host_descriptors.is_pinned())
            self.assertEqual(plan.causal_mask.device, lens.device)
            self.assertEqual(len(plan.resources), 4)
            workspace = plan.create_workspace()
            output = torch.empty_like(queries[0])

            def compute():
                """Replay one prepared host plan without refreshing it inside capture."""
                return xqa_decode_with_buckets(
                    queries[0],
                    *data,
                    *scales,
                    table,
                    plan,
                    workspace,
                    bmm1_scale=0.7 / 16,
                    bmm2_scale=1.3,
                    out=output,
                )

            compute()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                compute()
            graph.replay()
            torch.cuda.synchronize()
            _assert_output(
                output,
                _reference(queries[0], data, scales, table, lengths, 0.7 / 16, 1.3),
            )
            self.assertFalse(workspace.semaphores.any())

    def test_scale_overflow_public_registered_and_native(self):
        """Reject invalid double scales through public, registered and raw FFI paths."""
        from flashinfer.experimental.xqa_decode_buckets import backend
        from flashinfer.xqa import xqa_decode_with_buckets

        plan, workspaces, banks, args, _, _, _ = _fixture(direct=True)
        workspace = workspaces[0]
        q, k, v, ks, vs, output = banks[0]
        prefix = (
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
            output,
            plan.ctas,
            plan.mode == "aligned",
        )
        module, _ = backend.get_module(False, q.device, direct=True)
        for invalid in (1e300, -1e300, float("inf"), float("nan")):
            for scales in ((invalid, 1.0), (1.0, invalid)):
                with self.subTest(scales=scales):
                    with patch.object(
                        backend,
                        "get_module",
                        side_effect=AssertionError("invalid scale reached JIT"),
                    ):
                        with self.assertRaisesRegex(ValueError, "finite float32"):
                            xqa_decode_with_buckets(
                                q,
                                k,
                                v,
                                ks,
                                vs,
                                args[0],
                                plan,
                                workspace,
                                bmm1_scale=scales[0],
                                bmm2_scale=scales[1],
                                out=output,
                            )
                        with self.assertRaisesRegex(ValueError, "finite float32"):
                            backend._compute_op(
                                *prefix,
                                *scales,
                                plan.enable_pdl,
                                plan.page_size,
                                plan.head_group_ratio,
                                plan.q_len_per_req,
                                plan.query_tile,
                            )
                    with self.assertRaisesRegex(
                        (ValueError, RuntimeError), "finite float32"
                    ):
                        module.run(
                            *prefix, *scales, plan.enable_pdl, plan.q_len_per_req
                        )


class BucketCompileTests(unittest.TestCase):
    """Dispatcher, compiler and alias-boundary coverage of the public bucket API."""

    def test_direct_public_inductor(self):
        """Keep direct compute visible as a real fullgraph operator through replay."""
        _compile_check(direct=True, opcheck=True)

    def test_prepared_public_inductor(self):
        """Capture assignment plus compute as one graph for both task schedulers."""
        for mode in ("persistent", "aligned"):
            _compile_check(mode=mode, opcheck=mode == "aligned")

    def test_host_prepared_consumer(self):
        """Compile the host-plan consumer while leaving CPU/H2D refresh outside it."""
        _compile_check(mode="aligned", planner="host")

    def test_shared_plan_two_layers(self):
        """Require one captured assignment to feed two independent layer workspaces."""
        _compile_check(mode="aligned", layers=2)

    def test_storage_guards(self):
        """Reject eager aliases/misalignment and verify compiled rejection or safe output.

        Poison outputs and vary the offending query so accepted compiled calls
        must prove numerical correctness rather than reuse a stale eager result.
        """
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
        """Reject eight output/metadata-workspace aliases before native kernel lookup."""
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
