# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-control/ABI contracts, runnable without importing Torch or CUTLASS.

Execute the actual host functions with CPU stand-ins. This isolates routing and
binding regressions from GPU compilation/numerics, which need their own lane.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = _ROOT / "flashinfer/attention/prims_ts"
_MLA = _PACKAGE / "mla_decode.py"
_KERNELS = _PACKAGE / "kernels/mla_decode"


def _node(path, name, owner=None):
    nodes = ast.parse(path.read_text()).body
    if owner is not None:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    return next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)


def _function(path, name, namespace=None, owner=None):
    function = copy.deepcopy(_node(path, name, owner))
    function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    scope = {} if namespace is None else dict(namespace)
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


class _Tensor:
    ndim = 4
    shape = (2, 1, 16, 576)
    dtype = "bf16"

    def __getitem__(self, _index):
        return self

    def permute(self, *_axes):
        return self


_TORCH = SimpleNamespace(Tensor=_Tensor, uint8="uint8")


class TestBalancedMLAHostContract(unittest.TestCase):
    def setUp(self):
        self.supported = _function(
            _MLA, "_balanced_mla_mode_is_supported", {"torch": _TORCH}
        )

    def test_qualified_modes_and_new_feature_exclusions(self):
        for page_size in (16, 32, 64, 128):
            with self.subTest(page_size=page_size):
                self.assertTrue(self.supported(page_size=page_size))
        for options in self._unsupported_modes():
            with self.subTest(options=options):
                self.assertFalse(self.supported(**options))

    @staticmethod
    def _unsupported_modes():
        return (
            {"cp_world": 2},
            {"lse_mode": "base2"},
            {"tuning": object()},
            {"qk_rope_head_dim": 0},
            {"kv_lora_rank": 256},
            {"page_size": 4},
            {"page_size": 8},
            {"softmax_scale": _Tensor()},
            {"output_scale": _Tensor()},
        )

    def test_auto_exclusions_do_not_query_gate_or_device(self):
        def unexpected_gate(**_kwargs):
            self.fail(
                "unsupported modes must keep ordinary kernels without a gate/device query"
            )

        auto = _function(
            _MLA,
            "plan_auto",
            {
                "_balanced_mla_mode_is_supported": self.supported,
                "should_use_prims_ts_balanced_mla": unexpected_gate,
            },
            "BatchMLADecodePagedTSWrapper",
        )
        for options in self._unsupported_modes():
            with self.subTest(options=options):
                observed = []
                wrapper = SimpleNamespace(
                    _plan=lambda *args, **kwargs: observed.append((args, kwargs))
                )
                kwargs = dict(
                    device="cuda:0",
                    batch_size=32,
                    num_heads=16,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    page_size=32,
                    max_kv_len=131072,
                    max_seq_len_q=1,
                    packed_query=False,
                    q_data_type="bf16",
                    kv_data_type="bf16",
                    o_data_type="bf16",
                    seq_lens=(128,) * 32,
                )
                kwargs.update(options)
                self.assertFalse(auto(wrapper, **kwargs))
                self.assertEqual(len(observed), 1)
                forwarded = observed[0][1]
                self.assertFalse(forwarded["balanced"])
                self.assertIsNone(forwarded["balanced_seq_lens"])
                for key in (
                    "cp_world",
                    "lse_mode",
                    "tuning",
                    "softmax_scale",
                    "output_scale",
                ):
                    if key in options:
                        self.assertEqual(forwarded[key], options[key])

    def test_forced_exclusions_fail_before_policy_or_allocation(self):
        plan = _function(
            _MLA,
            "_plan",
            {
                "validate_lse_mode": lambda value: value,
                "_balanced_mla_mode_is_supported": self.supported,
            },
            "BatchMLADecodePagedTSWrapper",
        )
        for options in self._unsupported_modes():
            with self.subTest(options=options):
                kwargs = dict(
                    device="cuda:0",
                    batch_size=32,
                    num_qo_heads=16,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    page_size=32,
                    max_kv_len=131072,
                    max_q_len=1,
                    packed_query=False,
                    q_dtype="bf16",
                    kv_dtype="bf16",
                    out_dtype="bf16",
                    balanced=True,
                    balanced_seq_lens=(128,) * 32,
                )
                kwargs.update(options)
                with self.assertRaisesRegex(
                    ValueError, "unsupported balanced MLA mode"
                ):
                    plan(SimpleNamespace(), **kwargs)

    def test_native_ffi_prefix_and_balanced_descriptor_suffix(self):
        launch = _function(_MLA, "_launch_mla_decode", {"torch": _TORCH})
        tensor = _Tensor()
        runtime = SimpleNamespace(
            query=tensor,
            normalized_cache=tensor,
            out=tensor,
            extra_cache=None,
            sparse_inputs=object(),
            softmax_scale=0.125,
            output_scale=0.25,
            output_quant_args=object(),
        )
        descriptors = tuple(object() for _ in range(4))
        balanced = SimpleNamespace(
            **dict(
                zip(
                    (
                        "work_descriptors",
                        "partition_offsets",
                        "combine_descriptors",
                        "num_combine_descriptors",
                    ),
                    descriptors,
                    strict=True,
                )
            )
        )
        for plan in (None, balanced):
            with self.subTest(balanced=plan is not None):
                calls = []
                result = launch(
                    runtime,
                    block_tables=SimpleNamespace(transpose=lambda *_: tensor),
                    seq_lens_kv=tensor,
                    qo_indptr=None,
                    packed_query=False,
                    kv_lora_rank=512,
                    split_kv=4,
                    workspace=SimpleNamespace(lse=tensor, kernel_workspace=object()),
                    compiled=lambda *args: calls.append(args),
                    max_q_len=1,
                    balanced_plan=plan,
                )
                self.assertIs(result, tensor)
                args = calls[0]
                self.assertEqual(len(args), 21)
                self.assertIs(args[11], runtime.sparse_inputs)
                self.assertEqual(args[12:14], (0.125, 0.25))
                self.assertIs(args[14], runtime.output_quant_args)
                self.assertEqual(args[15:17], (1, 0))
                if plan is not None:
                    self.assertEqual(args[17:], descriptors)
                else:
                    self.assertEqual(args[17:], (None, None, None, None))

    def test_task_cache_preserves_native_slots(self):
        make_cache = _function(
            _KERNELS / "throughput_latency_1cta/tasks.py",
            "make_task_cache",
            {"Int32": int},
            "MlaDecodeTask",
        )
        task = SimpleNamespace(
            _tmem_base_offset=0,
            _warp_grp_thread_idx=1,
            _local_warp_idx=2,
            _lane_idx=3,
            _seq_len_kv=4,
            _local_kv_tiles=5,
            _k_tile_start=6,
            _partial_idx=7,
            _split_count=8,
            _descriptor_tile_count=9,
        )
        task.cfg = SimpleNamespace(use_balanced_scheduler=False)
        self.assertEqual(make_cache(task), (0, 1, 2, 3, 4, 5, 0, 0))
        task.cfg.use_balanced_scheduler = True
        self.assertEqual(make_cache(task), tuple(range(10)))

    def test_shared_kernel_call_signatures_keep_native_prefix(self):
        expected = (
            "self",
            "q_latent",
            "q_rope",
            "c_latent",
            "c_rope",
            "page_offsets",
            "o",
            "lse",
            "workspace",
            "split_kv",
            "cache_seqs",
            "cu_seqlens_q",
            "block_split_kvs",
            "softmax_scale",
            "output_scale",
            "stream",
            "output_quant_args",
            "max_q_len",
            "cp_rank",
        )
        for family, owner in (
            ("throughput_2cta", "MlaDecodeTs"),
            ("throughput_latency_1cta", "ThroughputLatencyMlaDecodeTs"),
        ):
            with self.subTest(family=family):
                function = _node(_KERNELS / family / "kernel.py", "__call__", owner)
                names = tuple(arg.arg for arg in function.args.args)
                self.assertEqual(names[: len(expected)], expected)
                self.assertEqual(
                    names[len(expected) :],
                    (
                        "balanced_work_descriptors",
                        "balanced_partition_offsets",
                        "balanced_combine_descriptors",
                        "balanced_num_combine_descriptors",
                    ),
                )

    def test_shared_reducer_keeps_ordinary_abi(self):
        ordinary = _node(
            _KERNELS / "throughput_2cta/reduction.py", "run_reduction_kernel"
        )
        self.assertFalse(
            any(arg.arg.startswith("balanced_") for arg in ordinary.args.args)
        )
        self.assertIn("cp_rank", [arg.arg for arg in ordinary.args.args])
        _node(
            _KERNELS / "throughput_2cta/reduction.py", "run_balanced_reduction_kernel"
        )

    def test_balanced_fixed_query_extent_is_checked_on_trusted_runs(self):
        run = _function(_MLA, "run", owner="BatchMLADecodePagedTSWrapper")
        state = SimpleNamespace(balanced_plan=object(), packed_query=False, max_q_len=2)
        wrapper = SimpleNamespace(_plan_state=state)
        for validate in (True, False):
            with (
                self.subTest(validate=validate),
                self.assertRaisesRegex(ValueError, "balanced fixed-query runs"),
            ):
                run(
                    wrapper,
                    _Tensor(),
                    _Tensor(),
                    _Tensor(),
                    _Tensor(),
                    validate=validate,
                )

    @staticmethod
    def _metadata_validator():
        namespace = {
            "_validate_mla_metadata": lambda *_: ("cpu", 2, 1),
            "_validate_qo_indptr": lambda *_, **__: None,
            "_derive_max_seq_len_q": lambda *_, **__: (8, 11, (3, 8)),
            "_mla_cp_local_length": _function(_MLA, "_mla_cp_local_length"),
            "_ceil_div": _function(_MLA, "_ceil_div"),
        }
        validate = _function(_MLA, "_validate_mla_run_metadata", namespace)
        runtime = SimpleNamespace(
            query=SimpleNamespace(shape=(11, 96, 576)), num_physical_pages=2
        )
        options = dict(
            device="cpu",
            batch_size=2,
            required_page_columns=1,
            packed_query=True,
            max_q_len=8,
            max_kv_len=128,
            page_size=128,
            mask_type="causal",
        )
        return validate, runtime, options

    def test_ordinary_cp_metadata_does_not_require_balanced_state(self):
        validate, runtime, options = self._metadata_validator()
        tables = SimpleNamespace(tolist=lambda: ((-1,), (1,)))
        lengths = SimpleNamespace(tolist=lambda: (131, 1031))
        self.assertEqual(
            validate(
                runtime,
                tables,
                lengths,
                object(),
                cp_world=8,
                cp_interleave_size=128,
                cp_rank=7,
                **options,
            ),
            8,
        )
        with self.assertRaisesRegex(ValueError, "local request longer than max_kv_len"):
            validate(
                runtime,
                tables,
                lengths,
                object(),
                cp_world=8,
                cp_interleave_size=128,
                cp_rank=0,
                **options,
            )
        # Main's nonpositive inactive-request contract is unchanged without CP.
        lengths = SimpleNamespace(tolist=lambda: (0, -1))
        self.assertEqual(validate(runtime, tables, lengths, object(), **options), 8)

    def test_balanced_metadata_requires_matching_snapshot(self):
        validate, runtime, options = self._metadata_validator()
        tables = SimpleNamespace(tolist=lambda: ((0,), (-1,)))
        lengths = SimpleNamespace(tolist=lambda: (128, 0))
        state = SimpleNamespace(balanced_plan=object(), planned_seq_lens=None)
        with self.assertRaisesRegex(RuntimeError, "snapshot is unavailable"):
            validate(runtime, tables, lengths, object(), state=state, **options)
        state.planned_seq_lens = (128, 128)
        with self.assertRaisesRegex(ValueError, "must exactly match"):
            validate(runtime, tables, lengths, object(), state=state, **options)
        state.planned_seq_lens = (128, 0)
        self.assertEqual(
            validate(runtime, tables, lengths, object(), state=state, **options), 8
        )
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            validate(
                runtime,
                tables,
                SimpleNamespace(tolist=lambda: (128, -1)),
                object(),
                state=state,
                **options,
            )

    def test_calibration_uses_current_private_planner_keywords(self):
        accepted = {
            arg.arg
            for arg in _node(
                _MLA, "_plan", "BatchMLADecodePagedTSWrapper"
            ).args.kwonlyargs
        }
        tuner = _PACKAGE / "balanced_scheduler/tune_cost_model.py"
        calls = [
            n
            for n in ast.walk(ast.parse(tuner.read_text()))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_plan"
        ]
        self.assertTrue(calls)
        for call in calls:
            self.assertFalse(
                {kw.arg for kw in call.keywords if kw.arg is not None} - accepted
            )


if __name__ == "__main__":
    unittest.main()
