"""CPU checks for validation and launch ordering; runnable without CUDA imports."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

_PATH = (
    Path(__file__).resolve().parents[3]
    / "flashinfer/experimental/nemotron_mxfp8/plan.py"
)
_SPEC = importlib.util.spec_from_file_location("nemotron_plan_test.plan", _PATH)
plan_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = plan_module
_SPEC.loader.exec_module(plan_module)


class ContractTest(unittest.TestCase):
    def test_reject_invalid_configuration_before_cuda(self):
        for overrides in (
            {"max_tokens": 0},
            {"max_tokens": -1},
            {"max_tokens": 2**31},
            {"local_expert_offset": 1},
            {"workers": 0},
            {"group_hint": -1},
            {"hardware_tanh": "1"},
        ):
            args = dict(
                weights=None,
                max_tokens=32,
                local_expert_offset=0,
                workers=152,
                group_hint=1024,
                hardware_tanh=True,
            )
            args.update(overrides)
            with self.assertRaises(ValueError):
                plan_module.LocalExpertPlan(**args)

    def test_tensor_metadata_rejects_strides_dtype_and_shape(self):
        x = torch.zeros(4, 8)
        plan_module._tensor(x, (4, 8), torch.float32, x.device, "x")
        for bad in (x.t(), x[:, ::2], x.double(), x[:2]):
            with self.assertRaises(ValueError):
                plan_module._tensor(bad, (4, 8), torch.float32, x.device, "x")

    def test_setup_requires_selected_device_and_no_capture(self):
        with patch.object(torch.cuda, "current_device", return_value=0):
            with self.assertRaisesRegex(ValueError, "select"):
                plan_module._check_device(torch.device("cuda:1"))
            with (
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=True
                ),
                self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"),
            ):
                plan_module._check_device(torch.device("cuda:0"))

    def test_closed_plan_cannot_run_or_borrow(self):
        plan = object.__new__(plan_module.LocalExpertPlan)
        plan._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            plan.run(None, None, None, out=None)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            plan.contribution_view()
        plan.close()

    def test_plan_preserves_stage_compute_assembly_order_and_current_stream(self):
        # Test the real orchestration without substituting a research frontend.
        plan = object.__new__(plan_module.LocalExpertPlan)
        plan._closed = False
        plan.device = torch.device("cpu")
        plan.max_tokens = 2
        plan.local_expert_offset = 256
        plan._local_ids, plan._local_probs = object(), object()
        plan._terms = object()
        plan._stage_tail = ("stage buffers",)
        plan._args = {"workspace": object()}
        plan._input_view = None
        calls = []
        plan._stage = lambda *args: calls.append(("stage", args))
        plan._compiled = lambda **kwargs: calls.append(("compute_and_tail", kwargs))
        glue = types.ModuleType("glue")
        glue.route_mask = lambda *args: calls.append(("mask", args))
        glue.weighted_partial = lambda *args: calls.append(("assemble", args))
        cuda = types.ModuleType("cuda")
        bindings = types.ModuleType("cuda.bindings")
        driver = types.ModuleType("cuda.bindings.driver")
        driver.CUstream = lambda handle: handle
        cuda.bindings, bindings.driver = bindings, driver
        modules = {
            "cuda": cuda,
            "cuda.bindings": bindings,
            "cuda.bindings.driver": driver,
            "nemotron_plan_test.megamoe_kernel_mxfp8": glue,
        }
        x = torch.zeros(2, 768, dtype=torch.bfloat16)
        ids = torch.zeros(2, 10, dtype=torch.int64)
        probs, out = torch.zeros(2, 10), torch.empty(2, 768)
        with (
            patch.dict(sys.modules, modules),
            patch.object(plan_module, "__package__", "nemotron_plan_test"),
            patch.object(plan_module, "_cute", side_effect=lambda x: x),
            patch.object(torch.cuda, "current_device", return_value=None),
            patch.object(
                torch.cuda,
                "current_stream",
                return_value=types.SimpleNamespace(cuda_stream=17),
            ),
        ):
            self.assertIs(plan.run(x, ids, probs, out=out), out)
        self.assertEqual(
            [c[0] for c in calls], ["mask", "stage", "compute_and_tail", "assemble"]
        )
        self.assertEqual(calls[0][1][-1], 2)
        self.assertEqual(calls[1][1][-1], 17)
        self.assertEqual(calls[2][1]["stream"], 17)
        self.assertIs(calls[3][1][-1], out)


if __name__ == "__main__":
    unittest.main()
