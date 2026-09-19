# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

"""Check the actual compact registration's mapping, mutation, and graph contract."""

import ast
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTIONS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
# Keep the real operator bodies and mutation schemas in an isolated test namespace.
for node in ast.walk(TREE):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = node.value.replace("flashinfer::", "flashinfer_w4a16_contract::")
SEEN = []


def reference_launch(**kwargs):
    SEEN.append(kwargs)
    kwargs["fc2_out"].copy_(
        kwargs["a_input"] * kwargs["swiglu_alpha"] + kwargs["swiglu_beta"]
    )
    for name in ["fc1_out", "activated", "fc1_scratch", "fc2_scratch", "workspace"]:
        kwargs[name].zero_()


NAMESPACE = {"torch": torch, "_w4a16_fused_moe_launch_flat": reference_launch}
SELECTED = [
    "_w4a16_fused_moe_launch_op",
    "_w4a16_fused_moe_launch_fake",
    "_w4a16_fused_moe_launch_compact",
    "_w4a16_fused_moe_launch_compact_fake",
]
exec(
    compile(
        ast.Module(body=[FUNCTIONS[n] for n in SELECTED], type_ignores=[]),
        str(SOURCE),
        "exec",
    ),
    NAMESPACE,
)


def arguments(m=3):
    values = {}
    for i, arg in enumerate(FUNCTIONS["_w4a16_fused_moe_launch_op"].args.args):
        annotation = ast.unparse(arg.annotation)
        if annotation == "torch.Tensor":
            value = torch.full((m, 16), i + 1.0)
        elif annotation == "int":
            value = i + 1
        elif annotation == "float":
            value = i + 0.25
        elif annotation == "str":
            value = arg.arg
        elif annotation == "bool":
            value = bool(i % 2)
        else:
            raise AssertionError(annotation)
        values[arg.arg] = value
    return values


def grouped(values):
    # Derive positional grouping from the operator destructuring, while mapping
    # correctness is checked against the unchanged legacy entry point below.
    function = FUNCTIONS["_w4a16_fused_moe_launch_compact"]
    groups = {
        node.value.id: [values[e.id] for e in node.targets[0].elts]
        for node in function.body
        if isinstance(node, ast.Assign)
    }
    return [
        groups[arg.arg] if arg.arg in groups else values[arg.arg]
        for arg in function.args.args
    ]


class CompactOperatorContract(unittest.TestCase):
    def test_preserves_every_legacy_argument_and_tensor_identity(self):
        values = arguments()
        SEEN.clear()
        torch.ops.flashinfer_w4a16_contract.w4a16_fused_moe_launch(*values.values())
        before = SEEN[-1]
        torch.ops.flashinfer_w4a16_contract.w4a16_fused_moe_launch_compact(
            *grouped(values)
        )
        after = SEEN[-1]
        self.assertEqual(set(before), set(after))
        for key, expected in before.items():
            if isinstance(expected, torch.Tensor):
                self.assertIs(after[key], expected, key)
            else:
                self.assertEqual(after[key], expected, key)

    def test_mutation_schema_tracks_all_six_buffers(self):
        values = arguments()
        original = {
            name: tensor._version
            for name, tensor in values.items()
            if isinstance(tensor, torch.Tensor)
        }
        torch.ops.flashinfer_w4a16_contract.w4a16_fused_moe_launch_compact(
            *grouped(values)
        )
        mutated = {
            "fc1_out",
            "activated",
            "fc2_out",
            "fc1_scratch",
            "fc2_scratch",
            "workspace",
        }
        for name, version in original.items():
            self.assertEqual(values[name]._version > version, name in mutated, name)
        schema = str(
            torch.ops.flashinfer_w4a16_contract.w4a16_fused_moe_launch_compact.default._schema
        )
        self.assertIn("Tensor(a1!)[] buffers", schema)

    def test_fullgraph_aot_dynamic_shapes_preserve_live_buffer_writes(self):
        # Create fixed scalar/list structure outside compilation; the wrapper
        # below only receives runtime tensor inputs and outputs.
        args = grouped(arguments())

        def run(x, out, scratch):
            inputs = [x] + args[0][1:]
            buffers = [scratch[0], scratch[1], out, scratch[2], scratch[3], scratch[4]]
            torch.ops.flashinfer_w4a16_contract.w4a16_fused_moe_launch_compact(
                inputs, buffers, *args[2:]
            )
            return out + 1

        compiled = torch.compile(run, backend="aot_eager", fullgraph=True, dynamic=True)
        for m in [3, 7]:
            x = torch.randn(m, 16)
            out = torch.full_like(x, float("nan"))
            scratch = [torch.ones_like(x) for _ in range(5)]
            actual = compiled(x, out, scratch)
            expected = x * args[-1][1] + args[-1][2]
            torch.testing.assert_close(out, expected)
            torch.testing.assert_close(actual, expected + 1)
            self.assertTrue(all(torch.equal(v, torch.zeros_like(v)) for v in scratch))

    def test_fake_execution_does_not_invoke_real_launcher(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        before = len(SEEN)
        with FakeTensorMode():
            torch.ops.flashinfer_w4a16_contract.w4a16_fused_moe_launch_compact(
                *grouped(arguments())
            )
        self.assertEqual(len(SEEN), before)


if __name__ == "__main__":
    unittest.main()
