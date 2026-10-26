"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""CPU-only activation contracts; no CUDA import or artifact download required."""

import ast
from enum import IntEnum
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class Activation(IntEnum):
    Swiglu = 3
    SwigluStep = 7
    ClampedRelu2 = 12


class Quantization(IntEnum):
    MxFp8 = 1
    DeepSeekFp8 = 2
    PerTensorFp8 = 3


class ClampedRelu2ContractTest(unittest.TestCase):
    def setUp(self):
        namespace = {
            "torch": SimpleNamespace(float32="float32", Size=tuple),
            "ActivationType": Activation,
            "Fp8QuantizationType": Quantization,
        }
        load_function(
            "flashinfer/fused_moe/core.py", "_validate_clamped_relu2_params", namespace
        )
        self.validate = load_function(
            "flashinfer/fused_moe/core.py",
            "_validate_fp8_block_scale_gemm1_activation_params",
            namespace,
        )
        load_function("flashinfer/utils.py", "check_shape_dtype_device", namespace)
        namespace["_validate_shared_bf16_gemm1_activation_params"] = load_function(
            "flashinfer/fused_moe/shared/validation.py",
            "validate_bf16_gemm1_activation_params",
            namespace,
        )
        self.validate_bf16 = load_function(
            "flashinfer/fused_moe/core.py",
            "_validate_bf16_gemm1_activation_params",
            namespace,
        )
        self.scale = SimpleNamespace(
            dtype="float32",
            ndim=1,
            is_cuda=True,
            is_contiguous=lambda: True,
            shape=(4,),
            device="cuda:0",
        )

    def test_valid_scale_does_not_require_device_value_readback(self):
        # The stand-in intentionally has no item(), cpu(), or tensor arithmetic.
        self.validate(
            Quantization.MxFp8, Activation.ClampedRelu2, None, None, self.scale
        )

    def test_rejects_unsupported_quantization_and_parameters(self):
        for quant, alpha, beta, limit in (
            (2, None, None, self.scale),
            (3, None, None, self.scale),
            (1, None, None, None),
            (1, self.scale, None, self.scale),
            (1, None, self.scale, self.scale),
        ):
            with (
                self.subTest(quant=quant, alpha=alpha, beta=beta, limit=limit),
                self.assertRaises(ValueError),
            ):
                self.validate(quant, Activation.ClampedRelu2, alpha, beta, limit)

    def test_rejects_scale_metadata(self):
        for field, value in (
            ("dtype", "bfloat16"),
            ("ndim", 0),
            ("is_cuda", False),
            ("is_contiguous", lambda: False),
        ):
            scale = SimpleNamespace(**vars(self.scale))
            setattr(scale, field, value)
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "float32"),
            ):
                self.validate(1, Activation.ClampedRelu2, None, None, scale)

    def test_existing_activation_contracts(self):
        self.validate(2, Activation.Swiglu, self.scale, self.scale, self.scale)
        self.validate(1, Activation.SwigluStep, None, None, self.scale)
        with self.assertRaises(ValueError):
            self.validate(2, Activation.SwigluStep, None, None, self.scale)

    def test_bf16_scale_contract(self):
        self.validate_bf16(12, None, None, self.scale, 4, "cuda:0")
        for scale, count, device in (
            (None, 4, "cuda:0"),
            (self.scale, 3, "cuda:0"),
            (self.scale, 4, "cuda:1"),
        ):
            with (
                self.subTest(scale=scale, count=count, device=device),
                self.assertRaises(ValueError),
            ):
                self.validate_bf16(12, None, None, scale, count, device)
        for alpha, beta in ((self.scale, None), (None, self.scale)):
            with self.assertRaises(ValueError):
                self.validate_bf16(12, alpha, beta, self.scale, 4, "cuda:0")
        self.validate_bf16(3, self.scale, self.scale, self.scale, 4, "cuda:0")
        self.validate_bf16(7, None, None, self.scale, 4, "cuda:0")
        with self.assertRaises(ValueError):
            self.validate_bf16(7, self.scale, None, self.scale, 4, "cuda:0")

    def test_export_feature_detection(self):
        supports = load_function(
            "flashinfer/jit/fused_moe.py", "_has_trtllm_clamped_relu2", {"re": re}
        )
        self.assertTrue(
            supports(b"enum class EltwiseActType : int {None, Relu2, ClampedRelu2};")
        )
        self.assertFalse(supports(b"enum class EltwiseActType {None, Relu2};"))
        self.assertFalse(
            supports(b"// ClampedRelu2\nenum class EltwiseActType {Relu2};")
        )
        self.assertFalse(supports(b"enum class ActType {ClampedRelu2};"))
        self.assertFalse(
            supports(b"enum class EltwiseActType {/* ClampedRelu2 */ Relu2};")
        )


if __name__ == "__main__":
    unittest.main()
