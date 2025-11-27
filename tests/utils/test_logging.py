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

import os
import sys
import tempfile
from enum import Enum
from pathlib import Path

import pytest
import torch


# Test enum classes
class TestEnum(Enum):
    """Test enum with integer values."""

    OPTION_A = 0
    OPTION_B = 1
    OPTION_C = 2


class StringEnum(Enum):
    """Test enum with string values. Names are for testing purposes."""

    MODE_STANDARD = "standard"
    MODE_OPTIMIZED = "optimized"


class TestAPILogging:
    """Test suite for FlashInfer API logging infrastructure."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Reset environment and reimport logging module for each test."""
        # Store original environment
        original_level = os.environ.get("FLASHINFER_LOGLEVEL")
        original_dest = os.environ.get("FLASHINFER_LOGDEST")

        yield

        # Restore original environment
        if original_level is not None:
            os.environ["FLASHINFER_LOGLEVEL"] = original_level
        elif "FLASHINFER_LOGLEVEL" in os.environ:
            del os.environ["FLASHINFER_LOGLEVEL"]

        if original_dest is not None:
            os.environ["FLASHINFER_LOGDEST"] = original_dest
        elif "FLASHINFER_LOGDEST" in os.environ:
            del os.environ["FLASHINFER_LOGDEST"]

        # Force reimport to pick up new environment variables
        if "flashinfer.api_logging" in sys.modules:
            del sys.modules["flashinfer.api_logging"]

    def setup_logging(self, level: int, dest: str = "stdout"):
        """Helper to set up logging environment and reimport."""
        os.environ["FLASHINFER_LOGLEVEL"] = str(level)
        os.environ["FLASHINFER_LOGDEST"] = dest

        # Force reimport
        if "flashinfer.api_logging" in sys.modules:
            del sys.modules["flashinfer.api_logging"]

        from flashinfer.api_logging import flashinfer_api

        return flashinfer_api

    def test_level_0_zero_overhead(self):
        """Test that level 0 has truly zero overhead (returns original function)."""
        decorator = self.setup_logging(level=0)

        def original_func(x, y):
            return x + y

        decorated_func = decorator(original_func)

        # At level 0, decorator should return the original function unchanged
        assert decorated_func is original_func
        assert decorated_func(5, 3) == 8

    def test_level_1_function_name(self):
        """Test that level 1 logs function name only."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=1, dest=log_file)

            @decorator
            def test_function(x, y):
                return x + y

            result = test_function(10, 20)
            assert result == 30

            # Check log contents
            with open(log_file, "r") as f:
                log_contents = f.read()

            assert "FlashInfer API Call: test_function" in log_contents
            # Level 1 should not log inputs/outputs details
            assert "Positional input arguments" not in log_contents
            assert "Output value" not in log_contents
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_level_3_inputs_outputs(self):
        """Test that level 3 logs inputs and outputs with metadata."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=3, dest=log_file)

            @decorator
            def test_function(tensor, value):
                return tensor * value

            tensor = torch.tensor([1.0, 2.0, 3.0])
            test_function(tensor, 2.0)

            # Check log contents
            with open(log_file, "r") as f:
                log_contents = f.read()

            # Should log function name
            assert "FlashInfer API Call: test_function" in log_contents

            # Should log inputs
            assert "Positional input arguments" in log_contents
            assert "arg[0]" in log_contents
            assert "Tensor(" in log_contents
            assert "shape=(3,)" in log_contents
            assert "dtype=torch.float32" in log_contents

            # Should log outputs
            assert "Output value:" in log_contents

            # Should NOT log statistics (level 5 only)
            assert "min=" not in log_contents
            assert "max=" not in log_contents
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_level_5_statistics(self):
        """Test that level 5 logs tensor statistics."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=5, dest=log_file)

            @decorator
            def test_function(tensor):
                return tensor + 1.0

            tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            test_function(tensor)

            # Check log contents
            with open(log_file, "r") as f:
                log_contents = f.read()

            # Should log statistics
            assert "min=" in log_contents
            assert "max=" in log_contents
            assert "mean=" in log_contents
            assert "nan_count=" in log_contents
            assert "inf_count=" in log_contents
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_enum_logging(self):
        """Test that enum values are logged with name and value."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=3, dest=log_file)

            @decorator
            def test_function(mode: TestEnum, strategy: StringEnum):
                return f"{mode.name}_{strategy.name}"

            test_function(TestEnum.OPTION_B, StringEnum.MODE_OPTIMIZED)

            # Check log contents
            with open(log_file, "r") as f:
                log_contents = f.read()

            # Should show enum name and value
            assert "TestEnum.OPTION_B" in log_contents
            assert "(value=1)" in log_contents
            assert "StringEnum.MODE_OPTIMIZED" in log_contents
            assert (
                "(value=optimized)" in log_contents
                or "(value='optimized')" in log_contents
                or '(value="optimized")' in log_contents
            )
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_default_parameters(self):
        """Test that default parameters are logged separately."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=3, dest=log_file)

            @decorator
            def test_function(x, y=10, z=20, mode=TestEnum.OPTION_A):
                return x + y + z

            # Call with only required argument
            result = test_function(5)
            assert result == 35

            # Check log contents
            with open(log_file, "r") as f:
                log_contents = f.read()

            # Should show default parameters section
            assert "Default parameters (not explicitly provided)" in log_contents
            assert "[DEFAULT]" in log_contents

            # Should show the default values
            assert "y=" in log_contents
            assert "z=" in log_contents
            assert "mode=" in log_contents
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_explicit_vs_default_parameters(self):
        """Test that explicitly provided parameters are not shown in defaults."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=3, dest=log_file)

            @decorator
            def test_function(x, y=10, z=20):
                return x + y + z

            # Call with some explicit parameters
            test_function(5, y=100)

            # Check log contents
            with open(log_file, "r") as f:
                log_contents = f.read()

            # y should be in keyword arguments (explicit)
            assert "Keyword input arguments:" in log_contents

            # Only z should be in defaults
            lines = log_contents.split("\n")
            default_section_started = False
            defaults_found = []
            for line in lines:
                if "Default parameters" in line:
                    default_section_started = True
                if default_section_started and "=" in line and "[DEFAULT]" in line:
                    defaults_found.append(line)

            # Should have only one default parameter (z)
            assert len(defaults_found) == 1
            assert "z=" in defaults_found[0]
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_class_method_logging(self):
        """Test that class methods log with class name."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=1, dest=log_file)

            class TestWrapper:
                @decorator
                def run(self, x):
                    return x * 2

            wrapper = TestWrapper()
            result = wrapper.run(5)
            assert result == 10

            # Check log contents
            with open(log_file, "r") as f:
                log_contents = f.read()

            # Should log class name for Wrapper classes
            assert "TestWrapper.run" in log_contents
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_crash_safety_inputs_logged_before_execution(self):
        """Test that inputs are logged BEFORE execution (crash-safe)."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=3, dest=log_file)

            @decorator
            def crashing_function(x, y):
                raise RuntimeError("Simulated crash")

            # Call the function and expect it to crash
            with pytest.raises(RuntimeError, match="Simulated crash"):
                crashing_function(42, 99)

            # Check that inputs were still logged
            with open(log_file, "r") as f:
                log_contents = f.read()

            # Inputs should be in the log even though function crashed
            assert "FlashInfer API Call: crashing_function" in log_contents
            assert "Positional input arguments" in log_contents
            assert "arg[0]" in log_contents
            assert "42" in log_contents
            assert "arg[1]" in log_contents
            assert "99" in log_contents

            # Outputs should NOT be in the log (function crashed)
            assert "Output value:" not in log_contents
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_different_data_types(self):
        """Test logging of various data types."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=3, dest=log_file)

            @decorator
            def test_function(
                int_val,
                float_val,
                bool_val,
                str_val,
                list_val,
                tuple_val,
                dict_val,
                none_val,
            ):
                return "success"

            test_function(
                42, 3.14, True, "hello", [1, 2, 3], (4, 5, 6), {"key": "value"}, None
            )

            # Check log contents
            with open(log_file, "r") as f:
                log_contents = f.read()

            # Should log all types correctly
            assert "42" in log_contents
            assert "3.14" in log_contents
            assert "True" in log_contents
            assert "'hello'" in log_contents
            assert "None" in log_contents
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_tensor_metadata(self):
        """Test that tensor metadata is logged correctly."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=3, dest=log_file)

            @decorator
            def test_function(tensor):
                return tensor

            # Create a tensor with specific properties
            tensor = torch.randn(2, 3, 4, dtype=torch.float32, device="cpu")
            tensor = tensor.contiguous()
            tensor.requires_grad = False

            test_function(tensor)

            # Check log contents
            with open(log_file, "r") as f:
                log_contents = f.read()

            # Should log all metadata
            assert "shape=(2, 3, 4)" in log_contents
            assert "dtype=torch.float32" in log_contents
            assert "device=cpu" in log_contents
            assert "requires_grad=False" in log_contents
            assert "is_contiguous=True" in log_contents
            assert "stride=" in log_contents
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_nested_structures(self):
        """Test logging of nested data structures."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=3, dest=log_file)

            @decorator
            def test_function(nested):
                return nested

            # Create nested structure
            nested = {
                "list": [1, 2, 3],
                "dict": {"inner": "value"},
                "tuple": (4, 5),
            }

            test_function(nested)

            # Check log contents
            with open(log_file, "r") as f:
                log_contents = f.read()

            # Should handle nested structures
            assert "list" in log_contents
            assert "dict" in log_contents
            assert "tuple" in log_contents
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_decorator_with_and_without_parentheses(self):
        """Test that decorator works both as @decorator and @decorator()."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=1, dest=log_file)

            # Without parentheses
            @decorator
            def func1(x):
                return x + 1

            # With parentheses
            @decorator()
            def func2(x):
                return x + 2

            result1 = func1(10)
            result2 = func2(20)

            assert result1 == 11
            assert result2 == 22

            # Check log contents
            with open(log_file, "r") as f:
                log_contents = f.read()

            assert "func1" in log_contents
            assert "func2" in log_contents
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_multiple_calls_same_function(self):
        """Test that multiple calls to the same function are all logged."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=1, dest=log_file)

            @decorator
            def test_function(x):
                return x

            # Call multiple times
            for i in range(3):
                test_function(i)

            # Check log contents
            with open(log_file, "r") as f:
                log_contents = f.read()

            # Should have 3 log entries
            assert log_contents.count("FlashInfer API Call: test_function") == 3
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_kwargs_logging(self):
        """Test that keyword arguments are logged correctly."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=3, dest=log_file)

            @decorator
            def test_function(a, b, c):
                return a + b + c

            # Call with keyword arguments
            result = test_function(a=1, b=2, c=3)
            assert result == 6

            # Check log contents
            with open(log_file, "r") as f:
                log_contents = f.read()

            # Should log keyword arguments
            assert "Keyword input arguments:" in log_contents
            assert "a=" in log_contents
            assert "b=" in log_contents
            assert "c=" in log_contents
        finally:
            Path(log_file).unlink(missing_ok=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_graph_compatibility(self):
        """Test that level 5 logging is compatible with CUDA graph capture."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=5, dest=log_file)

            @decorator
            def test_cuda_function(tensor):
                return tensor * 2.0

            # Create a CUDA tensor
            tensor = torch.randn(10, 10, device="cuda")

            # Test 1: Normal execution (should have statistics)
            test_cuda_function(tensor)

            with open(log_file, "r") as f:
                log_normal = f.read()

            # Should have statistics in normal execution
            # (unless PyTorch version is too old)
            if hasattr(torch.cuda, "is_current_stream_capturing"):
                # Normal execution should have min/max OR statistics error
                has_stats = "min=" in log_normal or "statistics error" in log_normal
                assert has_stats, "Expected statistics or error in normal execution"

            # Clear log file
            with open(log_file, "w") as f:
                f.write("")

            # Test 2: CUDA graph capture (should skip statistics)
            if hasattr(torch.cuda, "CUDAGraph"):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    test_cuda_function(tensor)

                with open(log_file, "r") as f:
                    log_capture = f.read()

                # Should skip statistics during capture
                assert (
                    "[statistics skipped: CUDA graph capture in progress]"
                    in log_capture
                    or "statistics" not in log_capture
                ), "Expected statistics to be skipped during CUDA graph capture"
        finally:
            Path(log_file).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
