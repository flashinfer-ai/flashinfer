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
import json
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
        env_keys = [
            "FLASHINFER_LOGLEVEL",
            "FLASHINFER_LOGDEST",
            "FLASHINFER_DUMP_DIR",
        ]
        original_env = {key: os.environ.get(key) for key in env_keys}

        yield

        module = sys.modules.get("flashinfer.api_logging")
        if module is not None and hasattr(module, "_restore_cuda_graph_hooks"):
            module._restore_cuda_graph_hooks()

        # Restore original environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

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
    def test_cuda_graph_compatibility(self, capfd):
        """Level-5 logging produces stats both in eager mode and during CUDA
        graph capture/replay. During capture the host log records a correlation
        id; during replay the captured kernel emits the actual statistics via
        device-side printf.
        """
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            log_file = f.name

        try:
            decorator = self.setup_logging(level=5, dest=log_file)
            from flashinfer.api_logging import _get_api_log_stats_kernel

            assert _get_api_log_stats_kernel() is not None

            @decorator
            def test_cuda_function(tensor):
                return tensor * 2.0

            tensor = torch.randn(10, 10, device="cuda")

            # Test 1: Normal execution (should have statistics)
            test_cuda_function(tensor)

            with open(log_file, "r") as f:
                log_normal = f.read()

            if hasattr(torch.cuda, "is_current_stream_capturing"):
                has_stats = "min=" in log_normal or "statistics error" in log_normal
                assert has_stats, "Expected statistics or error in normal execution"

            # Clear log file
            with open(log_file, "w") as f:
                f.write("")

            # Test 2: CUDA graph capture should record a correlation id rather
            # than skip stats; replaying the graph should make the captured
            # kernel print "[flashinfer stats] id=N ...".
            if hasattr(torch.cuda, "CUDAGraph"):
                # Warmup so graph capture starts from a clean state.
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    test_cuda_function(tensor)
                torch.cuda.current_stream().wait_stream(s)

                # Reset log to capture only the graph-capture / replay output.
                with open(log_file, "w") as f:
                    f.write("")

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    test_cuda_function(tensor)

                with open(log_file, "r") as f:
                    log_capture = f.read()

                assert (
                    "[statistics skipped: CUDA graph capture in progress]"
                    not in log_capture
                )
                assert log_capture.count("[stats deferred to GPU kernel: id=") >= 2

                # Replaying should emit device-side printf lines on every
                # replay. Clear any prior captured output, mutate the input
                # before each replay, and require both replay values to appear.
                capfd.readouterr()
                tensor.fill_(3.0)
                graph.replay()
                torch.cuda.synchronize()
                tensor.fill_(4.0)
                graph.replay()
                torch.cuda.synchronize()
                replay_stdout = capfd.readouterr().out
                assert replay_stdout.count("[flashinfer stats] id=") >= 4
                assert "min=3.000000" in replay_stdout
                assert "min=4.000000" in replay_stdout
        finally:
            Path(log_file).unlink(missing_ok=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_level_10_cuda_graph_dumps(self, tmp_path):
        """Level 10 dumps work under CUDA graph capture and repeated flushes
        preserve every replay snapshot instead of only the latest values.
        """
        import sys

        # Configure level 10 dump dir before importing api_logging.
        os.environ["FLASHINFER_LOGLEVEL"] = "10"
        os.environ["FLASHINFER_LOGDEST"] = "stderr"
        os.environ["FLASHINFER_DUMP_DIR"] = str(tmp_path / "fi_dumps")
        sys.modules.pop("flashinfer.api_logging", None)
        from flashinfer.api_logging import (
            flashinfer_api,
            clear_graph_dumps,
        )

        @flashinfer_api
        def _add(x, y):
            return x + y

        x = torch.full((4, 4), 1.0, device="cuda")
        y = torch.full((4, 4), 2.0, device="cuda")

        # An eager dump before capture should not interfere with captured dumps.
        _add(x, y)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _ = _add(x, y)

        # Find the dump directory created during capture (the most recent one).
        dump_root = tmp_path / "fi_dumps"
        dump_dirs = sorted(
            (d for d in dump_root.iterdir() if d.is_dir()), key=lambda p: p.name
        )
        assert dump_dirs, "No dump directory was created during capture"
        capture_dir = dump_dirs[-1]

        # Before flush, only metadata.jsonl should exist for the captured call.
        assert (capture_dir / "metadata.jsonl").exists()
        assert not (capture_dir / "inputs.pt").exists()
        assert not (capture_dir / "outputs.pt").exists()

        # Replay 1: x=1, y=2 -> result=3
        graph.replay()

        ins = torch.load(capture_dir / "inputs.pt", weights_only=False)
        outs = torch.load(capture_dir / "outputs.pt", weights_only=False)
        snapshot1 = capture_dir / "graph_flushes" / "flush_0001"
        ins_snapshot1 = torch.load(snapshot1 / "inputs.pt", weights_only=False)
        outs_snapshot1 = torch.load(snapshot1 / "outputs.pt", weights_only=False)
        assert torch.allclose(ins["arg_0"], torch.full((4, 4), 1.0))
        assert torch.allclose(ins["arg_1"], torch.full((4, 4), 2.0))
        assert torch.allclose(outs["result"], torch.full((4, 4), 3.0))
        assert torch.allclose(ins_snapshot1["arg_0"], torch.full((4, 4), 1.0))
        assert torch.allclose(ins_snapshot1["arg_1"], torch.full((4, 4), 2.0))
        assert torch.allclose(outs_snapshot1["result"], torch.full((4, 4), 3.0))

        # Replay 2 with mutated inputs -> dump should reflect the new values.
        x.fill_(10.0)
        y.fill_(20.0)
        graph.replay()

        ins = torch.load(capture_dir / "inputs.pt", weights_only=False)
        outs = torch.load(capture_dir / "outputs.pt", weights_only=False)
        snapshot2 = capture_dir / "graph_flushes" / "flush_0002"
        ins_snapshot1 = torch.load(snapshot1 / "inputs.pt", weights_only=False)
        outs_snapshot1 = torch.load(snapshot1 / "outputs.pt", weights_only=False)
        ins_snapshot2 = torch.load(snapshot2 / "inputs.pt", weights_only=False)
        outs_snapshot2 = torch.load(snapshot2 / "outputs.pt", weights_only=False)

        # The compatibility files at the dump root contain the latest flush.
        assert torch.allclose(ins["arg_0"], torch.full((4, 4), 10.0))
        assert torch.allclose(ins["arg_1"], torch.full((4, 4), 20.0))
        assert torch.allclose(outs["result"], torch.full((4, 4), 30.0))

        # Earlier flushes must remain intact, otherwise graph replay dumps only
        # preserve the last values read from the graph's static buffers.
        assert torch.allclose(ins_snapshot1["arg_0"], torch.full((4, 4), 1.0))
        assert torch.allclose(ins_snapshot1["arg_1"], torch.full((4, 4), 2.0))
        assert torch.allclose(outs_snapshot1["result"], torch.full((4, 4), 3.0))
        assert torch.allclose(ins_snapshot2["arg_0"], torch.full((4, 4), 10.0))
        assert torch.allclose(ins_snapshot2["arg_1"], torch.full((4, 4), 20.0))
        assert torch.allclose(outs_snapshot2["result"], torch.full((4, 4), 30.0))

        n_cleared = clear_graph_dumps()
        assert n_cleared >= 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_level_10_cuda_graph_captures_without_eager_warmup(self, tmp_path):
        """Capturing a level-10 dump without prior eager warmup should not
        abort capture or inject dump copies into the captured graph.
        """
        import sys

        os.environ["FLASHINFER_LOGLEVEL"] = "10"
        os.environ["FLASHINFER_LOGDEST"] = "stderr"
        os.environ["FLASHINFER_DUMP_DIR"] = str(tmp_path / "fi_dumps")
        sys.modules.pop("flashinfer.api_logging", None)
        from flashinfer.api_logging import (
            flashinfer_api,
            clear_graph_dumps,
            replay_from_dump,
        )

        @flashinfer_api
        def _id(x, cache):
            return x

        x = torch.zeros(8, device="cuda")
        k = torch.zeros(2, 4, device="cuda")
        v = torch.ones(2, 4, device="cuda")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _id(x, (k, v))

        x.fill_(7)
        graph.replay()

        dump_root = tmp_path / "fi_dumps"
        dump_dirs = sorted(
            (d for d in dump_root.iterdir() if d.is_dir()), key=lambda p: p.name
        )
        assert dump_dirs
        capture_dir = dump_dirs[-1]
        inputs = torch.load(capture_dir / "inputs.pt", weights_only=False)
        outputs = torch.load(capture_dir / "outputs.pt", weights_only=False)
        assert torch.allclose(inputs["arg_0"], torch.full((8,), 7.0))
        assert torch.allclose(inputs["arg_1__0"], torch.zeros(2, 4))
        assert torch.allclose(inputs["arg_1__1"], torch.ones(2, 4))
        assert torch.allclose(outputs["result"], torch.full((8,), 7.0))

        metadata = json.loads(
            (capture_dir / "metadata.jsonl").read_text().splitlines()[0]
        )
        cache_metadata = metadata["input_metadata"]["arg_1"]
        assert cache_metadata["type"] == "tuple"
        assert cache_metadata["items"][0]["type"] == "torch.Tensor"
        assert cache_metadata["items"][0]["tensor_key"] == "arg_1__0"
        assert cache_metadata["items"][0]["shape"] == [2, 4]

        replay = replay_from_dump(str(capture_dir), device="cpu", run=False)
        assert isinstance(replay["args"][1], tuple)
        assert torch.allclose(replay["args"][1][0], torch.zeros(2, 4))
        assert torch.allclose(replay["args"][1][1], torch.ones(2, 4))

        n_cleared = clear_graph_dumps()
        assert n_cleared >= 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_level_10_cuda_graph_flushes_on_replay(self, tmp_path):
        """Level-10 CUDA graph dumping preserves every replay without
        application code calling flush_graph_dumps() directly."""
        import sys

        os.environ["FLASHINFER_LOGLEVEL"] = "10"
        os.environ["FLASHINFER_LOGDEST"] = "stderr"
        os.environ["FLASHINFER_DUMP_DIR"] = str(tmp_path / "fi_dumps")
        sys.modules.pop("flashinfer.api_logging", None)
        from flashinfer.api_logging import flashinfer_api, clear_graph_dumps

        @flashinfer_api
        def _add(x, y):
            return x + y

        x = torch.full((4, 4), 1.0, device="cuda")
        y = torch.full((4, 4), 2.0, device="cuda")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _add(x, y)

        dump_root = tmp_path / "fi_dumps"
        dump_dirs = sorted(
            (d for d in dump_root.iterdir() if d.is_dir()), key=lambda p: p.name
        )
        assert dump_dirs
        capture_dir = dump_dirs[-1]

        graph.replay()
        snapshot1 = capture_dir / "graph_flushes" / "flush_0001"
        assert (snapshot1 / "inputs.pt").exists()
        assert (snapshot1 / "outputs.pt").exists()

        x.fill_(10.0)
        y.fill_(20.0)
        graph.replay()
        snapshot2 = capture_dir / "graph_flushes" / "flush_0002"
        assert (snapshot2 / "inputs.pt").exists()
        assert (snapshot2 / "outputs.pt").exists()

        ins_snapshot1 = torch.load(snapshot1 / "inputs.pt", weights_only=False)
        outs_snapshot1 = torch.load(snapshot1 / "outputs.pt", weights_only=False)
        ins_snapshot2 = torch.load(snapshot2 / "inputs.pt", weights_only=False)
        outs_snapshot2 = torch.load(snapshot2 / "outputs.pt", weights_only=False)
        latest_inputs = torch.load(capture_dir / "inputs.pt", weights_only=False)
        latest_outputs = torch.load(capture_dir / "outputs.pt", weights_only=False)

        assert torch.allclose(ins_snapshot1["arg_0"], torch.full((4, 4), 1.0))
        assert torch.allclose(ins_snapshot1["arg_1"], torch.full((4, 4), 2.0))
        assert torch.allclose(outs_snapshot1["result"], torch.full((4, 4), 3.0))
        assert torch.allclose(ins_snapshot2["arg_0"], torch.full((4, 4), 10.0))
        assert torch.allclose(ins_snapshot2["arg_1"], torch.full((4, 4), 20.0))
        assert torch.allclose(outs_snapshot2["result"], torch.full((4, 4), 30.0))
        assert torch.allclose(latest_inputs["arg_0"], torch.full((4, 4), 10.0))
        assert torch.allclose(latest_outputs["result"], torch.full((4, 4), 30.0))

        n_cleared = clear_graph_dumps()
        assert n_cleared >= 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_level_10_cuda_graph_replay_flushes_only_matching_graph(self, tmp_path):
        """Replaying one CUDA graph must not snapshot stale buffers for another
        captured graph that has not replayed."""
        import sys

        os.environ["FLASHINFER_LOGLEVEL"] = "10"
        os.environ["FLASHINFER_LOGDEST"] = "stderr"
        os.environ["FLASHINFER_DUMP_DIR"] = str(tmp_path / "fi_dumps")
        sys.modules.pop("flashinfer.api_logging", None)
        from flashinfer.api_logging import flashinfer_api, clear_graph_dumps

        @flashinfer_api
        def _id(x):
            return x

        x1 = torch.full((4,), 1.0, device="cuda")
        x2 = torch.full((4,), 5.0, device="cuda")

        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1):
            _id(x1)

        dump_root = tmp_path / "fi_dumps"
        dump_dirs = sorted(
            (d for d in dump_root.iterdir() if d.is_dir()), key=lambda p: p.name
        )
        assert len(dump_dirs) == 1
        dir1 = dump_dirs[0]

        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            _id(x2)

        dump_dirs = sorted(
            (d for d in dump_root.iterdir() if d.is_dir()), key=lambda p: p.name
        )
        assert len(dump_dirs) == 2
        dir2 = dump_dirs[1]

        x1.fill_(2.0)
        g1.replay()

        graph1_snapshot = dir1 / "graph_flushes" / "flush_0001"
        assert (graph1_snapshot / "inputs.pt").exists()
        assert not (dir2 / "graph_flushes").exists()

        graph1_inputs = torch.load(graph1_snapshot / "inputs.pt", weights_only=False)
        assert torch.allclose(graph1_inputs["arg_0"], torch.full((4,), 2.0))

        x2.fill_(7.0)
        g2.replay()

        graph2_snapshot = dir2 / "graph_flushes" / "flush_0001"
        assert (graph2_snapshot / "inputs.pt").exists()
        assert not (dir1 / "graph_flushes" / "flush_0002").exists()

        graph2_inputs = torch.load(graph2_snapshot / "inputs.pt", weights_only=False)
        assert torch.allclose(graph2_inputs["arg_0"], torch.full((4,), 7.0))

        n_cleared = clear_graph_dumps()
        assert n_cleared >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
