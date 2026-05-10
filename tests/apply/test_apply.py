import json
import sys

import pytest
import torch

import flashinfer
from flashinfer.api_logging import flashinfer_api
from flashinfer.trace_apply import enable_apply_from_env, register_plan_run
from flashinfer.trace import (
    BuildSpec,
    Const,
    Scalar,
    Solution,
    SourceFile,
    Tensor,
    TraceTemplate,
    Var,
)
from flashinfer.trace.templates.norm import rmsnorm_trace

_test_apply_trace = TraceTemplate(
    op_type="apply_test",
    name_prefix="apply_test",
    axes={
        "batch_size": Var(),
        "hidden_size": Const(abbrev="h"),
    },
    inputs={
        "x": Tensor(["batch_size", "hidden_size"]),
    },
    outputs={
        "output": Tensor(["batch_size", "hidden_size"], dtype_from="x"),
    },
)


_plan_run_trace = TraceTemplate(
    op_type="plan_run_test",
    name_prefix="plan_run",
    axes={
        "batch_size": Var(),
        "hidden_size": Const(abbrev="h"),
        "factor": Const(abbrev="f"),
    },
    inputs={
        "x": Tensor(["batch_size", "hidden_size"]),
        "factor": Scalar("int32", optional=True),
        "scale": Scalar("float32", optional=True),
        "bias": Scalar("float32", optional=True),
    },
    outputs={
        "output": Tensor(["batch_size", "hidden_size"], dtype_from="x"),
    },
)


@flashinfer_api(trace=_test_apply_trace)
def _apply_test_api(x):
    return x - 1


class _PlanRunApplyTest:
    @flashinfer_api
    def plan(self, factor: int, scale: float = 1.0):
        self.factor = factor
        self.scale = scale

    @flashinfer_api(trace=_plan_run_trace)
    def run(self, x, bias: float = 0.0):
        return x * self.factor * self.scale + bias


@pytest.fixture(autouse=True)
def _reset_trace_apply():
    flashinfer.disable_apply()
    yield
    flashinfer.disable_apply()


def _python_solution(definition, content, *, dps=False, language="python"):
    return Solution(
        name=f"{definition}_python",
        definition=definition,
        author="test",
        spec=BuildSpec(
            language=language,
            target_hardware=["CPU"],
            entry_point="main.py::run",
            destination_passing_style=dps,
        ),
        sources=[SourceFile(path="main.py", content=content)],
    )


def _apply_config_dict(definition, expression):
    return {
        "solutions": {
            definition: {
                "name": f"{definition}_python",
                "definition": definition,
                "author": "test",
                "spec": {
                    "language": "python",
                    "target_hardware": ["CPU"],
                    "entry_point": "main.py::run",
                    "destination_passing_style": False,
                },
                "sources": [
                    {
                        "path": "main.py",
                        "content": f"def run(x):\n    return {expression}\n",
                    }
                ],
            }
        }
    }


def test_rmsnorm():
    x = torch.randn(3, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)
    calls = []

    def solution(hidden_states, weight):
        calls.append((hidden_states, weight))
        return hidden_states + weight

    flashinfer.enable_apply({"rmsnorm_h4": solution})
    out = flashinfer.rmsnorm(x, weight)

    torch.testing.assert_close(out, x + weight)
    assert calls == [(x, weight)]


def test_rmsnorm_positional_and_kwargs():
    x_positional = torch.randn(3, 4, dtype=torch.float32)
    x_kwargs = torch.randn(3, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)
    calls = []

    def solution(hidden_states, weight):
        calls.append((hidden_states, weight))
        return hidden_states + weight

    flashinfer.enable_apply({"rmsnorm_h4": solution})
    out_positional = flashinfer.rmsnorm(x_positional, weight)
    out_kwargs = flashinfer.rmsnorm(input=x_kwargs, weight=weight)

    torch.testing.assert_close(out_positional, x_positional + weight)
    torch.testing.assert_close(out_kwargs, x_kwargs + weight)
    assert calls == [(x_positional, weight), (x_kwargs, weight)]


def test_rmsnorm_out():
    x = torch.randn(3, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)
    out = torch.empty_like(x)

    def solution(hidden_states, weight):
        return hidden_states + weight

    flashinfer.enable_apply({"rmsnorm_h4": solution})
    result = flashinfer.rmsnorm(x, weight, out=out)

    assert result is out
    torch.testing.assert_close(out, x + weight)


def test_solution_object():
    x = torch.randn(3, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)
    solution = _python_solution(
        "rmsnorm_h4",
        "def run(hidden_states, weight):\n    return hidden_states + weight\n",
    )

    flashinfer.enable_apply({"rmsnorm_h4": solution})
    out = flashinfer.rmsnorm(x, weight)

    torch.testing.assert_close(out, x + weight)


def test_apply_config_from_dict():
    x = torch.randn(2, 4, dtype=torch.float32)
    config = flashinfer.ApplyConfig.from_dict(
        {
            "solutions": {
                "apply_test_h4": {
                    "name": "apply_test_h4_python",
                    "definition": "apply_test_h4",
                    "author": "test",
                    "spec": {
                        "language": "python",
                        "target_hardware": ["CPU"],
                        "entry_point": "main.py::run",
                        "destination_passing_style": False,
                    },
                    "sources": [
                        {
                            "path": "main.py",
                            "content": "def run(x):\n    return x + 1\n",
                        }
                    ],
                }
            }
        }
    )

    flashinfer.enable_apply(config)
    out = _apply_test_api(x)

    torch.testing.assert_close(out, x + 1)


def test_solution_dict_roundtrip():
    x = torch.randn(2, 4, dtype=torch.float32)
    config_solution = _python_solution(
        "apply_test_h4",
        "def run(x):\n    return x + 1\n",
    )
    config = flashinfer.ApplyConfig(solutions={"apply_test_h4": config_solution})

    loaded_config = flashinfer.ApplyConfig.from_dict(config.to_dict())
    flashinfer.enable_apply(loaded_config)
    out = _apply_test_api(x)

    torch.testing.assert_close(out, x + 1)


def test_enable_apply_from_env_file(tmp_path, monkeypatch):
    x = torch.randn(2, 4, dtype=torch.float32)
    path = tmp_path / "apply.json"
    path.write_text(json.dumps(_apply_config_dict("apply_test_h4", "x + 1")))
    monkeypatch.setenv("FLASHINFER_APPLY", "1")
    monkeypatch.setenv("FLASHINFER_APPLY_CONFIG", str(path))

    enable_apply_from_env()
    out = _apply_test_api(x)

    torch.testing.assert_close(out, x + 1)


def test_solution_language_allowlist():
    x = torch.randn(2, 4, dtype=torch.float32)
    solution = _python_solution(
        "apply_test_h4",
        "def run(x):\n    return x + 1\n",
        language="triton",
    )

    flashinfer.enable_apply({"apply_test_h4": solution})
    out = _apply_test_api(x)

    torch.testing.assert_close(out, x + 1)


def test_cpp_solution_rejected():
    solution = _python_solution(
        "apply_test_h4",
        "def run(x):\n    return x + 1\n",
        language="cpp",
    )

    with pytest.raises(ValueError, match="Unsupported apply solution language"):
        flashinfer.enable_apply({"apply_test_h4": solution})


def test_fallback_on_miss():
    x = torch.randn(2, 4, dtype=torch.float32)

    def solution(x):
        return x + 1

    flashinfer.enable_apply({"apply_test_h8": solution})
    out = _apply_test_api(x)

    torch.testing.assert_close(out, x - 1)


def test_enable_overrides_previous_apply_until_disable():
    x = torch.randn(2, 4, dtype=torch.float32)

    def solution_one(x):
        return x + 1

    def solution_two(x):
        return x + 2

    flashinfer.enable_apply({"apply_test_h4": solution_one})
    flashinfer.enable_apply({"apply_test_h4": solution_two})
    out = _apply_test_api(x)

    torch.testing.assert_close(out, x + 2)

    flashinfer.disable_apply()
    out = _apply_test_api(x)

    torch.testing.assert_close(out, x - 1)


def test_fallback_on_error():
    x = torch.randn(2, 4, dtype=torch.float32)

    def solution(x):
        raise RuntimeError("boom")

    flashinfer.enable_apply({"apply_test_h4": solution})
    with pytest.warns(UserWarning, match="apply failed"):
        out = _apply_test_api(x)

    torch.testing.assert_close(out, x - 1)


def test_fallback_on_definition_mismatch():
    x = torch.randn(2, 4, dtype=torch.float32)
    solution = _python_solution("wrong", "def run(x):\n    return x + 1\n")

    flashinfer.enable_apply({"apply_test_h4": solution})
    with pytest.warns(UserWarning, match="declares definition"):
        out = _apply_test_api(x)

    torch.testing.assert_close(out, x - 1)


def test_rmsnorm_quant_out():
    x = torch.randn(2, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)
    scale = torch.tensor([1.0], dtype=torch.float32)
    out = torch.empty_like(x)

    def solution(hidden_states, weight, scale):
        return hidden_states + weight + scale.reshape(())

    flashinfer.enable_apply({"rmsnorm_quant_h4": solution})
    result = flashinfer.rmsnorm_quant(out, x, weight, scale)

    assert result is None
    torch.testing.assert_close(out, x + weight + 1.0)


def test_solution_object_dps():
    x = torch.randn(2, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)
    scale = torch.tensor([2.0], dtype=torch.float32)
    out = torch.empty_like(x)
    solution = _python_solution(
        "rmsnorm_quant_h4",
        (
            "def run(hidden_states, weight, scale, out):\n"
            "    out.copy_(hidden_states + weight + scale.reshape(()))\n"
        ),
        dps=True,
    )

    flashinfer.enable_apply({"rmsnorm_quant_h4": solution})
    result = flashinfer.rmsnorm_quant(out, x, weight, scale)

    assert result is None
    torch.testing.assert_close(out, x + weight + 2.0)


def test_fused_add_rmsnorm_mutation():
    x = torch.randn(2, 4, dtype=torch.float32)
    residual = torch.randn(2, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)
    expected_x = x + 1
    expected_residual = residual + 2

    def solution(hidden_states, residual, weight):
        del weight
        hidden_states.copy_(hidden_states + 1)
        residual.copy_(residual + 2)
        return None

    flashinfer.enable_apply({"fused_add_rmsnorm_h4": solution})
    result = flashinfer.fused_add_rmsnorm(x, residual, weight)

    assert result is None
    torch.testing.assert_close(x, expected_x)
    torch.testing.assert_close(residual, expected_residual)


def test_build_definition():
    x = torch.randn(2, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)

    definition = rmsnorm_trace.build_fi_trace_fn("flashinfer.norm.rmsnorm")(
        _write=False, input=x, weight=weight
    )
    traced = flashinfer.rmsnorm.fi_trace(input=x, weight=weight)

    assert definition == traced


def test_plan_run_apply_uses_cached_plan_inputs():
    x = torch.randn(2, 4, dtype=torch.float32)
    op = _PlanRunApplyTest()
    op.factor = 10
    op.scale = 1.0
    calls = []

    def solution(x, factor, scale=1.0, bias=0.0):
        calls.append((factor, scale, bias))
        return x * factor * scale + bias + 1

    register_plan_run(
        plan_fi_api=f"{__name__}._PlanRunApplyTest.plan",
        run_fi_api=f"{__name__}._PlanRunApplyTest.run",
    )
    flashinfer.enable_apply({"plan_run_h4_f3": solution})

    fallback = op.run(x, bias=1.0)
    torch.testing.assert_close(fallback, x * 10 + 1.0)
    assert calls == []

    op.plan(3, scale=2.0)
    out = op.run(x, bias=5.0)

    torch.testing.assert_close(out, x * 3 * 2.0 + 6.0)
    assert calls == [(3, 2.0, 5.0)]


if __name__ == "__main__":
    pytest.main(sys.argv)
