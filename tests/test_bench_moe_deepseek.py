import sys

from benchmarks import bench_moe_deepseek


def test_execution_mode_defaults_to_graph(monkeypatch):
    calls = []
    monkeypatch.setattr(bench_moe_deepseek, "is_sm100_family", lambda: True)
    monkeypatch.setattr(
        bench_moe_deepseek.torch.cuda, "get_device_name", lambda _device: "B200"
    )
    monkeypatch.setattr(
        bench_moe_deepseek, "run_benchmark", lambda **kwargs: calls.append(kwargs)
    )

    monkeypatch.setattr(
        sys, "argv", ["bench_moe_deepseek.py", "--num-tokens", "8", "--quiet"]
    )
    assert bench_moe_deepseek.main() == 0
    assert calls[-1]["use_cuda_graph"] is True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_moe_deepseek.py",
            "--num-tokens",
            "8",
            "--quiet",
            "--execution-mode",
            "eager",
        ],
    )
    assert bench_moe_deepseek.main() == 0
    assert calls[-1]["use_cuda_graph"] is False
