"""SM100 kernel imports must not claim application-owned module names."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


_PACKAGE = "flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe"
_RUNNER_PACKAGES = (
    "moe_bf16_glu",
    "moe_mxfp8_glu",
    "moe_nvfp4_swapab",
    "moe_mxfp8_bf16_glu",
)


@pytest.mark.parametrize("preload_application_packages", [False, True])
def test_sm100_cutedsl_imports_are_namespaced(preload_application_packages):
    # A fresh interpreter catches import-order bugs hidden by pytest's module cache.
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                import types
                from pathlib import Path

                names = (
                    "common", "src", "moe_nvfp4_swapab", "moe_mxfp8_glu",
                    "moe_bf16_glu", "moe_mxfp8_bf16_glu",
                )
                foreign = {}
                if sys.argv[1] == "True":
                    for name in names:
                        module = types.ModuleType(name)
                        module.__file__ = f"/application/{name}/__init__.py"
                        module.__path__ = []
                        foreign[name] = module
                    sys.modules.update(foreign)

                from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg

                for name in pkg._LAZY_HELPERS:
                    getattr(pkg, name)

                for name in names:
                    if name in foreign:
                        assert sys.modules[name] is foreign[name], name
                    else:
                        assert name not in sys.modules, name
                root = Path(pkg.__file__).parent.resolve()
                assert not any(
                    Path(entry or ".").resolve().is_relative_to(root)
                    for entry in sys.path
                ), sys.path
                """
            ),
            str(preload_application_packages),
        ],
        check=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    "module",
    [
        f"{package}.{runner}"
        for package in _RUNNER_PACKAGES
        for runner in ("mega_runner", "runner_fc12")
    ]
    + ["src.inputs_process"],
)
def test_sm100_cutedsl_module_help(module, tmp_path):
    """Module entry points must reach argparse before any GPU initialization."""
    result = subprocess.run(
        [sys.executable, "-m", f"{_PACKAGE}.src.{module}", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize(
    "package,functional_selector,functional_tokens",
    [
        ("moe_bf16_glu", "M1_", "2400"),
        ("moe_mxfp8_glu", "M1_", "2400"),
        ("moe_nvfp4_swapab", "A1_", "1500"),
        ("moe_mxfp8_bf16_glu", "F01_", "256"),
    ],
)
@pytest.mark.parametrize("mode", ["functional", "single", "multi", "multinode"])
def test_sm100_cutedsl_shell_launchers(
    package, functional_selector, functional_tokens, mode, tmp_path
):
    """Exercise real launcher branches with recording Python/torchrun executables."""
    project_root = Path(__file__).resolve().parents[2]
    script_dir = project_root.joinpath(*_PACKAGE.split("."), "src", package)
    functional = mode == "functional"
    script = script_dir / (
        "run_functional_tests.sh" if functional else "run_mega_tests.sh"
    )
    runner = "runner_fc12" if functional else "mega_runner"
    if functional:
        selector = functional_selector
    elif package == "moe_mxfp8_bf16_glu":
        selector = "M01_single_e4m3" if mode == "single" else "M03_multi_epi_form_a"
    else:
        selector = (
            "M01_single_balanced_topk2" if mode == "single" else "M03_mr_balanced_topk3"
        )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "launch.jsonl"
    recorder = f"#!{sys.executable}\n" + textwrap.dedent(
        """
        import json, os, sys
        with open(os.environ["LAUNCH_CAPTURE"], "a") as output:
            output.write(json.dumps({
                "argv": sys.argv[1:],
                "pythonpath": os.environ["PYTHONPATH"],
                "no_dist": os.environ.get("MEGA_NO_DIST"),
            }) + "\\n")
        """
    )
    for executable in ("python", "torchrun"):
        path = bin_dir / executable
        path.write_text(recorder)
        path.chmod(0o755)
    application_path = str(tmp_path / "application")
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "PYTHON": str(bin_dir / "python"),
        "TORCHRUN": str(bin_dir / "torchrun"),
        "PYTHONPATH": application_path,
        "LAUNCH_CAPTURE": str(capture),
        "MEGA_NPROC": "2",
        "MEGA_NNODES": "2" if mode == "multinode" else "1",
        "MEGA_NODE_RANK": "1" if mode == "multinode" else "0",
        "MEGA_MASTER_ADDR": "127.0.0.1",
        "MEGA_MASTER_PORT": "29517",
        "MEGA_NO_DIST": "0",
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = subprocess.run(
        ["bash", str(script), selector, "--fail-fast"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert capture.is_file(), result.stdout
    launches = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(launches) == 1, launches
    launch = launches[0]
    argv = launch["argv"]
    if mode in ("multi", "multinode"):
        assert "--nproc_per_node=2" in argv
        if mode == "multinode":
            for flag in (
                "--nnodes=2",
                "--node_rank=1",
                "--master_addr=127.0.0.1",
                "--master_port=29517",
            ):
                assert flag in argv
        module_index = argv.index("--module") + 1
    else:
        assert argv[0] == "-m"
        module_index = 1
    assert argv[module_index] == f"{_PACKAGE}.src.{package}.{runner}"
    token_flag = "--tokens_after_topk" if functional else "--num_tokens_per_rank"
    tokens = functional_tokens if functional else "192" if mode == "single" else "256"
    assert argv[argv.index(token_flag) + 1] == tokens
    assert launch["no_dist"] == ("1" if mode == "single" else "0")
    assert launch["pythonpath"].split(os.pathsep) == [
        str(project_root),
        application_path,
    ]
