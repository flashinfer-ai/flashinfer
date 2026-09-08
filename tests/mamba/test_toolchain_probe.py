"""Compiler admission must follow the assembler selected by Triton."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from triton.backends.nvidia import compiler


@pytest.mark.parametrize("supported", [True, False])
def test_probe_uses_selected_assembler(request, monkeypatch, supported):
    conftest_path = Path(__file__).with_name("conftest.py").resolve()
    conftest = next(
        plugin
        for plugin in request.config.pluginmanager.get_plugins()
        if getattr(plugin, "__file__", None)
        and Path(plugin.__file__).resolve() == conftest_path
    )
    selected_arches = []
    commands = []

    def select(arch):
        selected_arches.append(arch)
        return SimpleNamespace(path="selected-blackwell-ptxas")

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            stderr="" if supported else "not defined for option 'gpu-name'",
            stdout="",
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 3))
    monkeypatch.setattr(compiler, "get_ptxas", select, raising=False)
    monkeypatch.setattr("subprocess.run", run)

    assert conftest._triton_supports_current_arch() is supported
    assert selected_arches == [103]
    assert commands
    assert all(command[0] == "selected-blackwell-ptxas" for command in commands)
