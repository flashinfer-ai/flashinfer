import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def wheel_builder(monkeypatch):
    path = Path(__file__).parents[1] / "scripts" / "build_jit_cache_wheel.py"
    spec = importlib.util.spec_from_file_location("build_jit_cache_wheel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", Mock())
    return module


def test_dependency_install_recovers_from_disconnect(wheel_builder):
    failure = subprocess.CalledProcessError(
        1, ["python", "-m", "pip"], stderr=b"IncompleteRead(32768 bytes read)"
    )
    env = Mock()
    env.install.side_effect = [failure, None]
    requirements = {"torch==2.13.0+cu129"}

    wheel_builder.install_build_requirements(env, requirements)

    assert env.install.call_count == 2
    assert all(call.args == (requirements,) for call in env.install.call_args_list)
    wheel_builder.time.sleep.assert_called_once_with(10)


def test_dependency_install_stops_after_three_attempts(wheel_builder):
    failure = subprocess.CalledProcessError(
        1, ["python", "-m", "pip"], output="RemoteDisconnected"
    )
    env = Mock()
    env.install.side_effect = failure

    with pytest.raises(subprocess.CalledProcessError) as caught:
        wheel_builder.install_build_requirements(env, {"torch"})

    assert caught.value is failure
    assert env.install.call_count == 3
    assert [call.args[0] for call in wheel_builder.time.sleep.call_args_list] == [
        10,
        20,
    ]


def test_dependency_resolution_error_is_not_retried(wheel_builder):
    failure = subprocess.CalledProcessError(
        1, ["python", "-m", "pip"], stderr=b"No matching distribution found"
    )
    env = Mock()
    env.install.side_effect = failure

    with pytest.raises(subprocess.CalledProcessError):
        wheel_builder.install_build_requirements(env, {"unavailable-package"})

    env.install.assert_called_once()
    wheel_builder.time.sleep.assert_not_called()
