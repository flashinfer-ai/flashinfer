"""Regression test: ``import flashinfer`` must not require ``nvidia-cutlass-dsl``.

The CuTe DSL is an optional dependency. The availability probes that decide
whether to use it live in ``flashinfer.cute_dsl.availability``, which imports
nothing from ``cutlass`` -- if a ``cutlass`` import ever creeps onto the eager
``import flashinfer`` path again, the package stops being installable-and-usable
for consumers that do not want the DSL. That is how this broke before: the probe
lived next to code that imported ``cutlass`` at module scope, so asking "is the
DSL installed?" required the DSL to be installed.

The check runs in a subprocess with ``cutlass`` blocked at the import-system
level, so it is meaningful whether or not the DSL happens to be installed in
the environment running the tests.
"""

import subprocess
import sys
import textwrap

# Hides ``cutlass`` and every submodule from the import system, mimicking an
# environment where ``nvidia-cutlass-dsl`` was never installed.
_BLOCKER = """
import importlib.machinery
import sys


# A PathFinder that cannot see ``cutlass``. Returning None (rather than
# raising) is what makes this faithful: importlib.util.find_spec("cutlass")
# then yields None and ``import cutlass`` raises ModuleNotFoundError, exactly
# as in an environment where nvidia-cutlass-dsl was never installed.
class _NoCutlassPathFinder(importlib.machinery.PathFinder):
    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if name == "cutlass" or name.startswith("cutlass."):
            return None
        return super().find_spec(name, path, target)


sys.meta_path = [
    f for f in sys.meta_path if f is not importlib.machinery.PathFinder
] + [_NoCutlassPathFinder]
for _name in [m for m in sys.modules if m == "cutlass" or m.startswith("cutlass.")]:
    del sys.modules[_name]
"""


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_import_flashinfer_without_cutlass():
    result = _run(
        """
        import flashinfer  # noqa: F401
        print("ok")
        """
    )
    assert result.returncode == 0, (
        "import flashinfer failed without nvidia-cutlass-dsl installed:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "ok" in result.stdout


def test_probes_report_unavailable_without_cutlass():
    """The probes must answer ``False``, not raise."""
    result = _run(
        """
        from flashinfer.cute_dsl.availability import (
            is_cute_dsl_arch_supported,
            is_cute_dsl_available,
            is_cute_dsl_experimental_available,
            is_rubin_cute_dsl_available,
        )

        assert is_cute_dsl_available() is False
        assert is_rubin_cute_dsl_available() is False
        assert is_cute_dsl_experimental_available() is False
        assert is_cute_dsl_arch_supported(10, 0) is False
        print("ok")
        """
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "ok" in result.stdout


def test_availability_module_imports_no_cutlass():
    """Importing the probe module must not pull ``cutlass`` into ``sys.modules``."""
    result = _run(
        """
        import sys

        import flashinfer.cute_dsl.availability  # noqa: F401

        assert not [m for m in sys.modules if m == "cutlass" or m.startswith("cutlass.")]
        print("ok")
        """
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "ok" in result.stdout
