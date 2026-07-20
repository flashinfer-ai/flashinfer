# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""CUTLASS DSL runtime patches: applied by default, disabled by kill switch.

Runs in subprocesses because the patches mutate the process-global cutlass
runtime.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("cutlass")

PROBE = """
import json, os
{env_setup}
from flashinfer.experimental.sm12x._lib import runtime_patches as rp
rp.apply_cutlass_runtime_patches()
rp.apply_cutlass_runtime_patches()  # second call must be a guarded no-op
print(json.dumps({{
    "warning_patched": rp._WARNING_PATCHED,
    "frameinfo_patched": rp._DIRECT_FRAMEINFO_PATCHED,
}}))
"""


def _run(env_setup: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", PROBE.format(env_setup=env_setup)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"subprocess failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_patches_apply_by_default():
    data = _run(
        'os.environ.pop("FLASHINFER_EXP_SM12X_DISABLE_CUTLASS_RUNTIME_PATCHES", None)'
    )
    assert data["warning_patched"] is True
    assert data["frameinfo_patched"] is True


def test_kill_switch_disables_all_patches():
    data = _run(
        'os.environ["FLASHINFER_EXP_SM12X_DISABLE_CUTLASS_RUNTIME_PATCHES"] = "1"'
    )
    assert data["warning_patched"] is False
    assert data["frameinfo_patched"] is False
