"""
Copyright (c) 2026 by FlashInfer team.

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

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


@pytest.mark.parametrize("visible_devices", [None, "", "0", "0,2,5"])
def test_single_node_comm_script_preserves_gpu_visibility(visible_devices):
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/task_test_single_node_comm_kernels.sh"
    ).read_text()
    # Exercise the actual preamble without installing packages or running GPUs.
    preamble, separator, _ = script.partition("\nsource ")
    assert separator
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("BASH_ENV", None)
    if visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
    probe = shlex.join(
        [
            sys.executable,
            "-c",
            "import json, os; print(json.dumps(os.getenv('CUDA_VISIBLE_DEVICES')))",
        ]
    )
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", preamble + "\n" + probe],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(result.stdout) == visible_devices
