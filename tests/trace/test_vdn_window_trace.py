# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import torch

from flashinfer.vdn import VDNWindowAttentionWrapper


def test_vdn_window_trace_signature(tmp_path):
    q = torch.empty(75, 7, 128, dtype=torch.bfloat16)
    definition = VDNWindowAttentionWrapper.run.fi_trace(
        query=q, key=q, value=q, save_dir=tmp_path
    )
    assert definition["axes"]["num_heads"]["value"] == 7
    assert definition["axes"]["head_dim"]["value"] == 128
    assert definition["inputs"]["query"]["dtype"] == "bfloat16"
    assert definition["outputs"]["output"]["dtype"] == "bfloat16"
    golden = Path(__file__).parent / "fi_trace_out/vdn_window_attention_h7_d128.json"
    assert definition == json.loads(golden.read_text())
