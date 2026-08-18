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

import torch

from flashinfer import cake_vsa


class _NoRuntimeReduction:
    def max(self):
        raise AssertionError("static mask reductions must not run in run_cake_vsa")


def test_run_cake_vsa_uses_planned_mask_reductions(monkeypatch):
    q = torch.empty((1,), dtype=torch.bfloat16)
    stats = torch.empty((1,), dtype=torch.float32)
    profiles = []
    plan = {
        "head_dim": 128,
        "R": 128,
        "num_qo_heads": 8,
        "num_kv_heads": 8,
        "mb": 2,
        "N": 512,
        "max_selected_blocks": 2,
        "uniform_selected_blocks": True,
        "row_counts": _NoRuntimeReduction(),
    }

    monkeypatch.setattr(cake_vsa, "_check_inputs", lambda *_args: None)
    monkeypatch.setattr(
        cake_vsa,
        "_outputs",
        lambda *_args: (q, stats),
    )

    def record_profile(profile, *_args, **_kwargs):
        profiles.append(profile)

    monkeypatch.setattr(cake_vsa, "_run_standard", record_profile)

    result = cake_vsa.run_cake_vsa(
        plan,
        q,
        q,
        q,
        out=None,
        lse=None,
        return_lse=False,
        backend="cake",
    )

    assert result is q
    assert profiles == ["blk128_compact"]
