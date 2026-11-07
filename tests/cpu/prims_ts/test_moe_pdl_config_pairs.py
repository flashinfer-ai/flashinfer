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

"""PDL settings of the Prims-TS MoE FC1/FC2 GEMM pairs."""

import pytest

_MOE_DTYPES = (
    "bf16",
    "nvfp4",
    "fp8_per_tensor",
    "mxfp4_mxfp8",
    "mxfp4_bf16",
    "mxfp8_mxfp8",
    "deepseek_fp8",
)
_SHAPE = dict(num_tokens=64, top_k=8, num_local_experts=32, num_experts=32)


def _sampled_tactics(valid_tactics, sm_arch, shape, count=8):
    tactics = valid_tactics(sm_arch=sm_arch, **shape)
    assert tactics
    return tactics[:: max(1, len(tactics) // count)][:count]


@pytest.mark.cpu_only
@pytest.mark.parametrize("enable_pdl", (False, True))
@pytest.mark.parametrize("sm_arch", (100, 107))
@pytest.mark.parametrize("dtype", _MOE_DTYPES)
def test_moe_tactic_applies_enable_pdl_to_both_gemms(dtype, sm_arch, enable_pdl):
    from flashinfer.prims_ts.moe import config_mapper

    valid_tactics = getattr(config_mapper, f"valid_prims_ts_{dtype}_moe_tactics")
    map_tactic = getattr(config_mapper, f"map_trtllm_{dtype}_moe_tactic")
    for tactic in _sampled_tactics(valid_tactics, sm_arch, _SHAPE):
        pair = map_tactic(tactic, sm_arch=sm_arch, enable_pdl=enable_pdl, **_SHAPE)
        fc1 = pair.fc1.cfg.kwargs
        fc2 = pair.fc2.cfg.kwargs
        assert fc1["use_pdl"] == fc2["use_pdl"] == int(enable_pdl), tactic
        # FC1 waits before it triggers, which is what lets FC2 and finalize
        # read routing outputs before their own wait.
        assert fc1["skip_early_exit_pdl"] == 0, tactic
        assert fc2["skip_early_exit_pdl"] == int(
            enable_pdl and bool(fc2["use_early_exit"])
        ), tactic


@pytest.mark.cpu_only
def test_wide_cluster_fc2_waits_in_every_cta():
    """Multicast loaders run on selected CTAs only; the epilogue runs on all."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        build_batched_gemm_task_manager,
    )
    from flashinfer.prims_ts.batched_gemm.mixed_cga_ts import CgaTask
    from flashinfer.prims_ts.batched_gemm.pdl_resources import PdlWaitBarrier
    from flashinfer.prims_ts.moe.config_mapper import (
        map_trtllm_nvfp4_moe_tactic,
        valid_prims_ts_nvfp4_moe_tactics,
    )

    shape = dict(num_tokens=8192, top_k=8, num_local_experts=32, num_experts=256)
    wide_fc2 = []
    for tactic in valid_prims_ts_nvfp4_moe_tactics(sm_arch=107, **shape):
        pair = map_trtllm_nvfp4_moe_tactic(
            tactic, sm_arch=107, enable_pdl=True, **shape
        )
        kwargs = pair.fc2.cfg.kwargs
        if kwargs.get("cluster_dim_x", 1) > 2 or kwargs.get("cluster_dim_y", 1) > 1:
            wide_fc2.append(kwargs)
    if not wide_fc2:
        pytest.skip("no packaged NVFP4 FC2 tactic uses a cluster wider than 2x1")

    for kwargs in wide_fc2:
        assert kwargs["skip_early_exit_pdl"] == 1
        task_manager = build_batched_gemm_task_manager(
            num_experts=shape["num_local_experts"],
            num_tokens=128,
            top_k=shape["top_k"],
            verbose=False,
            **kwargs,
        )

        def waits(task):
            return any(isinstance(r, PdlWaitBarrier) for r in task.src_resources)

        assert any(
            isinstance(task, CgaTask) and waits(task) for task in task_manager.tasks
        )
        (epilogue,) = [
            t for t in task_manager.tasks if t.name.startswith("EpilogueTask0")
        ]
        assert not isinstance(epilogue, CgaTask)
        assert epilogue.run_only_on_cta_id is None
        assert waits(epilogue)
