# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only skipped-tile safety coverage for the throughput 2CTA MLA kernel."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0",
)

import cutlass.pipeline as pipeline
from cutlass.experimental.task_scheduling.enums import (
    SignalingThreads,
    TileSchedulerType,
)
from cutlass.experimental.task_scheduling.resources import (
    PipelineConfig,
    TileSchedulerConfig,
)

from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.config import (
    make_mla_decode_config,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.kernel import (
    build_mla_decode_task_manager,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.resources import (
    MlaWorkQueue,
)


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _make_clc_work_queue(cfg) -> MlaWorkQueue:
    return MlaWorkQueue(
        tile_sched_params=None,
        cfg=cfg,
        static_split_kv=1,
        static_seq_len_k=128,
        groups_tokens_heads_q_ratio=1,
        logical_num_heads_q=128,
        logical_seq_len_q=1,
        static_problem_shape_b=1,
        static_problem_shape_s=1,
        use_clc_dynamic=True,
        name="mla_work_queue",
        tile_scheduler_config=TileSchedulerConfig(
            TileSchedulerType.ClcDynamicPersistent,
            None,
            None,
        ),
        pipeline_config=PipelineConfig.create_clc_fetch_async_pipeline_cfg(
            num_stages=2,
            num_bytes=16,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                cfg.threads_per_cta * cfg.num_mma_ctas - 2 * cfg.threads_per_warp,
            ),
            cta_layout_vmnk=(cfg.num_mma_ctas, 1, 1, 1),
            producer_signaling_threads=SignalingThreads.CtaLeader,
            consumer_signaling_threads=SignalingThreads.All,
        ),
    )


def _slot(entry):
    resource, schedule_stage, call_id, _ = entry
    return id(resource), schedule_stage, call_id


def test_bf16_clc_skipped_tiles_preserve_queue_progress_and_pipeline_balance() -> None:
    """Skip data work symmetrically while retaining register setup and CLC progress."""

    cfg = make_mla_decode_config(
        qkv_dtype="bf16",
        o_dtype="bf16",
        is_persistent=True,
    )
    work_queue = _make_clc_work_queue(cfg)
    task_manager, _, _ = build_mla_decode_task_manager(
        cfg,
        domain=1,
        work_queue=work_queue,
        exhaustive_deadlock_race_check=False,
    )

    queue_entries = 0
    throttle_stages = set()
    for task in task_manager.tasks:
        assert task.skip_if is not None
        for entries, skippable_slots in (
            (task.head_schedule_list, task.skippable_head_slots),
            (task.tail_schedule_list, task.skippable_tail_slots),
        ):
            for entry in entries:
                resource, stage, _, _ = entry
                is_skippable = _slot(entry) in skippable_slots
                if resource is work_queue:
                    # Even a zero-K logical tile must fetch and retire its CLC work.
                    assert not is_skippable
                    queue_entries += 1
                elif resource.name == "work_throttle":
                    # Both sides of the cross-CTA throttle disappear together.
                    assert is_skippable
                    throttle_stages.add(stage.name)
                elif not is_skippable:
                    # Pure register initializers remain outside the skipped body so
                    # their SSA values dominate the following loop and tail regions.
                    assert stage.name in {"ProducerAuxWork", "ConsumerAuxWork"}

    assert queue_entries
    assert {
        "ProducerTryAcquire",
        "ProducerAcquire",
        "ProducerCommit",
        "ConsumerWait",
        "ConsumerRelease",
    }.issubset(throttle_stages)
