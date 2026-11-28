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

"""SMEM Sage K block scales staged for the softmax."""

import cutlass.cute as cute
import cutlass
from cutlass import Float32, Int32
from cutlass.cutlass_dsl import Constexpr
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)
from dataclasses import dataclass, field, replace
from ..config import FmhaConfig
from ..device.windows import resolve_kv_tile_idx_context
from ....sage.kernel.index import sage_section_word, sage_sequence_words
from ....sage.layout import FLOAT4_WORDS
from typing import Any


# ---------------------------------------------------------------------------
# SmemSageKScalesResource -- SMEM ring of one K tile's Sage sfK words per stage
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SmemSageKScalesResource(MemoryResource):
    """Ring of one K tile's ``sfK`` words per stage, staged by the load warp.

    A tile of one-token K blocks has too many words to keep in the softmax
    lanes' registers or to broadcast-load from global memory
    (``SageKTransport.SMEM``). After each K tile's TMA load the load warp
    copies the tile's atoms from the prepared scales into the acquired stage
    with lane-parallel 16-byte ``cp.async`` copies; the pipeline's
    ``AsyncLoad`` commit completes the stage once they land. Both softmax
    groups wait the stage before their score wait, take its shared-memory
    offset from ``open_stage``, read the words after the score wait
    (``TmemSPResource.read_staged_sage_k_words``) and release the stage after
    their max pass: a release orders every read before it, so an earlier one
    would keep all the words in registers while the scores load instead of
    letting the compiler place each read next to its use.
    """

    # Static configuration
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    stage_words: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    # Runtime scalars
    num_kv_heads: Int32 | None = field(init=False, default=None)

    # GMEM pointers and tensors
    sage_scales: cute.Pointer | None = field(init=False, default=None)

    # Allocations
    alloc: Constexpr[SmemAllocation | None] = field(init=False, default=None)

    def __init__(
        self,
        *,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        sage_scales: cute.Pointer | None,
        num_kv_heads: Int32,
        **kwargs: Any,
    ) -> None:
        pipeline_config = replace(pipeline_config, advance_on_wait=True)
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.sage_scales = sage_scales
        self.num_kv_heads = num_kv_heads
        self.alloc = SmemAllocation(
            "smem_sage_k",
            size_bytes=pipeline_config.num_stages * cfg.sage_k_tile_bytes,
            alignment=16,
        )
        self.stage_words = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Shared-memory byte offset of the waited ``sfK`` ring stage.",
        )

    @producer_work
    @cute.jit
    def copy_tile(
        self,
        stage_info: StageInfo,
        *,
        kv_head_coord: Int32,
        batch_coord: Int32,
        kv_tile_start: Int32,
    ) -> None:
        """Copy the current K tile's ``sfK`` words into the acquired stage.

        The tile is the one the K load of this iteration staged
        (``_resolve_kv_tile_idx_context``); the K-token section is padded to
        whole tiles, so the copy never runs past the sequence. Lane ``l``
        copies float4 ``l``, ``l + 32``, ... of the tile.
        """
        cfg = self.cfg
        layout = cfg.sage_scale_layout
        tile_idx = resolve_kv_tile_idx_context(stage_info, kv_tile_start)
        sequence_words = sage_sequence_words(
            layout,
            batch_idx=batch_coord,
            kv_head_idx=kv_head_coord,
            num_kv_heads=self.num_kv_heads,
        )
        tile_words = cfg.sage_k_tile_words
        source_word = sage_section_word(
            sequence_words, layout.k_tokens
        ) + tile_idx * Int32(tile_words)
        stage_bytes = Int32(self.alloc.offset) + stage_info.stage_idx * Int32(
            cfg.sage_k_tile_bytes
        )
        stage_ptr = cutlass.Array(
            stage_info.context.smem_base.data_ptr() + stage_bytes,
            dtype=Float32,
            shape=(tile_words,),
            addrspace=3,
        ).data_ptr()
        lane_idx = cute.arch.thread_idx()[0] & Int32(cute.arch.WARP_SIZE - 1)
        num_float4s = tile_words // FLOAT4_WORDS
        for round_idx in cutlass.range_constexpr(
            (num_float4s + cute.arch.WARP_SIZE - 1) // cute.arch.WARP_SIZE
        ):
            float4_idx = lane_idx + Int32(round_idx * cute.arch.WARP_SIZE)
            word = float4_idx * Int32(FLOAT4_WORDS)
            if float4_idx < Int32(num_float4s):
                prims.cp_async_shared_global(
                    stage_ptr + word,
                    self.sage_scales + source_word + word,
                    FLOAT4_WORDS * 4,
                    "cg",
                )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        return [self.alloc]

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=stage_words)
    @cute.jit
    def open_stage(self, stage_info: StageInfo) -> Int32:
        """Publish the waited stage's shared-memory offset for the passes that read it."""
        return Int32(self.alloc.offset) + Int32(self.consumer_work_stage) * Int32(
            self.cfg.sage_k_tile_bytes
        )
