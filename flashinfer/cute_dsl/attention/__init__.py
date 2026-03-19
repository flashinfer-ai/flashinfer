# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Modular attention kernels for CuTe DSL.

Kernels live at the top level of this package.
Building blocks (config, tmem_layout, roles, fusion, scheduler, wrappers) are
one level below in subdirectories.
"""

# Kernels
from .prefill import BlackwellFusedMultiHeadAttentionForward

# Building blocks
from .config import AttentionConfig, AttentionFusion, HeadMapping, TileBounds
from .tmem_layout import TmemLayout
from .warp_schedule import WarpSchedule, PREFILL_SCHEDULE
from .pipeline_topology import (
    PipelineEdge,
    PipelineType,
    PipelineTopology,
    make_prefill_topology,
)
from .mainloop_spec import (
    MainloopSpec,
    make_prefill_mainloop_spec,
)
from .fusion.mask import MaskType
from .fusion.variant import (
    AttentionVariant,
    StandardAttention,
    AttentionWithSink,
    SigmoidAttention,
    ALiBiAttention,
    RPEAttention,
    SoftCappingAttention,
)
from .scheduler.persistent import (
    FmhaStaticTileScheduler,
    FmhaStaticTileSchedulerParams,
    create_fmha_static_tile_scheduler,
    create_fmha_static_tile_scheduler_params,
)

# Wrappers
from .wrappers.batch_prefill import (
    BatchPrefillCuteDSLWrapper,
    qkv_torch_2_cute,
    create_and_pad_tensor,
)
