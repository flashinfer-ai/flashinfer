# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Modular attention kernels for CuTe DSL.

Kernels live at the top level of this package.
Building blocks (config, tmem_layout, roles, fusion, scheduler, wrappers) are
one level below in subdirectories.
"""

# Kernels
from .prefill import BlackwellFusedMultiHeadAttentionForward
from .mla_decode import BlackwellMultiLatentAttentionForward
from .mla_decode_fp8 import BlackwellMultiLatentAttentionForwardFP8

# Building blocks — FMHA prefill
from .config import AttentionConfig, AttentionFusion, HeadMapping, TileBounds
from .tmem_layout import TmemLayout
from .warp_schedule import WarpSchedule, PREFILL_SCHEDULE
from .pipeline_topology import (
    PipelineEdge,
    PipelineType,
    PipelineTopology,
    make_prefill_topology,
    make_mla_topology,
    make_mla_fp8_topology,
)
from .mainloop_spec import (
    MainloopSpec,
    make_prefill_mainloop_spec,
    MLAMainloopSpec,
    make_mla_mainloop_spec,
    make_mla_fp8_mainloop_spec,
)
from .fusion.mask import MaskType
from .fusion.variant import (
    tanh_approx,
    AttentionVariant,
    StandardAttention,
    AttentionWithSink,
    SigmoidAttention,
    SigmoidTanhAttention,
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

# Building blocks — MLA decode
from .mla_config import MLAConfig
from .mla_warp_schedule import (
    MLAWarpSchedule,
    MLA_DECODE_SCHEDULE,
    MLAWarpScheduleFP8,
    MLA_DECODE_FP8_SCHEDULE,
)
from .scheduler.mla_persistent import (
    MLAStaticTileScheduler,
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler,
    create_mla_static_tile_scheduler_params,
    mla_get_split_kv,
    mla_get_split_kv_simplified,
    mla_get_workspace_size,
)

# Wrappers
from .wrappers.batch_prefill import (
    BatchPrefillCuteDSLWrapper,
)
from .wrappers.batch_mla import (
    BatchMLADecodeCuteDSLWrapper,
    cute_dsl_mla_decode,
)
