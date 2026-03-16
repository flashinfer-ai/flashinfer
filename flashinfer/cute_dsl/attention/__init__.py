# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Modular attention kernels for CuTe DSL.

Kernels (prefill, decode, mla_decode) live at the top level of this package.
Building blocks (config, tmem_layout, roles, fusion, scheduler, wrappers) are
one level below in subdirectories.
"""

# Kernels
from .prefill import BlackwellFusedMultiHeadAttentionForward
from .mla_decode import BlackwellMultiLatentAttentionForward
from .mla_config import MLAConfig, mla_can_implement
from .mla_warp_schedule import MLAWarpSchedule, MLA_DECODE_SCHEDULE

# Building blocks
from .config import AttentionConfig, AttentionFusion, HeadMapping, TileBounds
from .tmem_layout import TmemLayout
from .warp_schedule import WarpSchedule, PREFILL_SCHEDULE, MLA_SCHEDULE
from .pipeline_topology import (
    PipelineEdge,
    PipelineType,
    PipelineTopology,
    make_prefill_topology,
    make_mla_topology,
)
from .mainloop_spec import (
    MainloopSpec,
    make_prefill_mainloop_spec,
    MLAMainloopSpec,
    make_mla_mainloop_spec,
)
from .fusion.mask import MaskType
from .fusion.logits_transform import sigmoid_logits_transform
from .fusion.output_transform import dumb_output_transform
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
from .wrappers.batch_mla import BatchMLAPagedAttentionWrapperCuteDSL
