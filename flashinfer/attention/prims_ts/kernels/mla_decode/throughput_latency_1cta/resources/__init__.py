# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Resources for the throughput-latency 1CTA MLA task schedule."""

from .common import MlaResource, WorkIdThrottleResource
from .smem_p import SmemPResource
from .smem_resources import (
    SmemKResource,
    SmemKvResource,
    SmemPageOffsetsResource,
    SmemQResource,
    SmemVResource,
)
from .tmem_corr import TmemCorrResource
from .tmem_o import TmemOResource
from .tmem_p import TmemPResource
from .tmem_s import TmemSKeepsResource, TmemSResource
from .tmem_softmax_stats import (
    TmemSoftmaxGlobalResource,
    TmemSoftmaxLocalResource,
)

__all__ = [
    "MlaResource",
    "WorkIdThrottleResource",
    "SmemKResource",
    "SmemKvResource",
    "SmemPageOffsetsResource",
    "SmemQResource",
    "SmemVResource",
    "SmemPResource",
    "TmemCorrResource",
    "TmemOResource",
    "TmemPResource",
    "TmemSKeepsResource",
    "TmemSResource",
    "TmemSoftmaxGlobalResource",
    "TmemSoftmaxLocalResource",
]
