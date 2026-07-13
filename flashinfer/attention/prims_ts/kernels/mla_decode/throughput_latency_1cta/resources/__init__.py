# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
