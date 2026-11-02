# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility import for the shared SM100 unified indexer-score kernel."""

from ..score_recompute.indexer_score_unified_sm100 import (
    IndexerScoreUnifiedSm100 as IndexerForwardSm100,
)

__all__ = ["IndexerForwardSm100"]
