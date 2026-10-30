# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Rubin route staging with the existing launch-binding control."""

import os

from ..routing_staging import stage_routes as _stage_routes


def stage_routes(ids, scores, ids_out, scores_out, *, workspace) -> None:
    _stage_routes(
        ids,
        scores,
        ids_out,
        scores_out,
        workspace=workspace,
        use_tvm_ffi=os.environ.get("FLASHINFER_SM107_PREP_TVM_FFI", "1") != "0",
    )
