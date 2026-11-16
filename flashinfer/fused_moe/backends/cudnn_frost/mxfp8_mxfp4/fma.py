# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""MXFP4-weight specialization of the shared MXFP8-activation FMA kernels."""

from functools import partial

from ..mxfp8 import fma as common

_TAG = "cudnn_frost-mxfp8_mxfp4-fma-v1"
supported = common.supported
check_support = common.check_support
source_digest = partial(common.source_digest, mixed=True)
tactic = partial(common.tactic, mixed=True)
build = partial(common.build, mixed=True)
