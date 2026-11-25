# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging.version import Version


@pytest.fixture(scope="session")
def cudnn_batch_invariant_device():
    from flashinfer import batch_invariant as bi
    from flashinfer.gemm.batch_invariant.api import MM_BF16_ARCHS

    if not torch.cuda.is_available() or not bi.is_supported_device(
        torch.device("cuda"), MM_BF16_ARCHS
    ):
        pytest.skip("requires SM100, SM103 or SM107 with CuTe DSL support")

    try:
        import cudnn
    except ModuleNotFoundError as exc:
        if exc.name != "cudnn":
            raise
        pytest.skip("cuDNN Frontend is not installed")
    if not hasattr(getattr(cudnn, "numerical_note", None), "BATCH_INVARIANT"):
        if Version(cudnn.__version__) < Version("1.34.0"):
            pytest.skip(
                f"cuDNN Frontend {cudnn.__version__} lacks batch-invariant GEMM"
            )
        pytest.fail(f"cuDNN Frontend {cudnn.__version__} is missing BATCH_INVARIANT")
