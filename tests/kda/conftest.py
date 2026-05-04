"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
"""

import pytest
import torch

from flashinfer.utils import get_compute_capability


def skip_if_not_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] < 10:
        pytest.skip(f"KDA chunk prefill requires SM100+, but got SM{cc[0]}{cc[1]}")
