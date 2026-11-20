# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Blockscaled dispatch must agree with validation without copying its inputs."""

import pytest
import torch

from flashinfer.gemm import gemm_base


@pytest.mark.cpu_only
@pytest.mark.parametrize("cc", [100, 103, 107, 120, 121])
@pytest.mark.parametrize("m", [1, 128])
@pytest.mark.parametrize("skip_check", [False, True])
def test_blockscaled_single_row_dispatch(monkeypatch, cc, m, skip_check):
    monkeypatch.setattr(
        "flashinfer.utils.get_compute_capability", lambda _: divmod(cc, 10)
    )
    monkeypatch.setattr(gemm_base, "is_sm12x_supported", lambda _: cc in (120, 121))
    n, k = 128, 256
    a = torch.empty((m, k), dtype=torch.float8_e4m3fn)
    b = torch.empty((n, k), dtype=torch.float8_e4m3fn)
    sa = torch.empty((1, k // 128))
    sb = torch.empty((1, k // 128))
    out = torch.empty((m, n), dtype=torch.float16)
    validated = []

    def requirement(*args, scale_granularity_mnk, **kwargs):
        # Model the SM12x alignment requirement independently of kernel launch.
        if cc in (120, 121) and m % scale_granularity_mnk[0] != 0:
            raise ValueError("M does not satisfy the selected scale granularity")
        validated.append(scale_granularity_mnk)
        return True

    def groupwise(a_arg, b_arg, sa_arg, sb_arg, **kwargs):
        assert a_arg is a and b_arg is b and sa_arg is sa and sb_arg is sb
        assert kwargs["out"] is out
        assert kwargs["scale_major_mode"] == "K"
        assert kwargs["out_dtype"] == torch.float16
        assert kwargs["mma_sm"] == 1
        expected = (1, 128, 128) if cc in (120, 121) and m == 1 else (128, 128, 128)
        assert kwargs["scale_granularity_mnk"] == expected
        assert validated == ([] if skip_check else [expected])
        return out

    monkeypatch.setattr(
        gemm_base, "_cutlass_gemm_fp8_nt_groupwise_requirement", requirement
    )
    monkeypatch.setattr(gemm_base, "gemm_fp8_nt_groupwise", groupwise)
    result = gemm_base.gemm_fp8_nt_blockscaled(
        a,
        b,
        sa,
        sb,
        scale_major_mode="K",
        out=out,
        out_dtype=torch.float16,
        skip_check=skip_check,
    )
    assert result is out
