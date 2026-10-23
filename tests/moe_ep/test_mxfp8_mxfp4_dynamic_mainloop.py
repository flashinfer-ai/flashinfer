# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the native MXFP4 x MXFP8 MMA contract (host and device)."""

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
import cutlass.cute as cute  # noqa: E402
from cuda.bindings import driver as cuda  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402
from cutlass.cutlass_dsl import Int32  # noqa: E402

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.dynamic_mainloop import (  # noqa: E402
    A_DTYPE,
    B_DTYPE,
    SF_DTYPE,
    SF_VEC_SIZE,
    align_valid_tokens,
    build_idesc_host,
    build_static_idesc_base,
    compute_idesc,
    validate_mma_contract,
)


def test_swap_ab_dtype_and_scale_contract():
    validate_mma_contract(
        a_dtype=A_DTYPE,
        b_dtype=B_DTYPE,
        sf_dtype=SF_DTYPE,
        sf_vec_size=SF_VEC_SIZE,
        mma_tiler_mnk=(256, 256, 256),
    )
    with pytest.raises(ValueError, match="instruction A"):
        validate_mma_contract(
            a_dtype=B_DTYPE,
            b_dtype=A_DTYPE,
            sf_dtype=SF_DTYPE,
            sf_vec_size=SF_VEC_SIZE,
            mma_tiler_mnk=(256, 256, 256),
        )
    with pytest.raises(ValueError, match="sf_vec_size=32"):
        validate_mma_contract(
            a_dtype=A_DTYPE,
            b_dtype=B_DTYPE,
            sf_dtype=SF_DTYPE,
            sf_vec_size=16,
            mma_tiler_mnk=(256, 256, 256),
        )


@pytest.mark.parametrize(
    ("valid", "aligned"),
    [(1, 16), (16, 16), (17, 32), (127, 128), (241, 256), (256, 256)],
)
def test_valid_token_alignment(valid, aligned):
    assert align_valid_tokens(valid) == aligned


def test_static_descriptor_is_mxf8f6f4_e2m1_e4m3_ue8m0_k32():
    # a=E2M1(5), b=E4M3(0), scale=UE8M0(1), M=(M>>7), K=32(0).
    assert build_static_idesc_base(umma_m=128) == 0x08800280
    assert build_static_idesc_base(umma_m=256) == 0x10800280


def test_runtime_descriptor_encodes_aligned_n_and_scale_data_ids():
    desc = build_idesc_host(
        umma_m=256,
        valid_tokens_in_tile=17,
        sfa_tmem_addr=0x80000000,
        sfb_tmem_addr=0x40000000,
    )
    assert (desc >> 17) & 0x3F == 4  # align16(17) / 8
    assert (desc >> 29) & 0x3 == 2
    assert (desc >> 4) & 0x3 == 1


@pytest.mark.parametrize("valid", [0, 257])
def test_runtime_descriptor_rejects_unissuable_n(valid):
    with pytest.raises(ValueError, match="dynamic N"):
        build_idesc_host(umma_m=128, valid_tokens_in_tile=valid)


class _DeviceIdescHarness:
    """Build runtime descriptors on the device for every scale-factor data ID."""

    @cute.jit
    def __call__(self, addrs: cute.Tensor, out: cute.Tensor, stream):
        self._kernel(addrs, out).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def _kernel(self, addrs: cute.Tensor, out: cute.Tensor):
        tidx = cute.arch.thread_idx()[0]
        if tidx < 4:
            out[tidx] = compute_idesc(
                static_base=build_static_idesc_base(umma_m=128),
                n_dim_value=((Int32(addrs[8]) + Int32(15)) & Int32(-16)) >> Int32(3),
                sfa_tmem_addr_i32=addrs[tidx],
                sfb_tmem_addr_i32=addrs[tidx + 4],
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_descriptor_matches_host_for_every_scale_data_id():
    # Scale-factor data IDs 2 and 3 set bit 31 of the TMEM address; an
    # arithmetic shift of the masked value used to sign-extend ones over the
    # descriptor (illegal instruction parameter on K steps 2 and 3).
    pytest.importorskip("cutlass")
    from flashinfer.utils import is_sm100a_supported

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("requires SM100/SM103 and CUDA 12.8+")
    sfa = [(i << 30) | 0x80 for i in range(4)]
    sfb = [((3 - i) << 30) | 0x90 for i in range(4)]
    valid = 17
    addrs = torch.tensor(
        [a - (1 << 32) if a >= 1 << 31 else a for a in sfa + sfb] + [valid],
        dtype=torch.int32,
        device="cuda",
    )
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    addrs_cute = from_dlpack(addrs, assumed_align=4).mark_layout_dynamic()
    out_cute = from_dlpack(out, assumed_align=4).mark_layout_dynamic()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(_DeviceIdescHarness(), addrs_cute, out_cute, stream)
    compiled(addrs_cute, out_cute, stream)
    torch.cuda.synchronize()
    device = [v & 0xFFFFFFFF for v in out.tolist()]
    host = [
        build_idesc_host(
            umma_m=128, valid_tokens_in_tile=valid, sfa_tmem_addr=a, sfb_tmem_addr=b
        )
        for a, b in zip(sfa, sfb, strict=True)
    ]
    assert device == host
