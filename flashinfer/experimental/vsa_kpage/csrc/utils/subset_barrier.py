# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named barriers for selected warp roles (same protocol as blk64 forward)."""

from cutlass import Int32
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op
from flashinfer.experimental.vsa_kpage.csrc.utils import pipeline as pipeline_custom


class _SubsetNamedBarrier(pipeline_custom.NamedBarrier):
    """CTA named barrier whose participants occupy only selected warp roles."""

    @dsl_user_op
    def arrive(self, *, loc=None, ip=None):
        self.arrive_w_index(Int32(0), loc=loc, ip=ip)

    @dsl_user_op
    def arrive_and_wait(self, *, loc=None, ip=None):
        self.arrive_and_wait_w_index(Int32(0), loc=loc, ip=ip)

    @dsl_user_op
    def arrive_w_index(self, index: Int32, *, loc=None, ip=None):
        nvvm.barrier_cta_arrive(
            Int32(self.barrier_id + index).ir_value(loc=loc, ip=ip),
            Int32(self.num_threads).ir_value(loc=loc, ip=ip),
            aligned=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def arrive_and_wait_w_index(self, index: Int32, *, loc=None, ip=None):
        nvvm.barrier_cta_sync(
            Int32(self.barrier_id + index).ir_value(loc=loc, ip=ip),
            thread_count=Int32(self.num_threads).ir_value(loc=loc, ip=ip),
            aligned=False,
            loc=loc,
            ip=ip,
        )
