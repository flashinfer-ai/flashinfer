import math
import operator
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from b12x.attention._shared.contiguous import layout_utils
from b12x.attention._shared.cute import ops as cute_ops
from b12x.attention._shared.contiguous.cute_dsl_utils import (
    ParamsBase,
)


@dataclass
class Softmax(ParamsBase):
    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = 80
    softmax_scale: Float32 | None = None

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax(scale_log2, num_rows, row_max, row_sum, arch, softmax_scale)

    def _row_layout(self) -> cute.Layout:
        return cute.make_layout((self.num_rows,), stride=(1,))

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    def _compute_row_max(
        self, acc_S_row: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return cute_ops.fmax_reduce(acc_S_row, init_val, arch=self.arch)

    def _compute_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return cute_ops.fadd_reduce(acc_S_row_exp, init_val, arch=self.arch)

    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        row_scale = cute.make_rmem_tensor(self._row_layout(), Float32)

        for r in range(int(self.num_rows)):
            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur = cute_ops.fmax_reduce(
                acc_S_row,
                init_val=self.row_max[r] if cutlass.const_expr(not is_first) else None,
                arch=self.arch,
            )
            row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
            row_max_prev = self.row_max[r]
            self.row_max[r] = row_max_cur
            if cutlass.const_expr(check_inf):
                row_max_cur = Float32(
                    cutlass.select_(row_max_cur == -Float32.inf, 0.0, row_max_cur)
                )
            row_max_cur_scaled = row_max_cur * self.scale_log2
            acc_S_row_exp = cute.math.exp2(
                acc_S_row * self.scale_log2 - row_max_cur_scaled,
                fastmath=True,
            )
            if cutlass.const_expr(is_first):
                acc_S_row_sum = cute_ops.fadd_reduce(
                    acc_S_row_exp, init_val=None, arch=self.arch
                )
                row_scale[r] = 1.0
            else:
                row_scale[r] = cute.math.exp2(
                    (row_max_prev - row_max_cur) * self.scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = cute_ops.fadd_reduce(
                    acc_S_row_exp,
                    init_val=self.row_sum[r] * row_scale[r],
                    arch=self.arch,
                )

            self.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)

        return row_scale

    def finalize(
        self, final_scale: Float32 = 1.0, sink_val: Float32 | cute.Tensor | None = None
    ) -> cute.Tensor:
        if cutlass.const_expr(
            sink_val is not None and isinstance(sink_val, cute.Tensor)
        ):
            assert cute.size(sink_val) == self.num_rows
        if cutlass.const_expr(self.softmax_scale is not None):
            # Match the four-lane reduction order of FlashAttention 2.
            # Reversing these additions can change BF16 output rounding and
            # the LSE weights used to merge disjoint context chunks.
            for r in range(int(self.num_rows)):
                total = self.row_sum[r]
                total += cute.arch.shuffle_sync_bfly(total, offset=2)
                total += cute.arch.shuffle_sync_bfly(total, offset=1)
                self.row_sum[r] = total
        else:
            self.row_sum.store(
                cute_ops.warp_reduce(self.row_sum.load(), operator.add, width=4)
            )
        row_scale = cute.make_rmem_tensor(self._row_layout(), Float32)

        for r in range(int(self.num_rows)):
            if cutlass.const_expr(sink_val is not None):
                sink_val_cur = (
                    sink_val if not isinstance(sink_val, cute.Tensor) else sink_val[r]
                )
                self.row_sum[r] += cute.math.exp2(
                    sink_val_cur * math.log2(math.e)
                    - self.row_max[r] * self.scale_log2,
                    fastmath=True,
                )

            row_sum_cur = self.row_sum[r]
            row_sum_is_zero_or_nan = (row_sum_cur == 0.0) | (row_sum_cur != row_sum_cur)
            safe_row_sum = Float32(
                cutlass.select_(row_sum_is_zero_or_nan, 1.0, row_sum_cur)
            )
            row_scale[r] = cute.arch.rcp_approx(safe_row_sum) * final_scale
            if cutlass.const_expr(self.softmax_scale is not None):
                # Retain the original scale: a log2 -> natural-log round trip
                # needlessly changes the normalization of merged attention.
                row_lse = self.row_max[r] * self.softmax_scale + cute.math.log(
                    safe_row_sum, fastmath=True
                )
            else:
                row_lse = (
                    self.row_max[r] * self.scale_log2
                    + cute.math.log2(safe_row_sum, fastmath=True)
                ) * math.log(2.0)
            self.row_sum[r] = Float32(
                cutlass.select_(row_sum_is_zero_or_nan, -Float32.inf, row_lse)
            )
        return row_scale

    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        for r in range(int(self.num_rows)):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
