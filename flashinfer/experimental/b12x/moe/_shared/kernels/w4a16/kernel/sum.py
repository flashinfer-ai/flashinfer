"""W4A16 MoE top-k reduction kernels."""

from __future__ import annotations

from cutlass.cutlass_dsl import Int32
from cutlass.cutlass_dsl import Int64
import cuda.bindings.driver as cuda
import cutlass.cute as cute
import cutlass
from b12x.moe._shared.kernels.w4a16.kernel.config import (
    _covering_count,
    _materialize_w4a16_topk_route_f32,
)


class W4A16TopKSumKernel:
    def __init__(
        self,
        *,
        topk: int,
        hidden_size: int,
        element_dtype: str = "bf16",
        full_rotation: bool = False,
        intermediate_hadamard: bool = False,
        num_experts: int = 0,
        route_num_experts: int = 0,
        use_expert_map: bool = False,
        broadcast_svh: bool = False,
        float32_output: bool = False,
        apply_topk_weights: bool = False,
    ):
        if element_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"unsupported element_dtype {element_dtype!r}")
        if topk <= 0 or hidden_size <= 0:
            raise ValueError("topk and hidden_size must be positive")
        if apply_topk_weights and full_rotation:
            raise ValueError("full-rotation top-k sum already applies router weights")
        # Plain path: multiply each route by its FP32 router weight (FC2 then
        # stores unweighted outputs; see _FP32_TOPK_WEIGHTS).
        self.apply_topk_weights = bool(apply_topk_weights)
        self.topk = int(topk)
        self.hidden_size = int(hidden_size)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.full_rotation = bool(full_rotation)
        self.intermediate_hadamard = bool(intermediate_hadamard)
        self.num_experts = int(num_experts)
        self.route_num_experts = int(route_num_experts)
        self.use_expert_map = bool(use_expert_map)
        # svh_table holds a single row shared by every expert; index it with a
        # zero expert stride.
        self.broadcast_svh = bool(broadcast_svh)
        self.float32_output = bool(float32_output)
        if self.use_expert_map:
            if self.route_num_experts <= 0:
                raise ValueError("expert-map top-k sum requires route_num_experts > 0")
            if self.num_experts <= 0:
                raise ValueError("expert-map top-k sum requires num_experts > 0")
        if self.full_rotation:
            if self.element_dtype != "fp16":
                raise ValueError("full-rotation top-k sum requires fp16 route values")
            if self.hidden_size % 128 != 0:
                raise ValueError(
                    "full-rotation top-k sum requires hidden_size % 128 == 0"
                )
            if self.num_experts <= 0:
                raise ValueError("full-rotation top-k sum requires num_experts > 0")
        if self.intermediate_hadamard:
            if not self.full_rotation:
                raise ValueError(
                    "intermediate-Hadamard top-k sum requires full rotation"
                )
            if self.hidden_size % 512 != 0:
                raise ValueError(
                    "intermediate-Hadamard top-k sum requires hidden_size % 512 == 0"
                )
        self.route_warps = 8
        self.cta_threads = 256

    @cute.jit
    def _cast_elem(self, x: cutlass.Float32):
        if cutlass.const_expr(self.float32_output):
            return x
        if cutlass.const_expr(self.is_fp16):
            return cutlass.Float16(x)
        return cutlass.BFloat16(x)

    @cute.jit
    def __call__(
        self,
        fc2_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        route_expert_ids_ptr: cute.Pointer,
        expert_map_ptr: cute.Pointer,
        svh_ptr: cute.Pointer,
        weight_num_experts: cutlass.Int32,
        route_num_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        fc2_flat = cute.make_tensor(
            fc2_ptr,
            layout=cute.make_layout(
                (active_m * Int32(self.topk * self.hidden_size),), stride=(1,)
            ),
        )
        output_flat = cute.make_tensor(
            output_ptr,
            layout=cute.make_layout((active_m * Int32(self.hidden_size),), stride=(1,)),
        )
        topk_weights_flat = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout((active_m * Int32(self.topk),), stride=(1,)),
        )
        route_expert_ids_flat = cute.make_tensor(
            route_expert_ids_ptr,
            layout=cute.make_layout((active_m * Int32(self.topk),), stride=(1,)),
        )
        expert_map_flat = cute.make_tensor(
            expert_map_ptr,
            layout=cute.make_layout(
                (Int64(route_num_experts) + Int64(1),), stride=(1,)
            ),
        )
        svh_rows = Int64(weight_num_experts)
        if cutlass.const_expr(self.broadcast_svh):
            svh_rows = Int64(1)
        svh_flat = cute.make_tensor(
            svh_ptr,
            layout=cute.make_layout((svh_rows * Int64(self.hidden_size),), stride=(1,)),
        )
        if cutlass.const_expr(self.full_rotation):
            if cutlass.const_expr(self.intermediate_hadamard):
                total = active_m * Int32(self.hidden_size // 512)
            else:
                total = active_m * Int32(self.hidden_size // 128)
            # One CTA owns one H128 output slab.  The full-rotation kernel
            # parallelizes the top-k routes across its eight warps, then has
            # warp zero accumulate the staged route values in router order.
            # The previous launch assigned one slab to one warp; K3 decode
            # consequently launched only 28 active warps and serialized all
            # 16 routes in each of them.
            grid = (total, 1, 1)
        else:
            total = active_m * Int32(self.hidden_size)
            grid = (_covering_count(total, self.cta_threads), 1, 1)
        self.kernel(
            fc2_flat,
            output_flat,
            topk_weights_flat,
            route_expert_ids_flat,
            expert_map_flat,
            svh_flat,
            weight_num_experts,
            route_num_experts,
            active_m,
        ).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        fc2_flat: cute.Tensor,
        output_flat: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        route_expert_ids_flat: cute.Tensor,
        expert_map_flat: cute.Tensor,
        svh_flat: cute.Tensor,
        weight_num_experts: cutlass.Int32,
        route_num_experts: cutlass.Int32,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.intermediate_hadamard and self.broadcast_svh):
            # The output scale is shared by every expert.  Linearity therefore
            # permits route reduction before the ordinary H128 cancellation:
            #
            #   sum_r w_r * (H128(v_r) * s)
            #     = H128(sum_r w_r * v_r) * s.
            #
            # This applies the two H128 transforms once per output subblock
            # instead of applying the first transform once per routed expert.
            tid = Int32(tidx)
            lane = tid & Int32(31)
            warp = tid >> Int32(5)
            unit = Int32(bidx)
            nblk = Int32(self.hidden_size // 512)
            total_units = active_m * nblk
            if unit < total_units:
                token = unit // nblk
                blk = unit - token * nblk
                block_col = blk * Int32(512)
                reduced_ptr = cute.arch.alloc_smem(cutlass.Float32, 512)
                reduced = cute.make_tensor(reduced_ptr, cute.make_layout(512))

                if warp < Int32(4):
                    sub = warp
                    col0 = block_col + sub * Int32(128) + lane * Int32(4)
                    acc0 = cutlass.Float32(0.0)
                    acc1 = cutlass.Float32(0.0)
                    acc2 = cutlass.Float32(0.0)
                    acc3 = cutlass.Float32(0.0)
                    for route in cutlass.range_constexpr(self.topk):
                        row = token * Int32(self.topk) + Int32(route)
                        raw_expert = route_expert_ids_flat[row].to(Int32)
                        expert = raw_expert
                        if cutlass.const_expr(self.use_expert_map):
                            expert = Int32(-1)
                            if raw_expert >= Int32(0) and raw_expert < Int32(
                                route_num_experts
                            ):
                                expert = expert_map_flat[raw_expert].to(Int32)
                        if expert >= Int32(0) and expert < Int32(weight_num_experts):
                            weight = topk_weights_flat[row].to(cutlass.Float32)
                            base = row * Int32(self.hidden_size) + col0
                            acc0 += (
                                fc2_flat[base + Int32(0)].to(cutlass.Float32) * weight
                            )
                            acc1 += (
                                fc2_flat[base + Int32(1)].to(cutlass.Float32) * weight
                            )
                            acc2 += (
                                fc2_flat[base + Int32(2)].to(cutlass.Float32) * weight
                            )
                            acc3 += (
                                fc2_flat[base + Int32(3)].to(cutlass.Float32) * weight
                            )
                    acc0, acc1, acc2, acc3 = self._had128_quad(
                        acc0, acc1, acc2, acc3, lane
                    )
                    acc0 *= svh_flat[col0 + Int32(0)].to(cutlass.Float32)
                    acc1 *= svh_flat[col0 + Int32(1)].to(cutlass.Float32)
                    acc2 *= svh_flat[col0 + Int32(2)].to(cutlass.Float32)
                    acc3 *= svh_flat[col0 + Int32(3)].to(cutlass.Float32)
                    acc0, acc1, acc2, acc3 = self._had128_quad(
                        acc0, acc1, acc2, acc3, lane
                    )
                    reduced_base = sub * Int32(128) + lane * Int32(4)
                    reduced[reduced_base + Int32(0)] = acc0
                    reduced[reduced_base + Int32(1)] = acc1
                    reduced[reduced_base + Int32(2)] = acc2
                    reduced[reduced_base + Int32(3)] = acc3
                cute.arch.sync_threads()

                if warp == Int32(0):
                    elem = lane * Int32(4)
                    for reg in cutlass.range_constexpr(4):
                        off = elem + Int32(reg)
                        o0, o1, o2, o3 = self._had4_normalized(
                            reduced[off + Int32(0)],
                            reduced[off + Int32(128)],
                            reduced[off + Int32(256)],
                            reduced[off + Int32(384)],
                        )
                        out_base = token * Int32(self.hidden_size) + block_col + off
                        output_flat[out_base + Int32(0)] = o0
                        output_flat[out_base + Int32(128)] = o1
                        output_flat[out_base + Int32(256)] = o2
                        output_flat[out_base + Int32(384)] = o3
            return
        if cutlass.const_expr(self.intermediate_hadamard):
            tid = Int32(tidx)
            lane = tid & Int32(31)
            warp = tid >> Int32(5)
            unit = Int32(bidx)
            nblk = Int32(self.hidden_size // 512)
            total_units = active_m * nblk
            if unit < total_units:
                token = unit // nblk
                blk = unit - token * nblk
                block_col = blk * Int32(512)
                route_values_ptr = cute.arch.alloc_smem(
                    cutlass.Float32, self.topk * 512
                )
                route_values = cute.make_tensor(
                    route_values_ptr, cute.make_layout(self.topk * 512)
                )
                route_weights_ptr = cute.arch.alloc_smem(cutlass.Float32, self.topk)
                route_weights = cute.make_tensor(
                    route_weights_ptr, cute.make_layout(self.topk)
                )

                # Each task is one (route, H128 subblock).  Eight warps cover
                # the 4*topk tasks while retaining the existing independent
                # warp-local H128 implementation.
                for task_group in cutlass.range_constexpr(
                    (self.topk * 4 + self.route_warps - 1) // self.route_warps
                ):
                    task = warp + Int32(task_group * self.route_warps)
                    route = task >> Int32(2)
                    sub = task & Int32(3)
                    valid_route = Int32(0)
                    expert = Int32(-1)
                    row = token * Int32(self.topk) + route
                    if route < Int32(self.topk):
                        raw_expert = route_expert_ids_flat[row].to(Int32)
                        expert = raw_expert
                        if cutlass.const_expr(self.use_expert_map):
                            expert = Int32(-1)
                            if raw_expert >= Int32(0) and raw_expert < Int32(
                                route_num_experts
                            ):
                                expert = expert_map_flat[raw_expert].to(Int32)
                        if expert >= Int32(0) and expert < Int32(weight_num_experts):
                            valid_route = Int32(1)

                    hs0 = cutlass.Float32(0.0)
                    hs1 = cutlass.Float32(0.0)
                    hs2 = cutlass.Float32(0.0)
                    hs3 = cutlass.Float32(0.0)
                    col0 = block_col + sub * Int32(128) + lane * Int32(4)
                    if valid_route != Int32(0):
                        base = row * Int32(self.hidden_size) + col0
                        v0 = fc2_flat[base + Int32(0)].to(cutlass.Float32)
                        v1 = fc2_flat[base + Int32(1)].to(cutlass.Float32)
                        v2 = fc2_flat[base + Int32(2)].to(cutlass.Float32)
                        v3 = fc2_flat[base + Int32(3)].to(cutlass.Float32)
                        h0, h1, h2, h3 = self._had128_quad(v0, v1, v2, v3, lane)
                        if cutlass.const_expr(self.broadcast_svh):
                            sbase = col0
                        else:
                            sbase = expert * Int32(self.hidden_size) + col0
                        hs0 = h0 * svh_flat[sbase + Int32(0)].to(cutlass.Float32)
                        hs1 = h1 * svh_flat[sbase + Int32(1)].to(cutlass.Float32)
                        hs2 = h2 * svh_flat[sbase + Int32(2)].to(cutlass.Float32)
                        hs3 = h3 * svh_flat[sbase + Int32(3)].to(cutlass.Float32)
                    value_base = route * Int32(512) + sub * Int32(128) + lane * Int32(4)
                    if route < Int32(self.topk):
                        route_values[value_base + Int32(0)] = hs0
                        route_values[value_base + Int32(1)] = hs1
                        route_values[value_base + Int32(2)] = hs2
                        route_values[value_base + Int32(3)] = hs3
                        if lane == Int32(0) and sub == Int32(0):
                            weight = cutlass.Float32(0.0)
                            if valid_route != Int32(0):
                                weight = topk_weights_flat[row].to(cutlass.Float32)
                            route_weights[route] = weight
                cute.arch.sync_threads()

                # Four warps reduce one H128 subblock each.  The first 512
                # shared values become the weighted, ordinary-unrotated H512
                # vector, ready for the exact intermediate-Hadamard residual transform.
                if warp < Int32(4):
                    sub = warp
                    acc0 = cutlass.Float32(0.0)
                    acc1 = cutlass.Float32(0.0)
                    acc2 = cutlass.Float32(0.0)
                    acc3 = cutlass.Float32(0.0)
                    for route in cutlass.range_constexpr(self.topk):
                        value_base = (
                            Int32(route * 512) + sub * Int32(128) + lane * Int32(4)
                        )
                        weight = route_weights[Int32(route)]
                        acc0 += route_values[value_base + Int32(0)] * weight
                        acc1 += route_values[value_base + Int32(1)] * weight
                        acc2 += route_values[value_base + Int32(2)] * weight
                        acc3 += route_values[value_base + Int32(3)] * weight
                    acc0, acc1, acc2, acc3 = self._had128_quad(
                        acc0, acc1, acc2, acc3, lane
                    )
                    reduced_base = sub * Int32(128) + lane * Int32(4)
                    route_values[reduced_base + Int32(0)] = acc0
                    route_values[reduced_base + Int32(1)] = acc1
                    route_values[reduced_base + Int32(2)] = acc2
                    route_values[reduced_base + Int32(3)] = acc3
                cute.arch.sync_threads()

                if warp == Int32(0):
                    elem = lane * Int32(4)
                    for reg in cutlass.range_constexpr(4):
                        off = elem + Int32(reg)
                        o0, o1, o2, o3 = self._had4_normalized(
                            route_values[off + Int32(0)],
                            route_values[off + Int32(128)],
                            route_values[off + Int32(256)],
                            route_values[off + Int32(384)],
                        )
                        out_base = token * Int32(self.hidden_size) + block_col + off
                        output_flat[out_base + Int32(0)] = o0
                        output_flat[out_base + Int32(128)] = o1
                        output_flat[out_base + Int32(256)] = o2
                        output_flat[out_base + Int32(384)] = o3
            return
        if cutlass.const_expr(self.full_rotation):
            tid = Int32(tidx)
            lane = tid & Int32(31)
            warp = tid >> Int32(5)
            unit = Int32(bidx)
            nblk = Int32(self.hidden_size // 128)
            total_units = active_m * nblk
            if unit < total_units:
                token = unit // nblk
                blk = unit - token * nblk
                col0 = blk * Int32(128) + lane * Int32(4)
                route_values_ptr = cute.arch.alloc_smem(
                    cutlass.Float32, self.topk * 128
                )
                route_values = cute.make_tensor(
                    route_values_ptr, cute.make_layout(self.topk * 128)
                )
                route_weights_ptr = cute.arch.alloc_smem(cutlass.Float32, self.topk)
                route_weights = cute.make_tensor(
                    route_weights_ptr, cute.make_layout(self.topk)
                )
                for route_group in cutlass.range_constexpr(
                    (self.topk + self.route_warps - 1) // self.route_warps
                ):
                    route = warp + Int32(route_group * self.route_warps)
                    valid_route = Int32(0)
                    expert = Int32(-1)
                    row = token * Int32(self.topk) + route
                    if route < Int32(self.topk):
                        raw_expert = route_expert_ids_flat[row].to(Int32)
                        expert = raw_expert
                        if cutlass.const_expr(self.use_expert_map):
                            expert = Int32(-1)
                            if raw_expert >= Int32(0) and raw_expert < Int32(
                                route_num_experts
                            ):
                                expert = expert_map_flat[raw_expert].to(Int32)
                        if expert >= Int32(0) and expert < Int32(weight_num_experts):
                            valid_route = Int32(1)

                    hs0 = cutlass.Float32(0.0)
                    hs1 = cutlass.Float32(0.0)
                    hs2 = cutlass.Float32(0.0)
                    hs3 = cutlass.Float32(0.0)
                    if valid_route != Int32(0):
                        base = row * Int32(self.hidden_size) + col0
                        v0 = fc2_flat[base + Int32(0)].to(cutlass.Float32)
                        v1 = fc2_flat[base + Int32(1)].to(cutlass.Float32)
                        v2 = fc2_flat[base + Int32(2)].to(cutlass.Float32)
                        v3 = fc2_flat[base + Int32(3)].to(cutlass.Float32)
                        h0, h1, h2, h3 = self._had128_quad(v0, v1, v2, v3, lane)
                        if cutlass.const_expr(self.broadcast_svh):
                            sbase = col0
                        else:
                            sbase = expert * Int32(self.hidden_size) + col0
                        s0 = svh_flat[sbase + Int32(0)].to(cutlass.Float32)
                        s1 = svh_flat[sbase + Int32(1)].to(cutlass.Float32)
                        s2 = svh_flat[sbase + Int32(2)].to(cutlass.Float32)
                        s3 = svh_flat[sbase + Int32(3)].to(cutlass.Float32)
                        hs0 = h0 * s0
                        hs1 = h1 * s1
                        hs2 = h2 * s2
                        hs3 = h3 * s3
                    value_base = route * Int32(128) + lane * Int32(4)
                    if route < Int32(self.topk):
                        route_values[value_base + Int32(0)] = hs0
                        route_values[value_base + Int32(1)] = hs1
                        route_values[value_base + Int32(2)] = hs2
                        route_values[value_base + Int32(3)] = hs3
                        if lane == Int32(0):
                            weight = cutlass.Float32(0.0)
                            if valid_route != Int32(0):
                                weight = topk_weights_flat[row].to(cutlass.Float32)
                            route_weights[route] = weight
                cute.arch.sync_threads()

                if warp == Int32(0):
                    acc0 = cutlass.Float32(0.0)
                    acc1 = cutlass.Float32(0.0)
                    acc2 = cutlass.Float32(0.0)
                    acc3 = cutlass.Float32(0.0)
                    for route in cutlass.range_constexpr(self.topk):
                        value_base = Int32(route * 128) + lane * Int32(4)
                        weight = route_weights[Int32(route)]
                        acc0 += route_values[value_base + Int32(0)] * weight
                        acc1 += route_values[value_base + Int32(1)] * weight
                        acc2 += route_values[value_base + Int32(2)] * weight
                        acc3 += route_values[value_base + Int32(3)] * weight
                    out_base = token * Int32(self.hidden_size) + col0
                    output_flat[out_base + Int32(0)] = acc0
                    output_flat[out_base + Int32(1)] = acc1
                    output_flat[out_base + Int32(2)] = acc2
                    output_flat[out_base + Int32(3)] = acc3
            return
        idx = Int32(bidx) * Int32(self.cta_threads) + Int32(tidx)
        total = active_m * Int32(self.hidden_size)
        if idx < total:
            token = idx // Int32(self.hidden_size)
            col = idx - token * Int32(self.hidden_size)
            acc = cutlass.Float32(0.0)
            for route in cutlass.range_constexpr(self.topk):
                row = token * Int32(self.topk) + Int32(route)
                valid_route = Int32(1)
                if cutlass.const_expr(self.use_expert_map):
                    raw_expert = route_expert_ids_flat[row].to(Int32)
                    expert = Int32(-1)
                    if raw_expert >= Int32(0) and raw_expert < route_num_experts:
                        expert = expert_map_flat[raw_expert].to(Int32)
                    if expert < Int32(0) or expert >= weight_num_experts:
                        valid_route = Int32(0)
                elif weight_num_experts > Int32(0):
                    expert = route_expert_ids_flat[row]
                    if expert < 0 or expert >= weight_num_experts:
                        valid_route = Int32(0)
                if valid_route != Int32(0):
                    route_value = fc2_flat[
                        Int64(row) * Int64(self.hidden_size) + Int64(col)
                    ].to(cutlass.Float32)
                    if cutlass.const_expr(self.apply_topk_weights):
                        acc += _materialize_w4a16_topk_route_f32(
                            route_value
                        ) * topk_weights_flat[row].to(cutlass.Float32)
                    else:
                        acc += _materialize_w4a16_topk_route_f32(route_value)
            output_flat[idx] = self._cast_elem(acc)

    @cute.jit
    def _had128_quad(
        self,
        v0: cutlass.Float32,
        v1: cutlass.Float32,
        v2: cutlass.Float32,
        v3: cutlass.Float32,
        lane: Int32,
    ):
        s0 = v0 + v1
        d0 = v0 - v1
        s1 = v2 + v3
        d1 = v2 - v3
        h0 = s0 + s1
        h1 = d0 + d1
        h2 = s0 - s1
        h3 = d0 - d1
        for i in cutlass.range_constexpr(5):
            st = 1 << i
            p0 = cute.arch.shuffle_sync_bfly(h0, offset=st)
            p1 = cute.arch.shuffle_sync_bfly(h1, offset=st)
            p2 = cute.arch.shuffle_sync_bfly(h2, offset=st)
            p3 = cute.arch.shuffle_sync_bfly(h3, offset=st)
            if (lane & Int32(st)) != Int32(0):
                h0 = p0 - h0
                h1 = p1 - h1
                h2 = p2 - h2
                h3 = p3 - h3
            else:
                h0 = p0 + h0
                h1 = p1 + h1
                h2 = p2 + h2
                h3 = p3 + h3
        rs = cutlass.Float32(0.088388347648)
        return h0 * rs, h1 * rs, h2 * rs, h3 * rs

    @cute.jit
    def _had4_normalized(
        self,
        v0: cutlass.Float32,
        v1: cutlass.Float32,
        v2: cutlass.Float32,
        v3: cutlass.Float32,
    ):
        s0 = v0 + v1
        d0 = v0 - v1
        s1 = v2 + v3
        d1 = v2 - v3
        rs = cutlass.Float32(0.5)
        return (
            (s0 + s1) * rs,
            (d0 + d1) * rs,
            (s0 - s1) * rs,
            (d0 - d1) * rs,
        )
