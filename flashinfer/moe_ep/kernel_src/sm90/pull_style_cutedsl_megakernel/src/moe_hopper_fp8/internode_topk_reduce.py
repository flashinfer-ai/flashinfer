"""Internode (NCCL GIN) top-k combine reduce for the SM90 FP8 MegaMoE kernel.

Runs after the fused kernel on the same stream.  For every token of this rank:

* the cells ``combine[t][k]`` of the top-k slots whose expert lives on this
  rank's node (written by this node's expert owners) are accumulated in
  ascending ``k``;
* then, in ascending node order, the one BF16 partial each remote node returned
  through its rail peer (``remote_partial[slot][t]``, already the FP32 sum of
  that node's slots rounded to BF16 once) is added for every remote node the
  token hits;
* the FP32 accumulator is rounded to BF16 once.

This is the fixed reduction tree of the internode design (invariant I11): with
the same per-(token, k) FC2 rows it reproduces bit for bit.  Routing weights are
already folded in by FC1 (``apply_topk_in_fc1=True``), so no score is applied.
Cells of slots routed to remote experts are never written and never read.
"""

from typing import ClassVar

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32


class InternodeTopkReduce:
    """Node-ordered combine reduce for a fixed problem geometry."""

    _threads: ClassVar[int] = 128
    _hidden_per_thread: ClassVar[int] = 8  # one 16-byte BF16 load per term

    def __init__(
        self,
        hidden: int,
        num_topk: int,
        *,
        num_experts_per_rank: int,
        ranks_per_node: int,
        num_nodes: int,
        my_node: int,
        max_tokens_per_rank: int,
    ) -> None:
        self.hidden = int(hidden)
        self.num_topk = int(num_topk)
        if self.hidden % self._hidden_per_thread != 0:
            raise ValueError(f"hidden ({self.hidden}) must be a multiple of 8.")
        self.node_expert_count = int(num_experts_per_rank) * int(ranks_per_node)
        self.num_nodes = int(num_nodes)
        self.my_node = int(my_node)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.hidden_tiles = self.hidden // self._hidden_per_thread

    @cute.jit
    def __call__(
        self,
        combine: cute.Tensor,         # ((1+S)*T, topk, hidden) bf16; rows [0, T) used
        topk_idx: cute.Tensor,        # (T, topk) Int64
        remote_partial: cute.Tensor,  # (S, T, hidden) bf16
        reduced_output: cute.Tensor,  # (T, hidden) bf16
        stream: cuda.CUstream,
    ):
        threads = self._threads
        total_workers = self.max_tokens_per_rank * self.hidden_tiles
        grid = [(total_workers + threads - 1) // threads, 1, 1]
        block = [threads, 1, 1]
        reduced_output = cute.make_tensor(
            reduced_output.iterator,
            cute.make_layout(
                (self.max_tokens_per_rank, self.hidden),
                stride=reduced_output.stride,
            ),
        )
        self._kernel(combine, topk_idx, remote_partial, reduced_output).launch(
            grid=grid, block=block, stream=stream,
        )

    @cute.jit
    def _aligned(self, tensor: cute.Tensor) -> cute.Tensor:
        p = tensor.iterator
        return cute.make_tensor(
            cute.make_ptr(p.dtype, p.toint(), p.memspace, assumed_align=16),
            tensor.layout,
        )

    @cute.kernel
    def _kernel(
        self,
        combine: cute.Tensor,
        topk_idx: cute.Tensor,
        remote_partial: cute.Tensor,
        reduced_output: cute.Tensor,
    ):
        threads = self._threads
        hpt: cutlass.Constexpr[int] = self._hidden_per_thread
        hidden_tiles = self.hidden_tiles
        num_topk: cutlass.Constexpr[int] = self.num_topk
        num_nodes: cutlass.Constexpr[int] = self.num_nodes
        my_node: cutlass.Constexpr[int] = self.my_node
        node_expert_count: cutlass.Constexpr[int] = self.node_expert_count

        worker_idx = cute.arch.block_idx()[0] * Int32(threads) + cute.arch.thread_idx()[0]
        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        if token_idx < Int32(self.max_tokens_per_rank):
            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128,
            )
            # Node of every top-k slot (-1 for padding slots).
            slot_node = cute.make_rmem_tensor((num_topk,), cutlass.Int32)
            for k in cutlass.range_constexpr(num_topk):
                e = Int32(topk_idx[token_idx, k])
                node = Int32(-1)
                if e >= Int32(0):
                    node = e // Int32(node_expert_count)
                slot_node[k] = node

            acc = cute.make_rmem_tensor((hpt,), cutlass.Float32)
            for i in cutlass.range_constexpr(hpt):
                acc[i] = Float32(0.0)

            # Local cells, ascending k.
            for k in cutlass.range_constexpr(num_topk):
                if slot_node[k] == Int32(my_node):
                    src = cute.zipped_divide(
                        combine[token_idx, k, None], (hpt,),
                    )[(None,), (hidden_tile_idx,)]
                    term = cute.make_rmem_tensor((hpt,), cutlass.BFloat16)
                    cute.copy(load_atom, self._aligned(src), term)
                    for i in cutlass.range_constexpr(hpt):
                        acc[i] = acc[i] + Float32(term[i])

            # One partial per remote node, ascending node order.
            for node in cutlass.range_constexpr(num_nodes):
                if cutlass.const_expr(node != my_node):
                    slot: cutlass.Constexpr[int] = node if node < my_node else node - 1
                    hits = Int32(0)
                    for k in cutlass.range_constexpr(num_topk):
                        if slot_node[k] == Int32(node):
                            hits = hits + Int32(1)
                    if hits > Int32(0):
                        src = cute.zipped_divide(
                            remote_partial[slot, token_idx, None], (hpt,),
                        )[(None,), (hidden_tile_idx,)]
                        term = cute.make_rmem_tensor((hpt,), cutlass.BFloat16)
                        cute.copy(load_atom, self._aligned(src), term)
                        for i in cutlass.range_constexpr(hpt):
                            acc[i] = acc[i] + Float32(term[i])

            dst = cute.zipped_divide(
                reduced_output[token_idx, None], (hpt,),
            )[(None,), (hidden_tile_idx,)]
            out = cute.make_rmem_tensor((hpt,), cutlass.BFloat16)
            out.store(acc.load().to(cutlass.BFloat16))
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128,
                ),
                out,
                self._aligned(dst),
            )
