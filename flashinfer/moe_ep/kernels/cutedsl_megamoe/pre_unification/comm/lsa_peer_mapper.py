"""Fail-closed peer-pointer mapping for an NCCL LSA arena.

The mapper stores one address delta for each rank in the caller's logical
NVLink domain. Kernel call sites continue to pass global ranks, but ``map``
converts them to a node-local index and traps if a cross-node rank is supplied.
This prevents an absent LSA mapping from silently degrading to the local base.
"""


from dataclasses import dataclass
from typing import Any, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64, dsl_user_op
from cutlass.base_dsl.dsl import (
    extract_mlir_values,
    get_mlir_types,
    new_from_mlir_values,
)
from cutlass.base_dsl.runtime.jit_arg_adapters import JitArgAdapterRegistry
from cutlass.base_dsl.typing import get_c_pointers
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm

try:
    from cutlass.base_dsl.typing import MLIR_DYNAMIC_INDEX
except ImportError:  # pragma: no cover - fixed by the MLIR GEP encoding ABI.
    MLIR_DYNAMIC_INDEX = -(2**31)


# The alternative vector payload selected for R > 16 is implemented, but no
# MegaMoE runtime validation currently exercises that launch-ABI path.  Treat
# it as unsupported pending a focused all-peer mapping probe and exact run.
_BYVAL_RANK_LIMIT = 16


def _byval_struct_ty() -> Any:
    return ir.Type.parse(f"!llvm.struct<(array<{_BYVAL_RANK_LIMIT} x i64>)>")


@dsl_user_op
def _device_trap(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "trap;",
        "",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dataclass(frozen=True)
class LsaPeerMapperDevice:
    """Device mapper containing node-local offsets and the caller global rank."""

    val: Any
    rank_idx: Int32
    ranks_per_node: cutlass.Constexpr[int]

    def __extract_mlir_values__(self) -> list:
        return [self.val, self.rank_idx.ir_value()]

    def __new_from_mlir_values__(self, values: list) -> "LsaPeerMapperDevice":
        return LsaPeerMapperDevice(
            val=values[0],
            rank_idx=Int32(values[1]),
            ranks_per_node=self.ranks_per_node,
        )

    def __get_mlir_types__(self) -> list:
        if self.ranks_per_node <= _BYVAL_RANK_LIMIT:
            offset_ty = ir.Type.parse("!llvm.ptr")
        else:
            offset_ty = ir.Type.parse(f"vector<{self.ranks_per_node}xi64>")
        return [offset_ty, ir.Type.parse("i32")]

    def __extract_mlir_attributes__(self) -> list:
        if self.ranks_per_node <= _BYVAL_RANK_LIMIT:
            offset_attrs = ir.DictAttr.get({
                "cute_nvgpu.grid_constant": ir.UnitAttr.get(),
                "llvm.byval": ir.TypeAttr.get(_byval_struct_ty()),
            })
        else:
            offset_attrs = ir.DictAttr.get({})
        return [offset_attrs, ir.DictAttr.get({})]

    @cute.jit
    def _offset_for_local_peer(self, peer_local_idx: Int32) -> Int64:
        if cutlass.const_expr(self.ranks_per_node <= _BYVAL_RANK_LIMIT):
            i64_ty = ir.Type.parse("i64")
            off_ptr = llvm.getelementptr(
                ir.Type.parse("!llvm.ptr"),
                self.val,
                [peer_local_idx.ir_value()],
                [MLIR_DYNAMIC_INDEX],
                i64_ty,
                no_wrap_flags="None",
            )
            return Int64(llvm.load(i64_ty, off_ptr))
        return Int64(llvm.extractelement(self.val, peer_local_idx.ir_value()))

    @cute.jit
    def map(
        self,
        local_ptr: Int64,
        dst_rank_idx: Int32,
        byte_off: Int64 = Int64(0),
    ) -> Int64:
        """Map a local LSA-arena address to a same-node global rank.

        ``dst_rank_idx`` remains a global rank so existing routing call sites do
        not change. Invalid and cross-node ranks execute a device ``trap``;
        they never reuse offset zero or otherwise alias the local allocation.
        """
        if dst_rank_idx < Int32(0):
            _device_trap()
        my_node = self.rank_idx // Int32(self.ranks_per_node)
        dst_node = dst_rank_idx // Int32(self.ranks_per_node)
        if dst_node != my_node:
            _device_trap()

        peer_local_idx = dst_rank_idx % Int32(self.ranks_per_node)
        return local_ptr + self._offset_for_local_peer(peer_local_idx) + byte_off

    @cute.jit
    def ptr_map_to_rank(
        self,
        ptr,
        dst_rank_idx: Int32,
        byte_align: Optional[int] = None,
    ):
        if cutlass.const_expr(ptr.memspace != AddressSpace.gmem):
            raise ValueError(
                "ptr_map_to_rank: source pointer must live in the NCCL LSA "
                f"arena GMEM, got memspace={ptr.memspace}."
            )
        if cutlass.const_expr(byte_align is None):
            byte_align = ptr.max_alignment
        peer_addr = self.map(ptr.toint(), dst_rank_idx, Int64(0))
        return cute.make_ptr(
            ptr.dtype,
            peer_addr,
            ptr.memspace,
            assumed_align=byte_align,
        )


@dataclass(frozen=True)
class LsaPeerMapperHost:
    """Runtime launch payload for a node-local NCCL LSA peer mapper."""

    base_addr: Int64
    offsets: tuple
    rank_idx: Int32
    num_max_ranks: cutlass.Constexpr[int]

    @property
    def ranks_per_node(self) -> int:
        return int(self.num_max_ranks)

    def validate(self) -> None:
        """Validate the concrete host payload before passing it to CuTe.

        This is intentionally not ``__post_init__``: CuTe reconstructs the
        dataclass with symbolic DSL scalars while compiling, and those values
        cannot be converted to Python ``int`` objects.
        """
        ranks_per_node = int(self.num_max_ranks)
        rank_idx = int(self.rank_idx)
        if ranks_per_node <= 0:
            raise ValueError(
                f"ranks_per_node must be positive, got {ranks_per_node}."
            )
        if rank_idx < 0:
            raise ValueError(f"rank_idx must be non-negative, got {rank_idx}.")
        if len(tuple(self.offsets)) != ranks_per_node:
            raise ValueError(
                f"len(offsets)={len(tuple(self.offsets))} must equal "
                f"ranks_per_node={ranks_per_node}."
            )

    def map_peer_offset(self, dst_rank_idx: int) -> int:
        """Return a host-visible same-node delta or reject the peer."""
        peer = int(dst_rank_idx)
        rank = int(self.rank_idx)
        ranks_per_node = int(self.num_max_ranks)
        if peer < 0:
            raise ValueError(f"dst_rank_idx must be non-negative, got {peer}.")
        if peer // ranks_per_node != rank // ranks_per_node:
            raise ValueError(
                f"cross-node LSA mapping rejected: rank={rank} peer={peer} "
                f"ranks_per_node={ranks_per_node}."
            )
        return int(tuple(self.offsets)[peer % ranks_per_node])

    @staticmethod
    def _as_int64(value) -> Int64:
        return value if isinstance(value, Int64) else Int64(int(value))

    @staticmethod
    def _as_int32(value) -> Int32:
        return value if isinstance(value, Int32) else Int32(int(value))

    @dsl_user_op
    def make_device_obj(self, *, loc=None, ip=None) -> Any:
        offsets = tuple(self.offsets)
        ranks_per_node = self.num_max_ranks

        if ranks_per_node <= _BYVAL_RANK_LIMIT:
            ptr_ty = ir.Type.parse("!llvm.ptr")
            st_ty = _byval_struct_ty()
            i64_ty = ir.Type.parse("i64")
            one = arith.constant(
                value=ir.IntegerAttr.get(i64_ty, 1),
                result=i64_ty,
                loc=loc,
                ip=ip,
            )
            buf = llvm.alloca(
                res=ptr_ty,
                elem_type=st_ty,
                array_size=one,
                alignment=64,
                loc=loc,
                ip=ip,
            )
            for i, off in enumerate(offsets):
                slot = llvm.getelementptr(
                    ptr_ty,
                    buf,
                    [],
                    [i],
                    i64_ty,
                    no_wrap_flags="None",
                    loc=loc,
                    ip=ip,
                )
                llvm.store(self._as_int64(off).ir_value(), slot, loc=loc, ip=ip)
            val = buf
        else:
            i32_ty = ir.Type.parse("i32")
            vec_ty = ir.Type.parse(f"vector<{ranks_per_node}xi64>")
            val = llvm.mlir_zero(vec_ty, loc=loc, ip=ip)
            for i, off in enumerate(offsets):
                idx = arith.constant(
                    value=ir.IntegerAttr.get(i32_ty, i),
                    result=i32_ty,
                    loc=loc,
                    ip=ip,
                )
                val = llvm.insertelement(
                    val,
                    self._as_int64(off).ir_value(),
                    idx,
                    loc=loc,
                    ip=ip,
                )

        return LsaPeerMapperDevice(
            val=val,
            rank_idx=self._as_int32(self.rank_idx),
            ranks_per_node=self.num_max_ranks,
        )


@JitArgAdapterRegistry.register_jit_arg_adapter(LsaPeerMapperHost)
class _LsaPeerMapperHostAdapter:
    def __init__(self, arg: LsaPeerMapperHost) -> None:
        self._arg = arg
        self._fields = (
            Int64(arg.base_addr),
            *(Int64(x) for x in tuple(arg.offsets)),
            Int32(arg.rank_idx),
        )

    def __c_pointers__(self) -> list[Any]:
        out: list[Any] = []
        for field in self._fields:
            out.extend(get_c_pointers(field))
        return out

    def __get_mlir_types__(self) -> list[Any]:
        out: list[Any] = []
        for field in self._fields:
            out.extend(get_mlir_types(field))
        return out

    def __extract_mlir_values__(self) -> list[ir.Value]:
        out: list[ir.Value] = []
        for field in self._fields:
            out.extend(extract_mlir_values(field))
        return out

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> LsaPeerMapperHost:
        idx = 0
        base_field = self._fields[0]
        n = len(get_mlir_types(base_field))
        base_addr = new_from_mlir_values(base_field, values[idx:idx + n])
        idx += n

        offsets = []
        for field in self._fields[1:-1]:
            n = len(get_mlir_types(field))
            offsets.append(new_from_mlir_values(field, values[idx:idx + n]))
            idx += n

        rank_field = self._fields[-1]
        n = len(get_mlir_types(rank_field))
        rank_idx = new_from_mlir_values(rank_field, values[idx:idx + n])
        idx += n
        if idx != len(values):
            raise ValueError(
                f"LsaPeerMapperHost adapter consumed {idx} values, "
                f"got {len(values)}."
            )

        obj = object.__new__(LsaPeerMapperHost)
        object.__setattr__(obj, "base_addr", base_addr)
        object.__setattr__(obj, "offsets", tuple(offsets))
        object.__setattr__(obj, "rank_idx", rank_idx)
        object.__setattr__(obj, "num_max_ranks", self._arg.num_max_ranks)
        return obj
