"""Declarative GMEM workspace shared by host layout and device access."""

import copy
import dataclasses
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64

from ...common.utils import ceil_div, round_up
from .utils import (
    cosize_from_shape_stride_tuples,
    flatten_shape_stride,
    is_nested_shape,
    is_power_of_two,
    ordered_stride,
    row_major_stride,
    validate_static_integer_tuple,
)


BufferResetAttr = Literal["data", "zero_on_first_allocate", "tail_reset"]
# Regions register into "local" or "shared". "tmp" is never registered: a workspace built with
# separate_tmp=True derives it at finalize() from the per-launch data of the local space.
BufferSpace = Literal["local", "shared", "tmp"]

_registration_spaces = ("local", "shared")
_buffer_spaces = ("local", "shared", "tmp")

_reset_order = {"tail_reset": 0, "zero_on_first_allocate": 1, "data": 2}


def _footprint_from_shape_stride(shape: Tuple, stride: Tuple) -> int:
    """Elements a region must own, as opposed to the ones its layout can address."""
    leaf_pairs = flatten_shape_stride(shape, stride) if shape else []
    claimed = max((size * step for size, step in leaf_pairs), default=1)
    # Layouts whose leaves overlap can address past what any single leaf tiles,
    # so the cosize stays a floor.
    return int(max(claimed, cosize_from_shape_stride_tuples(shape, stride)))


@dataclasses.dataclass(frozen=True)
class DeviceRegion:
    """One typed region in the local workspace, the shared (symmetric) workspace or the tmp buffer."""

    name: str
    dtype: Type[cutlass.Numeric]
    shape: Tuple
    buffer_space: BufferSpace
    stride: Optional[Tuple] = None
    mem_order: Optional[Tuple[int, ...]] = None
    byte_alignment: int = 16
    reset: BufferResetAttr = "data"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A workspace region needs a non-empty name.")
        if self.buffer_space not in _buffer_spaces:
            raise ValueError(f"Invalid buffer space {self.buffer_space!r}.")
        if self.reset not in _reset_order:
            raise ValueError(f"Invalid reset policy {self.reset!r}.")
        if not is_power_of_two(self.byte_alignment):
            raise ValueError(f"Region {self.name!r} alignment must be a positive power of two.")
        validate_static_integer_tuple(self.shape, field_name=f"{self.name}.shape")
        if self.stride is not None and self.mem_order is not None:
            raise ValueError(f"Region {self.name!r} accepts either stride or mem_order, not both.")
        if self.stride is None and self.mem_order is None:
            if is_nested_shape(self.shape) or len(self.shape) != 1:
                raise ValueError(f"Region {self.name!r} needs an explicit layout.")
            object.__setattr__(self, "stride", row_major_stride(self.shape))
        if is_nested_shape(self.shape) and self.mem_order is not None:
            raise ValueError(f"Nested region {self.name!r} needs an explicit stride.")
        if self.stride is not None:
            validate_static_integer_tuple(self.stride, field_name=f"{self.name}.stride")
        if self.mem_order is not None:
            validate_static_integer_tuple(self.mem_order, field_name=f"{self.name}.mem_order")
            expected = tuple(range(len(self.shape)))
            if tuple(sorted(self.mem_order)) != expected:
                raise ValueError(
                    f"Region {self.name!r} mem_order must be a permutation of {expected}, got {self.mem_order}."
                )
            resolved_stride, _ = ordered_stride(self.shape, self.mem_order)
            object.__setattr__(self, "stride", resolved_stride)


def _placed_in(region: DeviceRegion, buffer_space: BufferSpace) -> DeviceRegion:
    """The same region in another buffer (dataclasses.replace would resolve its layout a second time)."""
    placed = copy.copy(region)
    object.__setattr__(placed, "buffer_space", buffer_space)
    return placed


class DeviceWorkspace:
    """Single source of truth for GMEM region layout and device pointer derivation.

    Regions are registered into the local (rank-private) or the shared (symmetric) space. Inside a
    space finalize() puts the kernel state first (tail_reset, then zero_on_first_allocate) and the
    per-launch data behind it.

    separate_tmp=True splits the local space at that boundary into two buffers:

    local  kernel state only: persistent, zeroed once by the host, maintained by the kernel.
    tmp    the per-launch data. No launch depends on what an earlier launch left there and
           nothing in it has to survive the launch, so the caller owns the buffer, may fill it
           with anything between launches and may pass a different one to every launch.

    The split follows the reset class, so registration is the same in both modes. With
    separate_tmp=False (the default) there is one local buffer, state first and data behind
    it, and the tmp buffer is empty.
    """

    def __init__(self, separate_tmp: bool = False) -> None:
        self._separate_tmp = bool(separate_tmp)
        self._registered: Dict[BufferSpace, List[DeviceRegion]] = {"local": [], "shared": [], "tmp": []}
        self._region_by_name: Dict[str, DeviceRegion] = {}
        self._offset: Dict[str, int] = {}
        self._stride: Dict[str, Tuple] = {}
        self._cosize: Dict[str, int] = {}
        self._nbytes: Dict[str, int] = {}
        self._byte_alignment: Dict[str, int] = {}
        self._total: Dict[BufferSpace, int] = {"local": 0, "shared": 0, "tmp": 0}
        self._zero_leading: Dict[BufferSpace, int] = {"local": 0, "shared": 0, "tmp": 0}
        self._tail_leading: Dict[BufferSpace, int] = {"local": 0, "shared": 0, "tmp": 0}
        self._base: Dict[BufferSpace, Any] = {"local": None, "shared": None, "tmp": None}
        self._finalized = False

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "DeviceWorkspace":
        return self

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def separate_tmp(self) -> bool:
        """True when the per-launch data of the local space lives in its own tmp buffer."""
        return self._separate_tmp

    def register(
        self,
        name: str,
        dtype: Type[cutlass.Numeric],
        shape: Tuple,
        *,
        buffer_space: BufferSpace,
        stride: Optional[Tuple] = None,
        mem_order: Optional[Tuple[int, ...]] = None,
        byte_alignment: int = 16,
        reset: BufferResetAttr = "data",
    ) -> None:
        """Add a region to the local or the shared space.

        reset="data" is a statement about lifetime: no launch depends on what an earlier launch left in
        the region. In a workspace built with separate_tmp=True such a local region lives in the
        caller-owned tmp buffer, so it must tolerate arbitrary content at launch and a different buffer
        on every launch, and it cannot carry anything from one launch to the next.
        """
        if self._finalized:
            raise RuntimeError("Cannot register a region after finalize().")
        if buffer_space not in _registration_spaces:
            raise ValueError(
                f"Region {name!r}: regions register into 'local' or 'shared', not {buffer_space!r}. "
                "The tmp buffer is derived from the reset class of the local regions."
            )
        if name in self._region_by_name or any(
            region.name == name for regions in self._registered.values() for region in regions
        ):
            raise ValueError(f"Duplicate workspace region {name!r}.")
        self._registered[buffer_space].append(
            DeviceRegion(
                name=name,
                dtype=dtype,
                shape=shape,
                buffer_space=buffer_space,
                stride=stride,
                mem_order=mem_order,
                byte_alignment=byte_alignment,
                reset=reset,
            )
        )

    def finalize(self) -> None:
        if self._finalized:
            raise RuntimeError("DeviceWorkspace.finalize() may only be called once.")
        if self._separate_tmp:
            # Registration order is kept inside both halves and the tmp offsets restart at zero. Where
            # the data of the single-buffer layout starts on a multiple of its largest alignment, the
            # tmp buffer is exactly that data suffix.
            local_regions = self._registered["local"]
            self._registered["local"] = [region for region in local_regions if region.reset != "data"]
            self._registered["tmp"] = [
                _placed_in(region, "tmp") for region in local_regions if region.reset == "data"
            ]
        for buffer_space in _buffer_spaces:
            ordered = sorted(self._registered[buffer_space], key=lambda region: _reset_order[region.reset])
            cursor = 0
            for region_index, region in enumerate(ordered):
                stride = region.stride
                if stride is None:
                    raise RuntimeError(f"Region {region.name!r} stride was not resolved.")
                cosize = cosize_from_shape_stride_tuples(region.shape, stride)
                nbytes = (_footprint_from_shape_stride(region.shape, stride) * int(region.dtype.width) + 7) // 8
                if region.reset == "tail_reset" and (
                    region_index == 0 or ordered[region_index - 1].reset != "tail_reset"
                ):
                    cursor = round_up(cursor, 16)
                byte_alignment = region.byte_alignment
                cursor = round_up(cursor, byte_alignment)
                self._region_by_name[region.name] = region
                self._offset[region.name] = cursor
                self._stride[region.name] = stride
                self._cosize[region.name] = cosize
                self._nbytes[region.name] = nbytes
                self._byte_alignment[region.name] = byte_alignment
                cursor += nbytes
                is_last_tail_reset_region = region.reset == "tail_reset" and (
                    region_index + 1 == len(ordered) or ordered[region_index + 1].reset != "tail_reset"
                )
                if is_last_tail_reset_region:
                    cursor = round_up(cursor, 16)
                if region.reset != "data":
                    self._zero_leading[buffer_space] = cursor
                if is_last_tail_reset_region:
                    self._tail_leading[buffer_space] = cursor
            self._total[buffer_space] = round_up(cursor, 16)
        self._finalized = True

    def regions(self, buffer_space: BufferSpace) -> Tuple[DeviceRegion, ...]:
        """Regions of one space: as registered before finalize(), as placed afterwards.

        With separate_tmp=True finalize() moves the local data regions to "tmp", so a placement read
        before finalize() must not be kept.
        """
        return tuple(self._registered[buffer_space])

    def region(self, name: str) -> DeviceRegion:
        self._require_finalized()
        return self._region_by_name[name]

    def offset(self, name: str) -> int:
        self._require_finalized()
        return self._offset[name]

    def stride(self, name: str) -> Tuple:
        self._require_finalized()
        return self._stride[name]

    def cosize(self, name: str) -> int:
        self._require_finalized()
        return self._cosize[name]

    def nbytes(self, name: str) -> int:
        self._require_finalized()
        return self._nbytes[name]

    def byte_alignment(self, name: str) -> int:
        self._require_finalized()
        return self._byte_alignment[name]

    def total_bytes(self, buffer_space: BufferSpace) -> int:
        self._require_finalized()
        return self._total[buffer_space]

    @property
    def local_and_shared_bytes(self) -> Tuple[int, int]:
        self._require_finalized()
        return self._total["local"], self._total["shared"]

    @property
    def tmp_bytes(self) -> int:
        """Size of the caller-owned tmp buffer; 0 unless the workspace separates it."""
        self._require_finalized()
        return self._total["tmp"]

    def base_alignment(self, buffer_space: BufferSpace) -> int:
        """Alignment the base pointer of a buffer needs: the largest one its regions assume."""
        self._require_finalized()
        return max((region.byte_alignment for region in self._registered[buffer_space]), default=1)

    def zero_on_allocate_bytes(self, buffer_space: BufferSpace) -> int:
        self._require_finalized()
        return self._zero_leading[buffer_space]

    def tail_reset_bytes(self, buffer_space: BufferSpace) -> int:
        self._require_finalized()
        return self._tail_leading[buffer_space]

    @property
    def require_zero_workspace_leading_bytes(self) -> Tuple[int, int]:
        self._require_finalized()
        return self._zero_leading["local"], self._zero_leading["shared"]

    @cute.jit
    def assign_device_members(
        self,
        local_workspace: cute.Pointer,
        shared_workspace: Optional[cute.Pointer] = None,
        tmp_buffer: Optional[cute.Pointer] = None,
    ) -> None:
        # A buffer that stays None is unbound: ptr() of a region placed there fails at trace time
        # instead of falling back to another base.
        self._base["local"] = local_workspace
        self._base["shared"] = shared_workspace
        self._base["tmp"] = tmp_buffer

    def remove_device_members(self) -> None:
        self._base = {"local": None, "shared": None, "tmp": None}

    @cute.jit
    def ptr(self, name: str) -> cute.Pointer:
        region = self._region_by_name[name]
        base = self._base[region.buffer_space]
        address = base.toint() + Int64(self._offset[name])
        return cute.make_ptr(region.dtype, address, AddressSpace.gmem, assumed_align=self._byte_alignment[name])

    @cute.jit
    def tensor(self, name: str) -> cute.Tensor:
        region = self._region_by_name[name]
        return cute.make_tensor(self.ptr(name), cute.make_layout(region.shape, stride=self._stride[name]))

    @cute.jit
    def reset_tail(self, tid: Int32, total_threads: int) -> None:
        self.reset_tail_space("local", tid, total_threads)
        self.reset_tail_space("shared", tid, total_threads)

    @cute.jit
    def reset_tail_space(self, buffer_space: BufferSpace, tid: Int32, total_threads: int) -> None:
        num_vectors = self._tail_leading[buffer_space] // 16
        if cutlass.const_expr(num_vectors > 0):
            vectors = cute.make_tensor(
                cute.make_ptr(Int32, self._base[buffer_space].toint(), AddressSpace.gmem, assumed_align=16),
                cute.make_layout((num_vectors, 4), stride=(4, 1)),
            )
            zero = cute.make_rmem_tensor((4,), Int32)
            for element in cutlass.range_constexpr(4):
                zero[element] = Int32(0)
            store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=128)
            reset_round_count = ceil_div(num_vectors, total_threads)
            for reset_round in cutlass.range_constexpr(reset_round_count):
                vector_index = Int32(reset_round * total_threads) + tid
                if vector_index < Int32(num_vectors):
                    cute.copy(store_atom, zero, vectors[vector_index, None])

    def _require_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError("DeviceWorkspace must be finalized first.")
