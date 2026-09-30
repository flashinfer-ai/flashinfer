"""Host PLE geometry and explicit checkpoint-buffer allocation.

Geometry calculation is separate from executable declarations so loaders can
register and fill the exact tensors later borrowed by prepared plans.
"""
from __future__ import annotations

from dataclasses import dataclass
import weakref

import torch

from b12x._lib.scratch_layout import align_up
from b12x.preparation import PersistentMemory
from .reference import is_prime_64, ple_multipliers, ple_table_geometry


_SIGNED_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class Geometry:
    prime_sizes: tuple[int, ...]
    table_offsets: tuple[int, ...]
    multipliers: tuple[int, ...]
    table_vocab_size: int
    padded_vocab_size: int

    def __post_init__(self):
        for name in ("prime_sizes", "table_offsets", "multipliers"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def key(self):
        return self.prime_sizes, self.table_offsets, self.multipliers, self.padded_vocab_size

    @property
    def nbytes(self):
        return 8 * (len(self.prime_sizes) + len(self.table_offsets) + len(self.multipliers))


@dataclass(frozen=True)
class GeometryTensors:
    geometry: Geometry
    prime_sizes: torch.Tensor
    table_offsets: torch.Tensor
    multipliers: torch.Tensor


def compute_geometry(caps, *, prime_sizes=None, table_offsets=None, multipliers=None) -> Geometry:
    """Calculate or validate immutable geometry, without allocating device storage.

    Explicit tensor values are read here, not in plan construction. A loader may
    call this on checkpoint metadata before supplying it to a declaration.
    """
    if (prime_sizes is None) != (table_offsets is None):
        raise ValueError("prime_sizes and table_offsets must be provided together")
    if prime_sizes is None:
        prime_sizes, table_offsets = ple_table_geometry(
            base_size=caps.base_table_size, dense_layer_ordinal=caps.dense_layer_ordinal,
            total_heads=caps.head_count,
        )
    if multipliers is None:
        multipliers = ple_multipliers(
            vocab_size=caps.vocab_size, max_order=caps.max_order,
            dense_layer_ordinal=caps.dense_layer_ordinal,
        )
    values = {}
    for name, tensor, count in (
        ("prime_sizes", prime_sizes, caps.head_count),
        ("table_offsets", table_offsets, caps.head_count),
        ("multipliers", multipliers, caps.max_order),
    ):
        if isinstance(tensor, torch.Tensor):
            if tuple(tensor.shape) != (count,):
                raise ValueError(f"{name} must have shape {(count,)}")
            if tensor.dtype != torch.int64:
                raise TypeError(f"{name} must have dtype torch.int64")
            value = tuple(int(item) for item in tensor.detach().cpu().tolist())
        else:
            value = tuple(tensor)
            if len(value) != count or any(type(item) is not int for item in value):
                raise ValueError(f"{name} must contain {count} integers")
        values[name] = value
    total = 0
    for head, (size, offset) in enumerate(zip(values["prime_sizes"], values["table_offsets"], strict=True)):
        if not is_prime_64(size):
            raise ValueError(f"prime_sizes[{head}]={size} is not prime")
        if offset != total:
            raise ValueError(f"table_offsets[{head}] must be {total}, got {offset}")
        total += size
        if total > _SIGNED_INT64_MAX:
            raise ValueError("cumulative PLE table extent must fit signed int64")
    maximum = _SIGNED_INT64_MAX // caps.vocab_size
    for index, factor in enumerate(values["multipliers"]):
        if factor <= 0 or factor % 2 != 1 or factor > maximum:
            raise ValueError(f"multipliers[{index}] must be positive, odd, and at most {maximum}")
    padded = align_up(total, caps.table_alignment)
    if padded > _SIGNED_INT64_MAX:
        raise ValueError("padded PLE table extent must fit signed int64")
    return Geometry(values["prime_sizes"], values["table_offsets"], values["multipliers"], total, padded)


def allocate_geometry(geometry: Geometry, *, device) -> GeometryTensors:
    """Allocate fresh checkpoint-owned geometry tensors before weight loading."""
    if not isinstance(geometry, Geometry):
        raise TypeError("allocate_geometry requires host Geometry")
    tensors = tuple(torch.tensor(value, dtype=torch.int64, device=device) for value in (
        geometry.prime_sizes, geometry.table_offsets, geometry.multipliers,
    ))
    return GeometryTensors(geometry, *tensors)


class _GeometryInputs:
    """Declaration-owned parameter references; generated device copies stay weak."""

    def __init__(self, caps, *, geometry=None, prime_sizes=None, table_offsets=None, multipliers=None):
        if (prime_sizes is None) != (table_offsets is None):
            raise ValueError("prime_sizes and table_offsets must be provided together")
        sources = (prime_sizes, table_offsets, multipliers)
        if geometry is None:
            if any(isinstance(value, torch.Tensor) and value.device.type != "cpu" for value in sources):
                raise ValueError("device checkpoint geometry requires explicit host compute_geometry metadata")
            geometry = compute_geometry(
                caps, prime_sizes=prime_sizes, table_offsets=table_offsets, multipliers=multipliers,
            )
        if not isinstance(geometry, Geometry):
            raise TypeError("geometry must be host Geometry")
        validated = compute_geometry(
            caps, prime_sizes=geometry.prime_sizes, table_offsets=geometry.table_offsets,
            multipliers=geometry.multipliers,
        )
        if validated != geometry:
            raise ValueError("host PLE geometry has inconsistent extents")
        for name, source, count in zip(
            ("prime_sizes", "table_offsets", "multipliers"), sources,
            (caps.head_count, caps.head_count, caps.max_order),
        ):
            if source is not None:
                if not isinstance(source, torch.Tensor) or source.dtype != torch.int64:
                    raise TypeError(f"{name} must be an int64 tensor")
                if tuple(source.shape) != (count,):
                    raise ValueError(f"{name} must have shape {(count,)}")
        self.caps = caps
        self.geometry = geometry
        self.sources = sources
        self.key = ("ple.geometry", object())
        self._storage = None
        self._validated = False

    def memory(self, device):
        cached = None if self._storage is None else self._storage()
        if cached is not None and cached.prime_sizes.device == device:
            resident = self.geometry.nbytes
        else:
            resident = sum(
                source.numel() * source.element_size()
                for source in self.sources
                if source is not None and source.device == device and source.is_contiguous()
            )
        return PersistentMemory(self.key, self.geometry.nbytes, resident)

    def materialize(self, device):
        if not self._validated:
            values = tuple(
                expected if source is None else source
                for source, expected in zip(self.sources, (
                    self.geometry.prime_sizes, self.geometry.table_offsets, self.geometry.multipliers,
                ))
            )
            actual = compute_geometry(
                self.caps, prime_sizes=values[0], table_offsets=values[1], multipliers=values[2],
            )
            if actual != self.geometry:
                raise ValueError("checkpoint tensors disagree with declared PLE geometry")
            self._validated = True
        cached = None if self._storage is None else self._storage()
        if cached is not None and cached.prime_sizes.device == device:
            return cached
        tensors = tuple(
            source.to(device=device).contiguous() if source is not None
            else torch.tensor(values, dtype=torch.int64, device=device)
            for source, values in zip(self.sources, (
                self.geometry.prime_sizes, self.geometry.table_offsets, self.geometry.multipliers,
            ))
        )
        storage = GeometryTensors(self.geometry, *tensors)
        self._storage = weakref.ref(storage)
        return storage
