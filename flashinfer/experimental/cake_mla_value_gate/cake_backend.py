"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Host of the Cake experimental MLA value projection + sigmoid output gate
(SM100 / SM103): validation of the bound weight and of every launch, the
logical argument values of the program and the runner that launches it.

For ``t < B``, ``h < H`` and ``v < 128``, one launch computes::

    S[t, h, v]        = sum_k latent[t, h, k] * w_uv[h, k, v]          (FP32 accumulation, k < 512)
    out[t, h*128 + v] = bf16(f32(bf16(S[t, h, v])) * sigmoid(f32(gate[t, h*128 + v])))
                                                       if valid_lengths[t // q] > 0
    out[t, h*128 + v] = +0.0                           if valid_lengths[t // q] <= 0

The sigmoid is evaluated in FP32 and is not rounded to BF16
(``sigmoid_precision="fp32"``, the only mode offered).  :func:`prepare_mla_value_gate`
runs outside CUDA graph capture: it checks the device and the weight, decides
which program serves each row count ``B`` and loads (or JIT-builds) those
programs.  :meth:`MlaValueGateRunner.launch` checks host metadata only (shapes,
strides, dtypes, devices, alignment, overlap), looks up the program decided for
``B``, computes its grid on the host and submits one kernel on the current
torch stream without allocating or synchronising, so it can be captured in a
CUDA graph.  The kernel reads ``w_uv`` from the storage bound at prepare and
the other tensors from the pointers of each launch.

An architecture may have two programs that differ only in the token rows one
CTA owns (``grid_rule["rows"]``: 16 or 8).  They give identical bits; the
choice affects time only.  :func:`select_tile_rows` is the rule: the 8-row
program when ``enable_pdl`` is on or when its grid fits one wave
(``4 * H * ceil(B / 8)`` CTAs, at most the device's SM count), else the 16-row
program.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import tvm_ffi

from .cake_jit import (
    LAUNCH_ARGUMENTS,
    LOGICAL_SCALARS,
    LOGICAL_TENSORS,
    MODULES,
    VALUE_DIM,
    device_arch,
    device_sm_count,
    launch_grid,
    load,
    records_for,
)
from .cake_jit import (
    tile_rows as record_tile_rows,
)

__all__ = [
    "LATENT_DIM",
    "MAX_ROWS",
    "SIGMOID_PRECISIONS",
    "VALUE_DIM",
    "MlaValueGateRunner",
    "prepare_mla_value_gate",
    "select_tile_rows",
]

#: The latent width of one head (the ``K`` of the projection).
LATENT_DIM = 512
#: The largest token-row count a runner accepts (the qualified decode widths).
MAX_ROWS = 64
#: The sigmoid numerics the API offers: ``"fp32"`` -- evaluated in FP32, not rounded to BF16.
SIGMOID_PRECISIONS = ("fp32",)
#: ``w_uv`` is the ``[H, 512, 128]`` view of a contiguous ``[H, 128, 512]`` tensor (latent axis contiguous).
_W_UV_STRIDES = (LATENT_DIM * VALUE_DIM, 1, LATENT_DIM)
#: ``latent``, ``w_uv`` and ``out`` start on 16-byte boundaries; gate rows on 4-byte boundaries.
_ALIGN = 16
_GATE_ALIGN = 4
#: The kernel indexes its operands with 32-bit integers.
_INT32_LIMIT = 2**31
#: CUDA's limit on the y and z grid dimensions (``heads`` is grid y).
_GRID_YZ_LIMIT = 65535


def _int_arg(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}") from None


def _check_tensor(x: Any, name: str, dtype: torch.dtype) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {x.dtype}")
    if x.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor, got device {x.device}")


def _byte_range(x: torch.Tensor) -> tuple[int, int]:
    """``[first, end)`` bytes of the storage a non-empty view with non-negative strides covers."""
    start = x.data_ptr()
    elements = 1 + sum(
        (int(size) - 1) * int(stride)
        for size, stride in zip(x.shape, x.stride(), strict=True)
        if int(size) > 1
    )
    return start, start + elements * x.element_size()


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _argument_plan(record: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """``(kind, host name)`` of every argument of the record's entry, in order: buffers and
    parameters by their logical name, the launch argument and the grid axes by their own.
    A name outside the host vocabulary is a registry/host mismatch and is refused here,
    before anything is loaded."""
    logical = record["logical"]
    plan: list[tuple[str, str]] = []
    for kind, key in record["arg_plan"]:
        if kind in ("buffer", "parameter"):
            name = logical.get(key)
            vocabulary = LOGICAL_TENSORS if kind == "buffer" else LOGICAL_SCALARS
            if name not in vocabulary:
                raise RuntimeError(
                    f"record {record['library']} binds {kind} {key!r} to {name!r}, "
                    f"outside the host vocabulary {vocabulary}"
                )
        elif kind == "launch":
            if key not in LAUNCH_ARGUMENTS:
                raise RuntimeError(
                    f"record {record['library']} takes the launch argument {key!r}; "
                    f"the host supplies {LAUNCH_ARGUMENTS}"
                )
            name = key
        elif kind == "grid":
            if key not in ("grid_x", "grid_y", "grid_z"):
                raise RuntimeError(
                    f"record {record['library']} has an unknown grid argument {key!r}"
                )
            name = key
        else:
            raise RuntimeError(
                f"record {record['library']} has an unknown argument kind {kind!r}"
            )
        plan.append((kind, name))
    return tuple(plan)


def select_tile_rows(
    records: Mapping[int, dict[str, Any]],
    *,
    rows: int,
    heads: int,
    enable_pdl: bool,
    sm_count: int,
) -> int:
    """Row tile of the program that serves a launch of ``rows`` token rows of ``heads`` heads.

    ``records`` maps each row tile to its record (the programs of one architecture).  With one
    program, that program.  With two, the smaller tile when ``enable_pdl`` is on or when its grid
    fits one wave (``prod(launch_grid)`` -- ``4 * H * ceil(B / 8)`` for the 8-row program -- is at
    most ``sm_count``), else the larger tile.  The programs give identical bits; the rule
    affects time only."""
    tiles = sorted(records)
    if len(tiles) == 1:
        return tiles[0]
    if len(tiles) != 2:
        raise RuntimeError(f"expected one or two row tiles, got {tiles}")
    small, large = tiles
    if enable_pdl:
        return small
    grid_x, grid_y, grid_z = launch_grid(records[small], rows=rows, heads=heads)
    return small if grid_x * grid_y * grid_z <= sm_count else large


@dataclass(frozen=True, eq=False)
class _Program:
    """One loaded program: its record, argument plan and entry."""

    name: str
    tile_rows: int
    record: dict[str, Any] = field(repr=False)
    plan: tuple[tuple[str, str], ...] = field(repr=False)
    entry: Any = field(repr=False)


@dataclass(frozen=True, eq=False)
class MlaValueGateRunner:
    """The prepared programs of one bound ``w_uv``.

    ``launch(latent, gate, valid_lengths, out)`` (or a call) checks host
    metadata only, submits one kernel on the current torch stream -- no
    allocation, no synchronisation -- and returns ``out``.  Which program serves
    each row count ``B`` was decided at prepare (:func:`select_tile_rows`) and
    does not change: :meth:`program_for` and :meth:`tile_rows_for` report it.  A
    launch may be captured in a CUDA graph; a replay re-reads the contents of
    the captured tensors (and of ``w_uv``), so their storage must stay alive and
    in place.  The runner keeps a reference to ``w_uv``: in-place updates of its
    contents are seen by the next launch, while a new weight tensor (or a
    re-aliased one) needs a new :func:`prepare_mla_value_gate` and a new
    capture.  With ``enable_pdl`` the kernel may start before the preceding
    kernel of the stream completes: it reads ``w_uv`` before it waits on that
    kernel and reads ``latent``, ``gate`` and ``valid_lengths`` and writes
    ``out`` only after, so the preceding kernel must not write ``w_uv``; a
    following kernel launched with programmatic dependent launch must wait
    (``griddepcontrol.wait``) before it reads ``out``.
    """

    w_uv: torch.Tensor
    max_rows: int
    rows_per_request: int
    sigmoid_precision: str
    enable_pdl: bool
    arch: str
    sm_count: int
    programs: tuple[str, ...]
    _programs: tuple[_Program, ...] = field(repr=False)
    _selection: tuple[int, ...] = field(repr=False)
    _w_uv_physical: torch.Tensor = field(repr=False)
    _w_uv_bytes: tuple[int, int] = field(repr=False)

    @property
    def heads(self) -> int:
        return int(self.w_uv.shape[0])

    def _program(self, rows: int) -> _Program:
        rows = _int_arg(rows, "rows")
        if not 1 <= rows <= self.max_rows:
            raise ValueError(f"B = {rows} is outside [1, max_rows = {self.max_rows}]")
        return self._programs[self._selection[rows - 1]]

    def program_for(self, rows: int) -> str:
        """Name of the program a launch of ``rows`` token rows runs."""
        return self._program(rows).name

    def tile_rows_for(self, rows: int) -> int:
        """Token rows per CTA of the program a launch of ``rows`` token rows runs (16 or 8)."""
        return self._program(rows).tile_rows

    @property
    def program(self) -> str:
        """Name of the program a launch of ``max_rows`` rows runs (see :meth:`program_for`)."""
        return self.program_for(self.max_rows)

    @property
    def device(self) -> torch.device:
        return self.w_uv.device

    def _check(
        self,
        latent: torch.Tensor,
        gate: torch.Tensor,
        valid_lengths: torch.Tensor,
        out: torch.Tensor,
    ) -> tuple[int, int]:
        """Validate one launch's tensors; returns ``(rows, gate_pitch)``."""
        heads, q, device = self.heads, self.rows_per_request, self.device
        width = heads * VALUE_DIM
        for name, x, dtype in (
            ("latent", latent, torch.bfloat16),
            ("gate", gate, torch.bfloat16),
            ("valid_lengths", valid_lengths, torch.int32),
            ("out", out, torch.bfloat16),
        ):
            _check_tensor(x, name, dtype)
            if x.device != device:
                raise ValueError(f"{name} is on {x.device}; w_uv is on {device}")
        if torch.cuda.current_device() != device.index:
            raise ValueError(
                f"the tensors are on {device} but the current CUDA device is "
                f"{torch.cuda.current_device()}; launch under torch.cuda.device({device.index})"
            )
        if latent.ndim != 3 or tuple(latent.shape[1:]) != (heads, LATENT_DIM):
            raise ValueError(
                f"latent must be [B, {heads}, {LATENT_DIM}], got {tuple(latent.shape)}"
            )
        rows = int(latent.shape[0])
        if not 1 <= rows <= self.max_rows:
            raise ValueError(f"B = {rows} is outside [1, max_rows = {self.max_rows}]")
        if rows % q:
            raise ValueError(f"B = {rows} is not a multiple of rows_per_request = {q}")
        if not latent.is_contiguous() or latent.data_ptr() % _ALIGN:
            raise ValueError(
                f"latent must be contiguous and start on a {_ALIGN}-byte boundary"
            )
        if tuple(out.shape) != (rows, width):
            raise ValueError(f"out must be [{rows}, {width}], got {tuple(out.shape)}")
        if not out.is_contiguous() or out.data_ptr() % _ALIGN:
            raise ValueError(
                f"out must be contiguous and start on a {_ALIGN}-byte boundary"
            )
        if tuple(gate.shape) != (rows, width):
            raise ValueError(f"gate must be [{rows}, {width}], got {tuple(gate.shape)}")
        if gate.stride(1) != 1:
            raise ValueError(f"gate needs unit column stride, got {gate.stride(1)}")
        pitch = int(gate.stride(0)) if rows > 1 else width
        if pitch < width or pitch % 2:
            raise ValueError(
                f"gate's row pitch {pitch} must be even and at least {width} elements"
            )
        if gate.data_ptr() % _GATE_ALIGN:
            raise ValueError(
                f"gate rows must start on {_GATE_ALIGN}-byte boundaries (an even element offset)"
            )
        if (rows - 1) * pitch + width >= _INT32_LIMIT:
            raise ValueError(
                f"gate's rows span {(rows - 1) * pitch + width} elements; the kernel's "
                "32-bit indices need fewer than 2**31"
            )
        if (
            tuple(valid_lengths.shape) != (rows // q,)
            or not valid_lengths.is_contiguous()
        ):
            raise ValueError(
                f"valid_lengths must be a contiguous [{rows // q}] tensor, "
                f"got shape {tuple(valid_lengths.shape)}"
            )
        written = _byte_range(out)
        for name, x in (
            ("latent", latent),
            ("gate", gate),
            ("valid_lengths", valid_lengths),
        ):
            if _overlap(written, _byte_range(x)):
                raise ValueError(f"out overlaps {name}")
        if _overlap(written, self._w_uv_bytes):
            raise ValueError("out overlaps w_uv")
        return rows, pitch

    def launch(
        self,
        latent: torch.Tensor,
        gate: torch.Tensor,
        valid_lengths: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Run the program on ``latent`` ``[B, H, 512]``, ``gate`` ``[B, H * 128]``
        (row pitch ``gate.stride(0)``), ``valid_lengths`` ``[B // q]`` and the
        caller-owned ``out`` ``[B, H * 128]``; returns ``out``."""
        rows, pitch = self._check(latent, gate, valid_lengths, out)
        program = self._programs[self._selection[rows - 1]]
        heads = self.heads
        span = (rows - 1) * pitch + heads * VALUE_DIM
        values: dict[str, Any] = {
            "latent": latent,
            "w_uv": self._w_uv_physical,
            # The kernel reads row t of the gate at element t * gate_pitch of this span.
            "gate": gate.as_strided((span,), (1,), gate.storage_offset()),
            "valid_lengths": valid_lengths,
            "out": out,
            "rows": rows,
            "heads": heads,
            "rows_per_request": self.rows_per_request,
            "gate_pitch": pitch,
            "enable_pdl": int(self.enable_pdl),
        }
        grid = launch_grid(program.record, rows=rows, heads=heads)
        values["grid_x"], values["grid_y"], values["grid_z"] = grid
        arguments = tuple(values[name] for _kind, name in program.plan)
        with tvm_ffi.use_torch_stream():
            program.entry(*arguments)
        return out

    __call__ = launch


def prepare_mla_value_gate(
    w_uv: torch.Tensor,
    *,
    max_rows: int,
    sigmoid_precision: str,
    rows_per_request: int = 1,
    enable_pdl: bool = True,
    tile_rows: int | None = None,
) -> MlaValueGateRunner:
    """Bind ``w_uv`` (logical BF16 ``[H, 512, 128]`` with strides ``(65536, 1, 512)``,
    16-byte aligned), decide which program serves each ``B`` in ``[1, max_rows]``
    (:func:`select_tile_rows`, with the device's SM count) and load those programs; see
    :func:`flashinfer.cake_mla_value_gate.prepare_mla_value_gate` for the contract.
    ``tile_rows`` (backend only, for tests and measurements) forces the program with that
    row tile for every ``B`` instead of the rule.
    Raises ``RuntimeError`` under CUDA graph capture, ``TypeError`` / ``ValueError``
    on invalid arguments and ``NotImplementedError`` when the device is not an
    exact SM100a / SM103a part or the checkout registers no program for it."""
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "prepare_mla_value_gate must run outside CUDA graph capture (it validates the "
            "weight and loads or builds the program); capture runner.launch instead"
        )
    if sigmoid_precision not in SIGMOID_PRECISIONS:
        raise ValueError(
            f"sigmoid_precision must be one of {SIGMOID_PRECISIONS} ('fp32': the sigmoid is "
            f"evaluated in FP32 and not rounded to BF16), got {sigmoid_precision!r}"
        )
    max_rows = _int_arg(max_rows, "max_rows")
    if not 1 <= max_rows <= MAX_ROWS:
        raise ValueError(f"max_rows must be in [1, {MAX_ROWS}], got {max_rows}")
    q = _int_arg(rows_per_request, "rows_per_request")
    if not 1 <= q <= max_rows:
        raise ValueError(
            f"rows_per_request must be in [1, max_rows = {max_rows}], got {q}"
        )
    if not isinstance(enable_pdl, bool):
        raise TypeError(f"enable_pdl must be a bool, got {type(enable_pdl).__name__}")
    if tile_rows is not None:
        tile_rows = _int_arg(tile_rows, "tile_rows")
    _check_tensor(w_uv, "w_uv", torch.bfloat16)
    if w_uv.ndim != 3 or tuple(w_uv.shape[1:]) != (LATENT_DIM, VALUE_DIM):
        raise ValueError(
            f"w_uv must be [H, {LATENT_DIM}, {VALUE_DIM}], got {tuple(w_uv.shape)}"
        )
    heads = int(w_uv.shape[0])
    if not 1 <= heads <= _GRID_YZ_LIMIT or heads * _W_UV_STRIDES[0] >= _INT32_LIMIT:
        raise ValueError(
            f"w_uv has {heads} heads; the kernel takes 1 to "
            f"{(_INT32_LIMIT - 1) // _W_UV_STRIDES[0]}"
        )
    strides = tuple(int(s) for s in w_uv.stride())
    if strides[1:] != _W_UV_STRIDES[1:] or (
        heads > 1 and strides[0] != _W_UV_STRIDES[0]
    ):
        raise ValueError(
            f"w_uv must have strides {_W_UV_STRIDES} (the [H, {LATENT_DIM}, {VALUE_DIM}] view "
            f"of a contiguous [H, {VALUE_DIM}, {LATENT_DIM}] tensor, latent axis contiguous), "
            f"got {strides}"
        )
    if w_uv.data_ptr() % _ALIGN:
        raise ValueError(f"w_uv must start on a {_ALIGN}-byte boundary")
    device_index = int(w_uv.device.index)
    arch = device_arch(device_index)
    if arch is None:
        major, minor = torch.cuda.get_device_capability(device_index)
        raise NotImplementedError(
            "the Cake MLA value gate requires an exact SM100a or SM103a device "
            f"(compute capability 10.0 / 10.3), got {major}.{minor}"
        )
    names = records_for(arch, sigmoid_precision)
    if not names:
        raise NotImplementedError(
            f"this checkout registers no Cake MLA value gate program for {arch} with "
            f"sigmoid_precision={sigmoid_precision!r}"
        )
    by_tile: dict[int, str] = {}
    for name in names:
        tile = record_tile_rows(MODULES[name])
        if tile in by_tile:
            raise RuntimeError(
                f"the registry delivers two {tile}-row programs for ({arch}, "
                f"{sigmoid_precision!r}): {by_tile[tile]}, {name}"
            )
        by_tile[tile] = name
    if len(by_tile) > 2:
        raise RuntimeError(
            f"the registry delivers {len(by_tile)} programs for ({arch}, {sigmoid_precision!r}): "
            f"{sorted(by_tile.values())}; the host selects between at most two row tiles"
        )
    for name in by_tile.values():
        use_pdl = MODULES[name]["launch"]["use_pdl"]
        if use_pdl != "runtime" and bool(use_pdl) != enable_pdl:
            raise NotImplementedError(
                f"program {name} always launches with use_pdl={use_pdl}; "
                f"enable_pdl={enable_pdl} is not available"
            )
    sm_count = device_sm_count(device_index)
    if tile_rows is None:
        records = {tile: MODULES[name] for tile, name in by_tile.items()}
        chosen = [
            select_tile_rows(
                records,
                rows=rows,
                heads=heads,
                enable_pdl=enable_pdl,
                sm_count=sm_count,
            )
            for rows in range(1, max_rows + 1)
        ]
    elif tile_rows in by_tile:
        chosen = [tile_rows] * max_rows
    else:
        raise NotImplementedError(
            f"this checkout registers no {tile_rows}-row Cake MLA value gate program for "
            f"{arch}; its row tiles: {sorted(by_tile)}"
        )
    # Load only the programs some B uses (in registry order).
    programs = []
    for tile in [tile for tile in by_tile if tile in chosen]:
        name = by_tile[tile]
        record = MODULES[name]
        plan = _argument_plan(record)
        entry = getattr(load(name, arch), record["ffi_entry"])
        programs.append(
            _Program(name=name, tile_rows=tile, record=record, plan=plan, entry=entry)
        )
    index = {program.tile_rows: i for i, program in enumerate(programs)}
    physical = w_uv.transpose(1, 2)
    return MlaValueGateRunner(
        w_uv=w_uv,
        max_rows=max_rows,
        rows_per_request=q,
        sigmoid_precision=sigmoid_precision,
        enable_pdl=enable_pdl,
        arch=arch,
        sm_count=sm_count,
        programs=tuple(program.name for program in programs),
        _programs=tuple(programs),
        _selection=tuple(index[tile] for tile in chosen),
        _w_uv_physical=physical,
        _w_uv_bytes=_byte_range(physical),
    )
