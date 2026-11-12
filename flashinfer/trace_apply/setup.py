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

Setup: an untimed, weight-reading preparation step a Solution runs once per
weight set, outside the call path and outside CUDA-graph capture.

A Solution declares it on its spec: ``setup_entry_point`` (``setup(context) ->
state``), an optional ``teardown_entry_point`` (``teardown(state)``), the
``setup_inputs`` whose live tensors setup receives, and the ``setup_state_param``
keyword the entry point takes the state under.
Setup runs with the engine's live tensors (one layer's weights, read-only by
contract) and its state reaches every later call of the entry point by
keyword; nothing in trace_apply derives or converts a layout, the Solution
does, once, in its own step. The ``context`` is a plain dict with the keys
``axes``, ``device``, ``rank``, ``world_size``, ``inputs``, ``input_values``,
``setup_inputs`` and ``outputs``. With a Definition document (a sealed
``definition.json``) ``inputs`` lists every Definition input in declared order,
a scalar input with ``shape`` None, and ``axes`` and ``outputs`` are filled from
it; without one, ``inputs`` is the call's tensors and the other two are empty.

State is keyed on the identity of the ``setup_inputs`` tensors — address,
shape, stride, dtype and device — never on their contents. A Definition name
is a shape class, not a layer: every MoE layer of a model shares one name and
holds its own weights, so one state per key is one state per weight set, and
the view of the same storage an engine passes call after call keys the same.
Identity cannot see bytes change behind a stable address; an engine mode that
rewrites weights in place is the consumer's to refuse.

Refused, with ``SetupStateError`` (a ``RuntimeError``, never ``ImportError``):
a weight set first seen inside CUDA-graph capture (setup allocates and
synchronizes; run every weight set eagerly before capturing), a new weight set
after the first capture (the key set freezes, so a misdeclared activation read
cannot grow state without bound), more than ``MAX_WEIGHT_SETS`` weight sets, a
positional call, a call from another process (a state does not survive fork),
a setup input that is absent or not a tensor, a setup that raises, a setup
that calls its own Definition with a new weight set (refused, not left waiting
on the lock it holds), a ``{"state", "input_buffers"}`` result envelope with
non-empty ``input_buffers`` (serving has no harness copy phase), and ``release()`` of a
binding a capture has observed (captured graphs replay against the state
addresses, so the state is kept for the life of the process).

What a setup retains is a second resident copy for the life of the process,
allocated at the first eager call of each weight set; sizing it against the
engine's memory budget is the Solution's responsibility, and ``stats()``
reports what was retained. Torch is imported inside functions only, as in the
sealed-package layer that binds this class.
"""

from __future__ import annotations

import functools
import inspect
import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["MAX_WEIGHT_SETS", "SetupBoundCallable", "SetupStateError"]

#: Upper bound on the weight sets one bound callable retains. A Definition is
#: shared by every layer of a model, so the bound is a layer count with room,
#: not a call count: a key set that grows past it is reading an activation.
MAX_WEIGHT_SETS = 1024

_StateKey = tuple[tuple[str, int, tuple[int, ...], tuple[int, ...], str, str], ...]


class SetupStateError(RuntimeError):
    """A setup-bearing Solution could not be served its state.

    Deliberately a RuntimeError, never ImportError, for the reason
    ``SealedPackageError`` gives: an engine must not read a refusal as an
    unavailable optional integration and silently fall back to stock kernels.
    """


def _is_capturing(device: Any) -> bool:
    """Whether a CUDA-graph capture is open on the call's device or on the current one.

    Asked on the call's device, so a capture on another GPU than the current
    one is seen, and on the current device too when it differs: a setup
    allocates and synchronizes, which a capture open anywhere in the process
    may not tolerate. Seen from the calling thread: CUDA answers for the
    current stream only and offers no process-wide query, so a capture another
    thread holds is the serving contract's to keep away from a first use
    (engines capture in a single-threaded warm-up, after every weight set ran
    eagerly)."""
    if device.type != "cuda":
        return False
    import torch

    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return True
        index = torch.cuda.current_device()
    return index != torch.cuda.current_device() and (
        torch.cuda.is_current_stream_capturing()
    )


def _call_device(tensors: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    """The device the call runs on: that of the first tensor off the CPU among the
    setup inputs, then the call's other tensors (a host-side table may come
    first), else of the first tensor at all, else CPU."""
    import torch

    devices = [
        value.device
        for value in (*tensors, *kwargs.values())
        if isinstance(value, torch.Tensor)
    ]
    accelerated = [device for device in devices if device.type != "cpu"]
    return (accelerated or devices or [torch.device("cpu")])[0]


def _section(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return ``document[key]`` when it is a mapping, else an empty one."""
    value = document.get(key)
    return value if isinstance(value, Mapping) else {}


def _tensor_metadata(tensor: Any) -> dict[str, Any]:
    from flashinfer.trace.template import _dtype_str  # the Definition's dtype spelling

    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": _dtype_str(tensor.dtype),
        "device": tensor.device.type,
    }


def _declared_metadata(spec: Any, axes: Mapping[str, int]) -> dict[str, Any]:
    """Metadata of a Definition output with no live buffer: the declared shape
    with every axis the call resolved, stride unknown, the declared dtype."""
    shape = spec.get("shape") if isinstance(spec, Mapping) else None
    return {
        "shape": [
            axes.get(axis, axis) if isinstance(axis, str) else axis for axis in shape
        ]
        if isinstance(shape, list)
        else None,
        "stride": None,
        "dtype": spec.get("dtype") if isinstance(spec, Mapping) else None,
        "device": spec.get("device", "cuda") if isinstance(spec, Mapping) else "cuda",
    }


def _resolve_axes(
    definition: Mapping[str, Any], kwargs: Mapping[str, Any]
) -> dict[str, int]:
    """Axes of a Definition document for this call: the const values it declares,
    then every bare axis name in an input shape, read off the live tensor."""
    import torch

    axes: dict[str, int] = {}
    for name, axis in _section(definition, "axes").items():
        if isinstance(axis, Mapping) and isinstance(axis.get("value"), int):
            axes[name] = axis["value"]
    for name, spec in _section(definition, "inputs").items():
        tensor = kwargs.get(name)
        shape = spec.get("shape") if isinstance(spec, Mapping) else None
        if (
            isinstance(tensor, torch.Tensor)
            and isinstance(shape, list)
            and len(shape) == tensor.dim()
        ):
            for axis, size in zip(shape, tensor.shape, strict=True):
                if isinstance(axis, str) and axis.isidentifier():
                    axes.setdefault(axis, int(size))
    return axes


class SetupBoundCallable:
    """A Solution's entry point bound to its setup lifecycle.

    Calls go through by keyword exactly as trace_apply makes them, plus one
    keyword, ``state_param``, holding the state setup returned for the weight
    set the call's ``setup_inputs`` tensors identify. The published signature
    (``inspect.signature``) is the entry point's minus that parameter, so a
    consumer binding the Definition ABI sees what it would see without a setup.
    """

    def __init__(
        self,
        run: Callable[..., Any],
        setup: Callable[[dict[str, Any]], Any],
        teardown: Callable[[Any], Any] | None,
        *,
        name: str,
        setup_inputs: tuple[str, ...],
        state_param: str,
        definition: Mapping[str, Any] | None = None,
    ) -> None:
        """Bind ``run`` to its setup; it must take the keyword ``state_param``."""
        try:
            signature = inspect.signature(run)
        except (TypeError, ValueError) as error:
            raise SetupStateError(
                f"{name}: the entry point's signature is not introspectable, so "
                f"the state parameter {state_param!r} cannot be bound"
            ) from error
        parameter = signature.parameters.get(state_param)
        if parameter is None or parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise SetupStateError(
                f"{name}: the entry point takes no keyword parameter "
                f"{state_param!r} to receive the setup state"
            )
        functools.update_wrapper(self, run)
        self.__signature__ = signature.replace(
            parameters=[
                item
                for item in signature.parameters.values()
                if item.name != state_param
            ]
        )
        self.name = name
        self.setup_inputs = tuple(setup_inputs)
        self.state_param = state_param
        self._run = run
        self._setup = setup
        self._teardown = teardown
        self._definition = definition
        self._pid = os.getpid()
        # Re-entrant so that a setup calling its own Definition is refused by
        # ``_preparing`` below, instead of waiting forever on the lock it holds.
        self._lock = threading.RLock()
        self._preparing = False
        self._states: dict[_StateKey, tuple[Any, tuple[Any, ...]]] = {}
        self._frozen = False
        self._released = False
        self._stats: dict[str, Any] = {
            "runs": 0,
            "weight_sets": 0,
            "seconds": 0.0,
            "retained_bytes": 0,
            "refused_in_capture": 0,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the entry point with the state prepared for the call's setup inputs."""
        if args:
            raise SetupStateError(
                f"{self.name}: a setup-bearing Solution is called by keyword only, "
                f"not with {len(args)} positional argument(s)"
            )
        if os.getpid() != self._pid:
            raise SetupStateError(
                f"{self.name}: an installed setup state does not survive fork "
                f"(bound in process {self._pid}, called in process {os.getpid()})"
            )
        if self.state_param in kwargs:
            raise SetupStateError(
                f"{self.name}: the call already passes {self.state_param!r}, "
                "the parameter reserved for the setup state"
            )
        if self._released:
            # A route resolved before the release (a wrapper a caller kept, a
            # teardown re-entering its own Definition) must not prepare a state
            # nothing tracks; the live binding is the one a later import makes.
            raise SetupStateError(
                f"{self.name}: this binding was released; its states are torn down "
                "and a later import binds afresh"
            )
        tensors = self._setup_tensors(kwargs)
        key: _StateKey = tuple(
            (
                name,
                tensor.data_ptr(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                str(tensor.dtype),
                str(tensor.device),
            )
            for name, tensor in zip(self.setup_inputs, tensors, strict=True)
        )
        device = _call_device(tensors, kwargs)
        entry = self._states.get(key)
        if entry is None:
            entry = self._prepare(key, tensors, kwargs, device)
        elif not self._frozen and _is_capturing(device):
            # The first capture freezes the key set: a weight set that appears
            # later cannot have been prepared before the graphs were recorded.
            self._frozen = True
        kwargs[self.state_param] = entry[0]
        return self._run(**kwargs)

    def _setup_tensors(self, kwargs: Mapping[str, Any]) -> tuple[Any, ...]:
        """Collect the call's setup-input tensors; a missing or non-tensor refuses."""
        import torch

        tensors = []
        for name in self.setup_inputs:
            value = kwargs.get(name)
            if not isinstance(value, torch.Tensor):
                why = "absent from the call" if name not in kwargs else "not a tensor"
                raise SetupStateError(
                    f"{self.name}: setup input {name!r} is {why}; the retained "
                    "state is keyed on its identity"
                )
            tensors.append(value)
        return tuple(tensors)

    def _prepare(
        self,
        key: _StateKey,
        tensors: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        device: Any,
    ) -> tuple[Any, tuple[Any, ...]]:
        """Return the state for ``key``, running setup once under the capture rules."""
        import torch

        with self._lock:
            entry = self._states.get(key)
            if entry is not None:
                return entry
            if self._preparing:
                raise SetupStateError(
                    f"{self.name}: setup re-entered its own bound callable with a new "
                    "weight set; a setup must not call the Definition it prepares"
                )
            if _is_capturing(device):
                self._frozen = True
                self._stats["refused_in_capture"] += 1
                raise SetupStateError(
                    f"{self.name}: a weight set was first seen inside CUDA-graph "
                    "capture; setup allocates and synchronizes, so run every weight "
                    "set eagerly before capturing"
                )
            if self._frozen:
                raise SetupStateError(
                    f"{self.name}: a new weight set appeared after the first "
                    "CUDA-graph capture; the retained key set is frozen from then on"
                )
            if len(self._states) >= MAX_WEIGHT_SETS:
                raise SetupStateError(
                    f"{self.name}: setup would retain more than MAX_WEIGHT_SETS = "
                    f"{MAX_WEIGHT_SETS} weight sets; a setup input is probably an "
                    "activation, not a weight"
                )
            cuda = device.type == "cuda"
            before = torch.cuda.memory_allocated(device) if cuda else 0
            started = time.monotonic()
            self._preparing = True
            try:
                state = self._unpack(
                    self._setup(self._context(tensors, kwargs, device))
                )
            except SetupStateError:
                raise
            except Exception as error:
                raise SetupStateError(
                    f"{self.name}: setup raised {type(error).__name__}: {error}"
                ) from error
            finally:
                self._preparing = False
            if cuda:
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            after = torch.cuda.memory_allocated(device) if cuda else 0
            # The tensors are kept beside the state so their storage cannot be
            # recycled, and their addresses re-keyed, while the state lives.
            entry = self._states[key] = (state, tensors)
            self._stats["runs"] += 1
            self._stats["weight_sets"] = len(self._states)
            self._stats["seconds"] += elapsed
            self._stats["retained_bytes"] += max(0, after - before)
            return entry

    def _context(
        self, tensors: tuple[Any, ...], kwargs: Mapping[str, Any], device: Any
    ) -> dict[str, Any]:
        """Build the context setup receives: axes, device, rank, inputs, outputs."""
        import torch
        import torch.distributed as dist

        rank, world_size = 0, 1
        if dist.is_available() and dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
        definition = self._definition
        outputs = _section(definition, "outputs") if definition is not None else {}
        # Every Definition input in declared order when a Definition is known:
        # a scalar input (``shape`` None) with its
        # declared dtype and device, a tensor input with the live tensor's
        # metadata; then the call's other tensors, which are all of them
        # without a Definition.
        inputs: dict[str, Any] = {}
        if definition is not None:
            for name, spec in _section(definition, "inputs").items():
                value = kwargs.get(name)
                if isinstance(spec, Mapping) and spec.get("shape") is None:
                    inputs[name] = {
                        "shape": None,
                        "stride": None,
                        "dtype": spec.get("dtype"),
                        "device": spec.get("device", "cuda"),
                    }
                elif isinstance(value, torch.Tensor):
                    inputs[name] = _tensor_metadata(value)
        for name, value in kwargs.items():
            if (
                isinstance(value, torch.Tensor)
                and name not in inputs
                and name not in outputs
            ):
                inputs[name] = _tensor_metadata(value)
        axes = _resolve_axes(definition, kwargs) if definition is not None else {}
        return {
            "axes": axes,
            "device": str(device),
            "rank": rank,
            "world_size": world_size,
            "inputs": inputs,
            "input_values": {
                name: value
                for name, value in kwargs.items()
                if not isinstance(value, torch.Tensor)
            },
            "setup_inputs": dict(zip(self.setup_inputs, tensors, strict=True)),
            # A declared output is described from its live buffer when the call
            # passes one (destination passing) and from the Definition and the
            # resolved axes when it does not (a value-returning Solution).
            "outputs": {
                name: _tensor_metadata(kwargs[name])
                if isinstance(kwargs.get(name), torch.Tensor)
                else _declared_metadata(spec, axes)
                for name, spec in outputs.items()
            },
        }

    def _unpack(self, result: Any) -> Any:
        """The state from a ``{"state", "input_buffers"}`` envelope or a bare state."""
        if not isinstance(result, Mapping) or "input_buffers" not in result:
            return result
        if set(result) != {"state", "input_buffers"}:
            raise SetupStateError(
                f"{self.name}: setup result must contain exactly state and "
                f"input_buffers, got {sorted(result)}"
            )
        buffers = result["input_buffers"]
        if not isinstance(buffers, Mapping):
            raise SetupStateError(
                f"{self.name}: setup input_buffers must be a mapping of input "
                "names to tensors"
            )
        if buffers:
            raise SetupStateError(
                f"{self.name}: setup bound input_buffers {sorted(buffers)}; serving "
                "has no harness copy phase, so a setup cannot rebind inputs"
            )
        return result["state"]

    def stats(self) -> dict[str, Any]:
        """``runs``, ``weight_sets``, ``seconds``, ``retained_bytes`` (torch
        allocator delta across setup, CUDA only) and ``refused_in_capture``."""
        return dict(self._stats)

    @property
    def frozen(self) -> bool:
        """Whether a CUDA-graph capture has observed this binding: its key set is
        frozen and its states are kept for the life of the process."""
        return self._frozen

    def release(self) -> None:
        """Tear down every retained state (when a teardown is declared), then forget them.

        Refused once a CUDA-graph capture has observed the binding: captured
        graphs replay against the state addresses, so freeing them would serve
        recycled memory as weights. A shutdown-time operation: the caller
        quiesces calls first, since a call in flight already holds the state
        it was handed and a teardown may empty it. The binding refuses every
        call from here on, before the first teardown runs, so a teardown that
        re-enters its Definition prepares nothing. Every teardown is
        attempted; the first failure is raised afterwards, naming how many of
        them failed."""
        with self._lock:
            if self._frozen:
                raise SetupStateError(
                    f"{self.name}: a CUDA-graph capture observed the retained state; "
                    "captured graphs replay against its addresses, so it is kept for "
                    "the life of the process"
                )
            self._released = True
            states, self._states = self._states, {}
            self._stats["weight_sets"] = 0
            self._stats["retained_bytes"] = 0
            if self._teardown is None:
                return
            failures: list[Exception] = []
            for state, _ in states.values():
                try:
                    self._teardown(state)
                except Exception as error:
                    failures.append(error)
            if failures:
                first = failures[0]
                raise SetupStateError(
                    f"{self.name}: teardown raised {type(first).__name__}: {first} "
                    f"({len(failures)} of {len(states)} teardowns raised)"
                ) from first
