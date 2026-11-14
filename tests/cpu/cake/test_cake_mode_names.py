# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""Self-test of the Cake mode-name checks (``tests/cake/test_helpers/cake_mode_names.py``).

The modules are synthetic, built here from source; the fresh-interpreter check
imports standard-library names and packages written to ``tmp_path``.  No flashinfer
import, no GPU.
"""

from __future__ import annotations

import re
import textwrap
import types
from pathlib import Path

import pytest

from tests.cake.test_helpers.cake_mode_names import (
    assert_cake_mode_module,
    assert_mode_names,
    assert_no_deterministic_keyword,
    assert_old_names_fail,
    assert_same_object_same_name,
    public_names,
)

pytestmark = pytest.mark.cpu_only


def _module(name: str, source: str = "", **attrs: object) -> types.ModuleType:
    """A module ``name`` with ``attrs`` bound, then ``source`` run in it."""
    module = types.ModuleType(name)
    vars(module).update(attrs)
    exec(textwrap.dedent(source), vars(module))
    return module


DET_MODULE = "toy.cake_toy_det_train"
KCP_MODULE = "toy.cake_toy_nondet_train_kcp"


# Stands in for torch.autograd.Function, found by its module and name: its constructor
# is (*args, **kwargs), as torch's.
FUNCTION = """
class Function:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def apply(cls, *args):
        return cls.forward(None, *args)


Function.__module__ = "torch.autograd.function"
"""

# A det single-rank facade: constants, an autograd class, a context type, the autograd
# functional, a verb and the support query.
DET = (
    FUNCTION
    + """
ARCHES = ("sm_100a", "sm_103a")
SM_100A_TILE = 128


class CakeToyDetFunction(Function):
    @staticmethod
    def forward(ctx, x, eps=1e-6):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class CakeToyDetContext:
    def __init__(self, rank, world_size):
        self.rank, self.world_size = rank, world_size

    def split(self, n):
        return n // self.world_size

    def options(self, **kwargs):
        return kwargs


def cake_toy_det(x, eps=1e-6):
    return CakeToyDetFunction.apply(x, eps)


def cake_toy_det_train_forward(x, *, workspace=None):
    return x


def is_cake_toy_det_train_supported(device=None):
    return True


__all__ = [
    "ARCHES",
    "SM_100A_TILE",
    "CakeToyDetContext",
    "CakeToyDetFunction",
    "cake_toy_det",
    "cake_toy_det_train_forward",
    "is_cake_toy_det_train_supported",
]
"""
)

# A nondet facade with several autograd entries, one class per entry.
LM_HEAD = """
DEFAULT_CHUNK = 4096


class CakeLMHeadNondetLossFunction:
    @staticmethod
    def forward(ctx, hidden, weight, labels):
        return hidden


class CakeLMHeadNondetLogprobFunction:
    @staticmethod
    def forward(ctx, hidden, weight, labels):
        return hidden


def cake_lm_head_nondet_train_loss(hidden, weight, labels):
    return hidden


def cake_lm_head_nondet_train_logprob(hidden, weight, labels):
    return hidden


def is_cake_lm_head_nondet_train_supported(device=None):
    return True


__all__ = [
    "DEFAULT_CHUNK",
    "CakeLMHeadNondetLogprobFunction",
    "CakeLMHeadNondetLossFunction",
    "cake_lm_head_nondet_train_logprob",
    "cake_lm_head_nondet_train_loss",
    "is_cake_lm_head_nondet_train_supported",
]
"""

# A nondet KCP facade.
KCP = """
class CakeToyNondetKcpMultiFunction:
    @staticmethod
    def forward(ctx, xs, context):
        return xs


class CakeToyNondetKcpRecomputeHandle:
    pass


def cake_toy_nondet_kcp(x, context):
    return x


def cake_toy_nondet_kcp_multi(xs, context):
    return xs


def cake_toy_nondet_train_kcp_build_context(group):
    return group


def is_cake_toy_nondet_train_kcp_supported(device=None):
    return True


__all__ = [
    "CakeToyNondetKcpMultiFunction",
    "CakeToyNondetKcpRecomputeHandle",
    "cake_toy_nondet_kcp",
    "cake_toy_nondet_kcp_multi",
    "cake_toy_nondet_train_kcp_build_context",
    "is_cake_toy_nondet_train_kcp_supported",
]
"""

CONFORMING = {
    DET_MODULE: DET,
    "toy.cake_lm_head_nondet_train": LM_HEAD,
    KCP_MODULE: KCP,
}


@pytest.mark.parametrize("name", sorted(CONFORMING))
def test_conforming_module_passes(name: str) -> None:
    module = _module(name, CONFORMING[name])
    assert public_names(module) == module.__all__
    assert_cake_mode_module(module)


@pytest.mark.parametrize(
    "module_name, name, error",
    [
        (DET_MODULE, "cake_toy_train_forward", "lacks the mode token 'det'"),
        (DET_MODULE, "route_key", "lacks the mode token 'det'"),
        (DET_MODULE, "cake_toy_deterministic", "lacks the mode token 'det'"),
        (DET_MODULE, "CakeToyFunction", "lacks the mode part 'Det'"),
        (DET_MODULE, "CakeToyDetermineFunction", "lacks the mode part 'Det'"),
        (DET_MODULE, "Arches", "lacks the mode part 'Det'"),
        (DET_MODULE, "cake_toy_nondet", "carries 'nondet'"),
        (DET_MODULE, "CakeToyNondetFunction", "carries 'Nondet'"),
        (KCP_MODULE, "cake_toy_kcp", "lacks the mode token 'nondet'"),
        (KCP_MODULE, "CakeToyKcpContext", "lacks the mode part 'Nondet'"),
        (KCP_MODULE, "cake_toy_det_kcp", "carries 'det'"),
        (KCP_MODULE, "CakeToyDetKcpFunction", "carries 'Det'"),
    ],
)
def test_name_without_the_mode_word_of_its_module_fails(
    module_name: str, name: str, error: str
) -> None:
    module = _module(module_name)
    expected = re.escape(f"{module_name}.{name}: {error}")
    with pytest.raises(AssertionError, match=expected):
        assert_mode_names(module, names=[name])


# CamelCase names of the target table, the mode part after an acronym (with digits
# too): each passes in its module, fails in the module of the other mode.
@pytest.mark.parametrize(
    "leaf, mode, name",
    [
        ("kda_{}_train_kcp", "det", "CakeKDADetKcpContext"),
        ("moe_grouped_gemm_{}_train", "nondet", "CakeMoEGroupedGemmNondetFunction"),
        ("rmsnorm_silu_gate_{}_train", "det", "CakeRMSNormSiLUGateDetFunction"),
        ("mhc_head_{}_train", "det", "CakeMHCHeadDetFunction"),
        ("gqa_{}_train", "nondet", "CakeGQANondetFunction"),
        ("rmsnorm_fp8_{}_train", "det", "CakeRMSNormFP8DetFunction"),
        ("gqa_hd256_{}_train", "nondet", "CakeGQAHD256NondetFunction"),
    ],
)
def test_camel_case_mode_part_after_an_acronym(leaf: str, mode: str, name: str) -> None:
    module_name = f"toy.cake_{leaf}"
    assert_mode_names(_module(module_name.format(mode)), names=[name])
    other = module_name.format("nondet" if mode == "det" else "det")
    error = f"{other}.{name}: carries {mode.capitalize()!r}, the word of the other mode"
    with pytest.raises(AssertionError, match=re.escape(error)):
        assert_mode_names(_module(other), names=[name])


def test_all_caps_constants_are_exempt() -> None:
    module = _module(DET_MODULE)
    assert_mode_names(module, names=["ARCHES", "SM_100A_TILE", "WORKSPACE_BYTES", "X"])


OUT_OF_ORDER = (
    "cake_toy_train_det_forward",
    "toy_det_forward",
    "is_toy_det_supported",
    "CakeDetToyFunction",
    "ToyDetFunction",
)


def test_mode_word_out_of_order_fails_unless_exempt() -> None:
    module = _module(DET_MODULE)
    with pytest.raises(AssertionError) as info:
        assert_mode_names(module, names=OUT_OF_ORDER)
    message = str(info.value)
    assert message.startswith(f"{len(OUT_OF_ORDER)} "), message
    for name in OUT_OF_ORDER:
        assert f"cake_toy_det_train.{name}: does not start with" in message, name
    assert_mode_names(module, names=OUT_OF_ORDER, stem=False)
    assert_mode_names(module, names=OUT_OF_ORDER, stem_exceptions=OUT_OF_ORDER)


def test_mode_comes_from_the_module_name_or_the_argument() -> None:
    other = _module("toy.helpers", '__all__ = ["cake_toy_det", "SCALE"]')
    with pytest.raises(ValueError, match="pass mode"):
        assert_mode_names(other)
    with pytest.raises(ValueError, match="pass stem=False"):
        assert_mode_names(other, "det")
    assert_mode_names(other, "det", stem=False)
    with pytest.raises(AssertionError, match="cake_toy_det: carries 'det'"):
        assert_mode_names(other, "nondet", stem=False)
    with pytest.raises(ValueError, match="a det module, not nondet"):
        assert_mode_names(_module(DET_MODULE), "nondet")


def test_without_all_the_attributes_defined_in_the_module_are_public() -> None:
    module = _module(
        DET_MODULE,
        """
        import json
        from textwrap import dedent

        LIMIT = 3
        _private = 1


        def cake_toy_det(x):
            return json.dumps(x)


        def helper(x):
            return dedent(x)
        """,
    )
    assert public_names(module) == ["LIMIT", "cake_toy_det", "helper"]
    with pytest.raises(AssertionError) as info:
        assert_mode_names(module)
    message = str(info.value)
    assert message.startswith("1 "), message
    assert "toy.cake_toy_det_train.helper: lacks the mode token 'det'" in message


# Every callable here takes ``deterministic``, by name or through a ``**kwargs`` of an
# entry point; the constructor of the autograd class is torch's (*args, **kwargs).
DETERMINISTIC = (
    FUNCTION
    + """

class CakeToyDetFunction(Function):
    @staticmethod
    def forward(ctx, x, deterministic=True):
        return x

    @staticmethod
    def backward(ctx, grad, **kwargs):
        return grad, None


class CakeToyDetContext:
    def __init__(self, world_size, deterministic=True):
        self.world_size = world_size


class CakeToyDetOptions:
    def __init__(self, **options):
        self.options = options


class CakeToyDetPlan:
    @classmethod
    def build(cls, shape, *, deterministic=True):
        return cls()


def cake_toy_det(x, deterministic=True):
    return x


def cake_toy_det_train_forward(x, **kwargs):
    return x
"""
)


@pytest.mark.parametrize(
    "name, where, through",
    [
        ("cake_toy_det", "cake_toy_det", ""),
        ("cake_toy_det_train_forward", "cake_toy_det_train_forward", "**kwargs"),
        ("CakeToyDetFunction", "CakeToyDetFunction.forward", ""),
        ("CakeToyDetFunction", "CakeToyDetFunction.backward", "**kwargs"),
        ("CakeToyDetContext", "CakeToyDetContext", ""),
        ("CakeToyDetOptions", "CakeToyDetOptions", "**options"),
        ("CakeToyDetPlan", "CakeToyDetPlan.build", ""),
    ],
)
def test_deterministic_parameter_fails(name: str, where: str, through: str) -> None:
    module = _module(DET_MODULE, DETERMINISTIC)
    error = f"toy.cake_toy_det_train.{where}: takes a 'deterministic' parameter"
    error += f" through {through}\n" if through else "\n"
    for objs in (module, [getattr(module, name)]):
        with pytest.raises(AssertionError) as info:
            assert_no_deterministic_keyword(objs)
        assert error in f"{info.value}\n", info.value


def test_deterministic_parameter_report_lists_every_entry() -> None:
    module = _module(DET_MODULE, DETERMINISTIC)
    with pytest.raises(AssertionError) as info:
        assert_no_deterministic_keyword(module)
    message = str(info.value)
    assert message.startswith("7 "), message
    # the constructor of the autograd class, (*args, **kwargs), is not an entry point
    assert "toy.cake_toy_det_train.CakeToyDetFunction:" not in message


# A facade entry that wraps a backend function and adds ``deterministic``: the
# signature through ``__wrapped__`` hides it.
WRAPPER = """
import functools


def backend_forward(x):
    return x


@functools.wraps(backend_forward)
def cake_toy_det(x, deterministic=True):
    return backend_forward(x)


__all__ = ["cake_toy_det"]
"""


def test_deterministic_parameter_of_a_wrapper_fails() -> None:
    module = _module(DET_MODULE, WRAPPER)
    error = "toy.cake_toy_det_train.cake_toy_det: takes a 'deterministic' parameter"
    with pytest.raises(AssertionError, match=re.escape(error)):
        assert_no_deterministic_keyword(module)


# A wrapper's own (*args, **kwargs): inspect.signature reads the function it wraps.
PLAIN_WRAPPER = """
import functools


def backend_forward(x):
    return x


@functools.wraps(backend_forward)
def cake_toy_det(*args, **kwargs):
    return backend_forward(*args, **kwargs)


__all__ = ["cake_toy_det"]
"""


def test_callables_without_deterministic_parameter_pass() -> None:
    module = _module(DET_MODULE, DET)
    # the autograd constructor and the public method with **kwargs pass
    assert_no_deterministic_keyword(module)
    assert_no_deterministic_keyword([len, 3, "text", module.cake_toy_det])
    assert_no_deterministic_keyword(_module(DET_MODULE, PLAIN_WRAPPER))


# Type aliases a facade exports: typing's subscripted generics are callable, with a
# (*args, **kwargs) signature, but they name types, not entry points.
TYPE_ALIASES = """
import collections.abc
import typing


CakeToyDetSource = typing.Callable[[], typing.Union[tuple, list]]
CakeToyDetMaybeSource = typing.Optional[typing.Callable[[], tuple]]
CakeToyDetHook = collections.abc.Callable[[int], int]
CakeToyDetShapes = list[int]


class CakeToyDetPlan:
    Source = typing.Callable[[], tuple]

    def build(self, shape):
        return shape


__all__ = [
    "CakeToyDetSource",
    "CakeToyDetMaybeSource",
    "CakeToyDetHook",
    "CakeToyDetShapes",
    "CakeToyDetPlan",
]
"""


def test_type_aliases_are_not_entry_points() -> None:
    module = _module(DET_MODULE, TYPE_ALIASES)
    assert callable(module.CakeToyDetSource)
    assert callable(module.CakeToyDetShapes)
    assert_no_deterministic_keyword(module)
    assert_no_deterministic_keyword(
        [module.CakeToyDetSource, module.CakeToyDetHook, module.CakeToyDetShapes]
    )


def test_entry_beside_type_aliases_still_fails() -> None:
    source = (
        TYPE_ALIASES
        + """

def cake_toy_det(x, **options):
    return x


__all__ = [*__all__, "cake_toy_det"]
"""
    )
    module = _module(DET_MODULE, source)
    error = (
        "toy.cake_toy_det_train.cake_toy_det: takes a 'deterministic' parameter"
        " through **options"
    )
    with pytest.raises(AssertionError) as info:
        assert_no_deterministic_keyword(module)
    message = str(info.value)
    assert error in message, message
    assert message.startswith("1 "), message


# An entry that takes **options and checks the keys at call time, with a
# ``__signature__`` that lists every option keyword-only: inspect.signature reads it,
# in the wrapper too (functools.wraps copies it), so no **options is left to bind.
SIGNED_OPTIONS = """
import functools
import inspect

_OPTIONS = {"chunk": 64, "stages": 2}


def backend_forward(x, **options):
    unknown = sorted(set(options) - set(_OPTIONS))
    if unknown:
        raise TypeError(f"unknown options {unknown}")
    return x


backend_forward.__signature__ = inspect.Signature(
    [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    + [
        inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY, default=value)
        for key, value in _OPTIONS.items()
    ]
)


@functools.wraps(backend_forward)
def cake_toy_det_train_forward(x, **options):
    return backend_forward(x, **options)


__all__ = ["cake_toy_det_train_forward"]
"""


def test_signature_without_var_keyword_replaces_options() -> None:
    assert_no_deterministic_keyword(_module(DET_MODULE, SIGNED_OPTIONS))
    unsigned = SIGNED_OPTIONS.replace(
        "backend_forward.__signature__ = ", "_unused_signature = "
    )
    assert unsigned != SIGNED_OPTIONS
    error = (
        "toy.cake_toy_det_train.cake_toy_det_train_forward: takes a 'deterministic'"
        " parameter through **options"
    )
    with pytest.raises(AssertionError) as info:
        assert_no_deterministic_keyword(_module(DET_MODULE, unsigned))
    assert error in str(info.value), info.value


BACKEND = """
class CakeToyDetFunction:
    @staticmethod
    def forward(ctx, x):
        return x


def cake_toy_det_train_forward(x):
    return x
"""

# The facade re-exports the backend's class and wraps its function.
FACADE = """
import functools

CakeToyDetFunction = backend.CakeToyDetFunction


@functools.wraps(backend.cake_toy_det_train_forward)
def cake_toy_det_train_forward(x):
    return backend.cake_toy_det_train_forward(x)


__all__ = ["CakeToyDetFunction", "cake_toy_det_train_forward"]
"""
BACKEND_NAME = "toy.experimental.cake.cake_toy_det_train.cake_backend"


def _facade_and_backend() -> tuple[types.ModuleType, types.ModuleType]:
    backend = _module(BACKEND_NAME, BACKEND)
    return _module(DET_MODULE, FACADE, backend=backend), backend


def test_namespaces_bind_the_module_objects_under_its_names() -> None:
    facade, backend = _facade_and_backend()
    top = _module(
        "toy",
        CakeToyDetFunction=facade.CakeToyDetFunction,
        cake_toy_det_train_forward=facade.cake_toy_det_train_forward,
        unrelated=len,
    )
    assert_same_object_same_name(facade, [top, backend])


def test_an_alias_in_any_namespace_fails() -> None:
    facade, backend = _facade_and_backend()
    backend.CakeToyFunction = backend.CakeToyDetFunction
    # the function the facade entry wraps
    backend.cake_toy_train_forward = backend.cake_toy_det_train_forward
    backend._former = backend.cake_toy_det_train_forward  # private: not checked
    facade.cake_toy = facade.cake_toy_det_train_forward  # not in __all__
    top = _module("toy", cake_toy_forward=facade.cake_toy_det_train_forward)
    with pytest.raises(AssertionError) as info:
        assert_same_object_same_name(facade, [top, backend])
    message = str(info.value)
    assert message.startswith("4 "), message
    for alias in (
        f"{BACKEND_NAME}.CakeToyFunction: binds",
        f"{BACKEND_NAME}.cake_toy_train_forward: binds",
        "toy.cake_toy_det_train.cake_toy: binds",
        "toy.cake_toy_forward: binds",
    ):
        assert alias in message, alias
    assert "_former" not in message


# A facade that serves its public names from the backend through a module
# ``__getattr__`` (nothing bound), an ALL_CAPS name for its function among them.
LAZY_FACADE = """
def __getattr__(name):
    if name == "TOY_FORWARD":
        return backend.cake_toy_det_train_forward
    if name in __all__:
        return getattr(backend, name)
    raise AttributeError(name)


__all__ = ["CakeToyDetFunction", "cake_toy_det_train_forward"]
"""


def test_lazy_public_names_of_the_module_are_seen() -> None:
    backend = _module(BACKEND_NAME, BACKEND)
    facade = _module(DET_MODULE, LAZY_FACADE, backend=backend)
    top = _module("toy", CakeToyDetFunction=facade.CakeToyDetFunction)
    assert_same_object_same_name(facade, [top, backend])
    assert "CakeToyDetFunction" not in vars(facade)
    backend.CakeToyFunction = backend.CakeToyDetFunction
    facade.__all__ = [*facade.__all__, "TOY_FORWARD"]
    with pytest.raises(AssertionError) as info:
        assert_same_object_same_name(facade, [top, backend])
    message = str(info.value)
    assert message.startswith("2 "), message
    for alias in (
        f"{BACKEND_NAME}.CakeToyFunction: binds {DET_MODULE}.CakeToyDetFunction",
        f"{DET_MODULE}.TOY_FORWARD: binds {DET_MODULE}.cake_toy_det_train_forward",
    ):
        assert alias in message, alias


# The ``_LAZY_EXPORTS`` rows of the top level (``name -> (module, attribute)``, as in
# flashinfer/__init__.py) that serve the facade, its names and other modules.
LAZY_EXPORTS = {
    "cake_toy_det_train_forward": (
        ".cake_toy_det_train",
        "cake_toy_det_train_forward",
    ),
    "CakeToyDetFunction": ("toy.cake_toy_det_train", "CakeToyDetFunction"),
    "cake_toy_det_train": (".cake_toy_det_train", None),
    # private, other modules, a relative name above the top level: not checked
    "_former": (".cake_toy_det_train", "cake_toy_det_train_forward"),
    "cake_toy": (".cake_toy_train", "cake_toy"),
    "toy_backend": (".experimental.cake.cake_toy_det_train", None),
    "far": ("..cake_toy_det_train", None),
}


def test_lazy_export_rows_are_read_statically() -> None:
    facade, backend = _facade_and_backend()
    top = _module("toy", _LAZY_EXPORTS=dict(LAZY_EXPORTS))
    forward = "cake_toy_det_train_forward"
    # relative to toy.sub, "." is toy.sub: a row of another module
    sub = _module(
        "toy.sub", _LAZY_EXPORTS={"cake_toy": (".cake_toy_det_train", forward)}
    )
    namespaces = [top, sub, backend]
    assert_same_object_same_name(facade, namespaces)
    top._LAZY_EXPORTS.update(
        cake_toy_forward=(".cake_toy_det_train", forward),
        CakeToyFunction=("toy.cake_toy_det_train", "CakeToyDetFunction"),
        cake_toy_train=(".cake_toy_det_train", None),
    )
    # relative to toy.sub, ".." is toy
    sub._LAZY_EXPORTS["cake_toy_fwd"] = ("..cake_toy_det_train", forward)
    with pytest.raises(AssertionError) as info:
        assert_same_object_same_name(facade, namespaces)
    message = str(info.value)
    assert message.startswith("4 "), message
    for alias in (
        f"toy.cake_toy_forward: _LAZY_EXPORTS binds {DET_MODULE}.{forward} under "
        "another name",
        f"toy.CakeToyFunction: _LAZY_EXPORTS binds {DET_MODULE}.CakeToyDetFunction",
        f"toy.cake_toy_train: _LAZY_EXPORTS binds the module {DET_MODULE} under",
        f"toy.sub.cake_toy_fwd: _LAZY_EXPORTS binds {DET_MODULE}.{forward} under",
    ):
        assert alias in message, alias
    # a row its first access has bound is reported once, as a binding
    top.cake_toy_forward = facade.cake_toy_det_train_forward
    with pytest.raises(AssertionError) as info:
        assert_same_object_same_name(facade, namespaces)
    message = str(info.value)
    assert message.startswith("4 "), message
    assert f"toy.cake_toy_forward: binds {DET_MODULE}.{forward} under" in message
    assert "toy.cake_toy_forward: _LAZY_EXPORTS" not in message


# A facade that forwards to its backend by a call, importing it inside its functions
# (``backend`` stands in for that import): no attribute links the two.
CALLING_FACADE = """
def cake_toy_det(x):
    return backend.cake_toy_det(x)


def cake_toy_det_train_forward(x):
    return backend.cake_toy_det_train_forward(x)


__all__ = ["cake_toy_det", "cake_toy_det_train_forward"]
"""
CALLED_BACKEND = """
def cake_toy_det(x):
    return x


def cake_toy_det_train_forward(x):
    return x
"""


def test_forwards_tie_the_backend_objects_a_facade_calls() -> None:
    backend = _module(BACKEND_NAME, CALLED_BACKEND)
    facade = _module(DET_MODULE, CALLING_FACADE, backend=backend)
    forward = "cake_toy_det_train_forward"
    forwards = [
        ("cake_toy_det", backend.cake_toy_det),
        (forward, getattr(backend, forward)),
    ]
    assert_same_object_same_name(facade, [backend], forwards=forwards)
    # the backend keeps an old name of a function the facade calls
    backend.cake_toy_train_forward = getattr(backend, forward)
    top = _module("toy", cake_toy=backend.cake_toy_det)
    # unseen without forwards
    assert_same_object_same_name(facade, [top, backend])
    with pytest.raises(AssertionError) as info:
        assert_same_object_same_name(facade, [top, backend], forwards=forwards)
    message = str(info.value)
    assert message.startswith("2 "), message
    for alias in (
        f"{BACKEND_NAME}.cake_toy_train_forward: binds the backend object of "
        f"{DET_MODULE}.{forward} under another name",
        f"toy.cake_toy: binds the backend object of {DET_MODULE}.cake_toy_det under",
    ):
        assert alias in message, alias
    with pytest.raises(ValueError, match="'cake_toy_fwd', not a public name"):
        assert_same_object_same_name(
            facade, [backend], forwards=[("cake_toy_fwd", backend.cake_toy_det)]
        )


def test_the_module_is_bound_only_under_its_last_name_part() -> None:
    facade, backend = _facade_and_backend()
    # importing the submodule binds it under its last name part
    top = _module("toy", cake_toy_det_train=facade)
    assert_same_object_same_name(facade, [top, backend])
    # from . import cake_toy_det_train as cake_toy_train
    top.cake_toy_train = facade
    with pytest.raises(AssertionError) as info:
        assert_same_object_same_name(facade, [top, backend])
    message = str(info.value)
    assert message.startswith("1 "), message
    alias = f"toy.cake_toy_train: binds the module {DET_MODULE} under another name"
    assert alias in message, message


def test_old_names_that_are_gone_pass() -> None:
    assert_old_names_fail(
        old_modules=(
            "json.no_such_module_8928",
            "no_such_module_8928",
            "email.no_such_package_8928.child",
        ),
        old_names=(
            ("json", "no_such_name_8928"),
            ("json.decoder", "no_such_name_8928"),
            # the module of the name, or a parent below the top-level package, is gone
            ("json.no_such_module_8928", "name"),
            ("email.no_such_package_8928.child", "name"),
        ),
    )


def test_old_names_that_still_import_fail() -> None:
    with pytest.raises(AssertionError) as info:
        assert_old_names_fail(
            old_modules=("json", "json.decoder", "no_such_package_8928.child"),
            old_names=(("json", "dumps"), ("no_such_module_8928", "name")),
        )
    message = str(info.value)
    assert message.startswith("5 "), message
    for failure in (
        "json: still imports",
        "json.decoder: still imports",
        "no_such_package_8928.child: ModuleNotFoundError for 'no_such_package_8928'",
        "from json import dumps: still imports",
        "from no_such_module_8928 import name: "
        "the top-level module 'no_such_module_8928' is not found",
    ):
        assert failure in message, failure


def test_old_package_directories(tmp_path: Path) -> None:
    package = tmp_path / "toy_8928"
    (package / "stale" / "__pycache__").mkdir(parents=True)
    (package / "stale" / "__pycache__" / "old.cpython-312.pyc").write_bytes(b"")
    (package / "shim").mkdir()
    (package / "shim" / "entry.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "moved.py").write_text('raise ImportError("moved to toy_8928.new")\n')
    (package / "broken.py").write_text("import toy_8928.absent_8928\n")
    # a directory a removed package left behind (no source) counts as gone
    assert_old_names_fail(old_modules=("toy_8928.stale",), cwd=tmp_path)
    with pytest.raises(AssertionError) as info:
        assert_old_names_fail(
            old_modules=("toy_8928.shim", "toy_8928.moved"),
            # a module that imports a missing module other than itself or a parent
            old_names=(("toy_8928.broken", "name"),),
            cwd=tmp_path,
        )
    message = str(info.value)
    assert message.startswith("3 "), message
    assert "toy_8928.shim: still imports" in message
    assert "toy_8928.moved: raises ImportError: moved to toy_8928.new" in message
    assert (
        "from toy_8928.broken import name: ModuleNotFoundError for "
        "'toy_8928.absent_8928', not for toy_8928.broken or a parent"
    ) in message


def test_a_timeout_is_one_failure_among_the_others(tmp_path: Path) -> None:
    (tmp_path / "toy_8928_slow.py").write_text("import time\n\ntime.sleep(60)\n")
    module = _module(DET_MODULE, '__all__ = ["route_key"]\nroute_key = len\n')
    with pytest.raises(AssertionError) as info:
        assert_cake_mode_module(
            module, old_modules=("toy_8928_slow",), cwd=tmp_path, timeout=1
        )
    message = str(info.value)
    assert message.startswith("2 "), message
    assert "toy.cake_toy_det_train.route_key: lacks the mode token 'det'" in message
    assert (
        "the fresh interpreter timed out after 1 s on "
        "old_modules=['toy_8928_slow'], old_names=[]"
    ) in message


def test_all_checks_report_together() -> None:
    facade, backend = _facade_and_backend()
    facade.__all__ = [*facade.__all__, "route_key", "cake_toy_det"]
    facade.route_key = lambda deterministic=True: deterministic
    backend.cake_toy_train_forward = backend.cake_toy_det_train_forward
    # a facade function that calls a backend function kept under an old name
    backend.cake_toy = lambda x: x
    facade.cake_toy_det = lambda x: backend.cake_toy(x)
    with pytest.raises(AssertionError) as info:
        assert_cake_mode_module(
            facade,
            namespaces=[backend],
            forwards=[("cake_toy_det", backend.cake_toy)],
            old_modules=("json",),
            old_names=(("json", "dumps"),),
        )
    message = str(info.value)
    assert message.startswith("6 "), message
    for failure in (
        "toy.cake_toy_det_train.route_key: lacks the mode token 'det'",
        "toy.cake_toy_det_train.route_key: takes a 'deterministic' parameter",
        f"{BACKEND_NAME}.cake_toy_train_forward: binds",
        f"{BACKEND_NAME}.cake_toy: binds the backend object of {DET_MODULE}.",
        "json: still imports",
        "from json import dumps: still imports",
    ):
        assert failure in message, failure
