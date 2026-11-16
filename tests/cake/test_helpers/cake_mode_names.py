# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""Mode names of the Cake training modules (flashinfer issue #8928).

A Cake training module is ``flashinfer.cake_<op>_<mode>_train[_kcp]``, its ``mode``
``det`` or ``nondet``.  Its public names are its ``__all__`` or, without one, the
public attributes defined in it (:func:`public_names`).  The checks:

* :func:`assert_mode_names`: every public name carries the mode word of the module,
  the ``_``-delimited token ``det`` / ``nondet`` of a snake_case name or the part
  ``Det`` / ``Nondet`` of a CamelCase name; the word of the other mode fails.
  ALL_CAPS constants (:data:`CONSTANT`) are exempt.  With ``stem=True`` the mode also
  comes right after the op: a snake_case name starts with ``cake_<op>_<mode>`` (the
  support query with ``is_<module>``), a CamelCase name with ``Cake<Op><Mode>``.
* :func:`assert_no_deterministic_keyword`: no public callable accepts
  ``deterministic``: ``inspect.signature(obj).bind_partial(deterministic=False)``
  raises ``TypeError`` (so a ``**kwargs`` fails too) for a function, the constructor
  of a class and the ``forward`` / ``backward`` of a ``torch.autograd.Function``
  subclass, and no public method has a parameter named ``deterministic``.  A type
  alias (``Callable[[], Tensor]``, ``list[int]``) names a type and is skipped.
* :func:`assert_same_object_same_name`: the module and the other namespaces given
  (the ``flashinfer`` top level, the backend package or modules) bind the module, an
  object of it or a backend object it forwards to (``forwards``) only under the name
  the module gives it, so no alias of an old name is left.  The module's lazy public
  names and the rows of a ``_LAZY_EXPORTS`` table count too.
* :func:`assert_old_names_fail`: old module paths and old names fail at import, in a
  fresh interpreter.

:func:`assert_cake_mode_module` runs all of them for one module and reports every
failure in one ``AssertionError``.  This module imports neither flashinfer nor torch:
a family's names test passes its modules in.

Nothing here is keyed on the existence of ``flashinfer.experimental.cake.<module>``:
no check looks a backend up by its path, so none skips silently when the path is
absent.  Each family test passes its facade and, through ``namespaces``, its backend
package or modules explicitly (the conv1d KCP facade, for one, passes the ``kcp``
module of the single-rank backend package).  A call is not seen: a facade function
that calls its backend (the facade imports its backend inside its functions or its
``__getattr__``) ties the backend object to its name only through the
``(name, object)`` pair the family test lists in ``forwards``.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
import subprocess
import sys
import typing
from collections.abc import Iterable
from os import PathLike
from types import ModuleType

MODES = ("det", "nondet")
# module constants: exempt from the mode word
CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]*$")
# the last part of the name of a Cake training module
MODULE = re.compile(r"^cake_(?P<op>[a-z0-9_]+?)_(?P<mode>det|nondet)_train(?:_kcp)?$")
# the parts of a CamelCase name: an acronym before a word (digits allowed: FP8, HD256),
# a word, a run of capitals
_CAMEL_PARTS = re.compile(r"[A-Z][A-Z0-9]*(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z0-9]+")
# the line of the fresh interpreter's result (printed by _OLD_NAMES_SCRIPT)
_MARKER = "CAKE_OLD_NAMES "
_TIMEOUT = 600.0
# torch.autograd.Function by module and qualified name: torch is not imported
_AUTOGRAD_FUNCTION = ("torch.autograd.function", "Function")


def public_names(module: ModuleType) -> list[str]:
    """The ``__all__`` of ``module``; without one, its public attributes defined in it.

    "Defined in it" drops submodules and the objects whose ``__module__`` names another
    module (imports); values without ``__module__`` (numbers, tuples) stay.
    """
    exported = getattr(module, "__all__", None)
    if exported is not None:
        return list(exported)
    return sorted(
        name
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and not isinstance(obj, ModuleType)
        and getattr(obj, "__module__", module.__name__) == module.__name__
    )


def assert_mode_names(
    module: ModuleType,
    mode: str | None = None,
    *,
    names: Iterable[str] | None = None,
    stem: bool = True,
    stem_exceptions: Iterable[str] = (),
) -> None:
    """Every public name of ``module`` carries the mode word, and (``stem``) in order.

    ``mode`` defaults to the mode in the module's name; ``names`` defaults to
    :func:`public_names`.  The order check needs a module named
    ``cake_<op>_<mode>_train[_kcp]``; pass ``stem=False`` for another module.  The
    names in ``stem_exceptions`` (exceptions a family documents) skip the order check
    only, not the mode word.
    """
    _raise(_mode_name_errors(module, mode, names, stem, stem_exceptions))


def assert_no_deterministic_keyword(objs: ModuleType | Iterable[object]) -> None:
    """No callable of ``objs`` (a module: its public names) takes ``deterministic``.

    An entry point fails when ``inspect.signature(entry).bind_partial(
    deterministic=False)`` does not raise ``TypeError``, so a ``**kwargs`` fails too.
    The entry points are a callable that is not a class, the constructor of a class,
    and the ``forward`` and ``backward`` of a ``torch.autograd.Function`` subclass
    (inherited ones included).  Such a subclass is found by the module and name of
    its base, torch not imported; its constructor is torch's ``(*args, **kwargs)``,
    not an entry point.  Each of these, the ``forward`` and ``backward`` of another
    class and the public methods a class body defines also fail with a parameter
    named ``deterministic``, in its own signature or in the one of the function it
    wraps (``__wrapped__``).  A callable without a Python signature (a builtin type)
    is skipped, and so is a type alias: a subscripted generic such as
    ``Callable[[], Tensor]``, ``Optional[Tensor]`` or ``list[int]``
    (``typing.get_origin`` gives its origin) names a type, not an entry point,
    although it is callable with typing's ``(*args, **kwargs)``.  A class it
    subscripts is checked when the class itself is public.
    """
    _raise(_deterministic_errors(objs))


def assert_same_object_same_name(
    module: ModuleType,
    namespaces: Iterable[ModuleType] = (),
    *,
    forwards: Iterable[tuple[str, object]] = (),
) -> None:
    """``module`` and ``namespaces`` bind ``module`` and each object of it only under
    its name in ``module``.

    The module itself has the last part of its name (``flashinfer`` binds
    ``cake_x_nondet_train`` under that name only, eagerly or lazily).  The objects of
    the module are its public callables, the functions they wrap (``__wrapped__``, as
    :func:`functools.wraps` sets) and the object of each ``(name, obj)`` pair of
    ``forwards``: the backend object that the public ``name`` of ``module`` calls, a
    link no attribute shows.  So a facade entry that wraps or calls a backend
    function ties the backend's name for it too.  A name of ``forwards`` that is not
    a public name of ``module`` raises ``ValueError``.  ALL_CAPS constants are left
    out.  Names starting with ``_`` are private and skipped.

    A namespace binds its attributes (``vars``); ``module`` also binds the public names
    its module ``__getattr__`` serves lazily, resolved with ``getattr``.  Another
    namespace is read through ``vars`` only: pass a backend package that serves its
    names through its own module ``__getattr__`` as the submodules it serves from
    (the MoE grouped-GEMM backend package as its ``cake_backend`` and
    ``cake_autograd``).  A namespace that defines a dict ``_LAZY_EXPORTS``
    (``name -> (module, attribute)``, as in ``flashinfer/__init__.py``) has its rows
    read statically, nothing imported: a row whose module is ``module`` serves
    ``attribute`` under the same name, or with ``attribute`` None the module itself
    under the last part of the module's name; another name is an alias.  A relative
    module name resolves against the namespace's ``__name__``, as
    ``importlib.import_module(name, __name__)`` does; rows of other modules are
    ignored.
    """
    _raise(_same_object_errors(module, namespaces, forwards))


def assert_old_names_fail(
    old_modules: Iterable[str] = (),
    old_names: Iterable[tuple[str, str]] = (),
    *,
    cwd: str | PathLike[str] | None = None,
    timeout: float = _TIMEOUT,
) -> None:
    """The old names fail at import, checked in a fresh ``sys.executable`` interpreter.

    Each path of ``old_modules`` raises ``ModuleNotFoundError`` for itself or for a
    parent below the top-level package; a directory a removed package left behind
    (a namespace package without a ``.py`` file) counts as gone.  For each
    ``(module, name)`` of ``old_names``, ``from module import name`` raises
    ``ImportError``: the module imports without the name, or the module or a parent
    of it below the top-level package is not found.  The top-level package must
    import, so a wrong ``sys.path`` does not pass.  ``cwd`` is the working directory
    of the interpreter (the first entry of its ``sys.path``); a ``timeout`` of the
    interpreter is one failure.
    """
    _raise(_old_name_errors(old_modules, old_names, cwd, timeout))


def assert_cake_mode_module(
    module: ModuleType,
    mode: str | None = None,
    *,
    namespaces: Iterable[ModuleType] = (),
    forwards: Iterable[tuple[str, object]] = (),
    old_modules: Iterable[str] = (),
    old_names: Iterable[tuple[str, str]] = (),
    stem: bool = True,
    stem_exceptions: Iterable[str] = (),
    cwd: str | PathLike[str] | None = None,
    timeout: float = _TIMEOUT,
) -> None:
    """All the checks for one public facade module, every failure in one report.

    ``namespaces`` are the other modules that bind objects of ``module``: the
    ``flashinfer`` top level and the backend package or modules the facade forwards
    to or re-exports, each passed explicitly (nothing is found by path, see the
    module docstring; a lazy backend package as the submodules it serves from).
    ``forwards`` are the ``(public name, backend object)`` pairs of the facade
    functions that call their backend.  Both are those of
    :func:`assert_same_object_same_name`.  ``old_modules`` and ``old_names`` are the
    module paths and the ``(module, name)`` pairs before the rename; ``cwd`` and
    ``timeout`` are those of :func:`assert_old_names_fail`.
    """
    _raise(
        [
            *_mode_name_errors(module, mode, None, stem, stem_exceptions),
            *_deterministic_errors(module),
            *_same_object_errors(module, namespaces, forwards),
            *_old_name_errors(old_modules, old_names, cwd, timeout),
        ]
    )


def _raise(errors: list[str]) -> None:
    if errors:
        raise AssertionError(
            f"{len(errors)} Cake mode-name failure(s):\n  " + "\n  ".join(errors)
        )


def _mode_name_errors(
    module: ModuleType,
    mode: str | None,
    names: Iterable[str] | None,
    stem: bool,
    stem_exceptions: Iterable[str],
) -> list[str]:
    leaf = module.__name__.rpartition(".")[2]
    match = MODULE.match(leaf)
    if match is not None:
        if mode not in (None, match["mode"]):
            raise ValueError(
                f"{module.__name__} is a {match['mode']} module, not {mode}"
            )
        mode = match["mode"]
    if mode not in MODES:
        raise ValueError(
            f"{module.__name__} is not named cake_<op>_<mode>_train[_kcp]: "
            f"pass mode, one of {MODES}"
        )
    if stem and match is None:
        raise ValueError(
            f"{module.__name__} is not named cake_<op>_<mode>_train[_kcp]: "
            "pass stem=False"
        )
    prefix = f"cake_{match['op']}_{mode}" if stem else None
    support = f"is_{leaf}"
    exempt = set(stem_exceptions)
    errors = []
    for name in public_names(module) if names is None else names:
        if CONSTANT.match(name):
            continue
        skip_order = name in exempt
        error = _name_error(name, mode, None if skip_order else prefix, support)
        if error is not None:
            errors.append(f"{module.__name__}.{name}: {error}")
    return errors


def _name_error(name: str, mode: str, prefix: str | None, support: str) -> str | None:
    """Why ``name`` breaks the rule of a ``mode`` module, or None.

    ``prefix`` is ``cake_<op>_<mode>`` (None: no order check) and ``support`` the
    prefix ``is_<module>`` of the support query.
    """
    other = "nondet" if mode == "det" else "det"
    camel = name[:1].isupper()
    if camel:
        words = _CAMEL_PARTS.findall(name)
        own, wrong, kind = mode.capitalize(), other.capitalize(), "part"
    else:
        words = name.split("_")
        own, wrong, kind = mode, other, "token"
    if wrong in words:
        return f"carries {wrong!r}, the word of the other mode"
    if own not in words:
        return f"lacks the mode {kind} {own!r}"
    if prefix is None:
        return None
    if camel:
        if not name.lower().startswith(prefix.replace("_", "")):
            return f"does not start with Cake<Op><Mode> ({prefix!r} in CamelCase)"
    elif not any(name == p or name.startswith(p + "_") for p in (prefix, support)):
        return f"does not start with {prefix!r} (or {support!r} for the support query)"
    return None


def _deterministic_errors(objs: ModuleType | Iterable[object]) -> list[str]:
    if isinstance(objs, ModuleType):
        items = [
            (f"{objs.__name__}.{name}", getattr(objs, name))
            for name in public_names(objs)
        ]
    else:
        items = [(_label(obj), obj) for obj in objs]
    errors = []
    for label, obj in items:
        if not callable(obj) or _is_type_alias(obj):
            continue
        for where, target, entry in _signature_targets(label, obj):
            error = _deterministic_error(target, entry)
            if error is not None:
                errors.append(f"{where}: {error}")
    return errors


def _deterministic_error(target: object, entry: bool) -> str | None:
    """Why ``target`` takes ``deterministic``, or None.

    The parameter names are read in its own signature and in the one of the function
    it wraps (``__wrapped__``).  An ``entry`` point also fails when the signature
    ``inspect.signature`` gives binds ``deterministic=False`` (a ``**kwargs``).  A
    callable without a Python signature has none.
    """
    names: set[str] = set()
    signature = None
    for follow_wrapped in (True, False):
        try:
            found = inspect.signature(target, follow_wrapped=follow_wrapped)
        except (TypeError, ValueError):  # no Python signature
            continue
        names.update(found.parameters)
        if follow_wrapped:
            signature = found
    if "deterministic" in names:
        return "takes a 'deterministic' parameter"
    if not entry or signature is None:
        return None
    try:
        signature.bind_partial(deterministic=False)
    except TypeError:
        return None
    # no parameter has the name, so a **kwargs takes it
    kwargs = next(
        (p.name for p in signature.parameters.values() if p.kind is p.VAR_KEYWORD),
        "kwargs",
    )
    return f"takes a 'deterministic' parameter through **{kwargs}"


def _label(obj: object) -> str:
    qualname = getattr(obj, "__qualname__", None)
    if qualname is None:
        return repr(obj)
    return f"{getattr(obj, '__module__', '?')}.{qualname}"


def _signature_targets(label: str, obj: object) -> list[tuple[str, object, bool]]:
    """``(label, callable, entry point)`` for ``obj`` and, for a class, its forward,
    backward and public methods.
    """
    if not inspect.isclass(obj):
        return [(label, obj, True)]
    autograd = _is_autograd_function(obj)
    targets = [(label, obj, not autograd)]
    attrs = {n for n in vars(obj) if not n.startswith("_")}
    for attr in sorted(attrs | {"forward", "backward"}):
        member = getattr(obj, attr, None)
        if (
            callable(member)
            and not inspect.isclass(member)
            and not _is_type_alias(member)
        ):
            entry = autograd and attr in ("forward", "backward")
            targets.append((f"{label}.{attr}", member, entry))
    return targets


def _is_type_alias(obj: object) -> bool:
    """``obj`` is a subscripted generic (``Callable[[], Tensor]``, ``list[int]``):
    a type, not an entry point."""
    return typing.get_origin(obj) is not None


def _is_autograd_function(cls: type) -> bool:
    """``cls`` subclasses ``torch.autograd.Function``, found without importing torch."""
    return any(
        (getattr(base, "__module__", None), base.__qualname__) == _AUTOGRAD_FUNCTION
        for base in cls.__mro__
    )


def _same_object_errors(
    module: ModuleType,
    namespaces: Iterable[ModuleType],
    forwards: Iterable[tuple[str, object]],
) -> list[str]:
    # the public names of the module, the lazy ones resolved by its __getattr__
    public = {name: getattr(module, name) for name in public_names(module)}
    # id of an object of the module -> (the object, kept alive; the names it may be
    # bound under; how a report shows it)
    owned: dict[int, tuple[object, set[str], set[str]]] = {}

    def own(obj: object, name: str, shown: str) -> None:
        entry = owned.setdefault(id(obj), (obj, set(), set()))
        entry[1].add(name)
        entry[2].add(shown)

    own(module, module.__name__.rpartition(".")[2], f"the module {module.__name__}")
    for name, obj in public.items():
        if CONSTANT.match(name) or not callable(obj):
            continue
        for target in (obj, inspect.unwrap(obj)):
            own(target, name, f"{module.__name__}.{name}")
    for name, obj in forwards:
        if name not in public:
            raise ValueError(
                f"forwards names {name!r}, not a public name of {module.__name__}"
            )
        for target in (obj, inspect.unwrap(obj)):
            own(target, name, f"the backend object of {module.__name__}.{name}")
    errors = []
    seen: set[int] = set()
    for namespace in (module, *namespaces):
        if id(namespace) in seen:
            continue
        seen.add(id(namespace))
        bindings = dict(vars(namespace))
        if namespace is module:  # its lazy public names too
            for name, obj in public.items():
                bindings.setdefault(name, obj)
        reported: set[str] = set()
        for name, obj in bindings.items():
            entry = owned.get(id(obj))
            if name.startswith("_") or entry is None or name in entry[1]:
                continue
            shown = " / ".join(sorted(entry[2]))
            errors.append(
                f"{namespace.__name__}.{name}: binds {shown} under another name"
            )
            reported.add(name)
        # a lazy row already bound (and reported) above is reported once
        errors.extend(
            error
            for name, error in _lazy_export_errors(module, namespace)
            if name not in reported
        )
    return errors


def _lazy_export_errors(
    module: ModuleType, namespace: ModuleType
) -> list[tuple[str, str]]:
    """``(name, error)`` for each row of the ``_LAZY_EXPORTS`` dict of ``namespace``
    that serves ``module`` or an attribute of it under another name.

    The table is read from ``vars`` and nothing is imported.
    """
    table = vars(namespace).get("_LAZY_EXPORTS")
    if not isinstance(table, dict):
        return []
    leaf = module.__name__.rpartition(".")[2]
    errors = []
    for name, (module_name, attribute) in table.items():
        if name.startswith("_"):
            continue
        try:
            # as importlib.import_module(module_name, namespace.__name__) resolves it
            target = importlib.util.resolve_name(module_name, namespace.__name__)
        except ImportError:  # a relative name above the top-level package
            continue
        expected = leaf if attribute is None else attribute
        if target != module.__name__ or name == expected:
            continue
        if attribute is None:
            bound = f"the module {module.__name__}"
        else:
            bound = f"{module.__name__}.{attribute}"
        errors.append(
            (
                name,
                f"{namespace.__name__}.{name}: _LAZY_EXPORTS binds {bound} "
                "under another name",
            )
        )
    return errors


# Run by ``sys.executable -c`` with one argument, the JSON ``[old_modules, old_names]``;
# prints _MARKER and the JSON list of failures.
_OLD_NAMES_SCRIPT = r"""
import importlib
import json
import sys
from pathlib import Path

modules, names = json.loads(sys.argv[1])
failures = []


def gone(path, error):
    # the module, or a parent of it below the top-level package, is missing
    missing = error.name or ""
    return missing == path or ("." in missing and path.startswith(missing + "."))


def stale(module):
    # a namespace package without source: a directory a removed package left behind
    paths = getattr(module, "__path__", None)
    if getattr(module, "__file__", None) is not None or paths is None:
        return False
    return not any(next(Path(p).glob("*.py"), None) for p in paths)


for path in modules:
    try:
        module = importlib.import_module(path)
    except ModuleNotFoundError as error:
        if not gone(path, error):
            failures.append(
                f"{path}: ModuleNotFoundError for {error.name!r}, not for the module "
                "or a parent below the top-level package"
            )
        continue
    except Exception as error:
        failures.append(
            f"{path}: raises {type(error).__name__}: {error}, "
            "not ModuleNotFoundError"
        )
        continue
    if not stale(module):
        failures.append(f"{path}: still imports ({getattr(module, '__file__', None)})")

for path, name in names:
    statement = f"from {path} import {name}"
    try:
        importlib.import_module(path)
    except ModuleNotFoundError as error:
        # the module, or a parent of it below the top-level package, is gone, so is
        # the name; a missing top-level module proves nothing (a wrong sys.path)
        missing = error.name or ""
        if "." in missing and gone(path, error):
            continue
        if missing and "." not in missing:
            reason = f"the top-level module {missing!r} is not found"
        else:
            reason = (
                f"ModuleNotFoundError for {error.name!r}, not for {path} or a parent"
            )
        failures.append(f"{statement}: {reason}")
        continue
    except Exception as error:
        failures.append(
            f"{statement}: {path} does not import ({type(error).__name__}: {error})"
        )
        continue
    try:
        exec(statement, {})
    except ModuleNotFoundError as error:
        if error.name != f"{path}.{name}":
            failures.append(
                f"{statement}: ModuleNotFoundError for {error.name!r}, "
                "not ImportError for the name"
            )
    except ImportError:
        pass
    except Exception as error:
        failures.append(
            f"{statement}: raises {type(error).__name__}: {error}, not ImportError"
        )
    else:
        failures.append(f"{statement}: still imports")

print("CAKE_OLD_NAMES " + json.dumps(failures))
"""


def _old_name_errors(
    old_modules: Iterable[str],
    old_names: Iterable[tuple[str, str]],
    cwd: str | PathLike[str] | None,
    timeout: float,
) -> list[str]:
    modules = list(old_modules)
    names = [tuple(pair) for pair in old_names]
    if not modules and not names:
        return []
    try:
        result = subprocess.run(
            [sys.executable, "-c", _OLD_NAMES_SCRIPT, json.dumps([modules, names])],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as error:
        # the output captured so far is bytes, text=True or not
        captured = error.stderr or b""
        stderr = (
            captured.decode(errors="replace")
            if isinstance(captured, bytes)
            else captured
        )
        return [
            f"the fresh interpreter timed out after {timeout:g} s on "
            f"old_modules={modules!r}, old_names={names!r}: {stderr[-2000:]}"
        ]
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_MARKER):
            return json.loads(line[len(_MARKER) :])
    return [
        f"the fresh interpreter exited {result.returncode} without a result: "
        f"{result.stderr[-2000:]}"
    ]
