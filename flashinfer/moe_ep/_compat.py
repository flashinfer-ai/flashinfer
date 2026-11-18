"""Compatibility for renamed MegaMoE host arguments."""

import warnings
from functools import wraps
from inspect import signature


def deprecated_aliases(**aliases):
    """Accept old keywords and expose live aliases on dataclasses.

    Old keywords take precedence, including in ``dataclasses.replace``.
    Only canonical names are dataclass fields, so aliases cannot become stale.
    """

    def decorate(target):
        is_class = isinstance(target, type)
        function = target.__init__ if is_class else target

        @wraps(function)
        def wrapped(*args, **kwargs):
            for old, new in aliases.items():
                if old in kwargs:
                    value = kwargs.pop(old)
                    if value is not None:
                        warnings.warn(
                            f"{old} is deprecated; use {new}.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        kwargs[new] = value
            return function(*args, **kwargs)

        if not is_class:
            return wrapped
        target.__init__ = wrapped
        for old, new in aliases.items():
            setattr(
                target,
                old,
                property(
                    lambda self, name=new: getattr(self, name),
                    lambda self, value, name=new: setattr(self, name, value),
                ),
            )
        return target

    return decorate


def legacy_finalize(cls):
    """Keep the former ``do_finalize`` keyword, positional bool, and attribute."""
    from .config import FinalizeMode

    init = cls.__init__
    position = list(signature(init).parameters).index("finalize_mode") - 1

    def to_mode(value):
        return FinalizeMode.AUTO if value else FinalizeMode.NONE

    @wraps(init)
    def wrapped(self, *args, **kwargs):
        value = kwargs.pop("do_finalize", None)
        if value is not None:
            warnings.warn(
                "do_finalize is deprecated; use finalize_mode.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["finalize_mode"] = to_mode(value)
        if len(args) > position and isinstance(args[position], bool):
            warnings.warn(
                "positional do_finalize is deprecated; use finalize_mode=.",
                DeprecationWarning,
                stacklevel=2,
            )
            args = (*args[:position], to_mode(args[position]), *args[position + 1 :])
        init(self, *args, **kwargs)

    cls.__init__ = wrapped
    cls.do_finalize = property(
        lambda self: self.finalize_mode is not FinalizeMode.NONE,
        lambda self, value: setattr(self, "finalize_mode", to_mode(value)),
    )
    return cls
