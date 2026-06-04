import types
import typing

import cutlass.cute as cute


_in_mem_compile_cache: dict = {}


def _as_options_tuple(options):
    if options is None:
        return ()
    if isinstance(options, tuple):
        return options
    return (options,)


class KeyedCompileMixin:
    def _get_compile_key(self):
        compile_key = getattr(self, "_KeyedCompileMixin_compile_key", None)
        if compile_key is None:
            collected_attrs = []
            for attr_name in sorted(dir(self)):
                attr_value = getattr(self, attr_name)
                if attr_name.startswith("__"):
                    continue
                if isinstance(
                    attr_value,
                    (
                        types.MethodType,
                        types.BuiltinMethodType,
                        types.MethodWrapperType,
                    ),
                ):
                    continue

                if isinstance(attr_value, typing.Hashable):
                    collected_attrs.append((attr_name, attr_value))

            compile_key = (str(type(self).__mro__),) + tuple(collected_attrs)
            setattr(self, "_KeyedCompileMixin_compile_key", compile_key)  # noqa: B010

        return compile_key


def _compile_options_key(options):
    if options is None:
        return None
    options = _as_options_tuple(options)
    return tuple((type(option), option.value) for option in options)


def cached_compile(func, *args, compile_options=None, **kwargs):
    cache_key = (func._get_compile_key(), _compile_options_key(compile_options))
    compiled_fn = _in_mem_compile_cache.get(cache_key)

    if compiled_fn is None:
        compiler = cute.compile
        effective_compile_options = compile_options
        if effective_compile_options:
            compiler = cute.compile[effective_compile_options]
        compiled_fn = compiler(func, *args, **kwargs)
        _in_mem_compile_cache[cache_key] = compiled_fn

    return compiled_fn
