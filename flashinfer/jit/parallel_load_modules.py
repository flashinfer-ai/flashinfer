import threading
from typing import Any, Callable, List, Tuple

from .core import logger


def parallel_load_modules(
    load_module_func_args: List[Tuple[Callable, List[Any]]],
):
    # TODO(lequn): Change callers. Remove this function. Replace by build_jit_specs().
    from .core import JitSpec

    threads = []
    exceptions = []

    def wrapper(func, args):
        try:
            ret = func(*args)
            if isinstance(ret, JitSpec):
                ret.build_and_load()
        except Exception as e:
            exceptions.append((func, e))

    for func, args in load_module_func_args:
        thread = threading.Thread(target=wrapper, args=(func, args))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if exceptions:
        for func, e in exceptions:
            print(f"Exception occurred in {func.__name__}: {e}")
        raise RuntimeError("One or more exceptions occurred during module loading")

    logger.info("Finished loading modules")
