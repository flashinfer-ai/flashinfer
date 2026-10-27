import functools
import threading

from .jit.core import logger
from .jit.tllm_utils import gen_trtllm_utils_module


# functools.cache does not stop concurrent first calls from each running the
# loader, so serialize them; otherwise every waiting thread would rerun a
# failed build and log its own warning.
_load_lock = threading.Lock()


@functools.cache
def _load_trtllm_utils_module():
    # The autotuner calls delay_kernel() for every tactic and only logs
    # per-tactic failures at DEBUG. functools.cache does not cache exceptions
    # and JitSpec.build() reruns ninja, so remember a failed build instead of
    # rebuilding it for every tactic, and report it once.
    try:
        return gen_trtllm_utils_module().build_and_load(), None
    except Exception as e:
        logger.warning(
            "Failed to build JIT module trtllm_utils; delay_kernel() will raise "
            "for the rest of this process and autotuning skips the tactics "
            "that need it. %s: %s",
            type(e).__name__,
            e,
        )
        return None, e


def get_trtllm_utils_module():
    with _load_lock:
        module, error = _load_trtllm_utils_module()
    if error is not None:
        raise RuntimeError(
            "JIT module trtllm_utils failed to build earlier in this process"
        ) from error
    return module


def delay_kernel(stream_delay_micro_secs: int):
    get_trtllm_utils_module().delay_kernel(stream_delay_micro_secs)
