# Conftest for communication tests
import torch.distributed as dist


def pytest_sessionfinish(session, exitstatus):
    """Cleanup torch.distributed at the end of pytest session.

    This runs after all tests complete but before Python shutdown,
    avoiding the "destroy_process_group() was not called" warning.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
