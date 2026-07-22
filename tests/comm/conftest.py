# Copyright (c) 2024 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch.distributed as dist


def pytest_sessionfinish(session, exitstatus):
    """Cleanup torch.distributed at the end of pytest session.

    This runs after all tests complete but before Python shutdown,
    avoiding the "destroy_process_group() was not called" warning.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


"""
Shared test utilities for comm tests.
"""

from flashinfer.comm.mnnvl import MnnvlMemory


def mnnvl_available() -> bool:
    """Check if MNNVL is available on this host (all-NVLink-up hardware).

    The POSIX-FD handle exchange no longer needs CAP_SYS_PTRACE / pidfd_getfd
    (it uses SCM_RIGHTS over an abstract-namespace socket), so a container
    capability probe is intentionally not part of this gate -- MNNVL tests must
    run in default-capability containers, the exact scenario the exchange fix
    enables.
    """
    return MnnvlMemory.supports_mnnvl()


def pytest_addoption(parser):
    parser.addoption("--num_nodes", type=int, default=1)
    parser.addoption("--node_id", type=int, default=0)
    parser.addoption("--dist_init_method", type=str, default="tcp://localhost:29501")


@pytest.fixture
def num_nodes(request):
    return request.config.getoption("--num_nodes")


@pytest.fixture
def node_id(request):
    return request.config.getoption("--node_id")


@pytest.fixture
def dist_init_method(request):
    return request.config.getoption("--dist_init_method")
