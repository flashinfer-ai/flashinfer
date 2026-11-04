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
"""

"""Previous import path of :mod:`flashinfer.experimental.cake_dsa_indexer_det`.

The package and its submodules ``cake_backend``, ``cake_build``, ``cake_jit`` and ``cake_policy`` resolve to the
module objects of ``flashinfer.experimental.cake_dsa_indexer_det``.
"""

from .. import cake_dsa_indexer_det as _canonical


def __getattr__(name):
    return getattr(_canonical, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_canonical)))
