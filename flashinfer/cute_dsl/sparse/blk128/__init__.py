# Copyright (c) 2025 by FlashInfer team.
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

try:
    import quack  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The VSA Blackwell (blk128) backend requires the `quack` package "
        "from https://github.com/Dao-AILab/quack.\n"
        "Note: the `quack` package on PyPI is an unrelated project — install "
        "from source instead:\n"
        "    pip install git+https://github.com/Dao-AILab/quack.git"
    ) from e
