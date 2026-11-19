# Copyright (c) 2026 by FlashInfer team.
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

"""Per-row live counts bounded by the list width. Not part of the copied sources: added for the classes."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32


class LiveCounts:
    """``counts[row]`` read as ``min(counts[row], width)``, so a row never continues past its own list."""

    def __init__(self, values):
        self.values = values
        self.counts, self.width = values
        self.shape = self.counts.shape

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return LiveCounts(cutlass.new_from_mlir_values(self.values, values))

    @cute.jit
    def __getitem__(self, row):
        return cute.math.min(Int32(self.counts[row]), self.width)
