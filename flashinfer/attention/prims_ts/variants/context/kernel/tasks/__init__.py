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

"""Task definitions for the TS FMHA kernel.

Tasks own ordering, not data movement bodies. Each schedule below sequences
resource waits, acquires, work calls, commits, and releases for one warp role.
The resource methods contain the actual TMA, MMA, softmax, correction, and
epilogue work.

Schedule phase terms follow TS schedule-builder naming. HEAD is the one-time
schedule before the repeated K/V tile loop, LOOP is the repeated K/V tile body,
and TAIL is the one-time cleanup and drain after LOOP exits.
"""
