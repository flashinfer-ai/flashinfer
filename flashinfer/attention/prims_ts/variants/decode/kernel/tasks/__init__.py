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

"""Task definitions for the FMHA decode TS kernel.

Shared K/V streams follow the SwapsMmaAb decode pipeline shape:

- HEAD: initial Q / K setup and first BMM1 wave
- LOOP: staggered K/V loads with Qk0 -> Pv0 -> Qk1 -> Pv1
- TAIL: final V loads, final BMM2, final correction/output

Separate-operand conversion creates independent K and V tasks and rings;
the MMA task still issues QK and PV serially.

Each schedule is written as explicit TS resource transitions. Producer
resources use acquire/work/commit; consumer resources use wait/work/release.
The comments below name the logical pipeline step so the ordering can be read
without expanding the decorators on each resource method.
"""
