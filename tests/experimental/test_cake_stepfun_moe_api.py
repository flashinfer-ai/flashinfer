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

flashinfer public-API tests of the Cake StepFun fused-MoE package (placeholder).

The suite is delivered byte-for-byte from ``exports/stepfun_moe/test_api_template.py``
of the Cake repository together with the generated programs: it drives
``flashinfer.fused_moe`` with ``CakeStepFunConfig(backend="cake_stepfun")``,
compares every family against the native trtllm-gen kernels and writes the
per-family error table.  Until that delivery lands this module skips.
"""

import pytest

pytest.skip(
    "placeholder: the API suite of flashinfer.experimental.cake_stepfun_moe is "
    "delivered with the generated programs (exports/stepfun_moe/test_api_template.py)",
    allow_module_level=True,
)
