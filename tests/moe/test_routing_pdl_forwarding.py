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

"""Routing PDL propagation and numerical/graph regression coverage.

Reuse the odd-tile accuracy tests with both PDL settings. The spy also checks
that the core forwards the requested flag, so silently using moe_sort's default
cannot pass merely because the numerical output is still correct.
"""

import importlib
from unittest.mock import patch

import pytest

from .test_cute_dsl_fused_moe import TestOddTileCountBoundsContract as _Bounds


@pytest.mark.parametrize("routing_pdl", [False, True], ids=["pdl-off", "pdl-on"])
class TestRoutingPdlForwarding(_Bounds):
    @pytest.fixture(autouse=True)
    def _select_pdl(self, routing_pdl):
        self.routing_pdl = routing_pdl

    def _run_eager(self, tensors, buffers, num_experts, top_k, tactic_kwargs):
        module = importlib.import_module("flashinfer.fused_moe.cute_dsl.fused_moe")
        with patch.object(module, "moe_sort", wraps=module.moe_sort) as routing:
            output = super()._run_eager(
                tensors,
                buffers,
                num_experts,
                top_k,
                dict(tactic_kwargs, enable_pdl=self.routing_pdl),
            )
            routing.assert_called_once()
            assert routing.call_args.kwargs["enable_pdl"] is self.routing_pdl
        return output
