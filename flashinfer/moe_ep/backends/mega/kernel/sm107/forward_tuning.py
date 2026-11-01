"""The forward measured by Rubin online and offline tuning."""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .....tensors import MoEEpTensors
    from .automatic import RubinBackend


class ForwardExecution:
    def __init__(
        self,
        backend: RubinBackend,
        tensors: MoEEpTensors,
        *,
        quantize_input: bool,
        output: torch.Tensor | None,
    ):
        self.backend, self.tensors = backend, tensors
        self.quantize_input, self.output = quantize_input, output

    def contract(self):
        def layout(value):
            if value is None:
                return None
            return [str(value.dtype), list(value.shape[1:]), list(value.stride())]

        inputs = {
            name: getattr(self.tensors, name)
            for name in ("hidden_states", "topk_ids", "topk_weights", "scales")
        }
        for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
            value = getattr(self.tensors, name)
            inputs[name] = (
                value
                if value is not None
                else getattr(self.backend._kernel_config, name, None)
            )
        return {
            "quantize_input": self.quantize_input,
            "input_norm_const": getattr(
                self.backend._kernel_config, "input_norm_const", 1.0
            ),
            "inputs": {name: layout(value) for name, value in inputs.items()},
            "output": layout(self.output),
            "staging": {
                name: os.environ.get(name, default)
                for name, default in (
                    ("FLASHINFER_MEGA_FUSED_STAGE", "1"),
                    ("FLASHINFER_SM107_MEGA_CHECK_ROUTES", "0"),
                    ("FLASHINFER_SM107_FUSED_ROUTE_CHECK", "1"),
                    ("FLASHINFER_SM107_PREP_TVM_FFI", "1"),
                )
            },
        }

    def make_trial(self, workspace):
        from .....kernel_src.sm107.next_cutedsl_megamoe import KNOB_KEYS

        config = dataclasses.replace(
            self.backend._kernel_config,
            kernel_variant=workspace.config.kernel_variant,
            knobs={name: getattr(workspace.config, name) for name in KNOB_KEYS},
        )
        backend = type(self.backend)(config)
        backend.bind_ep_bootstrap(self.backend._ep_bootstrap)
        return _ForwardTrial(self, backend, workspace)

    def tune(self, l1, l2, workspace, *, force_tune=False, **kwargs):
        from .....kernel_src.sm107.next_cutedsl_megamoe.shim.autotune import (
            _choose_protocol,
        )
        from .....kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_protocol import (
            RubinTuningProtocol,
        )

        protocol = RubinTuningProtocol(
            self.output,
            l1,
            l2,
            workspace,
            make_trial=self.make_trial,
            execution_contract=self.contract(),
            **kwargs,
        )
        return _choose_protocol(protocol, force_tune=force_tune)


class _ForwardTrial:
    def __init__(self, execution, backend, workspace):
        self.execution, self.backend, self.workspace = execution, backend, workspace
        self.config, self.device = workspace.config, workspace.device

    def launch(self, l1, l2):
        execution = self.execution
        self.backend.stage_inputs(
            execution.tensors, self.workspace, quantize_input=execution.quantize_input
        )
        self.backend.compute(self.workspace, (l1, l2), output=execution.output)
