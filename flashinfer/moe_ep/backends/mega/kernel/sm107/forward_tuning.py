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
        from .....batch_invariant.spec import f32_bits

        def layout(value, *, batched=False):
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return {"scalar_f32": f32_bits(value)}
            return [
                str(value.dtype),
                list(value.shape[1:] if batched else value.shape),
                list(value.stride()),
                value.device.type,
            ]

        batched = (
            "hidden_states",
            "topk_ids",
            "topk_weights",
            "scales",
            "fc1_activation_per_token_scale",
        )
        inputs = {name: getattr(self.tensors, name) for name in batched}
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
            "inputs": {
                name: layout(value, batched=name in batched)
                for name, value in inputs.items()
            },
            "num_valid_tokens": layout(
                getattr(self.backend._kernel_config, "num_valid_tokens_tensor", None)
            ),
            "output": layout(self.output, batched=True),
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
