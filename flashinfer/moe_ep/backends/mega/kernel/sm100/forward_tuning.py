"""The complete forward used by explicit Blackwell inner EP searches."""

from __future__ import annotations

import dataclasses
import os


class ForwardExecution:
    def __init__(self, backend, tensors, *, quantize_input, output):
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

        cfg = self.backend._config
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
            inputs[name] = value if value is not None else getattr(cfg, name, None)
        return {
            "quantize_input": self.quantize_input,
            "input_norm_const": getattr(cfg, "input_norm_const", 1.0),
            "inputs": {
                name: layout(value, batched=name in batched)
                for name, value in inputs.items()
            },
            "num_valid_tokens": layout(getattr(cfg, "num_valid_tokens_tensor", None)),
            "output": layout(self.output, batched=True),
            "staging": {
                "FLASHINFER_MEGA_FUSED_STAGE": os.environ.get(
                    "FLASHINFER_MEGA_FUSED_STAGE", "1"
                )
            },
        }

    def make_trial(self, workspace):
        from .....kernel_src.sm100.next_cutedsl_megamoe.shim.tuning import knob_keys

        config = dataclasses.replace(
            self.backend._config,
            knobs={
                name: getattr(workspace.config, name)
                for name in knob_keys(workspace.config)
            },
        )
        backend = type(self.backend)(config)
        backend.bind_ep_bootstrap(self.backend._ep_bootstrap)
        return _ForwardTrial(self, backend, workspace)


class _ForwardTrial:
    def __init__(self, execution, backend, workspace):
        self.execution, self.backend, self.workspace = execution, backend, workspace
        self.config, self.device = workspace.config, workspace.device

    def launch(self, l1, l2):
        self.backend.stage_inputs(
            self.execution.tensors,
            self.workspace,
            quantize_input=self.execution.quantize_input,
        )
        self.backend.compute(self.workspace, (l1, l2), output=self.execution.output)
