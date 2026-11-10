"""The sealed kernel: ``run`` receives the Definition inputs by keyword, the
prepared weight under ``state`` and the destination buffer ``output``.

It converts nothing per call: ``prepare.py`` ran once for this weight set and
left the layout the kernel consumes in ``state``. ``weight`` still arrives (it
is a Definition input and the tensor the state is keyed on) but is not read.
"""

import torch


def run(hidden_states, weight, state, output):
    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    output.copy_((x * inv_rms * state["weight_f32"]).to(output.dtype))
