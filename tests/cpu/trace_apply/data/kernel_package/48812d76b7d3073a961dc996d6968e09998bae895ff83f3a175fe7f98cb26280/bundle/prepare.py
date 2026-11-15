"""Weight preparation for the sealed rmsnorm kernel.

``setup(context)`` runs once per weight set, at the first eager call and never
inside CUDA-graph capture, with the live tensors named by ``setup_inputs``
under ``context["setup_inputs"]``. It returns the state ``run`` receives by
keyword: here the weight converted once into the layout the kernel consumes
(contiguous float32), so no call converts it again. ``teardown(state)`` runs
only on an explicit release, never at process exit.
"""

import torch


def setup(context):
    weight = context["setup_inputs"]["weight"]
    return {"weight_f32": weight.to(torch.float32).contiguous()}


def teardown(state):
    state.clear()
