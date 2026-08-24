.. _apikda_backward:

flashinfer.kda_backward
=======================

Frozen recurrent Kimi Delta Attention (KDA) training backward for SM100a and
SM103a.
The operation reconstructs the forward recurrence and returns gradients for
Q, K, V, raw gate, raw beta, ``A_log``, ``dt_bias``, and initial state.

.. currentmodule:: flashinfer.kda_backward

.. autosummary::
    :toctree: ../generated

    recurrent_kda_backward
    RecurrentKDABackwardWorkspace

Supported contract
------------------

The implementation requires an NVIDIA compute-capability 10.0 or 10.3 GPU.
SM100a requires CUDA 12.8 or newer and SM103a requires CUDA 12.9 or newer.
Token tensors and output adjoint are contiguous BF16,
parameters and state tensors are contiguous FP32, and both key and value
dimensions are 128. State tensors use value-first ``[N,H,V,K]`` layout. Q and
K use L2 normalization with epsilon ``1e-6``. The decay and beta
transformations are ``exp(-5 * sigmoid(exp(A_log) * (g + dt_bias)))`` and
``sigmoid(beta)`` respectively. The output scale is fixed at
``1 / sqrt(128)``.

Only these shapes are admitted:

* fixed ``B=1,T=17,H=1``;
* packed sequence lengths ``[17,33,65]``, ``H=4``;
* fixed ``B=1,T=17,H=16``;
* fixed ``B=1,T=1024,H=4``;
* fixed ``B=1,T=4096,H=32``;
* fixed ``B=1,T=8192,H=96``;
* packed sequence lengths ``[1300,547,2048,963,271,3063]``, ``H=96``; and
* eight packed sequences of length 1024, ``H=96``.

Packed offsets must be a contiguous CUDA int64 tensor with the exact values
for the selected shape. The eager warm call synchronizes once to validate
those values; the launch path then consumes the CUDA tensor directly.

Workspace and CUDA Graphs
-------------------------

Eager calls without an explicit workspace use stream-local reusable scratch.
For CUDA Graph capture, construct one
``RecurrentKDABackwardWorkspace(device)`` per captured invocation and provide
eight preallocated tensors through ``out=``. Eagerly invoke the same call on
the intended capture stream with the exact input and output tensors, then
synchronize that stream before capture. This allocates every intermediate and
prepares the route's TMA descriptors. Capture accepts only that exact
pointer, shape, stride, dtype, scale, and lower-bound signature and performs
no allocation or descriptor preparation.

The aligned, uniform eight-by-1024 shape uses a persistent C16 forward
checkpoint kernel, a persistent C16 reverse kernel, and two parameter-gradient
reduction kernels. The other documented shapes retain the C32 or low-head
routes.

The low-head route stores an FP32 checkpoint tensor with shape
``[T,H,128,128]``. Its largest supported configuration, fixed
``T=1024,H=4``, occupies 256 MiB (about 268 MB) per workspace. An explicit
workspace used by a captured graph retains that storage for the workspace and
graph lifetime.

The workspace binds to its first stream and must outlive the graph and all
replays. Once a workspace participates in capture it cannot be passed through
Python again. Graph replay does not re-enter Python.
