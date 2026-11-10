Kimi K3-family MegaMoE
=====================

``Sm100_KimiK3_MegaMoeConfig`` selects an isolated Kimi K3-family provider through
:class:`flashinfer.moe_ep.MoEEpMegaLayer`. It supports SM100/SM103, 896 experts,
routed hidden size 3584, intermediate size 3072, top-16 routing, and EP sizes 1, 8,
or 16. Each rank owns a contiguous range of experts.

All EP ranks must share one NVLink domain. Across GB300 nodes, the container
must expose the assigned IMEX channel so PyTorch can map peer memory through
CUDA fabric handles. Two separate eight-GPU NVLink domains do not meet this
requirement.

The provider accepts NVFP4 weights and activations (``quant_kind="nvfp4"``)
or MXFP4 weights with MXFP8 activations (``quant_kind="mxfp4_mxfp8"``).
Both paths produce BF16 output. ``combine_dtype`` selects BF16 or MXFP8
communication. SiTU and late routing weights are required.

Kimi K3 and K3.1 Pro
-------------------

The published `Kimi K3 config <https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json>`_
and `Kimi K3.1 Pro config <https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/config.json>`_
share this routed MoE contract. Both use the same config class and selector;
there is no model-name switch. Map fields from each checkpoint's
``text_config`` and retain its own weights, routes and calibration.

.. list-table:: Checkpoint fields used by the routed kernel
   :header-rows: 1

   * - Checkpoint field
     - API field
     - Both models
   * - ``num_experts``
     - ``FleetParams.num_experts``
     - 896
   * - ``routed_expert_hidden_size``
     - ``FleetParams.token_hidden_size``
     - 3584
   * - ``moe_intermediate_size``
     - ``intermediate_size``
     - 3072
   * - ``num_experts_per_token``
     - ``top_k``
     - 16
   * - ``hidden_act``
     - ``activation``
     - ``situ``
   * - ``activation_situ_beta`` / ``activation_situ_linear_beta``
     - ``situ_beta`` / ``situ_linear_beta``
     - 4.0 / 25.0

Use ``routed_expert_hidden_size``, not the full model ``hidden_size=7168``.
The API validates the routed dimensions and rejects a 7168-wide fleet.
The framework still owns routing, the 7168-to-3584 input projection, the
RMSNorm after routed MoE, the 3584-to-7168 output projection, and the two
shared experts. ``topk_weights`` must already include the model's selected-score
normalization and routed scaling factor. Apply these weights only once,
after expert computation, as the kernel does by default.

The following application helper uses the existing API. ``bootstrap``
identifies the EP group and local CUDA device. ``local_weights`` is a canonical
:class:`flashinfer.moe_ep.MoEWeightPack` for this rank's routed experts.
Checkpoint loading and packing remain the caller's work.

.. code-block:: python

   from flashinfer import moe_ep

   def make_routed_layer(text, local_weights, bootstrap, capacity,
                         *, quant_kind="mxfp4_mxfp8", **calibration):
       config = moe_ep.Sm100_KimiK3_MegaMoeConfig(
           intermediate_size=text["moe_intermediate_size"],
           top_k=text["num_experts_per_token"],
           activation=text["hidden_act"],
           situ_beta=text["activation_situ_beta"],
           situ_linear_beta=text["activation_situ_linear_beta"],
           quant_kind=quant_kind,
           combine_dtype="bf16",
           **calibration,
       )
       return moe_ep.MoEEpMegaLayer(
           bootstrap=bootstrap,
           fleet_params=moe_ep.FleetParams(
               num_experts=text["num_experts"],
               token_hidden_size=text["routed_expert_hidden_size"],
               max_tokens_per_rank=capacity,
           ),
           weights=local_weights,
           backend=moe_ep.MegaConfig(megakernel=config, quantize_input=True),
       )

Both published checkpoints use MXFP4 routed weights; this example selects
MXFP8 activations. The two model integrations differ only in the checkpoint
config and weights they supply:

.. code-block:: python

   # Kimi K3
   k3_layer = make_routed_layer(
       k3_config["text_config"], k3_local_weights, bootstrap, capacity=256,
   )

   # Kimi K3.1 Pro
   k31_layer = make_routed_layer(
       k31_config["text_config"], k31_local_weights, bootstrap, capacity=256,
   )

For either layer, pass the projected BF16 routed input and the model's routes:

.. code-block:: python

   layer = k31_layer  # Use k3_layer for Kimi K3.
   workspace = layer.create_workspace(256)
   tensors = moe_ep.MoEEpTensors(routed_hidden_states, topk_ids, topk_weights)
   layer.warmup(tensors, workspace=workspace)
   routed_output = layer(tensors, workspace=workspace)  # [tokens, 3584]

For an NVFP4-converted checkpoint, set ``quant_kind="nvfp4"`` and pass its
``input_norm_const``, ``fc1_alpha``, ``fc2_alpha`` and ``fc1_norm_const`` as
calibration arguments. Do not reuse another checkpoint's calibration or
interpret MXFP4 packed bytes as NVFP4.

The shared contract is covered by CPU admission tests. The GPU tests below
validate the routed kernel; they are not an end-to-end K3.1 Pro checkpoint
accuracy or serving benchmark.

``tests/moe_ep/test_sm100_kimi_k31.py`` is a runnable K3.1 Pro usage example.
It maps the checkpoint fields into the public API and checks eager execution
and CUDA graph replay with sparse weights and normalized route scores:

.. code-block:: bash

   python -m pytest --confcutdir=tests/moe_ep -q tests/moe_ep/test_sm100_kimi_k31.py
   torchrun --standalone --nproc_per_node=8 -m pytest --confcutdir=tests/moe_ep -q tests/moe_ep/test_sm100_kimi_k31.py

For EP16, use the multi-node launch below with this test file.

Weights and scales
------------------

Supply a :class:`flashinfer.moe_ep.MoEWeightPack` in canonical
order: all FC1 gate rows, then all FC1 up rows. Let ``E`` be ``896 / EP``.
For prequantized weights, the packed E2M1 shapes are ``[E, 6144, 1792]`` for ``w13`` and
``[E, 3584, 1536]`` for ``w2``, both ``torch.uint8``. The low nibble holds
the even input element. Preprocessing applies the native gate/up layout
and scale swizzle.

Supply both unswizzled scale planes. NVFP4 uses E4M3 scales per 16 values:
``[E, 6144, 224]`` and ``[E, 3584, 192]``. MXFP4 uses E8M0 scales per 32
values: ``[E, 6144, 112]`` and ``[E, 3584, 96]``. Scales can use their typed
FP8 dtype or raw ``torch.uint8`` bytes. All tensors must be contiguous and
on the current CUDA device.

Unquantized BF16 or FP32 weights are also accepted: ``w13`` has shape
``[E, 6144, 3584]`` and ``w2`` has shape ``[E, 3584, 3072]``. Omit both
scale planes. Preprocessing quantizes each expert before execution.
For NVFP4, omitted FC1/FC2 alpha values use per-expert calibration from the
maximum absolute weight. Explicit alpha values instead set the global
weight scales through the equations below.

For NVFP4, pass the checkpoint calibration through ``input_norm_const``
and FP32 ``[E]`` tensors ``fc1_alpha``, ``fc2_alpha``, and
``fc1_norm_const``. For global quantization scales ``gs_x``, ``gs_w13``,
``gs_w2``, and FC1 activation normalization ``nc``, use:

.. code-block:: python

   input_norm_const = gs_x
   fc1_alpha = 1.0 / (gs_x * gs_w13)
   fc2_alpha = 1.0 / (nc * gs_w2)
   fc1_norm_const = nc

Defaults of one are valid only for weights and inputs calibrated that way.
``MoEEpTensors`` can override the three per-expert NVFP4 tensors for a call.
MXFP4/MXFP8 requires unit alpha and normalization values.

Inputs and activation
---------------------

With ``quantize_input=True``, supply BF16 ``[tokens, 3584]`` hidden states
and omit ``scales``. With ``quantize_input=False``, NVFP4 accepts packed
``uint8 [tokens, 1792]`` plus E4M3 scales ``[tokens, 224]``; MXFP8 accepts
``float8_e4m3fn [tokens, 3584]`` plus E8M0 scales ``[tokens, 112]``.
Input scales use the same typed-or-byte convention as weight scales.
Input quantization follows the pinned Kimi kernels. In NVFP4, a nonzero
block whose scale rounds to zero saturates its nonzero payload values to
signed six; the zero scale makes the decoded block zero. MXFP8 uses the
upstream unit reciprocal for scale code zero. These rules differ from
quantizers that clamp all scales to the smallest normal value.

BF16 input needs 32-byte alignment. Packed input, input scales, expert
IDs, and routing weights need 16-byte alignment.

Routing uses contiguous int32 or int64 ``[tokens, 16]`` expert IDs and
FP32 weights with the same shape. Active IDs must be distinct within each
token and have finite weights. Valid IDs are 0 through 895. An ID of
``-1`` masks a route; its weight is ignored, including NaN. Staging also
masks out-of-range IDs; it does not validate their values on the host.
Each rank may have a different
live token count, including zero, within the shared workspace capacity.

The SiTU gate is ``beta * tanh(gate / beta) * sigmoid(gate)``. The up branch
is ``linear_beta * tanh(up / linear_beta)``. Set both positive, finite
values from the model; neither has an implicit default. ``fast_situ=True`` enables the
source's faster arithmetic and can change output bits.

Workspace and graph lifetime
----------------------------

Create and destroy workspaces collectively on all EP ranks in the same
order. Capacity must be positive and satisfy
``capacity * 16 * 3584 < 2**31``. ``knobs=None`` uses the native
heuristic; a dictionary pins native settings. Online tuning,
workspace pooling, and live-token tuning profiles are not supported.
MXFP4/MXFP8 uses asynchronous epilogue flags; both ``epi_flag_batches``
values must be in ``[1, 4]``.

Warm each input shape before capture on every rank. CUDA graphs keep the
token shape fixed; update values in the captured input tensors before
replay. To change the live shape, use eager execution or capture another
graph. Keep tensor devices and dtypes fixed for each workspace.
``num_valid_tokens_tensor`` is not supported.

By default, each call returns an owned output. Set ``MoEEpTensors.output``
to a contiguous, 16-byte-aligned BF16 ``[tokens, 3584]`` CUDA tensor to reuse caller storage.
``return_workspace_view=True`` borrows workspace output until the next
call on that workspace. It cannot be combined with caller output.

Synchronize CUDA work and release all graphs and borrowed views before
``workspace.destroy()`` or ``layer.destroy()``. Handles are independent;
destroying one does not release another. Destroying the layer releases
all its remaining handles.

Only ``FinalizeMode.DEFAULT`` and the inference kernel are supported.
In-kernel FC2 reduction, early routing weights, per-token FC1 scales,
persistent finalization, and unfinalized output are rejected.

Validation and timing
---------------------

The original checks used B300 at EP1/EP8 and GB300 at EP16 (four nodes with four
GPUs each in one NVLink domain), with PyTorch
``2.14.0a0+b2c75dd062.nv26.09`` and CuTe DSL ``4.8.0``.
All four quantization/combine pairs passed the sparse FP64 layout oracle,
masked and empty inputs, changing eager counts, prequantized input, and
CUDA graph replay. The NVFP4/BF16 case also checks fast SiTU. Separate
tests compare exact quantizer bytes at zero, tiny values and midpoint ties,
and check masked reducer slots filled with NaNs. The sparse oracle checks
routing, gate/up order and scale layouts; it does not measure dense MMA
accumulation error.
The original seven distributed tests passed on every EP8 and EP16 rank, including
alternating eager calls and graph replays with the current standard provider
and separate caller buffers. The EP16 preflight also verified 16 distinct
GPU UUIDs, one fabric cluster/clique, and direct reads and writes to every peer.

The expanded eight-case suite in ``test_sm100_kimi_k3.py`` passed on B300
at EP1 and B200 at EP8. The added staged-output and flag-batch checks
have not been rerun at EP16.

Run the tests from the repository root:

.. code-block:: bash

   python -m pytest --confcutdir=tests/cpu/moe_ep -q tests/cpu/moe_ep/test_kimi_k3_api.py tests/cpu/moe_ep/test_kimi_k3_packaging.py
   python -m pytest --confcutdir=tests/moe_ep -q tests/moe_ep/test_kimi_k3_quantization.py
   torchrun --standalone --nproc_per_node=8 -m pytest --confcutdir=tests/moe_ep -q tests/moe_ep/test_sm100_kimi_k3.py

For EP16 on four GB300 nodes with four GPUs each, launch the following once
per node. Set ``NODE_RANK`` to 0 through 3 and use the same reachable
``MASTER_ADDR`` and ``MASTER_PORT`` on all four nodes:

.. code-block:: bash

   export NVSHMEM_DISABLE_NVLS=1 NVSHMEM_MAX_TEAMS=512
   torchrun --nnodes=4 --nproc_per_node=4 --node_rank="$NODE_RANK" \
       --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
       -m pytest --confcutdir=tests/moe_ep -q \
       tests/moe_ep/test_sm100_kimi_k3.py \
       tests/moe_ep/test_sm100_kimi_k3_coexist.py

The EP16 coexistence run used NVSHMEM ``3.7.2`` with
``NVSHMEM_DISABLE_NVLS=1`` and ``NVSHMEM_MAX_TEAMS=512`` on every rank.
Its default NVLS setup exhausted internal teams; increasing the team limit
alone did not complete setup during the retry. Disabling optional NVLS
collectives allowed the standard provider's direct peer-memory path to run.
Kimi alone does not use NVSHMEM and passed without these settings.

Set ``FLASHINFER_KIMI_TEST_WHEEL`` to a built wheel to enable the two clean
installed-package import tests. The pinned export inputs and local changes
are listed in the drop's ``VENDOR.md``.

The public API benchmark includes staging, BF16 input quantization, compute
and combine into caller output. The following warm-graph medians are in
milliseconds, measured with deterministic sparse weights,
16 valid routes per token and ``fast_situ=False``. Each result uses 30 samples
of ten graph replays; each sample uses the slowest EP rank. These adapter
measurements are separate from the earlier source-repository measurements.

.. list-table:: B300 EP8 warm-graph forward time (ms)
   :header-rows: 1

   * - Quantization
     - Combine
     - 256 tokens/rank
     - 2048 tokens/rank
     - 8192 tokens/rank
   * - NVFP4
     - BF16
     - 0.433
     - 0.702
     - 2.097
   * - NVFP4
     - MXFP8
     - 0.444
     - 0.651
     - 1.821
   * - MXFP4/MXFP8
     - BF16
     - 0.458
     - 0.991
     - 3.297
   * - MXFP4/MXFP8
     - MXFP8
     - 0.449
     - 0.994
     - 3.271

Three paired EP8 NVFP4/BF16 runs at 8192 tokens/rank measured current/Kimi
medians of ``2.564/2.105``, ``2.558/2.111`` and ``2.560/2.098`` ms.
The median paired speedup was ``1.218x``. The current standard selector
used the fixed tile ``(256, 128, 256)``; its default 8192-token tile failed
its own admission check. Both providers used the same supplied routes,
calibration, sparse weights, SiTU constants and default finalization, with
fast SiTU disabled. This comparison uses fixed knobs, without autotuning.

EP16 used 16 GB300 GPUs across four nodes in one NVLink domain, with the
same benchmark settings. Each case also checked the first eight output rows
on every rank against the sparse FP64 oracle. No current-provider timing
comparison was run at EP16.

.. list-table:: GB300 EP16 warm-graph forward time (ms)
   :header-rows: 1

   * - Quantization
     - Combine
     - 256 tokens/rank
     - 2048 tokens/rank
     - 8192 tokens/rank
   * - NVFP4
     - BF16
     - 0.232
     - 0.609
     - 2.102
   * - NVFP4
     - MXFP8
     - 0.241
     - 0.500
     - 1.627
   * - MXFP4/MXFP8
     - BF16
     - 0.249
     - 0.793
     - 2.743
   * - MXFP4/MXFP8
     - MXFP8
     - 0.244
     - 0.737
     - 2.614

The benchmark writes all samples, including each rank's samples, to JSON.

.. code-block:: bash

   torchrun --standalone --nproc_per_node=8 \
       -m benchmarks.bench_kimi_k3_moe_ep --output kimi-public-times.json

   torchrun --standalone --nproc_per_node=8 \
       -m benchmarks.bench_kimi_k3_moe_ep --provider current \
       --current-tile 256 128 256 --tokens 8192 --output current-public-times.json

.. autoclass:: flashinfer.moe_ep.Sm100_KimiK3_MegaMoeConfig
   :members:
