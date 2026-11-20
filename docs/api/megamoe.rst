.. _apimegamoe:

flashinfer.megamoe
==================

.. currentmodule:: flashinfer.megamoe

SM90 W4A8 expert-parallel MegaMoE with NVFP4 fused/split and MXFP4 fused kernels.
All paths use FP8 E4M3 activations and return BF16 outputs. Execution requires
NVLink-connected Hopper GPUs within each node and a symmetric buffer allocated
collectively before CUDA graph capture. NVFP4 supports four or eight ranks on
one node. MXFP4 supports power-of-two rank counts from one to 32, including
multiple nodes through :class:`MegaMoENcclTransport`. The inter-node path needs
NCCL device API headers and a GIN-capable NCCL installation; set ``NCCL_HOME``
when these are not provided by the ``nvidia-nccl`` wheel.

Buffer and execution
--------------------

.. autosummary::
    :toctree: ../generated

    MegaMoESymmBuffer
    MegaMoENcclTransport
    nvfp4_mega_moe
    mxfp4_mega_moe

Replica plan (MXFP4)
--------------------

Redundant expert slots per rank, the plan tables that route each logical
expert's token blocks onto its copies, and the weight copy of the slots whose
expert changed, either as a separate kernel or inside the MXFP4 fused kernel.
Replica planning and fused weight copies currently require one node and the
default MXFP4 weight storage.

.. autosummary::
    :toctree: ../generated

    MegaMoEExpertSlots

Weight preparation
------------------

.. autosummary::
    :toctree: ../generated

    transform_nvfp4_weights_for_mega_moe
    transform_mxfp4_weights_for_mega_moe

EP8 and EP16 reproduction
-------------------------

The ``flash`` preset is a generated routed-MoE layer fixture with the following
structure. It does not load a checkpoint or represent a complete model.
Its routed-layer dimensions match
`DeepSeek-V4-Flash-0731 config.json <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json>`_.
That checkpoint declares 43 hidden layers, 1 shared expert, 3 hash layers and
1 next-token prediction layer. It uses 64 attention heads, 1 key/value head
and a maximum context length of 1,048,576; these model components are outside
this benchmark. Its router uses ``sqrtsoftplus``, normalized top-k probabilities
and a scaling factor of 1.5; the generated fixture instead supplies the routing
indices and coefficients described below.

.. list-table:: Routed layer geometry
    :header-rows: 1
    :widths: 40 60

    * - Parameter
      - Value
    * - Input / output per rank
      - [M, 4096] / [M, 4096]
    * - Hidden width (H) / expert intermediate width (I)
      - 4096 / 2048
    * - Routed experts (E) / selected experts per token
      - 256 / 6
    * - FC1 weight per expert, concatenated gate and up projections
      - [2I, H] = [4096, 4096]
    * - Expert activation / FC2 weight per expert
      - Clamped SwiGLU (default clamp 10) / [H, I] = [4096, 2048]
    * - Weight values / activations / output
      - MXFP4 / FP8 E4M3 with FP32 scales per 128 values / BF16
    * - Experts per rank at EP8 / EP16
      - 32 / 16
    * - Deployment
      - 1 x 8 H20 / 2 x 8 H20, with NVLink within each node

The timed call includes token dispatch, both expert matrix multiplications,
SwiGLU, intermediate FP8 quantization, routing-weight application and output
combine, including inter-node transfers at EP16. Router evaluation, shared
experts, attention, normalization, residual operations, input quantization,
weight preparation, JIT compilation and graph capture are outside this interval.
The default benchmark also excludes replica planning; the separate
``--redundant-slots`` arm changes that scope. A routed-core result is not a
full-model latency or an end-to-end serving result.

Fixture and storage
~~~~~~~~~~~~~~~~~~~

The default ``--routing balanced --seed 7919`` fixture generates FP4 weights by
global expert ID and inputs by global token ID. For global token ``t`` and slot
``j``, balanced routing sets ``q = (6*t + j) % 256`` and
``expert = (q % 16)*16 + q//16``. It selects six distinct experts per token;
global expert assignment counts differ by at most one. Routing coefficients
are generated softmax weights, not the output of a learned router. Inputs use
FP8 E4M3 values with scale 1/128 per 128 elements; generated MXFP4 weight scale
bytes are 122 through 124. This is a synthetic balanced fixture, not the
historical R405 input or a measured customer routing trace.

Equal-total-token runs with the same seed therefore use the same global input
rows, routes, coefficients and source weights across EP partitions. Equal
per-rank-token runs have different total workloads. JSON records the input and
source-weight identities and every rank sample; preserve those records when
comparing runs. ``--routing random`` or ``--routing-trace`` selects a different
workload and must be reported separately.

.. list-table:: MXFP4 storage and selectable per-rank token counts
    :header-rows: 1
    :widths: 24 14 30 32

    * - ``--weight-storage``
      - Bytes per N128/K128 tile
      - EP8 M
      - EP16 M
    * - ``mxfp4`` (default)
      - 8704
      - General canonical selector
      - General canonical selector
    * - ``mxfp4_compact``
      - 8448
      - 1, 2, 4, 8
      - 1, 4, 8
    * - ``fp8``
      - 16384
      - 1, 2, 4, 8, 16, 32, 64, 128
      - 1, 4, 8, 16, 32, 64

All three layouts represent the same source MXFP4 values; ``fp8`` predecodes
those values and uses more weight memory. Compact and predecoded storage require
CUDA weight preparation and H20 with E256/H4096/I2048/top-6. Unsupported
shape/format plans raise an error rather than changing storage silently.
Compact M16 and larger are excluded; use canonical or predecoded storage for
those batches. These are API selection limits, not a claim that every listed
configuration has the same numerical or performance qualification. Replica
slots require single-node canonical storage.

Token counts and capacity
~~~~~~~~~~~~~~~~~~~~~~~~~

Let R be the EP rank count, M the tokens per rank, G = R*M the total tokens, and
T the measured latency. Aggregate throughput is G/T and throughput per GPU is
M/T. Write T8 and T16 for the EP8 and EP16 latencies measured with the same
statistic, storage, fixture and capacity policy.

.. list-table:: EP16 throughput relative to EP8
    :header-rows: 1
    :widths: 38 31 31

    * - Comparison
      - Aggregate throughput speedup
      - Per-GPU throughput speedup
    * - Same M per rank (EP16 processes twice as many total tokens)
      - 2*T8/T16
      - T8/T16
    * - Same G total (EP8 M = G/8, EP16 M = G/16)
      - T8/T16
      - T8/(2*T16)

For the six equal-total-token points, use G = 16, 64, 128, 256, 512, 1024.
They correspond to EP8 M = 2, 8, 16, 32, 64, 128 and EP16
M = 1, 4, 8, 16, 32, 64. ``--token-scope total`` requires exact division by EP;
it does not round or pad the logical token count.

The selected-capacity configuration uses the following per-rank allocations:

.. list-table:: M and ``--capacity``
    :header-rows: 1
    :widths: 50 50

    * - M
      - Capacity
    * - 1 or 2
      - 2
    * - 4
      - 4
    * - 8
      - 8
    * - 16, 32 or 64
      - 64
    * - 128 (EP8 canonical or predecoded)
      - 128

Capacity controls workspace allocation and JIT specialization, not the number
of logical tokens processed. The benchmark accepts one capacity per invocation
and does not select this table automatically; its default is 384. Run each point
separately to retain the selected capacities. A multi-point invocation needs a
capacity at least as large as its largest M and is a different configuration.

Commands and validation
~~~~~~~~~~~~~~~~~~~~~~~

The following commands use canonical storage and the selected capacities. On
one host with eight H20 GPUs::

    for mc in 1:2 4:4 8:8 16:64 32:64 64:64; do
      m=${mc%:*}; capacity=${mc#*:}
      torchrun --standalone --nproc-per-node=8 benchmarks/bench_megamoe_fp4_sm90.py \
        --shapes flash --tokens "$m" --capacity "$capacity" \
        --routing balanced --seed 7919 --repeats 2000 --warmup-graphs 10 --rounds 5 \
        --output "ep8-m${m}.json" --csv "ep8-m${m}.csv"
    done

On each of two hosts, set the same ``MASTER_ADDR`` and ``MASTER_PORT`` and a
different ``NODE_RANK`` (0 or 1). Execute the same loop in the same order::

    for mc in 1:2 4:4 8:8 16:64 32:64 64:64; do
      m=${mc%:*}; capacity=${mc#*:}
      torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
        --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
        benchmarks/bench_megamoe_fp4_sm90.py --shapes flash \
        --tokens "$m" --capacity "$capacity" --routing balanced --seed 7919 \
        --repeats 2000 --warmup-graphs 10 --rounds 5 \
        --output "ep16-m${m}.json" --csv "ep16-m${m}.csv"
    done

For equal total tokens, pass ``--token-scope total --tokens G`` for each total
listed above and choose capacity from its resulting M. To select another
storage, add ``--weight-storage`` explicitly and restrict M to its supported
set. Specify customer layer dimensions with
``--hidden H --intermediate I --experts E --topk K`` instead of ``--shapes flash``;
prepared formats retain the fixed geometry requirements above.

Each CUDA-event sample spans ``--repeats`` complete routed-core calls in a graph
and is divided by that count. ``--warmup-graphs`` sets the number of graph replays
before sampling. ``median_us`` in JSON (``median_max_rank_us`` in
CSV) is the median of the per-sample maximum rank latency.
``max_rank_median_us`` is the maximum of the rank-wise medians; these statistics
are not interchangeable. Input staging and the graph's initial collective gate
precede the measured events.

Run the independent dense-reference and repeated CUDA Graph checks with the
same storage, capacity, seed and ``torchrun`` topology before timing, for example::

    torchrun --standalone --nproc-per-node=8 \
      tests/moe/test_megamoe_mxfp4_transport_sm90.py \
      --hidden 4096 --intermediate 2048 --experts 256 --topk 6 \
      --tokens 8 --capacity 8 --seed 7919 --weight-storage mxfp4

For EP16, use the two-node launcher above. Cross-node tests use NCCL windows
and skip the single-node transport comparison. The benchmark's finite-output
check alone does not replace dense-reference or replay validation.
