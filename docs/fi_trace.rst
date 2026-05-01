.. _fi_trace:

fi_trace — Operation Schema Extraction
=======================================

``fi_trace`` is FlashInfer's operation schema extraction system.  Every
``@flashinfer_api``-decorated function automatically grows a ``.fi_trace()``
method that captures the *shape*, *dtype*, and *axis structure* of a call as a
portable JSON file — without running the GPU kernel.

These JSON files are the input format for `flashinfer-bench
<https://github.com/flashinfer-ai/flashinfer-bench>`_, the companion benchmark
toolkit.  Collecting them while running your production workload gives you a
precise benchmark suite that reflects your actual model and serving scenario.

Quick Start
-----------

Set two environment variables **before** importing FlashInfer:

.. code-block:: bash

    export FLASHINFER_TRACE_DUMP=1
    export FLASHINFER_TRACE_DUMP_DIR=./fi_trace_out   # default: ./fi_trace_out

    python my_inference_script.py

FlashInfer writes one ``.json`` file per unique (op, shape) combination.
Subsequent calls with the same shapes are deduplicated — no duplicate files.

.. code-block:: text

    fi_trace_out/
    ├── rmsnorm_h7168.json
    ├── gqa_paged_decode_h32_kv8_d128_ps16.json
    ├── moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json
    └── ...

Environment Variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 35 12 20 33

   * - Variable
     - Type
     - Default
     - Description
   * - ``FLASHINFER_TRACE_DUMP``
     - int
     - ``0``
     - Set to ``1`` to enable automatic JSON dumping on every API call.
   * - ``FLASHINFER_TRACE_DUMP_DIR``
     - str
     - ``./fi_trace_out``
     - Directory where JSON files are written.

Both variables are read **lazily at call time**, so they can be set after
``import flashinfer`` (e.g. when using ``python -m``).

JSON File Format
----------------

Each file describes one operation instance.  Here is an annotated example for
``rmsnorm`` with ``hidden_size=7168``:

.. code-block:: json

    {
      "name": "rmsnorm_h7168",
      "description": "Root Mean Square Normalization. Epsilon is fixed at 1e-6.",
      "op_type": "rmsnorm",
      "tags": [
        "fi_api:flashinfer.norm.rmsnorm",
        "status:verified"
      ],
      "axes": {
        "batch_size": { "type": "var" },
        "hidden_size": { "type": "const", "value": 7168 }
      },
      "inputs": {
        "hidden_states": { "shape": ["batch_size", "hidden_size"], "dtype": "bfloat16" },
        "weight":        { "shape": ["hidden_size"],               "dtype": "bfloat16" }
      },
      "outputs": {
        "output": { "shape": ["batch_size", "hidden_size"], "dtype": "bfloat16" }
      },
      "reference": "..."
    }

Key fields:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``name``
     - Auto-generated from ``op_type`` / ``name_prefix`` + const-axis values.
       Becomes the benchmark name in flashinfer-bench.
   * - ``op_type``
     - Identifies the kernel class (``rmsnorm``, ``gqa_paged``, ``moe``, …).
   * - ``tags``
     - List of key:value tags.  Always includes ``fi_api:<qualified.name>``
       and optional metadata like ``status:verified``.
   * - ``axes``
     - Symbolic dimensions.  ``"var"`` axes vary at runtime (batch size,
       sequence length).  ``"const"`` axes are fixed by model config (head
       dimension, hidden size) and carry a ``"value"``.
   * - ``inputs`` / ``outputs``
     - Each entry has ``"shape"`` (list of axis names) and a resolved
       ``"dtype"``.  Optional inputs carry ``"optional": true``.
   * - ``reference``
     - Source of a pure-PyTorch reference implementation for correctness
       checking (present on ``status:verified`` ops).

Calling ``.fi_trace()`` Directly
---------------------------------

Every decorated function exposes a ``.fi_trace()`` method.
You can call it without running the kernel:

.. code-block:: python

    import torch
    import flashinfer

    q = torch.zeros(32, 32, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(64, 16, 8, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(64, 16, 8, 128, dtype=torch.bfloat16, device="cuda")

    schema = flashinfer.norm.rmsnorm.fi_trace(
        hidden_states=torch.zeros(32, 7168, dtype=torch.bfloat16),
        weight=torch.ones(7168, dtype=torch.bfloat16),
    )
    print(schema["name"])   # rmsnorm_h7168
    print(schema["axes"])   # {'batch_size': {'type': 'var'}, 'hidden_size': {'type': 'const', 'value': 7168}}

To write to a specific directory, pass ``save_dir``:

.. code-block:: python

    schema = flashinfer.norm.rmsnorm.fi_trace(
        hidden_states=...,
        weight=...,
        save_dir="./my_traces",
    )

Covered Operations
------------------

The following FlashInfer operations have trace templates and will emit JSON
files when ``FLASHINFER_TRACE_DUMP=1``:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Module
     - Operation
     - ``op_type``
   * - ``flashinfer.norm``
     - ``rmsnorm``, ``fused_add_rmsnorm``
     - ``rmsnorm``
   * - ``flashinfer.sampling``
     - ``top_k_sampling_from_probs``,
       ``top_p_sampling_from_probs``,
       ``top_k_top_p_sampling_from_probs``
     - ``sampling``
   * - ``flashinfer.gemm``
     - ``mm_bf16``, ``mm_fp8``, ``mm_mxfp8``, ``mm_fp4``
     - ``gemm_bf16`` / ``gemm_fp8`` / ``gemm_mxfp8`` / ``gemm_fp4``
   * - ``flashinfer.decode``
     - ``BatchDecodeWithPagedKVCacheWrapper.run``
     - ``gqa_paged``
   * - ``flashinfer.prefill``
     - ``BatchPrefillWithPagedKVCacheWrapper.run``,
       ``BatchPrefillWithRaggedKVCacheWrapper.run``
     - ``gqa_paged`` / ``gqa_ragged``
   * - ``flashinfer.mla``
     - ``BatchMLAPagedAttentionWrapper.run``
     - ``mla_paged``
   * - ``flashinfer.gdn_decode``
     - ``gated_delta_rule_decode``, ``gated_delta_rule_mtp``
     - ``gdn``
   * - ``flashinfer.gdn_prefill``
     - ``chunk_gated_delta_rule``
     - ``gdn``
   * - ``flashinfer.fused_moe``
     - ``trtllm_fp8_block_scale_moe`` (6 routing types)
     - ``moe``
   * - ``flashinfer.fused_moe``
     - ``trtllm_fp4_block_scale_moe`` (6 routing types)
     - ``moe``

MoE Routing Types
-----------------

MoE operations dispatch to per-routing-type templates.  The output filename
encodes the routing method:

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Value
     - Name
     - Filename pattern (FP8 example)
   * - 0
     - Default (Softmax → TopK)
     - ``moe_fp8_block_scale_default_routing_topk8_e32_h7168_i2048.json``
   * - 1
     - Renormalize (TopK → Softmax)
     - ``moe_fp8_block_scale_renormalize_routing_topk8_e32_h7168_i2048.json``
   * - 2
     - DeepSeekV3 (Sigmoid + group selection)
     - ``moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json``
   * - 3
     - Llama4 (Top1 → Sigmoid)
     - ``moe_fp8_block_scale_llama4_routing_topk1_e32_h7168_i2048.json``
   * - 4
     - RenormalizeNaive (Softmax → TopK → Renormalize)
     - ``moe_fp8_block_scale_renormalize_naive_routing_topk8_e32_h7168_i2048.json``
   * - 5
     - TopK (no normalisation)
     - ``moe_fp8_block_scale_topk_routing_topk8_e32_h7168_i2048.json``

Example: Collecting Traces from a Real Workload
------------------------------------------------

The script below runs a representative set of FlashInfer ops and collects all
trace JSON files in one pass.  It covers the shapes used in DeepSeek-V3-style
models with expert-parallel MoE serving.

.. code-block:: bash

    python tests/trace/example.py

The generated files can be passed directly to ``flashinfer-bench``:

.. code-block:: bash

    flashinfer-bench --trace-dir fi_trace_out/ --backends fa2 cudnn cutlass

Adding Trace Support to a New Kernel
--------------------------------------

When adding a new kernel (see ``CLAUDE.md`` and ``.claude/skills/add-cuda-kernel/SKILL.md``
for the full tutorial), attach a ``TraceTemplate`` to the ``@flashinfer_api`` decorator:

.. code-block:: python

    from flashinfer.trace.template import Const, Tensor, TraceTemplate, Var
    from flashinfer.api_logging import flashinfer_api

    rmsnorm_trace = TraceTemplate(
        op_type="rmsnorm",
        name_prefix="rmsnorm",
        description="Root Mean Square Normalization.",
        axes={
            "batch_size":  Var(),
            "hidden_size": Const(abbrev="h"),
        },
        inputs={
            "hidden_states": Tensor(["batch_size", "hidden_size"]),
            "weight":        Tensor(["hidden_size"]),
        },
        outputs={
            "output": Tensor(["batch_size", "hidden_size"], dtype_from="hidden_states"),
        },
        tags=["status:verified"],
    )

    @flashinfer_api(trace=rmsnorm_trace)
    def rmsnorm(hidden_states, weight, eps=1e-6):
        ...

The template is registered automatically in ``_TRACE_REGISTRY`` at decoration
time and picked up by the consistency tests without any manual registration.

For operations whose template depends on a runtime parameter (e.g.
``routing_method_type`` for MoE), write a dispatch callable and attach a
``.templates`` attribute so the registry discovers all variants:

.. code-block:: python

    _TEMPLATES = {0: default_trace, 1: renorm_trace, ...}

    def my_dispatch(**kwargs):
        return _TEMPLATES.get(int(kwargs.get("routing_method_type", 0)))

    my_dispatch.templates = list(_TEMPLATES.values())

    @flashinfer_api(trace=my_dispatch)
    def my_moe_op(...):
        ...

Consistency Tests
-----------------

FlashInfer ships automated **linter-style tests** that validate every trace
template without running GPU kernels:

.. code-block:: bash

    pytest tests/trace/test_fi_trace_template_consistency.py -v

The tests check three properties for every registered template:

1. **Signature consistency** — every ``param=`` reference in the template
   matches a real parameter of the decorated function.
2. **Axes coverage** — every ``Const`` axis can be resolved from at least one
   tensor's shape or from a scalar kwarg.
3. **End-to-end completeness** — calling ``.fi_trace()`` with auto-generated
   minimal tensors returns a dict where all ``Const`` axes have values and
   no input/output has ``dtype == "unknown"``.

When you add a template, these tests run automatically with no manual
registration required.
