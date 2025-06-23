.. _apilogitsprocessor:

flashinfer.logits_processor
===========================

A declarative, pluggable framework for building processing pipelines for LLM outputs.

.. currentmodule:: flashinfer.logits_processor

Pipeline Construction
---------------------

Use :class:`LogitsPipe` to create processing pipelines:

.. code-block:: python

    import torch
    from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopP, Sample

    # Create a pipeline
    pipe = LogitsPipe([
        Temperature(),      # Scale logits by temperature
        Softmax(),          # Convert logits to probabilities
        TopP(),             # Apply top-p filtering
        Sample()            # Sample from the distribution
    ])

    # Apply the pipeline
    batch_size = 4
    vocab_size = 5
    logits = torch.randn(batch_size, vocab_size, device="cuda")
    output_ids = pipe(logits, temperature=0.7, top_p=0.9)

Pipeline
--------

.. autosummary::
    :toctree: ../generated

    LogitsPipe

Processors
----------

.. autosummary::
    :toctree: ../generated

    LogitsProcessor
    Temperature
    Softmax
    TopK
    TopP
    MinP
    Sample

Types
-----

.. autosummary::
    :toctree: ../generated

    TensorType
    TaggedTensor

Customization Features
-------------

Custom Logits Processor
^^^^^^^^^^^^^^^^^^^^^^^

You can create your own logits processor by subclassing :class:`LogitsProcessor`:

.. code-block:: python

    class CustomLogitsProcessor(LogitsProcessor):

        def __init__(self, **params: Any):
            super().__init__(**params)

        def legalize(self, input_type: TensorType) -> List["Op"]:
            return [CustomOp(**self.params)]

    class CustomOp(Op):
        # Define the input and output tensor types
        IN = TensorType.LOGITS
        OUT = TensorType.LOGITS

        def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
            pass

    pipe = LogitsPipe([CustomLogitsProcessor()])  # The pipe will be compiled into [CustomOp]

Custom Fusion Rules
^^^^^^^^^^^^^^^^^^^

You can register custom fusion rules to optimize specific processor combinations:

.. code-block:: python

    def custom_fusion_guard(window: List[Op]) -> bool:
        # Whether the fusion should be applied
        return True

    def build_custom_fusion(window: List[Op]) -> Op:
        # Create a fused operator by setting the parameters etc.
        return CustomOp()

    custom_rule = FusionRule(
        pattern=(Temperature, Softmax),
        guard=custom_fusion_guard,
        build=build_custom_fusion,
        prio=20
    )

    pipe = LogitsPipe(
        [Temperature(), Softmax(), Sample()],
        custom_fusion_rules=[custom_rule]
    )   # The compiled ops in the pipeline will be [CustomOp, Sample]
