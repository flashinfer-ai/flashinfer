.. _recursive-attention:

Attention States and Recursive Attention 
========================================


FlashInfer introduces the concept of **attention states**, which fully characterizes
the attention between a query and a set of key/value pairs. We further defines a 
**merge** operator on the **attention states**.  This merge operator facilitates the
computation of complete attention by allowing the recursive merging of attention states.

Suppose we define :math:`s_i = \mathbf{q}\mathbf{k}_i^T` as the pre-softmax attention
score between the query :math:`\mathbf{q}` and the key :math:`\mathbf{k}_i`. The Self-Attention
score on index :math:`i` can be generalized to index set :math:`I`:

.. math::

  s(I)=\log\left(\sum_{i\in I}\exp\left(s_i\right)\right)

We can also generalize the value on index :math:`i` to index set :math:`I`:

.. math::

    \mathbf{v}(I) = \sum_{i\in I}\textrm{softmax}(s_i) \mathbf{v}_i = \frac{\sum_{i\in I}\exp\left(s_i\right)\mathbf{v}_i}{\exp(s(I))}

The :math:`softmax` function is restricted to the index set :math:`I`. Note that :math:`\mathbf{v}(\{1,2,\cdots, n\})` is the self-attention output of the entire sequence. 
The *attention state* of the index set :math:`i` can be defined as a tuple :math:`(s(I), \mathbf{v}(I))`, then we can define a binary **merge** operator :math:`\oplus` of two attention states as ((in practice we will minus $s$ with maximum value to guarantee numerical stability and here we omit them for simplicity):

.. math::

  \begin{bmatrix}\mathbf{v}(I\cup J)\\s(I\cup J)\end{bmatrix}=\begin{bmatrix}\mathbf{v}(I)\\s(I)\end{bmatrix}\oplus\begin{bmatrix}\mathbf{v}(J)\\s(J)\end{bmatrix}=\begin{bmatrix} \frac{\mathbf{v}(I)\exp(s(I)) + \mathbf{v}(J)\exp(s(J))}{\exp(s(I)) + \exp(s(J))} \\  \log(\exp(s(I)) + \exp(s(J))) \end{bmatrix}

the **merge** operator can be generalized to any number of attention state inputs:

.. math::
    \begin{bmatrix}\mathbf{v}(\bigcup_{i=1}^{n}I_i) \\ s(\bigcup_{i=1}^{n}I_i) \end{bmatrix} = \bigoplus_{i=1}^{n}\begin{bmatrix}\mathbf{v}(I_i) \\ s(I_i)\end{bmatrix} = \begin{bmatrix} \sum_{i=1}^{n} \textrm{softmax}(s(I_i))\mathbf{v}(I_i) \\ \log(\sum_{i=1}^{n} \exp (s(I_i))) \end{bmatrix}

The above n-ary merge operator is consistent with the binary merge operator, and we can prove the operator is *communicative* and *associative*. There are different ways to get the attention state of the entire sequence by merging the attention states of index subsets, and the final outcome is mathematically equivalent:

.. image:: https://raw.githubusercontent.com/flashinfer-ai/web-data/main/tutorials/recursive-attention.png
    :width: 600
    :align: center
    :alt: Recurisve Attention

.. note::

  The generalized score :math:`s` is also known as log-sum-exp (``lse`` for short).

Applications
------------

Note that :math:`\oplus` operator is **commutative** and **associative**, which means we can 
safely offload the self-attention computation on a subset of KV to different devices
and **merge** the results **in any order**.

There are several interesting applications of this recursive form of self-attention in FlashInfer so far:

Shared-Prefix Batch Decoding
  Many LLM applications involves batch decoding with the shared long prompt, FlashInfer decomposes attention
  on the entire KV-Cache to shared prefix attention and unique suffixes attention.
  This decomposition enables the offloading of these components to different kernel implementations, resulting
  in a remarkable 30x acceleration in scenarios with long context and large batch-size.  
  Such decomposition accelerates the operator by 30 times in long context setting.
  Check `our blog post <https://flashinfer.ai/2024/01/08/cascade-inference.html>`_ on more details about this application,
  and :ref:`api-cascade-attention` on how to use this feature in FlashInfer.

KV Sequence Parallelism
  For long context LLM inference/serving, the batch size and number of heads per GPU is limited by the GPU memory,
  and the default parallelism strategy cannot use all SMs in GPUs, which results in suboptimal performance.
  Inspired by `Split-K <https://github.com/NVIDIA/cutlass/blob/8825fbf1efebac973d96730892919ab241b755bb/media/docs/efficient_gemm.md#parallelized-reductions>`_ trick
  in GEMM optimizations. FlashInfer partitions the KV sequence dimension and dispatches the attention computations to 
  different thread-blocks and merge them in a second pass. This same idea was also proposed in Flash-Decoding, you can 
  check their great `blog post <https://crfm.stanford.edu/2023/10/12/flashdecoding.html>`_ for visualizations and more details.

Related APIs
------------

FlashInfer exposes several APIs to facilitate the recursive attention computation:

- :ref:`api-merge-states` defines the operators to merge attention states.
- :ref:`apiprefill` and :ref:`apidecode` defines operators that returns attention states (APIs
  with suffix ``_return_lse`` returns both attention output :math:`v` and score :math:`s`). 

