"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from .op import Op
from .types import TensorType


class LogitsProcessor(ABC):
    """
    LogitsProcessor defines high-level transformations that can be applied to
    logits or probabilities. Each processor is automatically
    legalized into low-level :class:`Op` or :class:`ParameterizedOp` that can be type-checked, validated, and
    fused for optimal performance. Users can extend this class to implement their own processors.

    Parameters
    ----------
    **params : Any
        Processor-specific parameters at compile-time.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.logits_processor import LogitsPipe, TopK, Sample, TensorType
    >>> torch.manual_seed(42)
    >>>
    >>> # Create a pipeline that legalizes to a fused op.
    >>> pipe = LogitsPipe([
    ...     TopK(),         # Top-k filtering on logits
    ...     Sample()        # Sample from the filtered distribution
    ... ], input_type=TensorType.PROBS)  # assume the input is probabilities
    >>>
    >>> pipe
    LogitsPipe([TopK -> Sample], ops=[ProbsTopKOp -> ProbsSampleOp], compiled_ops=[FusedProbsTopKSampleOp])

    Notes
    -----
    Subclasses must implement the :meth:`legalize` method to convert the high-level
    processor into one or more low-level operators with specific input/output types
    """

    def __init__(self, **params: Any):
        """
        Initialize the processor.

        Parameters
        ----------
        **params : Any
            Processor-specific parameters at compile-time.
        """
        self.params = params

    @abstractmethod
    def legalize(self, input_type: TensorType) -> List[Op]:
        """
        Legalize the processor into a list of low-level operators.

        Parameters
        ----------
        input_type : TensorType
            The expected input tensor type of the processor.

        Returns
        -------
        List[Op]
            A list of low-level operators.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


class Temperature(LogitsProcessor):
    """
    Temperature scaling processor for logits.

    Scales logits by dividing by a temperature value.

    :attr:`TensorType.LOGITS` -> :attr:`TensorType.LOGITS`

    Parameters
    ----------
    temperature : float or torch.Tensor, Runtime
        Temperature value for scaling. Must be positive. Can be a scalar or per-batch tensor.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.logits_processor import LogitsPipe, Temperature, Sample
    >>> torch.manual_seed(42)
    >>> pipe = LogitsPipe([Temperature()])
    >>> logits = torch.randn(2, 2, device="cuda")
    >>> logits
    tensor([[ 0.1940,  2.1614], [ -0.1721,  0.8491]], device='cuda:0')
    >>> scaled_logits = pipe(logits, temperature=0.8)
    >>> scaled_logits
    tensor([[ 0.2425,  2.7017], [-0.2151,  1.0613]], device='cuda:0')
    """

    def __init__(self, **params: Any):
        """
        Constructor for Temperature processor. No compile-time parameters are needed.
        """
        super().__init__(**params)

    def legalize(self, input_type: TensorType) -> List[Op]:
        """
        Legalize the processor into a list of low-level operators.
        """
        from .operators import TemperatureOp

        if input_type != TensorType.LOGITS:
            raise ValueError(
                f"Temperature can only be applied to LOGITS, got {input_type}"
            )

        return [TemperatureOp(**self.params)]


class Softmax(LogitsProcessor):
    """
    Softmax processor to convert logits to probabilities.

    Applies the softmax function.

    :attr:`TensorType.LOGITS` -> :attr:`TensorType.PROBS`

    Parameters
    ----------
    enable_pdl : bool, optional, Compile-time
        Whether to enable PDL for the kernel implementation.
        Default is True.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.logits_processor import LogitsPipe, Softmax, Sample
    >>> torch.manual_seed(42)
    >>> pipe = LogitsPipe([Softmax()])
    >>> logits = torch.randn(2, 2, device="cuda")
    >>> logits
    tensor([[ 0.1940,  2.1614], [ -0.1721,  0.8491]], device='cuda:0')
    >>> probs = pipe(logits)
    >>> probs
    tensor([[0.1227, 0.8773], [0.2648, 0.7352]], device='cuda:0')

    Notes
    -----
    Can only appear once in a pipeline.
    """

    def __init__(self, enable_pdl: Optional[bool] = None, **params: Any):
        """
        Constructor for Softmax processor.

        Parameters
        ----------
        enable_pdl : bool, optional, Compile-time
            Whether to enable PDL for the kernel implementation.
            Default is None, which means the kernel will be automatically enabled if PDL is supported on the device.
        """
        super().__init__(enable_pdl=enable_pdl, **params)

    def legalize(self, input_type: TensorType) -> List[Op]:
        """
        Legalize the processor into a list of low-level operators.
        """
        from .operators import SoftmaxOp

        if input_type != TensorType.LOGITS:
            raise ValueError(f"Softmax can only be applied to LOGITS, got {input_type}")

        return [SoftmaxOp(**self.params)]


class TopK(LogitsProcessor):
    """
    Top-k filtering processor.

    Keeps only the top-k highest probability tokens and masks out the rest.

    :attr:`TensorType.LOGITS` -> :attr:`TensorType.LOGITS` | :attr:`TensorType.PROBS` -> :attr:`TensorType.PROBS`

    Parameters
    ----------
    joint_topk_topp : bool, optional, Compile-time
        Whether to enable joint top-k and top-p filtering when followed by TopP.
        Default is False.

    top_k : int or torch.Tensor, Runtime
        Number of top tokens to keep. Can be a scalar or per-batch tensor.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.logits_processor import LogitsPipe, TopK, Sample, TensorType
    >>> torch.manual_seed(42)
    >>>
    >>> # Top-k filtering on logits
    >>> pipe = LogitsPipe([TopK()], input_type=TensorType.LOGITS)
    >>> logits = torch.randn(2, 2, device="cuda")
    >>> logits
    tensor([[ 0.1940,  2.1614], [ -0.1721,  0.8491]], device='cuda:0')
    >>> topk_logits = pipe(logits, top_k=1)
    >>> topk_logits
    tensor([[  -inf, 2.1614], [  -inf, 0.8491]], device='cuda:0')
    >>>
    >>> # Top-k filtering on probabilities
    >>> pipe = LogitsPipe([TopK()], input_type=TensorType.PROBS)
    >>> probs = torch.randn(2, 2, device="cuda")
    >>> probs_normed = probs / probs.sum(dim=-1, keepdim=True)
    >>> probs_normed
    tensor([[  4.4998,  -3.4998], [-18.2893,  19.2893]], device='cuda:0')
    >>> topk_probs = pipe(probs_normed, top_k=1)
    >>> topk_probs
    tensor([[1., 0.], [0., 1.]], device='cuda:0')

    Notes
    -----
    When applied to :attr:`TensorType.LOGITS`, sets non-top-k values to -inf.
    When applied to :attr:`TensorType.PROBS`, zeros out non-top-k values and renormalizes.

    See Also
    --------
    :meth:`~flashinfer.sampling.top_k_mask_logits`
    :meth:`~flashinfer.sampling.top_k_renorm_probs`
    """

    def __init__(self, joint_topk_topp: bool = False, **params: Any):
        """
        Constructor for TopK processor.

        Parameters
        ----------
        joint_topk_topp : bool, optional, Compile-time
            Whether to enable joint top-k and top-p filtering when followed by TopP.
            Default is False.
        """
        super().__init__(joint_topk_topp=joint_topk_topp, **params)

    def legalize(self, input_type: TensorType) -> List[Op]:
        """
        Legalize the processor into a list of low-level operators.
        """
        from .operators import LogitsTopKOp, ProbsTopKOp

        if input_type == TensorType.LOGITS:
            return [LogitsTopKOp(**self.params)]
        elif input_type == TensorType.PROBS:
            return [ProbsTopKOp(**self.params)]
        else:
            raise ValueError(f"TopK cannot be applied to {input_type}")


class TopP(LogitsProcessor):
    """
    Top-p (nucleus) filtering processor.

    Keeps tokens with cumulative probability up to threshold p.

    :attr:`TensorType.PROBS` -> :attr:`TensorType.PROBS`

    Parameters
    ----------
    top_p : float or torch.Tensor, Runtime
        Cumulative probability threshold in (0, 1]. Can be a scalar or per-batch tensor.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.logits_processor import LogitsPipe, Softmax, TopP, Sample
    >>> torch.manual_seed(42)
    >>> pipe = LogitsPipe([TopP()])
    >>> probs = torch.randn(2, 2, device="cuda")
    >>> probs_normed = probs / probs.sum(dim=-1, keepdim=True)
    >>> probs_normed
    tensor([[ 0.0824,  0.9176], [-0.2541,  1.2541]], device='cuda:0')
    >>> topp_probs = pipe(probs_normed, top_p=0.9)
    >>> topp_probs
    tensor([[0., 1.], [0., 1.]], device='cuda:0')

    See Also
    --------
    :meth:`~flashinfer.sampling.top_p_renorm_probs`
    """

    def __init__(self, **params: Any):
        """
        Constructor for TopP processor. No compile-time parameters are needed.
        """
        super().__init__(**params)

    def legalize(self, input_type: TensorType) -> List[Op]:
        """
        Legalize the processor into a list of low-level operators.
        """
        from .operators import TopPOp

        if input_type != TensorType.PROBS:
            raise ValueError(f"TopP can only be applied to PROBS, got {input_type}")

        return [TopPOp(**self.params)]


class MinP(LogitsProcessor):
    """
    Min-p filtering processor.

    Keeps tokens with probability at least p times the maximum probability.

    :attr:`TensorType.PROBS` -> :attr:`TensorType.PROBS`

    Parameters
    ----------
    min_p : float or torch.Tensor, Runtime
        Minimum probability threshold as a ratio of max probability.
        Must be in (0, 1]. Can be a scalar or per-batch tensor.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.logits_processor import LogitsPipe, Softmax, MinP, Sample
    >>> torch.manual_seed(42)
    >>> pipe = LogitsPipe([MinP()])
    >>> probs = torch.randn(2, 2, device="cuda")
    >>> probs_normed = probs / probs.sum(dim=-1, keepdim=True)
    >>> probs_normed
    tensor([[ 0.0824,  0.9176], [-0.2541,  1.2541]], device='cuda:0')
    >>> minp_probs = pipe(probs_normed, min_p=0.05)
    >>> minp_probs
    tensor([[0.0824, 0.9176], [0.0000, 1.0000]], device='cuda:0')

    """

    def __init__(self, **params: Any):
        """
        Constructor for MinP processor. No compile-time parameters are needed.
        """
        super().__init__(**params)

    def legalize(self, input_type: TensorType) -> List[Op]:
        """
        Legalize the processor into a list of low-level operators.
        """
        from .operators import MinPOp

        if input_type != TensorType.PROBS:
            raise ValueError(f"MinP can only be applied to PROBS, got {input_type}")

        return [MinPOp(**self.params)]


class Sample(LogitsProcessor):
    """
    Sampling processor to generate token indices.

    Samples tokens from logits or probability distributions.

    :attr:`TensorType.LOGITS` -> :attr:`TensorType.INDICES` | :attr:`TensorType.PROBS` -> :attr:`TensorType.INDICES`

    Parameters
    ----------
    deterministic : bool, optional, Compile-time
        Whether to use deterministic kernel implementation.
        Default is True.

    indices : torch.Tensor, optional, Runtime
        Indices for batched sampling when probability tensors are shared.
    generator : torch.Generator, optional, Runtime
        Random number generator for reproducible sampling.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.logits_processor import LogitsPipe, Sample, TensorType
    >>> torch.manual_seed(42)
    >>>
    >>> # Sampling from logits
    >>> pipe = LogitsPipe([Sample(deterministic=True)], input_type=TensorType.LOGITS)
    >>> logits = torch.randn(2, 5, device="cuda")
    >>> logits
    tensor([[ 0.1940,  2.1614, -0.1721,  0.8491, -1.9244],
            [ 0.6530, -0.6494, -0.8175,  0.5280, -1.2753]], device='cuda:0')
    >>> tokens = pipe(logits, top_k=1)
    >>> tokens
    tensor([0, 1], device='cuda:0')
    >>>
    >>> # Sampling from probabilities
    >>> pipe = LogitsPipe([Sample(deterministic=True)], input_type=TensorType.PROBS)
    >>> probs = torch.randn(2, 5, device="cuda")
    >>> probs_normed = probs / probs.sum(dim=-1, keepdim=True)
    >>> probs_normed
    tensor([[ 2.8827,  0.0870,  0.2340, -3.2731,  1.0694],
            [ 0.3526,  0.0928,  0.1601, -0.1737,  0.5683]], device='cuda:0')
    >>> tokens = pipe(probs_normed, top_k=1)
    >>> tokens
    tensor([0, 0], device='cuda:0')

    Notes
    -----
    Outputs :attr:`TensorType.INDICES` - no operators can follow

    See Also
    --------
    :meth:`~flashinfer.sampling.sampling_from_logits`
    :meth:`~flashinfer.sampling.sampling_from_probs`
    """

    def __init__(self, deterministic: bool = True, **params: Any):
        """
        Constructor for Sample processor.

        Parameters
        ----------
        deterministic : bool, optional
            Whether to use deterministic kernel implementation.
            Default is True.
        """
        super().__init__(deterministic=deterministic, **params)

    def legalize(self, input_type: TensorType) -> List[Op]:
        """
        Legalize the processor into a list of low-level operators.
        """
        from .operators import LogitsSampleOp, ProbsSampleOp

        if input_type == TensorType.LOGITS:
            return [LogitsSampleOp(**self.params)]
        elif input_type == TensorType.PROBS:
            return [ProbsSampleOp(**self.params)]
        else:
            raise ValueError(f"Sampling cannot be applied to {input_type}")
