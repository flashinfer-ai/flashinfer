"""SM90-only launch contract; no imports from the SM100 validation/runtime path."""
from dataclasses import dataclass
import math
import torch
from ..sm90_gc import SM90GCSymmBuffer


@dataclass(frozen=True)
class ValidatedCall:
    ranks: int
    experts: int
    capacity: int
    topk: int
    hidden: int
    intermediate: int
    tokens: int
    bound: int
    clamp: float
    fast_math: bool
    shared_experts: int = 0


def validate_call(y, w1, w2, buffer, stats, recipe, activation, clamp,
                        fast_math, num_tokens_bound, shared_l1_weights=None,
                        shared_l2_weights=None, shared_recipe=None):
    if not isinstance(buffer, SM90GCSymmBuffer) or buffer.buffer is None:
        raise ValueError('expected live SM90GCSymmBuffer')
    device = buffer.buffer.device
    if (y.device != device or y.dtype != torch.bfloat16 or y.ndim != 2
            or not y.is_contiguous() or y.data_ptr() % 16 or y.shape[1] != buffer.hidden
            or not 0 <= y.shape[0] <= buffer.num_max_tokens_per_rank
            or torch.cuda.current_device() != device.index):
        raise ValueError('invalid SM90 y device, shape or dtype')
    if tuple(recipe) != (1,1,32) or activation != 'swiglu':
        raise ValueError('SM90 GC requires recipe (1,1,32) and swiglu')
    clamp = math.inf if clamp is None else float(clamp)
    if math.isnan(clamp) or clamp < 0:
        raise ValueError('activation_clamp must be nonnegative')
    bound = 0 if num_tokens_bound is None else int(num_tokens_bound)
    if bound < 0 or bound > buffer.num_max_tokens_per_rank or (bound and bound < y.shape[0]):
        raise ValueError('num_tokens_bound must cover every rank and fit capacity')
    local = buffer.num_experts // buffer.group.size()
    for name, weight, n, k in (('routed W1', w1, 2 * buffer.intermediate_hidden, buffer.hidden),
                               ('routed W2', w2, buffer.hidden, buffer.intermediate_hidden)):
        if not isinstance(weight, (tuple, list)) or len(weight) != 2:
            raise ValueError(f'{name} needs the record/secondary pair from '
                             'transform_weights_for_gc_moe_sm90')
        shapes = ((local, n // 256, k // 128, 16896), (local,))
        dtypes = (torch.int8, torch.float32)
        for tensor, shape, dtype in zip(weight, shapes, dtypes):
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                    or tensor.dtype != dtype or tensor.device != device
                    or not tensor.is_contiguous() or tensor.data_ptr() % 16):
                raise ValueError(f'invalid {name} SM90 tile layout, dtype or device; '
                                 'call transform_weights_for_gc_moe_sm90 first')
    if stats is not None and (stats.device != device or stats.dtype != torch.int32
                              or stats.shape != (local,) or not stats.is_contiguous()):
        raise ValueError('stats must be contiguous int32 [local_experts]')
    if (shared_l1_weights is None) != (shared_l2_weights is None):
        raise ValueError('both shared weight pairs must be supplied together')
    shared_experts = 0
    if shared_l1_weights is not None:
        for name, weight in (('shared W1', shared_l1_weights),
                             ('shared W2', shared_l2_weights)):
            if (not isinstance(weight, (tuple, list)) or len(weight) != 2
                    or not all(isinstance(tensor, torch.Tensor) for tensor in weight)):
                raise ValueError(f'{name} needs the data/scale pair from '
                                 'transform_weights_for_gc_moe_sm90')
        if shared_recipe is not None and tuple(shared_recipe) not in ((1, 1, 32), (1, 128, 128)):
            raise ValueError('unsupported shared recipe')
        if shared_l2_weights[0].ndim != 6:
            raise ValueError('shared W2 needs the tiled payload from '
                             'transform_weights_for_gc_moe_sm90')
        shared_i = shared_l2_weights[0].shape[1] * 128
        if shared_i <= 0 or shared_i % buffer.intermediate_hidden:
            raise ValueError('shared intermediate must contain whole shared experts')
        shared_experts = shared_i // buffer.intermediate_hidden
        if shared_experts > buffer.num_shared_experts:
            raise ValueError('workspace does not cover shared experts')
        for name, weight, n, k in (('shared W1', shared_l1_weights, 2 * shared_i, buffer.hidden),
                                   ('shared W2', shared_l2_weights, buffer.hidden, shared_i)):
            for tensor, shape, dtype in zip(weight,
                    ((n // 256, k // 128, 4, 4, 128, 16), (n // 256, k // 128, 4, 64, 4)),
                    (torch.uint8, torch.float32)):
                if (tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != device
                        or not tensor.is_contiguous() or tensor.data_ptr() % 16):
                    raise ValueError(f'invalid {name} FP8 tile/scale layout; '
                                     'call transform_weights_for_gc_moe_sm90 first')
    return ValidatedCall(buffer.group.size(), buffer.num_experts,
                          buffer.num_max_tokens_per_rank, buffer.num_topk,
                          buffer.hidden, buffer.intermediate_hidden,
                          y.shape[0], bound, clamp, bool(fast_math), shared_experts)


def sm90_fp8_fp4_gc_moe(y, w1, w2, buffer, stats, recipe, activation, clamp,
                        fast_math, num_tokens_bound, shared_l1_weights=None,
                        shared_l2_weights=None, shared_recipe=None):
    config = validate_call(y, w1, w2, buffer, stats, recipe, activation, clamp,
                           fast_math, num_tokens_bound, shared_l1_weights,
                           shared_l2_weights, shared_recipe)
    from ..jit_kernels.impls.sm90_fp8_fp4_gc_moe import launch_sm90_gc_moe
    launch_sm90_gc_moe(config, y, w1, w2, buffer, stats,
                       shared_l1_weights, shared_l2_weights)
