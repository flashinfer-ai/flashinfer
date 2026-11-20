"""SM120-only launch contract for the routed GC MoE path."""
from dataclasses import dataclass
import math
import torch
from ..sm120_gc import SM120GCSymmBuffer


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
    if not isinstance(buffer, SM120GCSymmBuffer) or buffer.buffer is None:
        raise ValueError('expected live SM120GCSymmBuffer')
    device = buffer.buffer.device
    if (y.device != device or y.dtype != torch.bfloat16 or y.ndim != 2
            or not y.is_contiguous() or y.data_ptr() % 16 or y.shape[1] != buffer.hidden
            or not 0 <= y.shape[0] <= buffer.num_max_tokens_per_rank
            or torch.cuda.current_device() != device.index):
        raise ValueError('invalid SM120 y device, shape or dtype')
    recipe = tuple(recipe)
    if recipe != (1, 1, 32) or activation != 'swiglu':
        raise ValueError('SM120 GC requires recipe (1,1,32) and swiglu')
    shared_recipe = recipe if shared_recipe is None else tuple(shared_recipe)
    if shared_recipe not in ((1, 1, 32), (1, 128, 128)):
        raise ValueError('shared recipe must be K32 or K128')
    clamp = math.inf if clamp is None else float(clamp)
    if math.isnan(clamp) or clamp < 0:
        raise ValueError('activation_clamp must be nonnegative')
    bound = 0 if num_tokens_bound is None else int(num_tokens_bound)
    if bound < 0 or bound > buffer.num_max_tokens_per_rank or (bound and bound < y.shape[0]):
        raise ValueError('num_tokens_bound must cover every rank and fit capacity')
    local = buffer.num_experts // buffer.group.size()
    for name, weight, n, k in (('routed W1', w1, 2 * buffer.intermediate_hidden, buffer.hidden),
                               ('routed W2', w2, buffer.hidden, buffer.intermediate_hidden)):
        payload_shape = (local, k // 128, n, 64)
        sf_shape = (local, n, k // 128)
        if (not isinstance(weight, (tuple, list)) or len(weight) != 2
                or not isinstance(weight[0], torch.Tensor)
                or tuple(weight[0].shape) != payload_shape
                or weight[0].dtype != torch.int8
                or not isinstance(weight[1], torch.Tensor)
                or tuple(weight[1].shape) != sf_shape
                or weight[1].dtype != torch.int32
                or weight[0].device != device or weight[1].device != device
                or not weight[0].is_contiguous()
                or weight[1].stride() != (n * (k // 128), 1, n)
                or weight[0].data_ptr() % 16 or weight[1].data_ptr() % 16):
            raise ValueError(f'invalid {name} SM120 tile layout, dtype or device; '
                             'call transform_weights_for_gc_moe_sm120 first')
    if stats is not None and (stats.device != device or stats.dtype != torch.int32
                              or stats.shape != (local,) or not stats.is_contiguous()):
        raise ValueError('stats must be contiguous int32 [local_experts]')
    if (shared_l1_weights is None) != (shared_l2_weights is None):
        raise ValueError('both shared weight pairs must be supplied together')
    shared_experts = 0
    if shared_l1_weights is not None:
        for name, pair in (('shared W1', shared_l1_weights), ('shared W2', shared_l2_weights)):
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or not all(isinstance(t, torch.Tensor) for t in pair)):
                raise ValueError(f'{name} requires the transformed data/scale pair')
        if shared_l2_weights[0].ndim != 3:
            raise ValueError('shared W2 requires the transformed SM120 tile layout')
        shared_w2 = shared_l2_weights[0]
        shared_i = shared_w2.shape[0] * 128
        if shared_i <= 0 or shared_i % buffer.intermediate_hidden:
            raise ValueError('shared intermediate must contain whole shared experts')
        shared_experts = shared_i // buffer.intermediate_hidden
        if shared_experts > buffer.num_shared_experts:
            raise ValueError('workspace does not cover shared experts')
        packed_gran_k = shared_recipe[2] * 4
        for name, pair, n, k in (
                ('shared W1', shared_l1_weights, 2 * shared_i, buffer.hidden),
                ('shared W2', shared_l2_weights, buffer.hidden, shared_i)):
            payload, sf = pair
            if (payload.shape != (k // 128, n, 128) or payload.dtype != torch.float8_e4m3fn
                    or sf.dtype != torch.int32 or not payload.is_contiguous()
                    or sf.stride() != (1, n) or sf.shape != (n, (k + packed_gran_k - 1) // packed_gran_k)
                    or any(t.device != device or t.data_ptr() % 16 for t in pair)):
                raise ValueError(f'invalid {name} SM100 packed weight layout')
    return ValidatedCall(buffer.group.size(), buffer.num_experts,
                          buffer.num_max_tokens_per_rank, buffer.num_topk,
                          buffer.hidden, buffer.intermediate_hidden,
                          y.shape[0], bound, clamp, bool(fast_math), shared_experts)
