"""Public allocation and checkpoint conversion for the independent SM90 GC path."""
from __future__ import annotations
import types
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from ..layout.sm90_gc_moe import SM90GCBufferLayout, align


class SM90GCSymmBuffer:
    """Own symmetric SM90 workspace. Serialize calls and retire graphs before destroy."""
    def __init__(self, group, num_experts, num_max_tokens_per_rank, num_topk,
                 hidden, intermediate_hidden, activation='swiglu', num_shared_experts=0):
        ranks = group.size()
        if torch.cuda.get_device_capability()[0] != 9:
            raise ValueError('SM90 GC requires Hopper')
        if not (1 <= ranks <= 32 and num_experts > 0 and num_experts % ranks == 0
                and 1 <= num_topk <= min(32, num_experts)
                and 512 <= hidden <= 8192 and hidden % 512 == 0
                and 256 <= intermediate_hidden <= 4096 and intermediate_hidden % 256 == 0
                and num_max_tokens_per_rank > 0 and activation == 'swiglu'):
            raise ValueError('invalid SM90 GC geometry or activation')
        self.group, self.rank_idx = group, dist.get_rank(group)
        self.num_experts, self.num_topk = num_experts, num_topk
        self.num_max_tokens_per_rank = align(num_max_tokens_per_rank)
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        if not isinstance(num_shared_experts, int) or num_shared_experts < 0:
            raise ValueError("num_shared_experts must be a nonnegative integer")
        self.num_shared_experts = num_shared_experts
        self.activation = activation
        self.layout = SM90GCBufferLayout(ranks, num_experts, self.num_max_tokens_per_rank,
                                         num_topk, hidden, intermediate_hidden, num_shared_experts)
        allocator = torch if ranks == 1 else symm_mem
        self.buffer = allocator.empty(self.layout.nbytes, dtype=torch.int8, device='cuda')
        self.handle = (types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()]) if ranks == 1
                       else symm_mem.rendezvous(self.buffer, group=group))
        self.peer_addresses = tuple(int(p) for p in self.handle.buffer_ptrs)
        if (len(self.peer_addresses) != ranks or any(p <= 0 or p % 16 for p in self.peer_addresses)
                or self.peer_addresses[self.rank_idx] != self.buffer.data_ptr()):
            raise RuntimeError('invalid SM90 symmetric peer mappings')
        self.buffer.zero_()
        for name, spec in self.layout.views.items():
            setattr(self, name, self.buffer.narrow(0, spec.offset, spec.nbytes)
                    .view(getattr(torch, spec.dtype)).view(spec.shape))
        self.peer_table.copy_(torch.tensor(self.peer_addresses, dtype=torch.int64, device=self.buffer.device))
        self.header[0] = 1
        group.barrier().wait()
        torch.cuda.synchronize()

    def destroy(self):
        for name in self.layout.views:
            setattr(self, name, None)
        self.peer_addresses = self.handle = self.buffer = self.group = None


def get_symm_buffer_for_gc_moe_sm90(group, num_experts, num_max_tokens_per_rank,
                                  num_topk, hidden, intermediate_hidden,
                                  activation='swiglu', num_shared_experts=0):
    return SM90GCSymmBuffer(group, num_experts, num_max_tokens_per_rank, num_topk,
                           hidden, intermediate_hidden, activation, num_shared_experts)


# Exact Humming code-space round-downscale mapping, including .25 -> .5.
_DOWNSCALE = (
    (0,1,2,3,4,5,6,7,0,9,10,11,12,13,14,15),
    (0,1,1,2,2,3,4,5,0,9,9,10,10,11,12,13),
    (0,0,1,1,1,2,2,3,0,8,9,9,9,10,10,11),
    (0,0,0,0,1,1,1,2,0,8,8,8,9,9,9,10),
    (0,0,0,0,0,0,1,1,0,8,8,8,8,8,9,9),
    (0,0,0,0,0,0,0,0,0,8,8,8,8,8,8,8),
)


def _scale_codes(scales):
    if scales.dtype in (torch.uint8, torch.float8_e8m0fnu):
        return scales.view(torch.uint8).to(torch.int32)
    if scales.dtype != torch.float32:
        raise ValueError('scales must be UE8M0 bytes or FP32 powers of two')
    bits = scales.contiguous().view(torch.int32)
    return ((bits >> 23) & 255).to(torch.int32)


def _transform(raw, scale_2=None, paired=False):
    packed, scales = raw
    if packed.ndim != 3 or packed.dtype not in (torch.int8, torch.uint8):
        raise ValueError('expected [E,N,K/2] packed MXFP4')
    e, n, half_k = packed.shape
    k = 2 * half_k
    if n % 256 or k % 128 or tuple(scales.shape) != (e, n, k // 32) or scales.device != packed.device:
        raise ValueError('invalid SM90 weight or scale shape')
    codes = _scale_codes(scales)
    if scale_2 is not None and (scale_2.shape != (e,) or scale_2.dtype != torch.float32
                               or scale_2.device != packed.device):
        raise ValueError('secondary checkpoint scale must be FP32 [E] on the weight device')
    # One contiguous stage record: unchanged FP4 bytes, then 512B of packed
    # relative exponents. Views only organize transform writes, with no copy.
    record = torch.empty((e, n // 256, k // 128, 16896), device=packed.device, dtype=torch.int8)
    data = record[..., :16384].view(e, n // 256, k // 128, 4, 256, 16)
    offsets = record[..., 16384:].view(torch.uint8).view(e, n // 256, k // 128, 128, 4)
    secondary = torch.empty(e, device=packed.device, dtype=torch.float32)
    lut = torch.tensor(_DOWNSCALE, dtype=torch.uint8, device='cpu',
                       pin_memory=True).to(packed.device, non_blocking=True)
    order = torch.tensor([0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15],
                         device='cpu', pin_memory=True).to(packed.device, non_blocking=True)
    for expert in range(e):
        sf = codes[expert]
        base = torch.maximum(sf.min(), sf.max() - 11)
        target = torch.maximum(sf, base)
        delta = target - sf
        source = packed[expert].contiguous().view(torch.uint8).reshape(n, k // 32, 16)
        d = delta.clamp_max(5).long().unsqueeze(-1)
        converted = (lut[d, (source & 15).long()] | (lut[d, (source >> 4).long()] << 4)).reshape(n, half_k)
        # Each lane owns four consecutive K values and four at K+16.
        # Move bytes, then interleave only the sign bits for two PRMT outputs.
        relative = (target - base + 1).to(torch.uint8)
        if paired:
            converted = converted.reshape(2, n // 256, 128, half_k).permute(1, 0, 2, 3).contiguous().reshape(n, half_k)
            relative = relative.reshape(2, n // 256, 128, k // 32).permute(1, 0, 2, 3).contiguous().reshape(n, k // 32)
        units = converted.reshape(n // 256, 256, k // 128, 4, 16)
        units = units.permute(0, 2, 3, 1, 4).contiguous()[..., order]
        words = units.reshape(-1, 4)
        moved = words & 0x77
        moved[:, 0] |= (words[:, 0] & 0x08) | ((words[:, 2] & 0x08) << 4)
        moved[:, 1] |= ((words[:, 0] & 0x80) >> 4) | (words[:, 2] & 0x80)
        moved[:, 2] |= (words[:, 1] & 0x08) | ((words[:, 3] & 0x08) << 4)
        moved[:, 3] |= ((words[:, 1] & 0x80) >> 4) | (words[:, 3] & 0x80)
        # Per M64: two 512B halves per warp, each lane owns a 16B
        # vector of {g0/s0,g0/s1,g1/s0,g1/s1} packed words. This is
        # an exact permutation of the same FP4 bytes, with no expansion.
        vectors = moved.reshape(n // 256, k // 128, 2, 2, 4, 4, 2, 8, 4, 4)
        vectors = vectors.permute(0, 1, 4, 5, 2, 7, 8, 3, 6, 9).contiguous()
        data[expert].copy_(vectors.reshape(data[expert].shape).view(torch.int8))
        sf_vectors = relative.reshape(n // 256, 4, 4, 2, 8, k // 128, 4)
        sf_vectors = sf_vectors.permute(0, 5, 1, 2, 4, 3, 6).contiguous()
        # Pack adjacent exponents (even byte -> low nibble, odd -> high nibble).
        # relative is in [1, 12] by construction, so this is lossless.
        flat = sf_vectors.reshape(n // 256, k // 128, 1024)
        nibbles = (flat[..., 0::2] & 0x0F) | (flat[..., 1::2] << 4)
        offsets[expert].copy_(nibbles.reshape(offsets[expert].shape))
        secondary[expert] = torch.exp2(base.float() - 128) * (1 if scale_2 is None else scale_2[expert])
    return record, secondary


def transform_weights_for_gc_moe_sm90(l1_weights, l2_weights,
                                      l1_weight_scale_2=None, l2_weight_scale_2=None,
                                      shared_recipe=(1, 1, 32)):
    """Convert checkpoint weights to SM90 tiles asynchronously on the current stream.

    Call outside CUDA Graph capture with valid quantized scales and source writes
    ordered on the current stream. Raw input references may be released on return.
    The caller must keep the returned tensors alive and address-stable until all
    launches and captured CUDA Graphs that reference them have retired.
    """
    pairs = (l1_weights, l2_weights)
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 or
           any(not isinstance(tensor, torch.Tensor) for tensor in pair)
           for pair in pairs):
        raise ValueError('SM90 transform needs two original (payload, scale) pairs')
    if l1_weights[0].dtype == torch.float8_e4m3fn:
        if l1_weight_scale_2 is not None or l2_weight_scale_2 is not None:
            raise ValueError('dense shared weights do not have secondary scales')
    elif (l1_weight_scale_2 is None) != (l2_weight_scale_2 is None):
        raise ValueError('both checkpoint secondary scales must be supplied together')
    inputs = tuple(tensor for pair in pairs for tensor in pair)
    if l1_weight_scale_2 is not None:
        if not isinstance(l1_weight_scale_2, torch.Tensor) or \
                not isinstance(l2_weight_scale_2, torch.Tensor):
            raise ValueError('secondary checkpoint scales must be tensors')
        inputs += (l1_weight_scale_2, l2_weight_scale_2)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('transform SM90 weights outside CUDA Graph capture')
    if any(not tensor.is_cuda for tensor in inputs):
        raise ValueError('SM90 transform inputs must use CUDA storage')
    current_device = torch.cuda.current_device()
    if any(tensor.device.index != current_device for tensor in inputs):
        raise ValueError('SM90 transform inputs must use the current CUDA device')
    stream = torch.cuda.current_stream(current_device)
    for tensor in inputs:
        tensor.record_stream(stream)
    if l1_weights[0].dtype == torch.float8_e4m3fn:
        return (_transform_shared(l1_weights, shared_recipe, True),
                _transform_shared(l2_weights, shared_recipe, False))
    if (l1_weights[0].ndim != 3 or l2_weights[0].ndim != 3
            or l1_weights[0].shape[0] != l2_weights[0].shape[0]
            or l1_weights[0].shape[1] != 4 * l2_weights[0].shape[2]
            or l2_weights[0].shape[1] != 2 * l1_weights[0].shape[2]):
        raise ValueError('W1/W2 geometry mismatch')
    return (_transform(l1_weights, l1_weight_scale_2, paired=True),
            _transform(l2_weights, l2_weight_scale_2))


def _transform_shared(pair, recipe, paired):
    """Lossless FP8 byte permutation and exact FP32 scale expansion to row/K32.

    GLM block N128/K128 scales are repeated over rows and four K32 groups.
    Flash per-row K32 scales are retained without rescaling any FP8 value.
    """
    data, sf = pair
    n, k = data.shape
    if recipe not in ((1, 128, 128), (1, 1, 32)) or n % 256 or k % 128:
        raise ValueError('invalid dense shared geometry or recipe')
    if data.dtype != torch.float8_e4m3fn or sf.dtype != torch.float32:
        raise ValueError('shared weights require E4M3 data and FP32 scales')
    if tuple(sf.shape) != (n // recipe[1], k // recipe[2]):
        raise ValueError('invalid dense shared scale shape')
    scales = sf.repeat_interleave(recipe[1], 0).repeat_interleave(recipe[2] // 32, 1)
    values = data.view(torch.uint8)
    if paired:
        values = values.reshape(2, n // 256, 128, k).permute(1, 0, 2, 3).reshape(n, k)
        scales = scales.reshape(2, n // 256, 128, k // 32).permute(1, 0, 2, 3).reshape(n, k // 32)
    lanes = torch.arange(128, device=data.device)
    rows = (lanes // 32) * 16 + (lanes % 32) // 4
    ks = (lanes % 4) * 4
    row_idx = (torch.arange(n // 256, device=data.device)[:, None, None, None, None, None] * 256
               + torch.arange(4, device=data.device)[None, None, :, None, None, None] * 64
               + rows[None, None, None, :, None, None])
    k_idx = (torch.arange(k // 128, device=data.device)[None, :, None, None, None, None] * 128
             + torch.arange(4, device=data.device)[None, None, None, None, :, None] * 32
             + ks[None, None, None, :, None, None])
    words = [values[row_idx + rr, k_idx + kk + torch.arange(4, device=data.device)]
             for rr, kk in ((0, 0), (8, 0), (0, 16), (8, 16))]
    # At fixed K32, adjacent lanes read contiguous LDS.128 vectors.
    tiled = torch.cat(words, dim=-1).permute(0, 1, 2, 4, 3, 5).contiguous()
    tiled_sf = scales.reshape(n // 256, 4, 64, k // 128, 4).permute(0, 3, 1, 2, 4).contiguous()
    return tiled, tiled_sf


def sm90_fp8_fp4_gc_moe(y, l1_weights, l2_weights, sym_buffer,
                        cumulative_local_expert_recv_stats=None, recipe=(1,1,32),
                        activation='swiglu', activation_clamp=None, fast_math=True,
                        num_tokens_bound=None, shared_l1_weights=None,
                        shared_l2_weights=None, shared_recipe=None):
    from ..apis.sm90_gc_moe import sm90_fp8_fp4_gc_moe as launch
    launch(y, l1_weights, l2_weights, sym_buffer, cumulative_local_expert_recv_stats,
           recipe, activation, activation_clamp, fast_math, num_tokens_bound,
           shared_l1_weights, shared_l2_weights, shared_recipe)
