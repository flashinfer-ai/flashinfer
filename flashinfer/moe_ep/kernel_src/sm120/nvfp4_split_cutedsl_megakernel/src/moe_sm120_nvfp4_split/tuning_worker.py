"""torchrun worker for offline tuning; no benchmark code enters the API."""
import os
import sys
import json
import time
import hashlib
from dataclasses import asdict
from pathlib import Path


def main():
    import torch
    import numpy as np
    from . import mega_runner as runner
    from .mega_reference import compute_megamoe_reference_nvfp4
    from .combine_reference import byte_difference, numerical_comparison
    from .combine_validation import rank_local_staging_report
    from .tuning_config import hardware_identity, implementation_fingerprint, config_from_dict
    from .api import KERNEL_CACHE_ABI
    spec = json.loads(Path(os.environ['SM120_TUNING_SPEC']).read_text())
    out = Path(spec['output']); out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['LOCAL_RANK'])
    data = dict(status='RUNNING', rank=rank, validations=[], samples=[],
                contract=spec['contract'], kernel_cache_abi=KERNEL_CACHE_ABI,
                implementation_sha256=implementation_fingerprint(), full_k3=True,
                mode='dynamic_correctness' if spec.get('dynamic') else 'uninstrumented_performance')
    def sha(t):
        return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    original_select = runner.select_megamoe_config
    def select(shape, overrides=None, **kwargs):
        if (torch.cuda.get_device_capability() != (12, 0)
                or shape.ep_cross_numa_peer_count or shape.ep_same_numa_peer_count != 3
                or (shape.expert_parallel_size, shape.tensor_parallel_size, shape.data_parallel_size) != (4, 1, 1)):
            raise ValueError('offline v1 requires SM120 same-NUMA EP4/TP1/DP1')
        cfg = (config_from_dict(spec['config'], shape, spec['contract']) if spec.get('config')
               else original_select(shape, overrides, **kwargs))
        data.update(shape=asdict(shape), config=asdict(cfg), hardware=hardware_identity())
        return cfg
    runner.select_megamoe_config = select
    runner.compute_megamoe_reference_nvfp4 = lambda **kw: compute_megamoe_reference_nvfp4(**kw, gemm_backend='native_qmma')
    def validate(self, *, force=False):
        expected = self.rank_local_output_ref if self.rank_local_combine else self.output_tensor_ref
        diff = byte_difference(self.output_tensor, expected)
        comparison = numerical_comparison(self.output_tensor, expected)
        transport = rank_local_staging_report(self) if self.rank_local_combine else self._validate_token_back_exact()
        row = dict(byte_differences=diff, nonfinite=comparison['nonfinite_elements'], transport=transport['status'])
        data['validations'].append(row)
        flag = torch.tensor([int(diff != 0 or row['nonfinite'] != 0 or transport['status'] == 'FAIL')], dtype=torch.int, device='cuda')
        flags = [torch.empty_like(flag) for _ in range(self.world_size)]
        self._ep_all_gather(flags, flag)
        if any(int(x) for x in flags):
            raise AssertionError(row)
    runner.MegaMoETester.validate = validate
    gen0 = runner.MegaMoETester.generate_inputs
    def generate(self):
        gen0(self)
        names = ('my_activation','my_activation_sf','my_topk_idx','my_topk_weights',
                 'my_fc1_weight','my_fc1_weight_sf','my_fc2_weight','my_fc2_weight_sf',
                 'my_fc1_alpha','my_fc2_alpha','my_fc1_norm_const')
        data['fixtures'] = {name:sha(getattr(self, name)) for name in names}
    runner.MegaMoETester.generate_inputs = generate
    # Preserve the existing RNG stream while avoiding the scalar FP8 factory.
    def scale(rng,n,k,*,blocksize):
        x=torch.rand((n,runner.ceil_div(k,blocksize)),dtype=torch.float32,device='cuda',generator=rng)
        return x.ge_(.8).mul_(.5).add_(1).to(runner.Nvfp4ScaleDtype)
    runner._make_nvfp4_scale_tensor = scale
    if spec.get('routing_trace'):
        def trace(*args, **kwargs):
            path = Path(spec['routing_trace'])
            array = np.load(path, allow_pickle=False)
            if isinstance(array, np.lib.npyio.NpzFile):
                with array as z: array = z[str(spec['tokens'])].copy()
            expected = (spec['ep'], spec['tokens'], spec['num_topk'])
            if array.shape != expected or array.dtype != np.int64:
                raise ValueError(f'routing trace must be int64 {expected}')
            if np.any(array < 0) or np.any(array >= spec['num_total_experts']):
                raise ValueError('routing trace expert ID outside bounds')
            if np.any(np.diff(np.sort(array, axis=-1), axis=-1) == 0):
                raise ValueError('routing trace contains duplicate experts for a token')
            return array
        runner._generate_topk_idx_balanced = trace
    gather0 = runner.MegaMoETester._ep_all_gather
    def gather(self, outputs, tensor, *args, **kwargs):
        result = gather0(self, outputs, tensor, *args, **kwargs)
        if tensor.dtype == torch.float64 and tensor.ndim == 1 and tensor.numel() == self._perf_iters:
            data['samples'] = [v.cpu().tolist() for v in outputs]
        return result
    runner.MegaMoETester._ep_all_gather = gather
    def timing_get(self): return self.__dict__.get('_tuning_timing', False)
    def timing_set(self, value):
        previous = timing_get(self); self.__dict__['_tuning_timing'] = value
        if value and not previous: data['measurement_start'] = time.time()
        elif previous and not value: data['measurement_end'] = time.time()
    runner.MegaMoETester._phase_timing_collect = property(timing_get, timing_set)
    run0 = runner.MegaMoETester.run
    def run(self):
        run0(self)
        data['output_sha256'] = sha(self.output_tensor)
        data['effective'] = {k:dict(tile=getattr(self,'_kernel_'+k).mma_tiler_mnk,
                                  stages=getattr(self,'_kernel_'+k).num_ab_stage)
                             for k in ('k1','k2')}
        if not data['validations'] or not data['samples']:
            raise AssertionError('missing native reference or full-pipeline samples')
        for k in ('k1', 'k2'):
            if (tuple(data['effective'][k]['tile']) != tuple(data['config'][k+'_tile'])
                    or data['effective'][k]['stages'] != (data['config'][k+'_stages'] or 2)):
                raise AssertionError('resolved configuration differs from compiled kernel')
        if spec.get('dynamic') and len(data['validations']) < spec['dynamic']:
            raise AssertionError('missing per-replay native validation')
        data['status'] = 'PASS'
        # mega_runner intentionally uses os._exit after communication teardown;
        # persist before returning control to it, rather than relying on finally.
        (out/f'rank{rank}.json').write_text(json.dumps(data, indent=2))
    runner.MegaMoETester.run = run
    argv=['tuning_worker','--num_tokens_per_rank',str(spec['tokens']),
          '--hidden',str(spec['hidden']),'--intermediate',str(spec['intermediate']),
          '--num_topk',str(spec['num_topk']),'--num_total_experts',str(spec['num_total_experts']),
          '--enable_static_expert_shape','--comm_backend','p2p_direct',
          '--split_launch','green_graph','--numerical_contract',spec['contract'],
          '--seed',str(spec['seed']),'--route_distribution',
          'balanced' if spec['route']=='trace' else spec['route'],
          '--gate_up_clamp','10.0','--perf_run','--use_cuda_events',
          '--perf_warmup',str(spec['warmups']),'--perf_iters',str(spec['samples'])]
    if spec.get('dynamic'):
        argv += ['--dynamic_route_replays',str(spec['dynamic'])]
        os.environ['MEGA_RANK_LOCAL_VALIDATE_EVERY_REPLAY']='1'
        os.environ['MEGA_DYNAMIC_ROUTE_PATTERN']='mixed_owner_stress'
    sys.argv = argv
    try:
        runner.main()
    finally:
        (out/f'rank{rank}.json').write_text(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
