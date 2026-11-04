from pathlib import Path
import subprocess
import shlex
import os
import tempfile
repo = Path(__file__).resolve().parents[2]
out = Path(os.getenv('CUTE_PREFILL_ARTIFACT_DIR', str(Path(tempfile.gettempdir()) / 'flashinfer-cute-prefill')))
out.mkdir(parents=True, exist_ok=True)
version = os.getenv('CUTE_PREFILL_VERSION', 'v6_sf_cache')
if version not in ('v1_whole', 'v2_micro', 'v3_vector', 'v4_sf_tile', 'v5_ldsm4', 'v6_sf_cache', 'v7_ring3', 'v8_ring2x5', 'v9_ring2', 'v10_pipeline3', 'v11_pipeline_copy', 'v12_pipeline_rolled', 'v13_pipeline_commit', 'v14_pipeline_noinc', 'v15_packed_p', 'v16_score_swizzle', 'v17_packed_score', 'v18_cached_indices', 'v19_role_pipeline', 'v20_scale_bits', 'v21_score_banks', 'v22_vector_q'):
    raise ValueError(version)
cmd=['/usr/local/cuda/bin/nvcc',str(Path(__file__).with_name('entry_' + version + '.cu')),'-o',str(out/'kernel.so'),'-shared','-Xcompiler=-fPIC','-std=c++17','-O3','--expt-relaxed-constexpr','-use_fast_math','-lineinfo','-gencode=arch=compute_120a,code=sm_120a','-Xptxas=-v', '-I'+str(repo/'include'),'-I'+str(repo/'3rdparty/cutlass/include')]
(out/'command.txt').write_text(shlex.join(cmd)+'\n')
with (out/'build.log').open('w') as f:
    result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
print((out/'build.log').read_text()[-12000:])
raise SystemExit(result.returncode)
