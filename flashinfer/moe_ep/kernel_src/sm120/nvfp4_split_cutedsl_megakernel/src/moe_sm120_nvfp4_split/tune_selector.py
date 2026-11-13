"""Offline full-pipeline tuning. Run with python -m ...tune_selector --help."""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from .heuristic import MegaMoEHeuristicInput
from .tuning_config import config_from_dict, make_document, implementation_fingerprint
from .tuning_search import auto_tokens, candidate_configs, qualify_pairs


def quantile(values, q):
    values=sorted(values); position=(len(values)-1)*q; i=int(position)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(position-i)


def query(kind, columns):
    output=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+columns,
                                    '--format=csv,noheader,nounits'],text=True)
    return [[v.strip() for v in row.split(',')] for row in output.splitlines() if row.strip()]


def descendant(pid, parent):
    while pid>1:
        if pid==parent: return True
        try: pid=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
        except FileNotFoundError: return True
    return False


def process_identity(pid):
    root=Path('/proc')/str(pid)
    return [root.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],
            hashlib.sha256(root.joinpath('cmdline').read_bytes()).hexdigest()]


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL); process.wait()


class Campaign:
    def __init__(self, args):
        self.args=args; self.root=Path(args.work_dir).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        # Never merge evidence from separate process/environment observations.
        if (self.root/'campaign.json').exists():
            raise ValueError('work directory already contains a campaign; choose a new directory')
        self.deadline=time.monotonic()+args.budget_seconds
        devices=query('gpu','index,uuid')
        requested=(args.devices or os.environ.get('CUDA_VISIBLE_DEVICES') or ','.join(str(i) for i in range(args.ep))).split(',')
        self.uuids=[]
        for device in requested:
            match=[row[1] for row in devices if device.strip() in row]
            if len(match)!=1: raise ValueError('unrecognized/ambiguous physical GPU: '+device)
            self.uuids.append(match[0])
        if len(self.uuids)!=args.ep or len(set(self.uuids))!=args.ep:
            raise ValueError('--devices must contain exactly EP distinct GPUs')
        self.allowed={pid:process_identity(pid) for pid in args.allow_resident_pid}
        self.counter=0; self.hardware=None
        self.source=implementation_fingerprint()
        self.record=dict(arguments=vars(args),gpus=self.uuids,allowed_residents=self.allowed,
                         implementation_sha256=self.source,start=time.time(),runs=[])
        self.save()

    def save(self):
        (self.root/'campaign.json').write_text(json.dumps(self.record,indent=2))

    def apps(self):
        return [r for r in query('compute-apps','gpu_uuid,pid,used_memory') if r[0] in self.uuids]

    def guard(self, parent=None):
        apps=self.apps()
        for row in apps:
            pid=int(row[1])
            if parent and descendant(pid,parent): continue
            if pid in self.allowed and process_identity(pid)==self.allowed[pid]: continue
            raise RuntimeError('unrecognized GPU process: '+str(row))
        return apps

    def run(self,tokens,route,seed,config=None,dynamic=0):
        if time.monotonic()>=self.deadline: raise TimeoutError('campaign budget exhausted')
        self.guard()
        self.counter+=1; directory=self.root/f'run_{self.counter:04d}'
        directory.mkdir()
        a=self.args
        spec=dict(tokens=tokens,route=route,seed=seed,config=config,hidden=a.hidden,
                  intermediate=a.intermediate,num_topk=a.num_topk,num_total_experts=a.num_total_experts,
                  ep=a.ep,contract=a.numerical_contract,samples=a.samples,warmups=a.warmups,
                  dynamic=dynamic,output=str(directory),routing_trace=a.routing_trace if route=='trace' else None)
        specpath=directory/'spec.json';specpath.write_text(json.dumps(spec,indent=2))
        env={k:v for k,v in os.environ.items() if not k.startswith('MEGA_')}
        env.update(CUDA_VISIBLE_DEVICES=','.join(self.uuids),SM120_TUNING_SPEC=str(specpath),
                   OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',MEGA_SPLIT_INCLUDE_K3='1',
                   MEGA_FC2_READY_BUNDLE_K_TILES='12',MEGA_K1_READY_MAJOR='1',MEGA_K2_TOKEN_STRIPE='2',
                   NCCL_NVLS_ENABLE='0',NCCL_P2P_DISABLE='1')
        env['PYTHONPATH']=str(Path(__file__).resolve().parent.parent)+os.pathsep+env.get('PYTHONPATH','')
        env['NATIVE_QMMA_CACHE']=str(self.root/'native_reference_cache')
        env['CUTE_DSL_CACHE_DIR']=str(self.root/'compile_cache')
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone',
             '--nproc_per_node='+str(a.ep),'--module','moe_sm120_nvfp4_split.tuning_worker']
        (directory/'command.json').write_text(json.dumps(dict(command=cmd,environment={k:env[k] for k in env if k.startswith(('MEGA_','MEGAMOE_','NCCL_'))}),indent=2))
        begin=time.monotonic(); row=dict(status='FAILED',spec=spec,evidence=str(directory))
        print(f'[{self.counter}] T{tokens} {route} seed={seed} '+('dynamic' if dynamic else 'timing'),flush=True)
        with (directory/'run.log').open('w') as log, (directory/'telemetry.jsonl').open('w') as tele, (directory/'power.csv').open('w') as power, (directory/'pmon.log').open('w') as pmon:
            sampler=subprocess.Popen(['nvidia-smi','-i',','.join(self.uuids),
                '--query-gpu=timestamp,index,clocks.sm,power.draw','--format=csv,noheader,nounits','-lms','20'],stdout=power,start_new_session=True)
            monitor=subprocess.Popen(['nvidia-smi','pmon','-s','u','-d','1'],stdout=pmon,stderr=subprocess.STDOUT,start_new_session=True)
            process=None
            try:
                process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while process.poll() is None:
                    tele.write(json.dumps(dict(time=time.time(),apps=self.guard(process.pid)))+'\n');tele.flush()
                    if time.monotonic()>=self.deadline or time.monotonic()-begin>a.case_timeout:
                        raise TimeoutError('campaign or case budget exhausted')
                    time.sleep(.5)
                if process.returncode: raise RuntimeError(f'worker exited {process.returncode}; see {directory}/run.log')
                if implementation_fingerprint.__wrapped__()!=self.source: raise RuntimeError('implementation changed')
                row=self.collect(directory,spec)
            except BaseException as exc:
                row.update(error=str(exc))
                raise
            finally:
                if process is not None: stop(process)
                stop(sampler);stop(monitor)
                row['elapsed_seconds']=time.monotonic()-begin
                (directory/'result.json').write_text(json.dumps(row,indent=2))
                self.record['runs'].append(row);self.save()
        # Resident compute activity invalidates the whole observation.
        for line in (directory/'pmon.log').read_text().splitlines():
            parts=line.split()
            if len(parts)>3 and parts[1].isdigit() and int(parts[1]) in self.allowed and parts[3].isdigit() and int(parts[3])>0:
                row.update(status='FAILED',error='authorized resident became compute-active')
                (directory/'result.json').write_text(json.dumps(row,indent=2));self.save()
                raise RuntimeError(row['error'])
        return row

    def collect(self,directory,spec):
        ranks=[json.loads((directory/f'rank{i}.json').read_text()) for i in range(self.args.ep)]
        if any(r['status']!='PASS' or r['implementation_sha256']!=self.source for r in ranks):
            raise ValueError('incomplete worker/source validation')
        if any(r['shape']!=ranks[0]['shape'] or r['config']!=ranks[0]['config'] or r['hardware']!=ranks[0]['hardware'] or r['samples']!=ranks[0]['samples'] for r in ranks):
            raise ValueError('rank configuration/hardware/sample disagreement')
        raw=ranks[0]['samples']; expected=spec['dynamic'] or spec['samples']
        if len(raw)!=self.args.ep or any(len(x)!=expected for x in raw):
            raise ValueError('incomplete max-rank samples')
        vals=[max(x) for x in zip(*raw)]
        power=[]
        start=min(r['measurement_start'] for r in ranks);end=max(r['measurement_end'] for r in ranks)
        for line in (directory/'power.csv').read_text().splitlines():
            try:
                t,g,clock,watts=[x.strip() for x in line.split(',')]
                stamp=datetime.strptime(t,'%Y/%m/%d %H:%M:%S.%f').timestamp()
                if start<=stamp<=end: power.append([int(g),float(clock),float(watts)])
            except (ValueError,TypeError): continue
        if len({r[0] for r in power})!=self.args.ep:
            raise ValueError('missing per-GPU timed-window PCLK/power telemetry')
        if self.hardware is None: self.hardware=ranks[0]['hardware']
        if self.hardware!=ranks[0]['hardware']: raise ValueError('hardware changed during campaign')
        return dict(status='PASS',shape=ranks[0]['shape'],config=ranks[0]['config'],hardware=self.hardware,
                    implementation_sha256=self.source,contract=spec['contract'],
                    fixtures=[r['fixtures'] for r in ranks],outputs=[r['output_sha256'] for r in ranks],
                    native_checks=sum(len(r['validations']) for r in ranks),
                    transport_statuses=sorted({v['transport'] for r in ranks for v in r['validations']}),
                    p50_us=quantile(vals,.5),p90_us=quantile(vals,.9),samples=vals,
                    pclk_mean_mhz=statistics.mean(r[1] for r in power),power_mean_w=statistics.mean(r[2] for r in power),
                    evidence=str(directory),spec=spec)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('hidden','intermediate','num_topk','num_total_experts'):
        p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--ep',type=int,default=4,choices=[4],help='v1 qualifies same-NUMA EP4/TP1/DP1')
    p.add_argument('--tokens',default='auto',help='comma-separated capacities, or auto (13 sweep points then tile boundaries)')
    p.add_argument('--numerical_contract',default='ordinary_topk',choices=['ordinary_topk','owner_bf16'])
    p.add_argument('--devices',default=None,help='physical GPU indices/UUIDs; defaults to CUDA_VISIBLE_DEVICES or first EP GPUs')
    p.add_argument('--allow-resident-pid',type=int,action='append',default=[])
    p.add_argument('--routing_trace',default=None,help='int64 NPY [EP,T,topk], or NPZ keyed by token capacity')
    p.add_argument('--max_candidates',type=int,default=12)
    p.add_argument('--budget_seconds',type=float,default=3600)
    p.add_argument('--case_timeout',type=float,default=600)
    p.add_argument('--samples',type=int,default=400)
    p.add_argument('--warmups',type=int,default=50)
    p.add_argument('--dynamic_replays',type=int,default=100)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--min_gain',type=float,default=.01)
    p.add_argument('--work_dir',default='selector_tuning_runs')
    p.add_argument('--output',default='tuned_selector.json')
    p.add_argument('--plan_only',action='store_true',help='print workload plan without accessing GPUs')
    a=p.parse_args(argv)
    if min(a.hidden,a.intermediate,a.num_topk,a.num_total_experts)<=0:
        p.error('model dimensions must be positive')
    tokens=(auto_tokens(a.num_total_experts,a.num_topk,a.ep) if a.tokens=='auto' else [int(x) for x in a.tokens.split(',')])
    if any(x<=0 for x in tokens) or len(tokens)!=len(set(tokens)): p.error('tokens must be distinct positive integers')
    if min(a.hidden,a.intermediate,a.num_topk,a.num_total_experts)<=0 or a.num_topk>a.num_total_experts or a.num_total_experts%a.ep or a.hidden%32 or a.intermediate%64:
        p.error('invalid model structure or NVFP4 alignment')
    if a.samples<100 or a.warmups<10 or a.dynamic_replays<100 or a.max_candidates<2 or not 0<a.min_gain<1 or a.budget_seconds<=0 or a.case_timeout<=0:
        p.error('require samples>=100, warmups>=10, dynamic_replays>=100, candidates>=2, positive budgets and 0<min_gain<1')
    routes=['balanced','power_law','zero_count_peer']+(['trace'] if a.routing_trace else [])
    if a.plan_only:
        print(json.dumps(dict(tokens=tokens,routes=routes,candidate_limit=a.max_candidates,contract=a.numerical_contract,scope='exact capacities, same-NUMA EP4/TP1/DP1; preserve reduction association',budget_seconds=a.budget_seconds),indent=2));return
    output=Path(a.output).resolve()
    if output.exists(): p.error('output already exists; choose a new path to preserve prior tuning')
    c=Campaign(a); entries=[]; decisions=[];failure=None
    try:
        for t in tokens:
            base=c.run(t,'balanced',a.seed)
            shape=MegaMoEHeuristicInput(**base['shape'])
            if shape.ep_cross_numa_peer_count or shape.ep_same_numa_peer_count!=3:
                raise ValueError('v1 tuner requires same-NUMA EP4')
            baseline=config_from_dict(base['config'],shape,a.numerical_contract)
            candidates=candidate_configs(shape,baseline,a.numerical_contract,a.max_candidates)
            screens=[]
            for cfg in candidates[1:]:
                result=c.run(t,'balanced',a.seed,asdict(cfg))
                if result['p50_us']<base['p50_us'] and result['p90_us']<=base['p90_us']:
                    screens.append(result)
            decision=dict(tokens=t,screened=len(candidates),adopted=False)
            for best in sorted(screens,key=lambda x:x['p50_us'])[:2]:
                pairs=[]
                # Independent holdout seeds, reverse order second time; screen
                # measurements never qualify a profile by themselves.
                for route in routes:
                    for repeat in range(2):
                        seed=a.seed+10007+repeat*104729
                        if repeat==0:
                            b=c.run(t,route,seed,base['config']);v=c.run(t,route,seed,best['config'])
                        else:
                            v=c.run(t,route,seed,best['config']);b=c.run(t,route,seed,base['config'])
                        pairs.append(dict(route=route,baseline=b,candidate=v))
                verdict=qualify_pairs(pairs,routes,a.min_gain)
                decision.setdefault('candidates',[]).append(dict(config=best['config'],verdict=verdict))
                if not verdict['qualified']: continue
                dynamic=c.run(t,'balanced',a.seed+999983,best['config'],a.dynamic_replays)
                entry=dict(shape=base['shape'],numerical_contract=a.numerical_contract,config=best['config'],
                           validation='qualified',evidence=dict(metrics=verdict['metrics'],
                           runs=[dict(route=x['route'],baseline=x['baseline']['evidence'],candidate=x['candidate']['evidence']) for x in pairs],
                           dynamic=dynamic['evidence'],dynamic_native_checks=dynamic['native_checks'],
                           arithmetic_fp64='not_run',transport_statuses=dynamic['transport_statuses'],
                           scope='bounded search; synthetic routes plus optional supplied trace'))
                entries.append(entry);decision['adopted']=True;break
            decisions.append(decision)
    except (Exception,KeyboardInterrupt) as exc:
        failure=str(exc) or type(exc).__name__
    finally:
        c.record.update(end=time.time(),decisions=decisions,error=failure,qualified_entries=len(entries));c.save()
        if c.hardware:
            document=make_document(c.hardware,entries,dict(campaign=str(c.root),status='partial' if failure else 'complete',error=failure))
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open('x') as f: json.dump(document,f,indent=2)
        print(json.dumps(dict(output=str(output) if c.hardware else None,qualified_entries=len(entries),error=failure),indent=2))
    if failure: raise SystemExit(2)


if __name__=='__main__': main()
