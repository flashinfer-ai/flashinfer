"""Synthetic CLI orchestration tests; no GPU performance claims or fixtures."""
import contextlib
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .test_tuning_config import shape
from .heuristic import _select_cached
from . import tune_selector as cli
from .tuning_config import implementation_fingerprint, load_tuning_config


class TuningCliTest(unittest.TestCase):
    def run_synthetic(self, directory, fail_at=None):
        s=shape(16384);base=_select_cached(s,None,'owner_bf16',None)
        candidate=replace(base,k2_stages=3)
        calls=[]
        class FakeCampaign:
            def __init__(self,args):
                self.root=Path(directory)/'synthetic_campaign';self.root.mkdir()
                self.hardware='synthetic-device';self.record={}
            def save(self):
                (self.root/'campaign.json').write_text(json.dumps(dict(self.record,synthetic=True)))
            def run(self,tokens,route,seed,config=None,dynamic=0):
                calls.append((tokens,route,seed,dynamic))
                if fail_at and len(calls)==fail_at:raise TimeoutError('synthetic budget exhaustion')
                config=config or asdict(base)
                latency=97. if config['k2_stages']==3 else 100.
                return dict(status='PASS',shape=asdict(s),config=config,hardware=self.hardware,
                    implementation_sha256=implementation_fingerprint(),contract='owner_bf16',
                    fixtures=[str(seed)],outputs=['synthetic-equal-output'],p50_us=latency,p90_us=latency+10,
                    native_checks=404 if dynamic else 4,transport_statuses=['PASS'],
                    evidence=str(self.root/f'synthetic_{len(calls)}'),
                    spec=dict(seed=seed,dynamic=dynamic))
        output=Path(directory)/'synthetic.json'
        args=['--num_topk','8','--hidden','4096','--intermediate','4096',
              '--num_total_experts','288','--tokens','16384','--max_candidates','2',
              '--numerical_contract','owner_bf16','--output',str(output)]
        exit_code=0
        with patch.object(cli,'Campaign',FakeCampaign),patch.object(cli,'candidate_configs',return_value=[base,candidate]),contextlib.redirect_stdout(io.StringIO()):
            try:cli.main(args)
            except SystemExit as exc:exit_code=exc.code
        return json.loads(output.read_text()),calls,exit_code,output

    def test_complete_positive_path_exports_after_holdouts_and_dynamic(self):
        with tempfile.TemporaryDirectory() as temp:
            document,calls,code,path=self.run_synthetic(temp)
            self.assertEqual(code,0)
            self.assertEqual(len(document['entries']),1)
            self.assertEqual(len(calls),15)  # baseline, screen, 12 holdouts, dynamic
            self.assertEqual(calls[-1][-1],100)
            self.assertEqual(document['entries'][0]['validation'],'qualified')
            self.assertFalse(load_tuning_config(path).error)

    def test_budget_expiry_cannot_export_partial_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            document,calls,code,path=self.run_synthetic(temp,fail_at=6)
            self.assertEqual(code,2)
            self.assertEqual(document['entries'],[])
            self.assertEqual(document['provenance']['status'],'partial')


if __name__=='__main__':unittest.main()
