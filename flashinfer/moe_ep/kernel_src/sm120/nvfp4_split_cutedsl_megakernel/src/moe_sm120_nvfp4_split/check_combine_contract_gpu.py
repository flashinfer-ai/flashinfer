"""Diagnostic EP regression wrapper; not a performance measurement.

Run via torchrun with the usual mega_runner arguments. Environment:
MEGA_CONTRACT_DYNAMIC=1: audit every one of 100 changing-route Green replays;
  independent expert reference at replays 0/49/99, transport on every replay.
MEGA_EXPECT_ORDINARY_FAILURE=1: negative test, require strict ordinary failure
  while grouped and staging checks pass on every rank.
MEGA_CONTRACT_OUTPUT_DIR: per-rank JSON evidence (optional).
"""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from moe_sm120_nvfp4_split import api, mega_runner as runner
from moe_sm120_nvfp4_split.combine_validation import rank_local_staging_report

dynamic = os.environ.get("MEGA_CONTRACT_DYNAMIC") == "1"
expect_failure = os.environ.get("MEGA_EXPECT_ORDINARY_FAILURE") == "1"
out = Path(os.environ.get("MEGA_CONTRACT_OUTPUT_DIR", os.environ.get("SOL_OUT", ".")))
out.mkdir(parents=True, exist_ok=True)
records = []
original_update = runner.MegaMoETester._update_runtime_route
original_run_kernel = runner.MegaMoETester.run_kernel
original_validate = runner.MegaMoETester.validate


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def validate(self, **kwargs):
    caught = False
    try:
        original_validate(self, **kwargs)
    except AssertionError as exc:
        if not expect_failure:
            raise
        report = self.last_combine_validation_report
        assert "ordinary_topk" in str(exc)
        assert report["ordinary_topk"]["status"] == "FAIL"
        assert report["grouped_implementation"]["status"] == "PASS"
        assert report["staging_transport"]["status"] == "PASS"
        caught = True
    if expect_failure and not caught:
        raise AssertionError("strict ordinary top-k failure was expected but not raised")
    record = dict(
        kind="arithmetic_and_transport", replay=getattr(self, "_contract_replay", None),
        rank=self.rank, abi=api.KERNEL_CACHE_ABI, runner=runner.__file__,
        report=self.last_combine_validation_report, expected_strict_failure=caught,
        output_sha=digest(self.output_tensor), route_sha=digest(self.my_topk_idx),
    )
    records.append(record)
    (out / f"validation_rank{self.rank}.json").write_text(json.dumps(records, indent=2) + "\n")


def capture(self):
    index = getattr(self, "_contract_replay", -1)
    if index < 0 or index == getattr(self, "_contract_captured", -1):
        return
    torch.cuda.synchronize()
    routes = [torch.empty_like(self.my_topk_idx) for _ in range(self.world_size)]
    self._ep_all_gather(routes, self.my_topk_idx)
    self._global_topk_idx.copy_(torch.stack(routes))
    transport = rank_local_staging_report(self)
    epoch = transport["epoch"]
    # Track actual graph replay, not repeated validation of the same output.
    if epoch <= getattr(self, "_contract_previous_epoch", 0):
        transport["status"] = "FAIL"
        transport["errors"]["epoch_not_advanced"] = 1
    self._contract_previous_epoch = epoch
    self._agree_combine_report({"staging_transport": transport})
    owners = self.my_topk_idx // self.problem.num_experts_per_rank
    counts = torch.stack([(owners == o).sum(1) for o in range(self.world_size)], 1)
    records.append(dict(
        kind="dynamic_transport", replay=index, epoch=epoch, report=transport,
        output_sha=digest(self.output_tensor), route_sha=digest(self.my_topk_idx),
        contributor_counts=[int((counts == c).sum()) for c in range(self.problem.num_topk + 1)],
    ))
    if index in (0, 49, 99):
        self.compute_reference(force=True)
        self.validate(force=True)
    self._contract_captured = index


def update(self, replay):
    capture(self)
    result = original_update(self, replay)
    self._contract_replay = replay
    return result


def run_kernel(self):
    result = original_run_kernel(self)
    capture(self)
    replays = [r for r in records if r["kind"] == "dynamic_transport"]
    assert len(replays) == 100 and len({r["route_sha"] for r in replays}) > 1
    (out / f"dynamic_rank{self.rank}.json").write_text(json.dumps(records, indent=2) + "\n")
    print("DYNAMIC_CONTRACT100_COMPLETE", self.rank, flush=True)
    return result


runner.MegaMoETester.validate = validate
if dynamic:
    runner.MegaMoETester._update_runtime_route = update
    runner.MegaMoETester.run_kernel = run_kernel
if __name__ == "__main__":
    runner.main()
