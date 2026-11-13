"""Run the usual MegaMoE CLI against the independent native QMMA reference.

Example: torchrun --standalone --nproc_per_node=4 -m
moe_sm120_nvfp4_split.check_native_reference [normal mega_runner arguments].
The normal runner remains available for the separate Torch arithmetic check.
"""
import json
import torch
from . import mega_runner as runner
from .combine_reference import byte_difference, numerical_comparison
from .combine_validation import rank_local_staging_report


def main():
    compute = runner.compute_megamoe_reference_nvfp4
    def native_reference(**kwargs):
        return compute(**kwargs, gemm_backend="native_qmma")
    runner.compute_megamoe_reference_nvfp4 = native_reference

    def validate(self, *, force=False):
        if self.misc.skip_ref_check and not force:
            return
        expected = (self.rank_local_output_ref if self.rank_local_combine
                    else self.output_tensor_ref)
        if expected is None:
            raise RuntimeError("native reference must be computed before validation")
        diff = byte_difference(self.output_tensor, expected)
        numeric = numerical_comparison(self.output_tensor, expected)
        transport = (rank_local_staging_report(self) if self.rank_local_combine
                     else self._validate_token_back_exact())
        report = dict(
            reference="independent_SM120_QMMA", byte_differences=diff,
            comparison=numeric, staging_transport=transport,
            independent_numerical_accuracy="NOT_CHECKED; run mega_runner separately",
        )
        self.last_combine_validation_report = report
        print(f"[rank {self.rank}] NATIVE_QMMA_VALIDATION " + json.dumps(report), flush=True)
        failure = torch.tensor(
            [int(diff != 0 or numeric["nonfinite_elements"] != 0
                 or transport["status"] == "FAIL")],
            device=self.output_tensor.device, dtype=torch.int32,
        )
        agreed = [torch.empty_like(failure) for _ in range(self.world_size)]
        if self.world_size == 1:
            agreed[0].copy_(failure)
        else:
            self._ep_all_gather(agreed, failure)
        if any(bool(x.item()) for x in agreed):
            raise AssertionError("native QMMA implementation/reference mismatch")

    runner.MegaMoETester.validate = validate
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runner.main()


if __name__ == "__main__":
    main()
