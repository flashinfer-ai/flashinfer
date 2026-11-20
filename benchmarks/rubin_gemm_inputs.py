# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Optional matched-input fixtures for the Rubin benchmark scripts only.

Fixture storage is packed K-major A/B, linear SFA, R128c4 SFB and U64/G64
weight rows. No production allocator, task or kernel is patched here.
"""

from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from benchmarks.bench_rubin_gemm import count_nonzero


def digest(tensor):
    h = hashlib.sha256()
    for part in tensor.contiguous().view(torch.uint8).flatten().split(64 * 1024**2):
        h.update(memoryview(part.cpu().numpy()))
    return h.hexdigest()


def sf_offsets(rows, cols):
    # R128c4 storage: [row-block, col-block, row%32, row//32%4, col%4].
    r = rows[:, None]
    j = torch.arange(cols, device=rows.device)[None, :]
    return (
        (r // 128 * (cols // 4) + j // 4) * 512
        + (r % 32 * 4 + r % 128 // 32) * 4
        + j % 4
    )


def unpack(packed):
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    return table[torch.stack((packed & 15, packed >> 4), -1).long()].flatten(-2)


def reference_samples(data, shape, norm, *, full, fc1=True):
    m, n, k, e = shape
    experts = list(range(e)) if full else sorted({0, 1, e // 2, e - 1})
    local_rows = (
        torch.arange(m, device="cuda")
        if full
        else torch.tensor([0, 1, 31, 127, 255, 256, 383, 511], device="cuda")
    )
    out_rows, refs = [], []
    for expert in experts:
        rows = local_rows + expert * m
        tokens = data["route"][rows].long()
        a = unpack(data["A"].reshape(-1, k // 2)[tokens])
        a *= (
            data["SFA"]
            .reshape(-1, k // 16)[tokens]
            .contiguous()
            .view(torch.float8_e4m3fn)
            .float()
            .repeat_interleave(16, -1)
        )
        b = unpack(data["B"].reshape(e, n, k // 2)[expert])
        sfb = data["SFB"][
            sf_offsets(torch.arange(n, device="cuda") + expert * n, k // 16)
        ]
        b *= (
            sfb.contiguous().view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
        )
        accum = (a @ b.T).reshape(-1, n // 128, 2, 64)
        up, gate = accum[:, :, 0], accum[:, :, 1]
        if fc1:
            refs.append((up * gate * torch.sigmoid(gate)).flatten(1) * norm)
        else:
            refs.append(torch.cat((up.flatten(1), gate.flatten(1)), dim=1))
        out_rows.append(rows)
    return torch.cat(out_rows), torch.cat(refs)


def check_output(c, sf, rows, ref, n_out):
    q = unpack(c.reshape(-1, n_out // 2)[rows])
    scales = (
        sf[sf_offsets(rows, n_out // 16)].contiguous().view(torch.float8_e4m3fn).float()
    )
    blocks = ref.reshape(ref.shape[0], -1, 16)
    ideal_scales = blocks.abs().amax(-1) / 6
    assert float(ideal_scales.max()) < 448, (
        "Output scale saturation invalidates comparison"
    )
    ref_scales = ideal_scales.to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(scales, ref_scales, atol=0, rtol=0)
    scaled = blocks / ref_scales.unsqueeze(-1)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    distances = (scaled.abs().unsqueeze(-1) - levels).abs()
    indices = distances.argmin(-1)
    # E2M1 uses round-to-nearest-even at exact midpoints.
    for j in (1, 3, 5):
        midpoint = (levels[j] + levels[j + 1]) * 0.5
        indices = torch.where(scaled.abs() == midpoint, j + 1, indices)
    expected = (levels[indices] * torch.sign(scaled)).flatten(1)
    mismatch = q != expected
    rate = float(mismatch.float().mean())
    assert rate < 0.0001, (rate, int(mismatch.sum()))
    boundary_error = 0.0
    if bool(mismatch.any()):
        # Permit only round-boundary differences from reciprocal/activation
        # approximations, never arbitrary errors hidden by a pass percentage.
        midpoint = ((q.abs() + expected.abs()) * 0.5)[mismatch]
        boundary_error = float(
            (scaled.flatten(1).abs()[mismatch] - midpoint).abs().max()
        )
        assert boundary_error < 2e-5, boundary_error
    assert int(torch.count_nonzero(q)) > 0
    return dict(
        elements=q.numel(),
        differing_fp4_values=int(mismatch.sum()),
        mismatch_fraction=rate,
        max_round_boundary_error=boundary_error,
        scales_exact=True,
        max_output_scale=float(scales.max()),
        nonzero_values=int(torch.count_nonzero(q)),
    )


class MatchedInputs:
    def __init__(self, fixture_dir, shape, *, fc1):
        self.path = Path(fixture_dir)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        m, n, k, e = self.manifest["shape"]
        assert (e, n, k) == (
            shape["num_experts"],
            shape["problem_n"],
            shape["problem_k"],
        )
        assert m * e == shape["num_tokens"] * shape["top_k"]
        self.shape = (m, n, k, e)
        self.fc1 = fc1
        self.data = {}
        self.audit = {}
        for name in ("A", "B", "SFA", "SFB", "route"):
            array = np.load(
                self.path / (name + ".npy"), mmap_mode="c", allow_pickle=False
            )
            expected_dtype = np.int32 if name == "route" else np.uint8
            assert array.dtype == expected_dtype
            fingerprint = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
            record = self.manifest["tensors"][name]
            assert (
                fingerprint == record["sha256"] and array.nbytes == record["bytes"]
            ), name
            tensor = torch.from_numpy(array).to(device="cuda")
            nonzero = count_nonzero(tensor)
            assert nonzero > 0, name
            self.data[name] = tensor
            self.audit[name] = dict(
                sha256=fingerprint, bytes=array.nbytes, nonzero=nonzero
            )
        assert self.data["A"].numel() == shape["num_tokens"] * k // 2
        assert self.data["SFA"].numel() == shape["num_tokens"] * k // 16
        assert self.data["B"].numel() == e * n * k // 2
        assert self.data["SFB"].numel() == e * n * k // 16
        assert self.data["route"].numel() == e * m
        assert int(self.data["route"].min()) >= 0
        assert int(self.data["route"].max()) < shape["num_tokens"]
        self.scale_c = 2.0**-15 if fc1 else 1.0
        torch.backends.cuda.matmul.allow_tf32 = False
        self.rows, self.reference = reference_samples(
            self.data, self.shape, self.scale_c, full=False, fc1=fc1
        )

    def build_io(self, shape, options):
        from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import SfLayout
        from flashinfer.prims_ts.batched_gemm.batched_gemm_quant import build_sf_buffer

        m, n, k, e = self.shape
        data = self.data
        # Restore the logical U-half/G-half ordering. The production runner
        # applies its configured weight/SF packing for either output type.
        source_rows = (
            torch.arange(n, device="cuda").reshape(-1, 2, 64).permute(1, 0, 2).flatten()
        )
        weights = data["B"].reshape(e, n, k // 2).index_select(1, source_rows)
        sf_weights = runner._permute_mma_sf_rows(
            data["SFB"],
            row_source_indices=source_rows,
            logical_cols=k,
            layout=int(SfLayout.R128c4),
            num_experts=e,
            sf_vec_size=16,
        )
        original_layout = runner._make_token_layout
        layout_holder = []

        def token_layout(**kwargs):
            layout = original_layout(**kwargs)
            assert layout.tokens_per_expert == [m] * e
            routes = data["route"].reshape(e, m).cpu().tolist()
            padded = [0] * layout.total_padded_tokens
            for expert, start in enumerate(layout.padded_starts[:-1]):
                padded[start : start + m] = routes[expert]
            layout = replace(layout, expanded_to_token=padded)
            layout_holder.append(layout)
            return layout

        def random_tensor(dims, **kwargs):
            return weights if len(dims) == 3 else data["A"].reshape(-1, k // 2)

        def mma_scales(**kwargs):
            if kwargs["num_experts"] == e:
                return sf_weights
            assert kwargs["num_experts"] == 1
            layout = layout_holder[0]
            expanded = runner._expand_bf16_activations(
                data["SFA"].reshape(-1, k // 16), layout
            )
            return build_sf_buffer(
                expanded,
                total_bytes=layout.total_padded_tokens * k // 16,
                layout=kwargs["layout"],
                data_blk_cols=k // 16,
                device="cuda",
            )

        with ExitStack() as stack:
            for name, function in (
                ("_make_token_layout", token_layout),
                ("_random_quantized_tensor", random_tensor),
                ("_create_sf_linear", lambda **kwargs: data["SFA"]),
                ("_create_sf_tensor_mma_layout", mma_scales),
            ):
                stack.enter_context(patch.object(runner, name, function))
            io = runner._build_launch_io(
                **shape,
                **options,
                routing="balanced_round_robin",
                scale_c_value=self.scale_c,
                scale_gate_value=1.0,
            )
        starts = torch.tensor(io["token_layout"].padded_starts[:-1], device="cuda")
        self.output_rows = starts[self.rows // m] + self.rows % m
        return io

    def check(self, buffers):
        n = self.shape[1]
        if self.fc1:
            result = check_output(
                buffers["C"], buffers["SFC"], self.output_rows, self.reference, n // 2
            )
            # The archived integer fixtures have no ambiguous rounding cases.
            assert result["differing_fp4_values"] == 0
            return result
        actual = buffers["C"].view(torch.bfloat16).reshape(-1, n)[self.output_rows]
        torch.testing.assert_close(
            actual, self.reference.to(torch.bfloat16), atol=0, rtol=0
        )
        return dict(elements=actual.numel(), bf16_exact=True)
