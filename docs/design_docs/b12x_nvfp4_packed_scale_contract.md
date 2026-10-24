# B12X packed NVFP4 scale contract

ModelOpt NVFP4 weights contain E2M1 values, nonnegative E4M3 block scales,
and FP32 per-expert global scales. Native storage retains every finite
nonnegative E4M3 block-scale value. W4A16 MMA-packed storage has a narrower
scale contract.

For BF16 execution, preparation chooses one power-of-two normalization
factor `f` per projection across all experts. The factor makes the largest
normalized block scale no greater than 448. Preparation encodes `s*f*128`
as FP16 bits 14 through 7 and compensates the global scale by `1/f`.
Nonzero scales with `s*f < 1/64` are discarded. These are the subnormal
values of the normalized E4M3 scale, corresponding to raw E4M3 codes 1–7
when `f=1`. Zero remains zero, and all retained scales are represented
exactly. FP16 preparation uses `f=1` and the same discard boundary.

Hybrid A4 execution consumes these same packed scales. Compressed scale
storage preserves the encoded packed bytes exactly; neither CSF expansion
nor hybrid A4 reconstruction restores discarded values. A lossless native
representation is required when preserving such subnormal scales matters.

The boundary test in `test_w4a16_packed_format.py` covers all finite
nonnegative E4M3 codes within the normalized range at factors 1, 2, 4, and 8.
It checks exact reconstruction of retained values and the respective 7, 3,
1, and 0 discarded positive source codes.

## Checkpoint audit

An audit of `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD`
metadata revision `dec48abd33efa73c3bb7c95b74eee10cad34f9be` examined all 42
routed layers, all 288 experts, both TP2 ranks, and both projection groups.
The config SHA-256 was
`d9d0b32d0fa38d0cfc7ac162670db17fe16fe02404e847e6b2aa07d71efd67f1`;
the safetensors index SHA-256 was
`568c770e4a083ab53c3d74deab348d541724f23ec9637d83513ecb8fa4a7ef73`.
The audit decoded the checkpoint's byte-window CSF fixed streams and
exception records before applying the preparation rule.

All 19,025,362,944 scales were retained. Every projection/rank normalization
factor was 1, the smallest positive scale was 0.5625, and the largest was
448. Packed scale preparation is therefore lossless for those audited
routed-expert weights. This result does not cover other checkpoints,
checkpoint revisions, or uninspected quantized tensor families.
