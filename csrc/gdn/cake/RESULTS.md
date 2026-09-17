# GDN prefill validation results

Validation: PASS on B200 and B300.

Manifest SHA-256: `836fa2ab739f31c9c037f2277b079a4593947363408caa350972be88d1d2517a`.

Each architecture covers 1713 original inputs plus one separately validated caller-normalized fixture, including 27 expected rejections. The unsupported original fixture invocation remains rejected.

Each architecture has 137 passing performance rows: 47 fresh measurements and 90 historical measurements bound through unchanged CUDA, host code, ABI, headers, compile options, routes and public API bytes. Correctness uses 83 fresh rows and 1631 retained rows. Historical seals keep their original identities.

The three real public API arms use CuTe revision `8044d94bf9acc5369857baf88d28906bb32bf264`, the exported backend, and the previous production backend. Each row uses three counterbalanced groups, 100 ms warmup and 1000 ms measurement per arm, CUDA Graph replay, CUPTI, cold L2 and clocks at least 1900 MHz. CuTe/exported must be at least 1.0, incumbent/exported at least 0.98; direction disagreement and endpoint drift must each be at most 5%.

The original B200 B4/S2048, Q/K/V heads 16/16/32, D128, BF16 input and FP32-state CLI repeats measured 0.0748635/0.0748000 ms. Both passed refcheck and the 0.0768 ms limit using CUDA Graph/CUPTI. These measurements remain applicable through exact generated-artifact equality. The CLI logs contain no paired CuTe medians.

The separate five-shape B200 matrix passed output and state checks before and after timing. Its CuTe baseline is public revision `85da10187476d1a33e560fddbdbf7447d4937d31`; it does not supply the pinned baseline portfolio measurements. No neighbor speed floor was added. Default prefill dispatch remains CuTe.

Sealed results are retained; a complete archive of all historical raw timing arrays is not claimed.

Physical turnaround through qualification: 4.86 hours. Kernel medians exclude setup, compilation, queueing and validation.

## Measured comparisons

| ISA | Minimum CuTe/exported | Minimum incumbent/exported |
|---|---:|---:|
| sm_100a | 1.000987 | 1.267218 |
| sm_103a | 1.008746 | 1.290698 |

### Separate five-shape B200 matrix

CuTe baseline: public revision `85da10187476d1a33e560fddbdbf7447d4937d31`.

| Shape | Exported medians, ms | CuTe medians, ms |
|---|---:|---:|
| B2-S2048 | 0.069056 / 0.068928 | 0.071840 / 0.071999 |
| B3-S2048 | 0.072704 / 0.072672 | 0.073984 / 0.073920 |
| B4-S2048 | 0.074719 / 0.074912 | 0.076128 / 0.075936 |
| B4-S2049 | 0.078112 / 0.078208 | 0.079712 / 0.079552 |
| B4-S128 | 0.015776 / 0.015744 | 0.016672 / 0.016672 |

## Integration and sanitizer checks

The registered source correctness and benchmark slices passed on B200 and B300.
The public API integration test files passed on both architectures.
Synccheck and racecheck each reached the enforced 20-second process timeout on
both architectures and are recorded as SKIP. No sanitizer error was reported
before those timeouts; a timeout is not a pass.

The measured public integration uses revision
`85da10187476d1a33e560fddbdbf7447d4937d31` with this export overlaid. The publication
is based on `24c30bddbe1628e225458380b476de756b506f69`. The GDN semantic wrappers,
JIT loader implementation, common header, and GDN tests are unchanged between
those revisions; this update changes the loader's manifest checksum. General
JIT environment changes are outside the measured checkout. The existing GPU
measurements qualify the exported artifacts; they are not a new full-checkout
GPU run at the publication head.
