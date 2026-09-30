# Packed NVFP4 FC1 scheduling — measured results

Fresh public qualification passed all 32 shapes. Hardware-limit evidence remains incomplete.

The packed FC1 schedule uses four-column stripes and removes redundant startup writes to activation scale-factor shared memory after complete producer coverage was verified. Clamp placement, projection arithmetic, routing, scale handling, quantization, rounding, synchronization and the public ABI retain their existing semantics. One generated packed-FC1 device unit changes; the other 12 device units and 13-module dispatch remain unchanged. Direct dispatch for T1–9 is unchanged, so timing variation there is not credited to this optimization.

Target: NVIDIA B200 (sm_100a), every integer T=1..32, H=4096, I=2048, E=256 routed experts, top-k=6 and swiglu_limit=10.0. Output is BF16; the shared expert is outside this routed-expert kernel. The named baseline is `flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`. All three arms use identical matched physical NVFP4 inputs, routes, scales and equivalent clamped-SwiGLU parameters. These are synthetic matched fixtures, not full actual-checkpoint pipeline performance measurements.

## Validation and timing

- 112 CPU tests passed. These test results are reused for the identical executable sources; only this result document is added after qualification.
- All 32 strict BF16 checks passed with atol=rtol=0.01, including source-bitwise eager and graph output comparisons.
- Seven public API/graph cases and eight routing cases passed.
- Six balanced complete-call captures per arm and shape used CUPTI with cold L2; all 576 timing postchecks passed. The timing boundary includes every required per-call pipeline step.
- Synccheck and racecheck for the corresponding source schedule were separately skipped after the mandatory 20-second process-group deadline, with no reported errors. These skips are not passes; no unchanged sanitizer check was repeated.

Public geometric-mean speedup is **1.024004388551×**; matched source control is **1.024027935471×**. Every public shape strictly beats FlashInfer, with no aggregate ties or regressions. Minimum: T15, 169.796310796 versus 170.639060655 µs, **1.004963298995×**. Maximum: T1, **1.173878999853×**.

Public is slower than its matched source control at 18 shapes: 1, 5, 6, 7, 10, 14, 15, 18, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32. Every comparison remains in the table. Capture-level wins/ties/losses are public/FI (192, 0, 0), source/FI (192, 0, 0) and source/public (86, 4, 102). Six balanced captures provide descriptive variation, not a confidence interval or independent randomized trials.

Successful measured-shard cohort physical turnaround: **4301.582590 seconds**, earliest selected shard submission to latest completion including scheduling gaps. Aggregate wrapper worker time: **5508.653243 seconds**. These spans exclude preparation/JIT, prior argument-only failed attempts and later CPU reductions, and are distinct from measured GPU latency.

Source-only paired and default registered measurements are separate cohorts; their results are not substituted for this exported artifact's fresh qualification. Actual source and public runtime-library identities remain distinct. Broader generator validation and hardware-limit evidence remain incomplete; the data does not establish a hardware ceiling.

| T | Source µs | Public µs | FlashInfer µs | Public speedup | Source speedup | Source/public |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.082184060 | 25.082313439 | 29.443601014 | 1.173878999853× | 1.173885054986× | 0.999994841801× |
| 2 | 41.359722292 | 41.253094663 | 44.943595440 | 1.089459974017× | 1.086651286537× | 1.002584718313× |
| 3 | 56.021097712 | 55.876921727 | 60.047497462 | 1.074638609398× | 1.071872917772× | 1.002580242098× |
| 4 | 68.063287879 | 67.897953899 | 72.682319831 | 1.070464066393× | 1.067863779373× | 1.002435036256× |
| 5 | 79.967636709 | 79.989133047 | 83.146179674 | 1.039468444106× | 1.039747866705× | 0.999731259271× |
| 6 | 91.650221812 | 91.829289334 | 93.535519155 | 1.018580453290× | 1.020570570431× | 0.998049995563× |
| 7 | 99.935489806 | 100.122145413 | 102.868614154 | 1.027431181473× | 1.029350177339× | 0.998135721052× |
| 8 | 108.729417954 | 108.585464442 | 110.270406144 | 1.015517193854× | 1.014172688663× | 1.001325716227× |
| 9 | 120.446985838 | 120.324137613 | 121.886551045 | 1.012985037436× | 1.011951857464× | 1.001020977396× |
| 10 | 127.140628542 | 127.199271787 | 128.074296527 | 1.006879164691× | 1.007343584780× | 0.999538965556× |
| 11 | 135.898126919 | 135.807138988 | 136.740600863 | 1.006873437449× | 1.006199305043× | 1.000669979002× |
| 12 | 145.887936818 | 145.711412551 | 146.724972716 | 1.006955942210× | 1.005737526462× | 1.001211464936× |
| 13 | 155.284311717 | 155.028981502 | 156.756606159 | 1.011143881877× | 1.009481282597× | 1.001646983762× |
| 14 | 161.567269831 | 161.615260856 | 162.505815739 | 1.005510339048× | 1.005809010133× | 0.999703053879× |
| 15 | 169.657819769 | 169.796310796 | 170.639060655 | 1.004963298995× | 1.005783646676× | 0.999184369636× |
| 16 | 179.193709413 | 179.183104897 | 180.313460178 | 1.006308380924× | 1.006248828536× | 1.000059182566× |
| 17 | 184.815291061 | 184.804446749 | 187.609564448 | 1.015178843087× | 1.015119275960× | 1.000058679929× |
| 18 | 189.460139775 | 189.497641976 | 192.127022141 | 1.013875529730× | 1.014076218715× | 0.999802096745× |
| 19 | 194.735492233 | 194.505964366 | 197.567285071 | 1.015738955438× | 1.014541739699× | 1.001180055677× |
| 20 | 202.815326521 | 202.553473587 | 205.311086620 | 1.013614247063× | 1.012305579373× | 1.001292759535× |
| 21 | 209.913599656 | 210.223977468 | 213.514099515 | 1.015650555595× | 1.017152294396× | 0.998523585101× |
| 22 | 221.471313177 | 221.620456458 | 224.713554532 | 1.013956735417× | 1.014639554479× | 0.999327032876× |
| 23 | 222.196643048 | 221.945468884 | 226.356199025 | 1.019873035314× | 1.018720156705× | 1.001131693136× |
| 24 | 233.957376082 | 234.269802755 | 237.336208575 | 1.013089206481× | 1.014442085775× | 0.998666380947× |
| 25 | 241.565469779 | 241.682620281 | 244.695894420 | 1.012467897509× | 1.012958907759× | 0.999515271305× |
| 26 | 247.395988000 | 247.790990529 | 250.894834953 | 1.012526058421× | 1.014142698843× | 0.998405904392× |
| 27 | 255.716483691 | 256.292460329 | 259.300389634 | 1.011736316009× | 1.014015154172× | 0.997752658672× |
| 28 | 263.535448177 | 264.111823853 | 267.663078381 | 1.013446026291× | 1.015662523704× | 0.997817683177× |
| 29 | 272.036155012 | 272.035965994 | 274.964220506 | 1.010764218258× | 1.010763515953× | 1.000000694826× |
| 30 | 281.241690801 | 281.407324254 | 284.073751110 | 1.009475328559× | 1.010069845268× | 0.999411410298× |
| 31 | 285.321145833 | 285.737484629 | 288.451993332 | 1.009500009097× | 1.010973065069× | 0.998542932524× |
| 32 | 289.620324691 | 289.622567532 | 293.279259183 | 1.012625713809× | 1.012633555658× | 0.999992255985× |
