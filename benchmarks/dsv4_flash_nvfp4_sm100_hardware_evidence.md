# B200 routed NVFP4 decode: hardware evidence

The [qualified public-export results](dsv4_flash_nvfp4_sm100_results.md) cover
every integer T=1..32 and retain **32/32 wins** against
`flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`: geometric mean
**1.025160423562×**, minimum **T14 1.006047396915×**
(161.422111 /162.398295 µs). The matched source-schedule result is
1.025096697119×. The report includes every public/source difference, including
17 shapes where the export is slightly slower than its source schedule.

Geometry is B200/sm_100a, H=4096, I=2048, 256 routed experts, top-k=6 and clamped
SwiGLU limit 10. The following profiles and prototypes concern the generated
CUDA source schedule in separate measured cohorts. They are **not a new
qualification or profile of the public export**, and their timings must not
be combined with the public all32 cohort to infer a causal gain.

## Large-T traffic and memory service

Each selected expert contributes 9 MiB of FC1 weights/block scales and
4.5 MiB for FC2. T16 has 96 routes selecting 77 distinct experts; T32 has
192 routes selecting 135. These are logical distinct-weight footprints,
not automatically HBM traffic.

| Standalone source-stage profile | Duration µs | Read bytes / logical footprint | DRAM TB/s | Sustained peak utilization |
|---|---:|---:|---:|---:|
| T16 FC1 |108.576|1.000365|6.729742|87.74808%|
| T16 FC2 |58.432|1.002064|6.242248|81.467485%|
| T32 FC1 |184.704|1.000379|6.922561|90.25%|
| T32 FC2 |96.000|1.000665|6.657651|86.81%|

The logical FC1/FC2 footprints are 726,663,168 /363,331,584 bytes at T16 and
1,274,019,840 /637,009,920 bytes at T32. Observed read excess is at most 0.21%.
T16 partition-normalized utilization is nearly uniform. These observations
support strong memory service with little redundant weight traffic in the
measured fixtures.

HBM clocks were approximately 3.99 GHz. Observed SM clocks were 1.814 /1.845 GHz
for T16 FC1/FC2 and 1.801 /1.831 GHz for T32. “Sustained peak” is the profiler's
clock-dependent counter reference, not a percentage of a universal complete-
call limit. Standalone profiler durations include their own replay conditions.

A separate read-only load/checksum calibration reached 6.046 /6.442 /6.729 TB/s
at 1.090 /1.911 /3.822 GB. Increasing the grid from eight to 32 blocks per SM
count improved only 1.758% /1.747% /1.217%. This is attainable service for that
load calibration, not an upper bound transferable to TMA/MMA schedules with
different instructions, residency, clocks and cache reuse.

## Three paired T1 scheduling probes

Each prototype was compared with its unchanged selected source control and
FlashInfer using six balanced three-arm captures, seven complete calls per
capture, CUPTI and cold L2 without fallback. Every required projection,
activation/quantization and finalization operation remained in the timed call.
Strict BF16 `atol=rtol=0.01` and bitexact consumed-intermediate checks passed.

| Prototype | Prototype / control / FlashInfer µs | Paired control/prototype GM | Descriptive 95% capture interval | Wins / ties / losses |
|---|---:|---:|---:|---:|
| One 32 KiB weight TMA instead of two 16 KiB requests |25.008 /25.104 /29.808|1.000887825|[0.994154767,1.005771522]|4 /1 /1|
| Weight-loader bypass of the TMEM allocation rendezvous |25.536 /25.168 /29.024|0.982076684|[0.974264352,0.988491245]|0 /0 /6|
| Preloaded-weight finalizer: 64-thread CTAs and 32 feature tiles |25.184 /25.0395 /29.4075|0.990115579|[0.984049833,0.996166166]|0 /0 /6|

Latencies are medians of capture medians. Ratios and exact bootstrap intervals
use the six paired capture-level log ratios, not 42 independent samples. They
describe these individual sessions. The first probe did not establish a gain;
the other two regressed in both order strata. None is selected in the public
implementation. The finalizer remap retained the six-route FMA order and BF16
rounding; its completion tail was 1.344 µs versus 1.312 µs control.

Physical turnaround /worker time for these three units was respectively
242.324485 /125.547575 s, 237.133255 /115.634397 s and
224.142534 /102.161654 s. These include orchestration and are distinct from
the microsecond kernel measurements.

## Interpretation and limits

The profiles and closed scheduling probes support a measured practical
frontier for the explored exact-semantic source schedules. They do not prove
an absolute optimum or establish a numerical whole-call hardware ceiling.
The selected public export remains qualified by its separate all32 report.

PDL permits overlapping kernel spans that include dependency waits. Stage
spans and completion tails cannot be added or independently subtracted as
achievable savings. T1's 81 MiB selected payload fits within the measured L2;
large-T bandwidth figures therefore cannot be assigned to it. Device-level
traffic sampling also does not establish ownership of every transferred byte.

No new benchmark, profile or public qualification was run to assemble this
summary. The rejected prototypes did not trigger new sanitizer runs.
Previously reported separate synccheck and racecheck 20-second timeout skips
remain skips, not passes. Broader validation remains incomplete.
