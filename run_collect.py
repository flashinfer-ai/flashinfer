import subprocess
import re

# Values to replace the last parameter (1)
values = [
    1, 2, 4, 8, 16, 24, 32, 48, 64,
    96, 128, 256, 512, 1024, 1536,
    2048, 3072, 4096
]

base_cmd = (
    "pytest -s "
    "'tests/test_trtllm_cutlass_fused_moe.py::"
    "test_moe_nvfp4[True-True-otype0-wtype0-256-8-256-7168-{}]'"
)

time_pattern = re.compile(r"Elapsed time: ([\d.]+) ms")

results = []

for v in values:
    print(f"Running with last param = {v}")
    cmd = base_cmd.format(v)
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        match = time_pattern.search(output)
        if match:
            elapsed_time = float(match.group(1))
        else:
            elapsed_time = None
            print(f"Warning: Elapsed time not found in output for {v}")
    except subprocess.CalledProcessError as e:
        output = e.output
        elapsed_time = None
        print(f"Error running test for {v}:\n{output}")

    results.append((v, elapsed_time))

# Print results as a table
print("\nResults:")
print(f"{'Value':>6} | {'Time (ms)':>10}")
print("-" * 20)
for val, time in results:
    time_str = f"{time:.2f}" if time is not None else "N/A"
    print(f"{val:6} | {time_str:>10}")

