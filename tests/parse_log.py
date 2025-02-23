import numpy as np

# Create 64x64 array to store s_frag values
s_frag_array = np.zeros((64, 64), dtype=int)

# Parse the log file and populate array
with open("LOG", "r") as f:
    for line in f:
        if line.startswith("q_idx"):
            # Extract q_idx, kv_idx and s_frag values
            parts = line.split()
            q = int(parts[1])
            kv = int(parts[3])
            s = int(float(parts[5]))
            s_frag_array[q][kv] = s

# Print the array
print("64x64 array of s_frag values:")
print(s_frag_array[0:64, 0:64])
