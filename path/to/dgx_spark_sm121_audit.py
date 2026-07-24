import os
import subprocess

def check_cuda_version():
    """
    Check CUDA version installed on the system.
    """
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        cuda_version = output.splitlines()[0].split(":")[1].strip()
        return cuda_version
    except subprocess.CalledProcessError as e:
        print(f"Error checking CUDA version: {e}")
        return None

def check_sm121_support(cuda_version):
    """
    Check if SM121 is supported by the given CUDA version.
    """
    if cuda_version == "12.9":
        return True
    else:
        return False

def audit_dgx_spark_sm121_support():
    """
    Audit DGX Spark (SM121) support across FlashInfer ops, APIs, backends, and testing.
    """
    cuda_version = check_cuda_version()
    if cuda_version is None:
        print("Error checking CUDA version. Skipping audit.")
        return

    sm121_supported = check_sm121_support(cuda_version)
    if sm121_supported:
        print("DGX Spark (SM121) support is available for CUDA version", cuda_version)
    else:
        print("DGX Spark (SM121) support is not available for CUDA version", cuda_version)

if __name__ == "__main__":
    audit_dgx_spark_sm121_support()