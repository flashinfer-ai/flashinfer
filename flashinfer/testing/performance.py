"""
Utility functions for performance analysis and optimization.

This module provides helpers for common performance tasks:
- GPU capability detection
- Optimal batch size calculation
- Memory estimation
- Performance metric calculations
"""

import torch
from typing import Dict, Any, Optional, Tuple


def get_gpu_info() -> Dict[str, Any]:
    """
    Get comprehensive GPU information.
    
    Returns:
        Dictionary with GPU properties
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        return {
            "available": True,
            "name": torch.cuda.get_device_name(device),
            "compute_capability": (props.major, props.minor),
            "total_memory_gb": props.total_memory / (1024 ** 3),
            "multi_processor_count": props.multi_processor_count,
            "max_threads_per_multi_processor": props.max_threads_per_multi_processor,
            "cuda_version": torch.version.cuda,
            "supports_fp16": True,
            "supports_bf16": props.major >= 8,
            "supports_fp8": props.major >= 9,
            "supports_fp4": props.major >= 10 or props.major == 12,
        }
    except Exception as e:
        return {"available": True, "error": str(e)}


def estimate_kv_cache_memory(
    num_layers: int,
    batch_size: int,
    max_seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16
) -> Dict[str, float]:
    """
    Estimate KV cache memory requirements.
    
    Args:
        num_layers: Number of transformer layers
        batch_size: Maximum batch size
        max_seq_len: Maximum sequence length
        num_kv_heads: Number of KV heads per layer
        head_dim: Dimension per head
        dtype: Data type for KV cache
    
    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_element = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    if dtype == torch.float8_e4m3fn:
        bytes_per_element = 1
    
    # Memory per layer per request
    memory_per_layer_per_request = (
        max_seq_len * num_kv_heads * head_dim * 2  # K and V
        * bytes_per_element
    )
    
    total_memory_bytes = (
        memory_per_layer_per_request * num_layers * batch_size
    )
    
    return {
        "total_gb": total_memory_bytes / (1024 ** 3),
        "per_layer_gb": (memory_per_layer_per_request * batch_size) / (1024 ** 3),
        "per_request_gb": (memory_per_layer_per_request * num_layers) / (1024 ** 3),
    }


def calculate_optimal_batch_size(
    gpu_memory_gb: float,
    num_layers: int,
    max_seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    safety_factor: float = 0.8,
    dtype: torch.dtype = torch.float16
) -> int:
    """
    Calculate optimal batch size for given GPU memory.
    
    Args:
        gpu_memory_gb: Available GPU memory in GB
        num_layers: Number of transformer layers
        max_seq_len: Maximum sequence length
        num_kv_heads: Number of KV heads
        head_dim: Dimension per head
        safety_factor: Memory safety factor (default: 0.8)
        dtype: Data type for KV cache
    
    Returns:
        Recommended batch size
    """
    available_memory_gb = gpu_memory_gb * safety_factor
    
    # Estimate memory per request
    mem_estimate = estimate_kv_cache_memory(
        num_layers, 1, max_seq_len, num_kv_heads, head_dim, dtype
    )
    
    memory_per_request_gb = mem_estimate["per_request_gb"]
    
    # Add overhead for model weights and activations (rough estimate: 2x)
    memory_per_request_gb *= 2
    
    batch_size = int(available_memory_gb / memory_per_request_gb)
    
    return max(1, batch_size)


def calculate_attention_flops(
    batch_size: int,
    seq_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int
) -> int:
    """
    Calculate FLOPs for attention operation.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_qo_heads: Number of query/output heads
        num_kv_heads: Number of key/value heads
        head_dim: Dimension per head
    
    Returns:
        Number of FLOPs
    """
    # QK^T: (batch_size * num_qo_heads * seq_len * head_dim) 
    #       @ (batch_size * num_kv_heads * head_dim * seq_len)
    # For GQA, each qo_head attends to kv_heads
    qk_flops = batch_size * num_qo_heads * seq_len * seq_len * head_dim * 2
    
    # PV: (batch_size * num_qo_heads * seq_len * seq_len) 
    #     @ (batch_size * num_kv_heads * seq_len * head_dim)
    pv_flops = batch_size * num_qo_heads * seq_len * seq_len * head_dim * 2
    
    return qk_flops + pv_flops


def calculate_tflops(
    flops: int,
    time_seconds: float
) -> float:
    """
    Calculate TFLOPS from FLOPs and time.
    
    Args:
        flops: Number of floating-point operations
        time_seconds: Execution time in seconds
    
    Returns:
        TFLOPS (teraFLOPS per second)
    """
    return flops / time_seconds / 1e12


def calculate_memory_bandwidth(
    bytes_transferred: int,
    time_seconds: float
) -> float:
    """
    Calculate memory bandwidth in GB/s.
    
    Args:
        bytes_transferred: Number of bytes transferred
        time_seconds: Transfer time in seconds
    
    Returns:
        Bandwidth in GB/s
    """
    return bytes_transferred / time_seconds / 1e9


def get_optimal_dtype(
    gpu_compute_capability: Tuple[int, int],
    prioritize_memory: bool = False
) -> torch.dtype:
    """
    Get the optimal data type for given GPU.
    
    Args:
        gpu_compute_capability: (major, minor) compute capability
        prioritize_memory: Prioritize memory efficiency over precision
    
    Returns:
        Recommended torch.dtype
    """
    major, minor = gpu_compute_capability
    
    # Blackwell (SM 10.0+, 12.0+)
    if major >= 10 or major == 12:
        if prioritize_memory:
            return torch.float8_e4m3fn  # FP8 for memory
        return torch.bfloat16
    
    # Hopper (SM 9.0)
    elif major == 9:
        if prioritize_memory:
            return torch.float8_e4m3fn
        return torch.bfloat16
    
    # Ampere (SM 8.0, 8.6, 8.9)
    elif major == 8:
        return torch.bfloat16  # BF16 for better numerical stability
    
    # Turing (SM 7.5)
    else:
        return torch.float16


def detect_performance_issues(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Analyze performance metrics and detect potential issues.
    
    Args:
        metrics: Dictionary with performance metrics:
                 - median_latency_ms
                 - std_latency_ms
                 - tflops (optional)
                 - bandwidth_gb_per_sec (optional)
                 - sm_efficiency (optional)
    
    Returns:
        Dictionary with detected issues and recommendations
    """
    issues = []
    recommendations = []
    
    # Check latency consistency
    if "std_latency_ms" in metrics and "median_latency_ms" in metrics:
        cv = (metrics["std_latency_ms"] / metrics["median_latency_ms"]) * 100
        if cv > 5.0:
            issues.append("High latency variance detected")
            recommendations.append(
                "Check for thermal throttling or background processes"
            )
    
    # Check compute efficiency
    if "sm_efficiency" in metrics:
        if metrics["sm_efficiency"] < 60:
            issues.append(f"Low SM efficiency ({metrics['sm_efficiency']:.1f}%)")
            recommendations.append(
                "Increase batch size or enable split-K for better GPU utilization"
            )
    
    # Check memory bandwidth
    if "bandwidth_gb_per_sec" in metrics:
        gpu_info = get_gpu_info()
        gpu_name = gpu_info.get("name", "").upper()
        
        # Rough bandwidth targets
        if "H100" in gpu_name:
            peak_bandwidth = 3350  # GB/s for H100 HBM3
        elif "A100" in gpu_name:
            peak_bandwidth = 1940  # GB/s for A100 HBM2e
        else:
            peak_bandwidth = None
        
        if peak_bandwidth:
            efficiency = (metrics["bandwidth_gb_per_sec"] / peak_bandwidth) * 100
            if efficiency < 50:
                issues.append(
                    f"Low memory bandwidth utilization ({efficiency:.1f}%)"
                )
                recommendations.append(
                    "Check memory access patterns and consider data layout optimization"
                )
    
    return {
        "has_issues": len(issues) > 0,
        "issues": issues,
        "recommendations": recommendations
    }
