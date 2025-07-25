#!/usr/bin/env python
"""
cuda_gelu.py – build and test a custom CUDA GELU op for PyTorch,
now with an *optional* L2-cache flush before every timed run.
"""

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"  # targeting A100 (SM 80)

from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------------------- #
# Optional low-level CUDA helpers (PyCUDA) to flush the GPU’s L2 cache
# --------------------------------------------------------------------------- #

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # creates and holds a context on device 0

    # Query L2-cache size and allocate a scratch buffer ≥ 2×L2
    _l2_size = cuda.Device(0).get_attribute(cuda.device_attribute.L2_CACHE_SIZE)
    _flush_bytes = _l2_size * 2
    _d_F = cuda.mem_alloc(_flush_bytes)

    def flush_l2() -> None:
        """Overwrite a 2×L2-sized buffer to evict previous data from cache."""
        cuda.memset_d8(_d_F, 0, _flush_bytes)
        cuda.Context.synchronize()

except ModuleNotFoundError:
    # PyCUDA not installed – fall back to a no-op flush
    print("[cuda_gelu] PyCUDA not found – L2-cache flush disabled")
    def flush_l2() -> None:
        return

# --------------------------------------------------------------------------- #
# Reference GELU in pure PyTorch
# --------------------------------------------------------------------------- #

def manual_gelu(x: torch.Tensor) -> torch.Tensor:
    """Tanh-based GELU approximation identical to torch.nn.functional.gelu."""
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

# --------------------------------------------------------------------------- #
# Build a small CUDA extension that implements GELU
# --------------------------------------------------------------------------- #

def create_cuda_gelu():
    """
    Compile a CUDA kernel for GELU and return it as a Python-callable function.
    Returns
    -------
    callable
        gelu(x: torch.Tensor[float32, CUDA & contiguous]) → torch.Tensor
    """
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # surface kernel errors

    cuda_src = r"""
    #include <math.h>
    #include <torch/extension.h>
    #include <c10/cuda/CUDAException.h>

    __global__ void gelu_kernel(const float* __restrict__ x,
                                float* __restrict__ y,
                                const int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            float v = x[i];
            y[i] = 0.5f * v *
                   (1.0f + tanhf(0.79788456f * (v + 0.044715f * v * v * v)));
        }
    }

    __host__ __device__ inline unsigned cdiv(unsigned a, unsigned b) {
        return (a + b - 1) / b;
    }

    torch::Tensor gelu(torch::Tensor x) {
        TORCH_CHECK(x.device().is_cuda(), "gelu: input must be on CUDA");
        TORCH_CHECK(x.is_contiguous(),    "gelu: input must be contiguous");
        TORCH_CHECK(x.dtype() == torch::kFloat32,
                    "gelu: only float32 is supported");

        const int n = x.numel();
        auto y     = torch::empty_like(x);

        const int block = 1024;
        const int grid  = cdiv(n, block);

        gelu_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }
    """

    cpp_stub = "torch::Tensor gelu(torch::Tensor x);"

    build_dir = Path("var") / "cuda_gelu"
    build_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available – cannot build cuda_gelu")

    module = load_inline(
        name="inline_gelu",
        cpp_sources=[cpp_stub],
        cuda_sources=[cuda_src],
        functions=["gelu"],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(build_dir),
        verbose=False,
    )
    return module.gelu

# --------------------------------------------------------------------------- #
# Improved timing helper with *optional* cold-cache runs
# --------------------------------------------------------------------------- #

def show_time(func, ntest: int = 10, flush: bool = False):
    """
    Time `func()` `ntest` times on the GPU.
    If `flush=True`, evict L2 cache via `flush_l2()` before every timed call.
    Returns
    -------
    list[float]  –  runtimes in micro-seconds (μs)
    """
    # JIT warm-up
    for _ in range(10):
        func()

    times = []
    for _ in range(ntest):
        if flush:
            flush_l2()  # cold-cache start
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        func()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # convert to µs
    return times

# --------------------------------------------------------------------------- #
# Quick test & benchmark
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    gelu_cuda = create_cuda_gelu()

    # Large test vector
    x = torch.randn(1 << 20, device="cuda", dtype=torch.float32)

    # Correctness checks
    y_ref  = torch.nn.functional.gelu(x, approximate="tanh")
    y_cuda = gelu_cuda(x)
    max_err = (y_ref - y_cuda).abs().max().item()
    print(f"max |error| cuda vs torch GELU: {max_err:.3e}")

    # Manual PyTorch GELU
    y_manual = manual_gelu(x)
    max_err_manual = (y_ref - y_manual).abs().max().item()
    print(f"max |error| manual vs torch GELU: {max_err_manual:.3e}")

    # Benchmark set-up
    def run_cuda():       gelu_cuda(x)
    def run_manual():     manual_gelu(x)
    def run_torch():      torch.nn.functional.gelu(x, approximate="tanh")

    print("\n--- Hot-cache timings (no flush) ---")
    cuda_hot   = show_time(run_cuda)
    manual_hot = show_time(run_manual)
    torch_hot  = show_time(run_torch)
    print(f"CUDA  GELU avg:  {np.mean(cuda_hot):8.3f} µs")
    print(f"Manual GELU avg:{np.mean(manual_hot):8.3f} µs")
    print(f"torch  GELU avg: {np.mean(torch_hot):8.3f} µs")

    print("\n--- Cold-cache timings (flush=True) ---")
    cuda_cold   = show_time(run_cuda, flush=True)
    manual_cold = show_time(run_manual, flush=True)
    torch_cold  = show_time(run_torch, flush=True)
    print(f"CUDA  GELU (cold) avg:  {np.mean(cuda_cold):8.3f} µs")
    print(f"Manual GELU (cold) avg:{np.mean(manual_cold):8.3f} µs")
    print(f"torch  GELU (cold) avg: {np.mean(torch_cold):8.3f} µs")

# --------------------------------------------------------------------------- #
# expected results
# max |error| cuda vs torch GELU: 5.603e-06
# max |error| manual vs torch GELU: 2.384e-07

# --- Hot-cache timings (no flush) ---
# CUDA  GELU avg:    63.414 µs
# Manual GELU avg: 227.694 µs
# torch  GELU avg:   38.324 µs

# --- Cold-cache timings (flush=True) ---
# CUDA  GELU (cold) avg:    46.065 µs
# Manual GELU (cold) avg: 238.569 µs
# torch  GELU (cold) avg:   45.015 µs
# --------------------------------------------------------------------------- #
