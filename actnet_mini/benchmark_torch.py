"""
Benchmark CPU vs GPU for active contraction simulation
Quick test to see speedup
"""

import time
import torch
import numpy as np

print("=" * 60)
print("Active Contraction - CPU vs GPU Benchmark")
print("=" * 60)

# Check CUDA availability
if torch.cuda.is_available():
    print(f"\n✓ CUDA available")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("\n✗ CUDA not available - comparison will use CPU only")
    exit()

print("\n" + "-" * 60)
print("Running benchmark...")
print("-" * 60)

# Small test: force calculations
N = 5000
iterations = 100

# CPU test
print(f"\nCPU test ({N} particles, {iterations} iterations):")
device_cpu = torch.device("cpu")

xcnt = torch.randn(N, 1, device=device_cpu)
ycnt = torch.randn(N, 1, device=device_cpu)
phi = torch.rand(N, 1, device=device_cpu) * 2 * np.pi
lens = torch.rand(N, 1, device=device_cpu) + 1.0
km = torch.rand(N, 1, device=device_cpu) * 10
Dx = torch.randn(N, 1, device=device_cpu)
Dy = torch.randn(N, 1, device=device_cpu)
Diag = torch.rand(N, 1, device=device_cpu)

t0 = time.time()
for _ in range(iterations):
    Dr = torch.sqrt(Dx**2 + Dy**2)
    Fx = -km * (torch.abs(Dx) - 0.05 * Diag) * (Dx / Dr)
    Fy = -km * (torch.abs(Dy) - 0.05 * Diag) * (Dy / Dr)

    gamma_x = 200 * (torch.cos(phi) ** 2 + lens * (torch.sin(phi) ** 2) / np.pi)
    gamma_y = 200 * (torch.sin(phi) ** 2 + lens * (torch.cos(phi) ** 2) / np.pi)

    Vx = Fx / gamma_x
    Vy = Fy / gamma_y

    xcnt += Vx * 0.1
    ycnt += Vy * 0.1
    phi += torch.randn(N, 1, device=device_cpu) * 0.01

cpu_time = time.time() - t0
print(f"  Time: {cpu_time:.3f} seconds")

# GPU test
print(f"\nGPU test ({N} particles, {iterations} iterations):")
device_gpu = torch.device("cuda")

xcnt = torch.randn(N, 1, device=device_gpu)
ycnt = torch.randn(N, 1, device=device_gpu)
phi = torch.rand(N, 1, device=device_gpu) * 2 * np.pi
lens = torch.rand(N, 1, device=device_gpu) + 1.0
km = torch.rand(N, 1, device=device_gpu) * 10
Dx = torch.randn(N, 1, device=device_gpu)
Dy = torch.randn(N, 1, device=device_gpu)
Diag = torch.rand(N, 1, device=device_gpu)

# Warmup
for _ in range(10):
    Dr = torch.sqrt(Dx**2 + Dy**2)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(iterations):
    Dr = torch.sqrt(Dx**2 + Dy**2)
    Fx = -km * (torch.abs(Dx) - 0.05 * Diag) * (Dx / Dr)
    Fy = -km * (torch.abs(Dy) - 0.05 * Diag) * (Dy / Dr)

    gamma_x = 200 * (torch.cos(phi) ** 2 + lens * (torch.sin(phi) ** 2) / np.pi)
    gamma_y = 200 * (torch.sin(phi) ** 2 + lens * (torch.cos(phi) ** 2) / np.pi)

    Vx = Fx / gamma_x
    Vy = Fy / gamma_y

    xcnt += Vx * 0.1
    ycnt += Vy * 0.1
    phi += torch.randn(N, 1, device=device_gpu) * 0.01

torch.cuda.synchronize()
gpu_time = time.time() - t0
print(f"  Time: {gpu_time:.3f} seconds")

# Results
print("\n" + "=" * 60)
print("Results")
print("=" * 60)
print(f"CPU:     {cpu_time:.3f}s")
print(f"GPU:     {gpu_time:.3f}s")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")

if cpu_time / gpu_time > 1:
    print(f"\n✓ GPU is {cpu_time/gpu_time:.2f}x faster!")
else:
    print(f"\n✗ GPU slower (overhead dominates for small problem)")

print("\nNote: Actual simulation includes sparse operations on CPU")
print("      Real speedup depends on particle count and connectivity")
print("=" * 60)
