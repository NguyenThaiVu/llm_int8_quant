import pandas as pd
import time
import threading
import torch
import pynvml

from utils_quant import *
import gemm_cutlass

class PowerSampler:
    def __init__(self, gpu_id=0, interval=0.01):
        """
        interval: power sampling interval in seconds.
                  0.01 = 10 ms
        """
        pynvml.nvmlInit()

        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.interval = interval

        self.running = False
        self.samples = []
        self.timestamps = []

    def _sample(self):
        """
        
        """
        start_time = time.perf_counter()

        while self.running:
            # NVML returns milliwatts
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            power_w = power_mw / 1000.0

            t = time.perf_counter() - start_time

            self.timestamps.append(t)
            self.samples.append(power_w)

            time.sleep(self.interval)

    def start(self):
        self.samples = []
        self.timestamps = []

        self.running = True

        self.thread = threading.Thread(target=self._sample)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def average_power(self):
        if len(self.samples) == 0:
            return 0.0

        return sum(self.samples) / len(self.samples)

    def energy(self):
        """
        Integrate power over time using trapezoidal integration.

        Power: W = J/s
        Energy: Joules
        """
        if len(self.samples) < 2:
            return 0.0

        energy_j = 0.0

        for i in range(1, len(self.samples)):
            dt = self.timestamps[i] - self.timestamps[i - 1]

            p_avg = (self.samples[i] + self.samples[i - 1]) / 2.0

            energy_j += p_avg * dt

        return energy_j

# def print_power_benchmark(n_iterations, n_sampler,\
#                 total_time, latency_per_op, avg_power, total_energy, energy_per_op):
#     print("\n" + "═" * 56)
#     print("                 BENCHMARK RESULTS")
#     print(f"  {'Iterations':<28} {n_iterations:>18,}")
#     print(f"  {'Samples':<28} {len(n_sampler):>18,}")
#     print(f"  {'Total time':<28} {total_time:>15.6f} s")
#     print(f"  {'Latency / operation':<28} {latency_per_op * 1000:>15.4f} ms")
#     print(f"  {'Average power':<28} {avg_power:>15.2f} W")
#     print(f"  {'Total energy':<28} {total_energy:>15.4f} J")
#     print(f"  {'Energy / operation':<28} {energy_per_op * 1000:>15.4f} mJ")
#     print("═" * 56 + "\n")

def measure_power(func, *args, n_iterations=100):
    """
    Measure the average power consumption of a function over a number of repetitions.

    Parameters:
    func (callable): The function to measure.
    *args: Arguments to pass to the function.
    n_iterations (int): Number of times to repeat the function call.

    Returns:
    float: Average power consumption in watts.
    """
    sampler = PowerSampler(interval=0.01)

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            func(*args)

    torch.cuda.synchronize()
    sampler.start()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(n_iterations):
            func(*args)

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    sampler.stop()
    
    # Compute output
    total_time = end_time - start_time
    avg_power = sampler.average_power()
    total_energy = sampler.energy()
    energy_per_op = total_energy / n_iterations
    latency_per_op = total_time / n_iterations
    
    print("\n" + "═" * 56)
    print("                 BENCHMARK RESULTS")
    print(f"  {'Iterations':<28} {n_iterations:>18,}")
    print(f"  {'Samples':<28} {len(sampler.samples):>18,}")
    print(f"  {'Total time':<28} {total_time:>15.6f} s")
    print(f"  {'Latency / operation':<28} {latency_per_op * 1000:>15.4f} ms")
    print(f"  {'Average power':<28} {avg_power:>15.2f} W")
    print(f"  {'Total energy':<28} {total_energy:>15.4f} J")
    print(f"  {'Energy / operation':<28} {energy_per_op * 1000:>15.4f} mJ")
    print("═" * 56 + "\n")


if __name__ == "__main__":

    device = "cuda"
    M = 4096
    K = 4096
    N = 4096
    d_type = torch.bfloat16

    x = torch.randn(M, K, device=device, dtype=torch.float16)
    w = torch.randn(N, K, device=device, dtype=torch.float16)

    print("Warm-up...")
    for _ in range(10):
        y = x @ w.T
    torch.cuda.synchronize()

    # ============================================================
    # 1. Measure power torch matmul
    # ============================================================
    print("\nMeasuring power for torch.matmul...")
    sampler = PowerSampler(interval=0.01)
    
    for _ in range(10):
        y = x @ w.T
    torch.cuda.synchronize()

    num_iterations = 500

    measure_power(torch.matmul, x, w, n_iterations=num_iterations)
    
    # Clear cache 
    time.sleep(1)
    torch.cuda.empty_cache()
    
    # ============================================================
    # 2. Measure power custom INT8 matmul
    # ============================================================
    print("\nMeasuring power for custom INT8 matmul...")
    x_i8, scale_x = quantize_row_int8_symmetric_nd(x)
    w_i8, scale_w = quantize_row_int8_symmetric_nd(w)
    
    for _ in range(10):
        Y_deq = gemm_cutlass.func_w8a8_matmul(x_i8, w_i8, scale_x, scale_w)
    torch.cuda.synchronize()
    
    measure_power(gemm_cutlass.func_w8a8_matmul,\
                x_i8, w_i8, scale_x, scale_w, n_iterations=num_iterations)