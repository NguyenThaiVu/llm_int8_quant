In this folder `profiler_torch`, we focus on profiling the latency on different scenario.

File `profile_rmsnorm.py`: compare the latency of RMSNorm layer between:
- Torch implementation (torch.nn.functional.rms_norm)
- INT8 RMSNorm - naive implementation.
- INT8 RMSNorm - Warp reduction implementation.

File `compare_rmsnorm.py`: profile (using Nsight Compute) the kernel criteria
(SMEM per block, achieved warps occupancy, Maximum resident blocks per SM), between:
- INT8 RMSNorm - naive implementation.
- INT8 RMSNorm - Warp reduction implementation.