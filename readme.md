CUDA HOME: /usr/local/cuda-12.4
or /usr/local/cuda-12.3

NVCC Path: /usr/local/cuda-12.4/bin/nvcc
or /usr/local/cuda-12.3/bin/nvcc

CONDA Enable: source /sciclone/apps/miniforge3-24.9.2-0/etc/profile.d/conda.sh
CONDA Enable: source /sciclone/apps/miniforge3-24.9.2-0/etc/profile.d/conda.sh


NSight System path: /usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys

nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  -o nsys_softmax \
  ./softmax


NSight Compute path: /usr/local/cuda-12.4/nsight-compute-2024.1.1/ncu
or /sciclone/apps/NVIDIA-Nsight-Compute-2024.3/ncu

NSight Compute
ncu \
  --kernel-name softmax_lastdim_3d_kernel \
  --set full \
  -o report_softmax \
  ./softmax


