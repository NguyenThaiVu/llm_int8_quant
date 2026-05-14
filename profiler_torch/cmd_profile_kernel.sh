/usr/local/cuda-12.4/nsight-compute-2024.1.1/ncu \
  --kernel-name rmsnorm_int8_kernel \
  --set full \
  -o report_rmsnorm_int8 \
  python profile_rmsnorm_bf16_int8.py