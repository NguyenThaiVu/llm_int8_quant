/usr/local/cuda-12.4/nsight-compute-2024.1.1/ncu \
  --kernel-name rmsnorm_kernel \
  --set full \
  -o report_rmsnorm_bf16 \
  python roofline_rmsnorm.py