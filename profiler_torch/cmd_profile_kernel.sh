/usr/local/cuda-12.4/nsight-compute-2024.1.1/ncu \
  --kernel-name hierarchical_silu_int8_vec4_kernel \
  --set full \
  -o report_silu_int8 \
  python NCU_silu.py