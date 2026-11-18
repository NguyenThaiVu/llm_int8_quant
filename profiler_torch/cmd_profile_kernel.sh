/usr/local/cuda-12.4/nsight-compute-2024.1.1/ncu \
  --kernel-name softmax_int8_naive_kernel \
  --set full \
  -o report_softmax_naive_int8 \
  python ncu_softmax.py