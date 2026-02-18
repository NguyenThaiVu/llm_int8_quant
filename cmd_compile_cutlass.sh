# /usr/local/cuda-12.4/bin/nvcc \
/usr/local/cuda-12.3/bin/nvcc \
  -O3 -lineinfo \
  -I /sciclone/home/tnguyen10/Desktop/GPU_learn/cutlass/include \
  -I /sciclone/home/tnguyen10/Desktop/GPU_learn/cutlass/tools/util/include \
  -I /sciclone/home/tnguyen10/Desktop/GPU_learn \
  -gencode arch=compute_80,code=sm_80 \
  -o matmul_update matmul_update.cu
  # -o matmul matmul.cu