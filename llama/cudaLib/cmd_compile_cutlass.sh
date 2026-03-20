# /usr/local/cuda-12.4/bin/nvcc \
# /usr/local/cuda-12.3/bin/nvcc \
/home/tnguyen10/cuda-12.1/bin/nvcc \
  -O3 -lineinfo \
  -I /home/tnguyen10/Desktop/llm_int8_quant/cutlass \
  -I /home/tnguyen10/Desktop/llm_int8_quant/cutlass/include \
  -I /home/tnguyen10/Desktop/llm_int8_quant/cutlass/tools/util/include \
  -I /home/tnguyen10/Desktop/llm_int8_quant \
  -gencode arch=compute_80,code=sm_80 \
  -o test_cutlass test_cutlass.cu