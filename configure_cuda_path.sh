#!/bin/bash

export CUDA_HOME="/sciclone/home/tnguyen10/cuda-12.8"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Clear Bash command path cache
hash -r