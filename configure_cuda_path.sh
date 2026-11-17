# CUDA 12.8
export CUDA_HOME="$HOME/cuda-12.8"
# export CUDA_HOME="/sciclone/home/tnguyen10/cuda-12.8"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX="$CUDA_HOME/bin/nvcc"