# CUDA 12.8
# export CUDA_HOME="$HOME/cuda-12.8"
export CUDA_HOME="/home/tnguyen10/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"

# Conda GCC 14
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"

# Blackwell GPU
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"