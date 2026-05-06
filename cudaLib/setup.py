import os
import pathlib

# Check CUDA HOME can be /usr/local/cuda-12.4 or /usr/local/cuda-12.3

if not os.environ.get("CUDA_HOME"):
    if pathlib.Path("/usr/local/cuda-12.4").exists():
        os.environ["CUDA_HOME"] = "/usr/local/cuda-12.4"
    elif pathlib.Path("/usr/local/cuda-12.3").exists():
        os.environ["CUDA_HOME"] = "/usr/local/cuda-12.3"
    elif pathlib.Path("/home/tnguyen10/cuda-12.1").exists():
        os.environ["CUDA_HOME"] = "/home/tnguyen10/cuda-12.1"
    else:
        raise EnvironmentError("CUDA_HOME is not set.")


os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ["PATH"]
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Check path /sciclone/home/tnguyen10/Desktop/GPU_learn
if pathlib.Path("/sciclone/home/tnguyen10/Desktop/GPU_learn").exists():
    path_root = "/sciclone/home/tnguyen10/Desktop/GPU_learn"
elif pathlib.Path("/home/tnguyen10/Desktop/llm_int8_quant").exists():
    path_root = "/home/tnguyen10/Desktop/llm_int8_quant"
else:
    raise EnvironmentError("Path to cutlass library is not found.")
include_dirs = [f"{path_root}/cutlass/include",\
                f"{path_root}/cutlass/tools/util/include",\
                f"{path_root}"]

# include_dirs=["/sciclone/home/tnguyen10/Desktop/GPU_learn/cutlass/include",\
#                 "/sciclone/home/tnguyen10/Desktop/GPU_learn/cutlass/tools/util/include",\
#                 "/sciclone/home/tnguyen10/Desktop/GPU_learn"]


cuda  = os.environ.get("CUDA_HOME", "")

setup(
    name="gemm_cutlass",
    ext_modules=[
        CUDAExtension(
            name="gemm_cutlass",
            sources=["gemm_cutlass.cu"],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-gencode=arch=compute_80,code=sm_80",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)