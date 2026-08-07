import subprocess
import os
import pathlib

# Check CUDA HOME can be /usr/local/cuda-12.4 or /usr/local/cuda-12.3
NVCC_PATH = None
if pathlib.Path("/usr/local/cuda-12.4").exists():
    NVCC_PATH = "/usr/local/cuda-12.4"
elif pathlib.Path("/usr/local/cuda-12.3").exists():
    NVCC_PATH = "/usr/local/cuda-12.3"
elif pathlib.Path("/home/tnguyen10/cuda-12.1").exists():
    NVCC_PATH = "/home/tnguyen10/cuda-12.1"
else:
    raise EnvironmentError("CUDA_HOME is not set.")

# Define the search space for our empirical tuning
BLOCK_SIZES = [32, 64, 128, 256, 512, 1024]

TEMPLATE_FILE = "kernel.cu.template"
TEMP_SRC_FILE = "temp_kernel.cu"
EXE_FILE = "temp_kernel.exe" if os.name == 'nt' else "./temp_kernel"

def read_template(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def run_tuning():
    if not os.path.exists(TEMPLATE_FILE):
        print(f"Error: {TEMPLATE_FILE} not found.")
        return

    template_content = read_template(TEMPLATE_FILE)
    results = {}

    print(f"--- Starting Empirical Auto-Tuning ---")
    
    for block_size in BLOCK_SIZES:
        print(f"Testing Block Size: {block_size}... ", end="", flush=True)
        
        # 1. Substitute parameters into the template
        source_code = template_content.replace("__BLOCK_SIZE_PLACEHOLDER__", str(block_size))
        with open(TEMP_SRC_FILE, 'w') as f:
            f.write(source_code)
        
        # 2. Compile via nvcc
        compile_cmd = [os.path.join(NVCC_PATH, "bin", "nvcc"), "-O3", TEMP_SRC_FILE, "-o", EXE_FILE]
        compile_res = subprocess.run(compile_cmd, capture_output=True, text=True)
        
        if compile_res.returncode != 0:
            print("Compilation Failed!")
            print(compile_res.stderr)
            continue
            
        # 3. Execute the binary and collect performance metrics
        run_res = subprocess.run([EXE_FILE], capture_output=True, text=True)
        
        if run_res.returncode != 0:
            print("Execution Failed!")
            continue
            
        # Parse the execution time printed by the binary
        avg_time = float(run_res.stdout.strip())
        results[block_size] = avg_time
        print(f"{avg_time:.4f} ms")

    # 4. Clean up temporary files
    if os.path.exists(TEMP_SRC_FILE): os.remove(TEMP_SRC_FILE)
    if os.path.exists(EXE_FILE): os.remove(EXE_FILE)
    if os.path.exists(EXE_FILE + ".exp"): os.remove(EXE_FILE + ".exp") # Windows clean-up
    if os.path.exists(EXE_FILE + ".lib"): os.remove(EXE_FILE + ".lib") # Windows clean-up

    # 5. Evaluate Cost Function and Report Optimal Configurations
    print("\n--- Tuning Results Summary ---")
    best_block_size = min(results, key=results.get)
    
    for b_size, execution_time in results.items():
        prefix = "-> " if b_size == best_block_size else "   "
        print(f"{prefix}Block Size {b_size:4d}: {execution_time:.4f} ms")
        
    print(f"\nOptimal configuration found: THREADS_PER_BLOCK = {best_block_size}")

if __name__ == "__main__":
    run_tuning()