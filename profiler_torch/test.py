K = 4096
N = 8192

AI = (2 * K * N) / (2 * K * N + 2 * K + 2 * N)
print(f"AI = {AI:.4f} (FLOP/Byte)")