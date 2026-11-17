## File Overview

This repository contains scripts for running Qwen3 inference and evaluating different quantization methods.

| File | Description |
|---|---|
| `qwen.py` | Standard Qwen3 model without quantization. |
| `qwen_quant.py` | Our proposed quantization method for single-sentence inference. |
| `qwen_quant-BATCH.py` | Our proposed quantization method for batched sentence. |
| `qwen_llm_int8.py` | Qwen3 using the LLM.int8() quantization technique as Benchmark 1. |
| `qwen_smoothquant.py` | Qwen3 using the SmoothQuant technique as Benchmark 2. |