#!/usr/bin/env bash

THRESHOLD_MB=10000
SLEEP_SECONDS=10
GPU_ID=0  # select GPU device 0, 1, 2,...
PYTHON_FILE="qwen_quant.py" # qwen_quant.py or qwen.py

CMD=(python $PYTHON_FILE)

echo "Waiting for GPU $GPU_ID memory usage to be under ${THRESHOLD_MB} MB..."

while true; do
    USED_MB=$(nvidia-smi \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits \
        -i "$GPU_ID" | tr -d ' ')

    echo "$(date): GPU $GPU_ID memory used: ${USED_MB} MB"

    if [ "$USED_MB" -lt "$THRESHOLD_MB" ]; then
        echo "$(date): GPU memory is available. Running: ${CMD[*]}"

        "${CMD[@]}"
        EXIT_CODE=$?

        echo "$(date): Command finished with exit code $EXIT_CODE"

        if [ "$EXIT_CODE" -eq 0 ]; then
            echo "$(date): ✅ Run finished successfully."
        else
            echo "$(date): ❌ Run failed."
        fi

        break
    fi

    sleep "$SLEEP_SECONDS"
done

echo "$(date): wait_for_gpu.sh finished."