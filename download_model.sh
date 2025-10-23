#!/bin/bash

# Configuration
# "mistralai/Mistral-7B-Instruct-v0.2"
MODELS=(
    "01ai/Yi-34B-Chat-8bits"
)
RETRY_DELAY=60  # Seconds to wait between retries
MAX_ATTEMPTS=0  # Set to a number >0 for max attempts; 0 means unlimited

# Function to check if download is successful (relies on command exit code)
download_model() {
    local model="$1"
    echo "Downloading model: $model"
    modelscope download --model "$model"
}

# Main loop
for model in "${MODELS[@]}"; do
    attempt=1
    success=false

    while [ $success = false ]; do
        echo "=== Attempt $attempt to download model: $model ==="

        if download_model "$model"; then
            echo "✓ Download completed successfully!"
            success=true
        else
            echo "✗ Download failed (likely due to timeout). Retrying in ${RETRY_DELAY} seconds..."
            sleep $RETRY_DELAY

            if [ $MAX_ATTEMPTS -gt 0 ] && [ $attempt -ge $MAX_ATTEMPTS ]; then
                echo "❌ Maximum attempts ($MAX_ATTEMPTS) reached. Exiting."
                exit 1
            fi

            attempt=$((attempt + 1))
        fi
    done

    echo "🎉 All files downloaded successfully after $attempt attempts."
done
