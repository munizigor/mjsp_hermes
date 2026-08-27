#!/bin/bash

network_name="hermes-network"

# 1. Check if the network exists; create with a compatible subnet if it does not
if ! sudo docker network inspect $network_name >/dev/null 2>&1; then
    echo "Network '$network_name' not found. Creating it..."
    sudo docker network create --subnet=172.20.0.0/16 $network_name
fi

# Define log file paths
VLLM_LOG="vllm_server.log"
WHISPER_LOG="whisper_docker.log"

echo "Starting inference servers in parallel..."

# 1. Start vLLM in the background and save logs
source info_extraction/vllm_server/start_ministral3b_light.sh > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "[+] vLLM server started (PID: $VLLM_PID). Logs: $VLLM_LOG"

# 2. Start Whisper in a subshell (so 'cd' doesn't affect the script) and save logs
(pwd && ls && cd asr/whisper/whisper_cpp && sudo docker compose up) > "$WHISPER_LOG" 2>&1 &
WHISPER_PID=$!
echo "[+] Whisper server started (PID: $WHISPER_PID). Logs: $WHISPER_LOG"

# 3. Handle Ctrl+C to kill the background servers cleanly
trap "echo -e '\nStopping servers...'; kill $VLLM_PID; sudo kill $WHISPER_PID; exit" SIGINT SIGTERM

echo "Both servers are running. Press [Ctrl+C] to stop them."

# 4. Wait for background processes to finish so the script doesn't exit immediately
wait