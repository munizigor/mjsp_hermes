# Configurações
API_KEY=A5EnrdmUTCGkGtYxxWKuOwa2wZjrWXCp3N5wESMkf_Y
HEADER="X-Hermes-API-Key: $API_KEY"

host=0.0.0.0
#host=127.0.0.1
#host=35.247.206.139
#host=hermes-backend-np.samare.com.br
port=8001

BASE_DIR="/mnt/remote-cc/segments"
processed_count=0
max_calls=24

ls "$BASE_DIR" | grep "\-solicitante$" | while read -r dirname; do
    FULL_PATH="$BASE_DIR/$dirname"

    if [ "$processed_count" -ge "$max_calls" ]; then
        echo "Reached the limit of $max_calls calls. Exiting."
        break
    fi

    # Check if it's a directory and count .wav files
    if [ -d "$FULL_PATH" ]; then
        WAV_COUNT=$(ls "$FULL_PATH"/*.wav 2>/dev/null | wc -l)
        
        if [ "$WAV_COUNT" -ge 3 ]; then
            # Extract variables from directory name
            src_number2=$(echo "$dirname" | cut -d'-' -f2)
            dst_number2=$(echo "$dirname" | cut -d'-' -f3)
            timestamp2=$(echo "$dirname" | cut -d'-' -f6)

            echo "Processing $dirname ($WAV_COUNT wav files found)..."
            
            curl -i -X POST "http://$host:$port/start_call/?source_number=$src_number2&destination_number=$dst_number2&call_timestamp=$timestamp2" \
              -H "$HEADER"
            ((processed_count++))
        else
            echo "Skipping $dirname: only $WAV_COUNT .wav files found."
        fi
    fi
done