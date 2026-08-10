# Configurações
API_KEY=A5EnrdmUTCGkGtYxxWKuOwa2wZjrWXCp3N5wESMkf_Y
HEADER="X-Hermes-API-Key: $API_KEY"
#host=10.0.0.7
#host=127.0.0.1
#host=35.247.206.139
host=35.247.206.139
port=8001
FULL_URL="http://${host}:${port}/list_emergencies"
echo "URL:"
echo "$FULL_URL"
echo "Header:"
echo "$HEADER"

curl -i -X GET $FULL_URL -H "$HEADER"