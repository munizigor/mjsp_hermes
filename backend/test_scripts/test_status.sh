API_KEY=A5EnrdmUTCGkGtYxxWKuOwa2wZjrWXCp3N5wESMkf_Y
HEADER="X-Hermes-API-Key: $API_KEY"

curl -i -H "Accept: application/json" -H "Content-Type: application/json" -H "$HEADER" \
  -X GET http://127.0.0.1:8001/status?last_hours=24

curl -i -H "Accept: application/json" -H "Content-Type: application/json" -H "$HEADER" \
  -X GET http://127.0.0.1:8001/status?last_hours=1