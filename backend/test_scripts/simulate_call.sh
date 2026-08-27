API_KEY=A5EnrdmUTCGkGtYxxWKuOwa2wZjrWXCp3N5wESMkf_Y
HEADER="X-Hermes-API-Key: $API_KEY"

curl -i -H "Accept: application/json" -H "Content-Type: application/json" -H "$HEADER" \
  -X GET http://127.0.0.1:8001/start_call_interpretation/?operator_unique_code=pitagoras

curl -X 'POST' 'http://127.0.0.1:8001/send_audio/?id_emergencia=1&sp_rate=16000' -H 'accept: application/json' -H "$HEADER" \
  -H 'Content-Type: multipart/form-data' -F 'audio_data=@test_audios/filho_assaltado.wav;type=audio/wav'

###Print the complete 'emergency_transcripts' table in the sqlite database using the sqlite3 python module and python -c command:
python3 -c "import sqlite3; conn = sqlite3.connect('sqlite.db/database.sqlite');\
 cursor = conn.cursor(); cursor.execute('SELECT * FROM emergency_transcripts');\
  rows = cursor.fetchall(); print(rows); conn.close()"