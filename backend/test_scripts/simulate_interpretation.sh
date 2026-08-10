export id_emergencia=34
echo "Creating emergency"
curl -i -H "Accept: application/json" -H "Content-Type: application/json" \
  -X GET http://127.0.0.1:8001/start_call_interpretation/?operator_unique_code=pitagoras
echo "Starting to send transcripts"
export transcript="Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. Ela está inconsciente. Meu nome é Maria Oliveira. Meu telefone é 99993-8888, moro na cidade de São Paulo. "
curl -i -X POST -G "http://127.0.0.1:8001/send_transcript/" \
  -H "Accept: application/json" \
  --data-urlencode "id_emergencia=$id_emergencia" \
  --data-urlencode "transcript=$transcript"
sleep 3
export transcript="Pera! Na verdade acho que o nome da rua é outro."
curl -i -X POST -G "http://127.0.0.1:8001/send_transcript/" \
  -H "Accept: application/json" \
  --data-urlencode "id_emergencia=$id_emergencia" \
  --data-urlencode "transcript=$transcript"
sleep 3

export transcript="Qual é a rua minha senhora?"
curl -i -X POST -G "http://127.0.0.1:8001/send_transcript/" \
  -H "Accept: application/json" \
  --data-urlencode "id_emergencia=$id_emergencia" \
  --data-urlencode "transcript=$transcript"
sleep 3

export transcript="Rua dos Gerânios número 899, perto do posto de saúde, a UBS do bairro."
curl -i -X POST -G "http://127.0.0.1:8001/send_transcript/" \
  -H "Accept: application/json" \
  --data-urlencode "id_emergencia=$id_emergencia" \
  --data-urlencode "transcript=$transcript"
sleep 3

export transcript="Okay, vou anotar aqui. Mas qual é o bairro? Aqui no sistema está dizendo que é no bairro Pajuçara."
curl -i -X POST -G "http://127.0.0.1:8001/send_transcript/" \
  -H "Accept: application/json" \
  --data-urlencode "id_emergencia=$id_emergencia" \
  --data-urlencode "transcript=$transcript"
sleep 3


export transcript="Anote mesmo! Eu errei feio! Ah... o nome do bairro? Pajuçara... Acho que não. Aqui é bairro Capim Macio."
curl -i -X POST -G "http://127.0.0.1:8001/send_transcript/" \
  -H "Accept: application/json" \
  --data-urlencode "id_emergencia=$id_emergencia" \
  --data-urlencode "transcript=$transcript"
sleep 3

###Print the complete 'resultados_inferencia' table in the sqlite database using the sqlite3 python module and python -c command:
python3 -c "import sqlite3; conn = sqlite3.connect('sqlite.db/database.sqlite');\
 cursor = conn.cursor(); cursor.execute('SELECT * FROM resultados_inferencia');\
  rows = cursor.fetchall(); print('\n'.join([str(row) for row in rows])); conn.close()"
python3 -c "import sqlite3; conn = sqlite3.connect('sqlite.db/database.sqlite');\
 cursor = conn.cursor(); cursor.execute('SELECT * FROM emergency_transcripts');\
  rows = cursor.fetchall(); print('\n'.join([str(row) for row in rows])); conn.close()"
python3 -c "import sqlite3; conn = sqlite3.connect('sqlite.db/database.sqlite');\
 cursor = conn.cursor(); cursor.execute('SELECT * FROM emergency_audios');\
  rows = cursor.fetchall(); print('\n'.join([str(row) for row in rows])); conn.close()"

curl -i -H "Accept: application/json" -H "Content-Type: application/json" -X GET http://127.0.0.1:8001/get_all_inference_results/?id_emergencia=$id_emergencia

curl -i -H "Accept: application/json" -H "Content-Type: application/json" \
  -X GET http://127.0.0.1:8001/get_cluster_metrics/