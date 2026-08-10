# Configurações
API_KEY=A5EnrdmUTCGkGtYxxWKuOwa2wZjrWXCp3N5wESMkf_Y
HEADER="X-Hermes-API-Key: $API_KEY"

src_number=107
dst_number=108
timestamp=1768416225.58
id_emergencia=1
#host=10.0.0.7
#host=127.0.0.1
#host=35.247.206.139
host=hermes-backend-np.samare.com.br
port=8001

echo "Starting call from $src_number to $dst_number at $timestamp" \
  && curl -i -X POST "http://$host:$port/start_call/?source_number=$src_number&destination_number=$dst_number&call_timestamp=$timestamp" \
  -H "$HEADER" \
  && sleep 6 \
  && echo "List of emergencies:" \
  && curl -i -X GET "http://$host:$port/list_emergencies/" \
  -H "$HEADER" \
  && sleep 3 \
  && echo "Emergency $id_emergencia:" \
  && curl -i -X GET "http://$host:$port/get_emergency/?id_emergencia=$id_emergencia" \
  -H "$HEADER" \
  && sleep 6 \
  && echo "Ending call from $src_number to $dst_number" \
  && curl -i -X POST "http://$host:$port/end_call_by_number/?source_number=$src_number&destination_number=$dst_number" \
  -H "$HEADER"