#/bin/bash
set -e
: "${HERMES_BACKEND_DIR:=$(pwd)}"

# Carrega variáveis do .env se existir (inclui LOCAL_SQL_DB_PATH)
if [ -f "${HERMES_BACKEND_DIR}/.env" ]; then
  # Sanitiza CRLF e exporta apenas linhas do tipo NOME=VALOR (ignora comentários/linhas vazias)
  while IFS= read -r line; do
    case "$line" in
      ''|\#*) continue;;
    esac
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      name=${BASH_REMATCH[1]}
      value=${BASH_REMATCH[2]}
      if [[ "$value" =~ ^\".*\"$|^'.*'$ ]]; then
        eval "export $name=$value"
      else
        value_escaped=$(printf '%s' "$value" | sed "s/'/'\"'\"'/g")
        eval "export $name='$value_escaped'"
      fi
    fi
  done < <(sed 's/\r$//' "${HERMES_BACKEND_DIR}/.env")
fi

if [ -z "${LOCAL_SQL_DB_PATH:-}" ]; then
  echo "Erro: LOCAL_SQL_DB_PATH não está definido. Verifique ${HERMES_BACKEND_DIR}/.env" >&2
  exit 1
fi


#cd ${HERMES_BACKEND_DIR} \
#    #&& sudo rm -rf ${LOCAL_SQL_DB_PATH}/* \
#    && sudo rm -f datasets/naturezas_cache_vllm.json \
#    && sudo docker compose build > build.log \
#    && sudo docker compose up > stdout.log 2> stderr.log

cd ${HERMES_BACKEND_DIR} \
    && sudo rm -rf ${LOCAL_SQL_DB_PATH}/* \
    && sudo rm -f datasets/naturezas_cache_vllm.json \
    && sudo docker compose build > build.log \
    && sudo docker compose up > stdout.log 2> stderr.log