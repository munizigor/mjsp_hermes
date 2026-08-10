#/bin/bash
set -e
: "${HERMES_BACKEND_DIR:=$(pwd)}"

cd ${HERMES_BACKEND_DIR} \ && sudo docker compose down > build.log