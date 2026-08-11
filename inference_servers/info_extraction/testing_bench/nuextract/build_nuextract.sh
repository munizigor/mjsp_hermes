#!/bin/bash

export NUEXTRACT_TOKEN=putarealsecret
export VERSION=4B
git clone https://github.com/pentalpha/nuextract.git
cd nuextract/dockerfiles && sudo docker build . \
    -t nuextract-2-${VERSION} \
    -f NuExtract-2.0-${VERSION}.dockerfile

export VERSION=2B
cd nuextract/dockerfiles && sudo docker build . \
    -t nuextract-2-${VERSION} \
    -f NuExtract-2.0-${VERSION}.dockerfile

#sudo docker run -it --rm --name nuextract --network=host \
#    --runtime nvidia --gpus all \
#    nuextract-2-${VERSION} --api-key putarealsecret --dtype half --max-model-len 8192 --enforce-eager