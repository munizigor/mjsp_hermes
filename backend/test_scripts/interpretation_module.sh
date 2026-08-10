docker build -t hermes-interpretation -f Dockerfile.interpretation .
docker run -e HF_HOME=/tmp/hf_cache_dir \
    -v $(pwd)/config.json:/config.json -v $(pwd)/logs:/logs -v /tmp:/tmp \
    -v $(pwd)/src/interpretation:/src -v $(pwd)/datasets:/datasets -v $(pwd)/src/init.sql:/init.sql \
    --runtime=nvidia --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    hermes-interpretation \
    /bin/bash