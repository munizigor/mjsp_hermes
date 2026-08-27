#!/bin/bash

export NUEXTRACT_TOKEN=putarealsecret

curl http://localhost:8000/health

export NUEXTRACT_TOKEN=putarealsecret && curl http://localhost:8000/v1/models -H "Authorization: Bearer $NUEXTRACT_TOKEN"

export NUEXTRACT_TOKEN=putarealsecret && curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $NUEXTRACT_TOKEN" \
    -d '{
    "model": "numind/NuExtract-2.0-4B",
    "messages": [{
        "role": "user",
        "content": "Liam is here contrary to Alex who is in Taïwan. Samuel will come tomorrow with Charles"
    }],
    "chat_template_kwargs": {
        "template": "{\"names\": [\"verbatim_string\"], \"countries\": [\"verbatim_string\"]}",
        "examples": [{
            "input": "Sam is CTO",
            "output": "{\"names\": [\"Sam\"]}"
        }]
    }
}'