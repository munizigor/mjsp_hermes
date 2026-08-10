import os
import sys
import json
from time import sleep

from asr_runner import AsrRunner

#for key, val in os.environ.items():
#    print(key, val)

asr_models = {
    'base': 'openai/whisper-base',
    'small': 'openai/whisper-small',
    'medium': 'my-north-ai/whisper-medium-pt',
    'large': 'nilc-nlp/distil-whisper-coraa-mupe-asr',
    'turbo': 'openai/whisper-large-v3-turbo',
    'turbo_cuda': 'turbo_cuda',
    'azure-fast': 'azure-fast-transcribe'
}

if __name__ == "__main__":
    config_json = json.load(open('/config.json', 'r'))
    asr_config = config_json.get('asr', {})
    hardware_config = asr_config.get('hardware_config', {})
    asr_model = asr_config.get('model', 'large')
    language = asr_config.get('language', 'pt')
    n_asr_workers = asr_config.get('n_asr_workers', 1)
    n_cpus = asr_config.get('n_cpus', 6)
    
    while not os.path.exists(os.environ["SQLITE_DB_PATH"]):
        print(f'asr/start.py waiting 4s for {os.environ["SQLITE_DB_PATH"]}')
        sleep(4)
    if asr_model in asr_models:
        full_mname = asr_models[asr_model]
    else:
        full_mname = asr_model

    whisperer = AsrRunner(whisper_mname=full_mname,
        n_asr_workers=n_asr_workers,
        hardware_config=hardware_config,
        language=language,
        n_cpus=n_cpus
    )

    while True:
        sleep(2)
    