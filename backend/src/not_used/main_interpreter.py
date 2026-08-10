

import json
import subprocess

import ollama

from pydantic import BaseModel

def stop_ollama():
    cmd = ["ollama", "stop", "cnmoro/gemma3-gaia-ptbr-4b:q8_0"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Saída:", result.stdout)
    if result.stderr:
        print("Erro:", result.stderr)

class EmergencyData(BaseModel):
    endereco_completo: str | None
    ponto_referencia: str | None
    nome_do_solicitante: str | None
    numero: str | None
    descricao_breve: str | None
    outras_observacoes: str | None
    tipo_chamada: str | None

class EmergencyDataNER(BaseModel):
    rua_ou_logradouro: str | None
    numero: str | None
    bairro: str | None
    municipio: str | None
    ponto_referencia: str | None
    nome_do_solicitante: str | None
    complemento: str | None

fields_to_find_ner = '''
rua_ou_logradouro: Nome da rua ou logradouro da emergência, caso seja citado durante a transcrição.
numero: Número da casa/prédio/local na rua da emergência, caso seja citado durante a transcrição.
bairro: Bairro da emegência, caso seja citado durante a descrição.
municipio: Cidade da emergência, caso seja citado durante a descrição.
ponto_referencia: Ponto de referência mais próximo do endereço, caso seja informado.
nome_do_solicitante: Nome do solicitante, caso ele tenha se identificado.
complemento: Complemento ao número do endereço na rua (número/código de apartmento, condomínio, etc...), quando aplicável.
'''

brackets = "{ and }"

prompt_a = f"""
You are an expert in interpreting call transcripts for emergency services. Your task is to extract relevant information
from the provided call transcript. The information you need to extract includes (field name: description):
{fields_to_find_ner}

In case an information is missing, do not write "Missing", "Não especificado", "Faltando" ou "Não encontrado".
In those cases, just report an empty string ("") or omit the field.
Please analyze the following call transcript and extract the required information:
<transcript_placeholder>
"""


"""
Json error correction example:
try:
    response = response.strip('`').strip('json')
    new_lines = []
    for line in response.split('\n'):
        if line.startswith('- "'):
            line = line.replace('- "', '"')
        new_lines.append(line)
    response_fixed = '\n'.join(new_lines)
    response_dict = json.loads(response_fixed)
except json.JSONDecodeError as err:
    print("Error decoding JSON response")
    print(err)
    print(err.__traceback__)
    print("Response was:")
    print(response)
    return {}
"""

def ollama_interpret_call_transcript(transcript: str, 
        llm_name = 'cnmoro/gemma3-gaia-ptbr-4b:q8_0', llm_provider = 'ollama') -> dict:
    """
    Interpret a call transcript and extract relevant information.

    Args:
        transcript (str): The call transcript to interpret.

    Returns:
        dict: A dictionary containing the extracted information.
    """
    prompt = prompt_a.replace('<transcript_placeholder>', transcript)
    #print(prompt)
    response_meta = {}
    result = None
    if llm_provider == 'ollama':
        format = EmergencyDataNER.model_json_schema()
        result = ollama.generate(model=llm_name, prompt=prompt, format=format)
        #print(result)
        total_time = (result['total_duration'] - result['load_duration']) / 1000000000
        prompt_time = result['prompt_eval_duration'] / 1000000000
        prompt_tokens = result['prompt_eval_count']
        result_time = result['eval_duration'] / 1000000000
        result_tokens = result['eval_count']

        response_meta = {
            'total_time': total_time,
            'prompt_time': prompt_time,
            'prompt_tokens': prompt_tokens,
            'result_tokens': result_tokens,
            'result_time': result_time,
            'time_other_operations': total_time - (prompt_time + result_time),
            'model': llm_name,
        }
        response_dict = EmergencyDataNER.model_validate_json(result['response']).model_dump()
        #print(json.dumps(result, indent=4, ensure_ascii=False))
        #response = result['response']
    
    empty_fields = [key for key, value in response_dict.items() if value in [None, '', ' ', 'null'] or value != value]
    for f in empty_fields:
        del response_dict[f]

    if 'prompt_time' in response_meta and 'prompt_tokens' in response_meta:
        response_meta['prompt_tokens_per_second'] = response_meta['prompt_tokens'] / response_meta['prompt_time']
    if 'result_time' in response_meta and 'result_tokens' in response_meta:
        response_meta['result_tokens_per_second'] = response_meta['result_tokens'] / response_meta['result_time']
    
    return response_dict, response_meta