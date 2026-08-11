import json
import subprocess
import pydantic
from typing import List, Optional
from enum import Enum
from pydantic import Field
import ollama
from pydantic import BaseModel

brackets = "{ and }"




class RespostaDeClassificacao(str, Enum):
    CERTAMENTE_SIM = "Certamente Sim"
    SIM = "Sim"
    PROVAVELMENTE_SIM = "Provavelmente Sim"
    NAO_SEI = "Não sei"
    PROVAVELMENTE_NAO = "Provavelmente Não"
    NAO = "Não"
    CERTAMENTE_NAO = "Certamente Não"

class ChanceDeClassificacao(str, Enum):
    Complete_Certainty_100 = "Certamente Sim"
    High_Probability_85 = "Sim"
    Moderate_Probability_65 = "Provavelmente Sim"
    Chances_Consideraveis_70 = "Chances Consideráveis"
    Talvez_65 = "Talvez"
    Complete_Uncertainty_50 = "Não sei"
    Low_Probability_35 = "Provavelmente Não"
    Very_Unlikely_10 = "Não"
    Impossible_0 = "Certamente Não"

class InformacoesOcorrencia(pydantic.BaseModel):
    #detalhes_gravidade_ocorrencia: DetalhesGravidadeOcorrencia
    fato_ocorrendo_neste_momento: Optional[ChanceDeClassificacao] = Field(
        None, 
        description="Indica se o fato está ocorrendo neste momento."
    )
    autor_do_fato_no_local: Optional[ChanceDeClassificacao] = Field(
        None, 
        description="Indica se o autor do fato está no local."
    )
    autor_do_fato_armado: Optional[ChanceDeClassificacao] = Field(
        None, 
        description="Indica se o autor do fato está armado."
    )
    feridos_com_risco_de_morte: Optional[ChanceDeClassificacao] = Field(
        None, 
        description="Indica se há feridos com risco de morte."
    )
    '''ferimentos_graves: Optional[ChanceDeClassificacao] = Field(
        None, 
        description="Indica se há ferimentos graves."
    )'''
    risco_de_tumulto: Optional[ChanceDeClassificacao] = Field(
        None, 
        description="Indica se a ocorrência pode causar risco de tumulto."
    )
    lei_maria_da_penha: Optional[ChanceDeClassificacao] = Field(
        None, 
        description="Indica se a ocorrência envolve violência doméstica ou familiar (Lei Maria da Penha)."
    )
    violencia_domestica: Optional[ChanceDeClassificacao] = Field(
        None, 
        description="Indica se a ocorrência envolve violência doméstica ou familiar (Lei Maria da Penha)."
    )
    rua_ou_logradouro: List[str] = pydantic.Field(default_factory=list)
    street_number: List[str] = pydantic.Field(default_factory=list)
    complemento: List[str] = pydantic.Field(default_factory=list)
    bairro: List[str] = pydantic.Field(default_factory=list)
    cidade: List[str] = pydantic.Field(default_factory=list)
    ponto_de_referencia: List[str] = pydantic.Field(default_factory=list)
    nome_do_solicitante: List[str] = pydantic.Field(default_factory=list)
    pessoa: List[str] = pydantic.Field(default_factory=list)

answer_to_value = {
    "Certamente Sim": 1.0,
    "Sim": 0.85,
    "Provavelmente Sim": 0.75,
    "Chances Consideráveis": 0.7,
    "Talvez": 0.65,
    "Não sei": 0.5,
    "Provavelmente Não": 0.35,
    "Não": 0.1,
    "Certamente Não": 0.0,
}

question_mapping = {
    "fato_ocorrendo_neste_momento": "Fato Ocorrendo Neste Momento ?",
    "autor_do_fato_no_local": "Autor Do Fato No Local ?",
    "autor_do_fato_armado": "Autor Do Fato Armado ?",
    "feridos_com_risco_de_morte": "Feridos Com Risco de Morte ?",
    #"ferimentos_graves": "Feridos Com Risco de Morte ?",
    "risco_de_tumulto": "Risco De Tumulto ?",
    "lei_maria_da_penha": "Lei Maria da Penha ?",
    "violencia_domestica": "Lei Maria da Penha ?",
}

prompt_a = f"""
You are an expert in interpreting call transcripts for emergency services. 
Your task is to extract relevant information from the provided call transcript.
Be informative and reliable. We need these informations for alocating forces.
There are several types of informations and classifications for you to do:
<template_placeholder>
Additional instructions for filling in details:
- When asked about the name of something, fill in only the name. No additional descriptions;
- If you don't have the answer, return an empty list. Do not say "Não sei" or "Não informado" or something like this.
- Avoid pretextual and postextual elements. Just use the transcript.
- When the field is a classification / chance, use one of the given options.
- When many elements, avoid concatenating verbally with "e" or "and". Just put them in the list.

Please analyze the following call transcript:
<transcript_placeholder>
"""

def stop_ollama(model_name):
    cmd = ["ollama", "stop", model_name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Saída:", result.stdout)
    if result.stderr:
        print("Erro:", result.stderr)
        return False
    return True

def ollama_interpret_call_transcript(transcript: str, fmt_class,
        llm_name = 'cnmoro/gemma3-gaia-ptbr-4b:q8_0') -> dict:
    """
    Interpret a call transcript and extract relevant information.

    Args:
        transcript (str): The call transcript to interpret.

    Returns:
        dict: A dictionary containing the extracted information.
    """
    prompt = prompt_a.replace('<transcript_placeholder>', transcript)
    response_meta = {}
    result = None
    fmd_json = fmt_class.model_json_schema()
    prompt = prompt.replace('<template_placeholder>\n', json.dumps(fmd_json, ensure_ascii=False))
    result = ollama.generate(model=llm_name, 
        prompt=prompt, 
        format=fmd_json,
        options={'temperature': 0, 'num_predict': 1000}
    )
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
    response_dict = fmt_class.model_validate_json(result['response']).model_dump()
    #print(json.dumps(result, indent=4, ensure_ascii=False))
    #response = result['response']

    if 'prompt_time' in response_meta and 'prompt_tokens' in response_meta:
        response_meta['prompt_tokens_per_second'] = response_meta['prompt_tokens'] / response_meta['prompt_time']
    if 'result_time' in response_meta and 'result_tokens' in response_meta:
        response_meta['result_tokens_per_second'] = response_meta['result_tokens'] / response_meta['result_time']
    
    return response_dict, response_meta

def consult_ollama_emergency(transcript: str, llm_name = 'cnmoro/gemma3-gaia-ptbr-4b:q8_0') -> dict:
    response_dict, response_meta =  ollama_interpret_call_transcript(transcript, 
        InformacoesOcorrencia, llm_name=llm_name)
    entities = {}
    clfs = response_dict.get('detalhes_gravidade_ocorrencia', {})
    if clfs == None:
        clfs = {}
    for clf_name, clf_val in clfs.items():
        if clf_val is None:
            clf_val = "Não sei"
        clf_float = answer_to_value.get(clf_val, 0.5)
        clf_name_full = question_mapping[clf_name]
        if clf_name_full in entities:
            entities[clf_name_full] = (entities[clf_name_full] + clf_float) / 2
        else:
            entities[clf_name_full] = clf_float

    entity_field_mapping = {
        "rua_ou_logradouro": "rua_ou_logradouro",
        "street_number": "numero",
        "complemento": "complemento",
        "bairro": "bairro",
        "cidade": "cidade",
        "ponto_de_referencia": "ponto_de_referencia",
        "nome_do_solicitante": "nome_do_solicitante",
        "pessoa": "pessoa",
    }

    for field_name, values in response_dict.items():
        if field_name == 'detalhes_gravidade_ocorrencia':
            continue
        if field_name not in entity_field_mapping:
            print(f"Field {field_name} not in entity_field_mapping")
            continue
        json_key = entity_field_mapping[field_name]
        entities[json_key] = [(str(item), 1.0) for item in values]

    for field_name, json_key in entity_field_mapping.items():
        if not field_name in entities:
            entities[field_name] = []

    result_data = {
        "entities": entities,
        "input_tokens": response_meta['prompt_tokens'],
        "output_tokens": response_meta['result_tokens'],
        "latency": response_meta['total_time'],
    }

    return result_data

if __name__ == '__main__':
    #Run an example
    transcript_example = '''
    Solicitante: é dos bombeiros? Minha casa está pegando fogo!
    Atendente: Sim, é bombeiros. Como está a situação?
    Solicitante: Minha casa está pegando fogo!
    Atendente: Ok, posso ajudar. Por favor, me diga o endereço da sua casa.
    Solicitante: É na rua floresta da tijuca, número 201.
    Atendente: Qual seu nome?
    Solicitante: Meu nome é Joana.
    Atendente: Ok, Joana. 
    Solicitante: Foi o meu marido que colocou o fogo! Ele disse que quer me matar!
    '''

    response_dict = consult_ollama_emergency(transcript_example)

    print(json.dumps(response_dict, indent=4, ensure_ascii=False))