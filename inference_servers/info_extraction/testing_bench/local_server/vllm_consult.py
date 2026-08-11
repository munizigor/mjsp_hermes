import sys
import json
import time
from openai import OpenAI, APITimeoutError, APIConnectionError

from local_server.ollama_consult import (
    InformacoesOcorrencia,
    answer_to_value,
    question_mapping,
    prompt_a,
)

vllm_client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")


def vllm_interpret_call_transcript(transcript: str, fmt_class, model_name: str) -> dict:
    """
    Interpret a call transcript and extract relevant information.

    Args:
        transcript (str): The call transcript to interpret.

    Returns:
        dict: A dictionary containing the extracted information.
    """
    prompt = prompt_a.replace("<transcript_placeholder>", transcript)
    response_meta = {}
    result = None
    fmd_json = fmt_class.model_json_schema()
    prompt = prompt.replace(
        "<template_placeholder>\n", json.dumps(fmd_json, ensure_ascii=False)
    )
    payload_extra = {
        "chat_template_kwargs": {
            "template": json.dumps(fmt_class.model_json_schema()),
            # You can add few-shot examples here if needed
            "examples": [],
        }
    }

    messages = [{"role": "user", "content": prompt}]

    max_retries = 2

    for attempt in range(1, max_retries + 1):
        start_time = time.time()
        try:

            response = vllm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                # enable_thinking=False,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "dados_ocorrencia",
                        "schema": fmt_class.model_json_schema(),
                    },
                },
                temperature=0,  # Low temp for extraction
                timeout=16,
                extra_body={"enable_thinking": False},
            )

            end_time = time.time()
            latency = end_time - start_time

            # 1. Parse Usage
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens

            # 2. Parse Content (The model returns a JSON string)
            content_str = response.choices[0].message.content
            try:
                recognized_entities = json.loads(content_str)
            except json.JSONDecodeError:
                # Fallback if model outputs markdown or bad JSON
                recognized_entities = {
                    "error": "Failed to parse JSON",
                    "raw_content": content_str,
                }

            return recognized_entities, input_tokens, output_tokens, latency

        except (APITimeoutError, APIConnectionError) as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {e}")
        except Exception as e:
            # Immediate fail for non-network errors (e.g., bad request)
            raise e


def process_vllm_response(response_dict, input_tokens, output_tokens, latency):
    entities = {}
    # print(json.dumps(response_dict, indent=4, ensure_ascii=False))
    for clf_name in question_mapping.keys():
        clf_val = response_dict.get(clf_name, None)
        if clf_val is None:
            clf_val = "Não sei"
        elif isinstance(clf_val, list):
            print("Value should not be a list: ", clf_val, file=sys.stderr)
            print(response_dict, file=sys.stderr)
            if len(clf_val) >= 1:
                if isinstance(clf_val[0], str):
                    clf_val = clf_val[0]
                elif isinstance(clf_val[0], dict):
                    clf_val = clf_val[0].get("value", "Não sei")
            else:
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
        if field_name == "detalhes_gravidade_ocorrencia":
            continue
        if field_name not in entity_field_mapping:
            # print(f"Field {field_name} not in entity_field_mapping")
            continue
        json_key = entity_field_mapping[field_name]
        entities[json_key] = [(str(item), 1.0) for item in values]

    for field_name, json_key in entity_field_mapping.items():
        if not field_name in entities:
            entities[field_name] = []

    result_data = {
        "entities": entities,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency": latency,
    }

    return result_data


def consult_vllm_emergency(transcript: str, model_name: str) -> dict:
    response_dict, input_tokens, output_tokens, latency = (
        vllm_interpret_call_transcript(transcript, InformacoesOcorrencia, model_name)
    )
    result_data = process_vllm_response(
        response_dict, input_tokens, output_tokens, latency
    )
    return result_data


if __name__ == "__main__":
    # Run an example
    model_name = sys.argv[1]  # or CohereLabs/tiny-aya-global
    transcript_example = """
    Solicitante: é dos bombeiros? Minha casa está pegando fogo!
    Atendente: Sim, é bombeiros. Como está a situação?
    Solicitante: Minha casa está pegando fogo!
    Atendente: Ok, posso ajudar. Por favor, me diga o endereço da sua casa.
    Solicitante: É na rua floresta da tijuca, número 201.
    Atendente: Qual seu nome? 
    Solicitante: Meu nome é Joana.
    Atendente: Ok, Joana.
    Eu também estou pegando fogo! Vou desmaiar
    
    """

    response_dict = consult_vllm_emergency(transcript_example, model_name)

    print(json.dumps(response_dict, indent=4, ensure_ascii=False))
