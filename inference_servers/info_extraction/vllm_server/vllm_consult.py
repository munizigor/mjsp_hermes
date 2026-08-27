import sys
import json
import time
from typing import List, Optional
from openai import OpenAI, APITimeoutError, APIConnectionError
from pydantic import BaseModel, Field


class EmergencyInterpretationA(BaseModel):
    descricao_breve: str = Field(
        description="Breve descrição da ocorrência da chamada de emergência (máximo de 200 caracteres)."
    )
    outras_observacoes: str = Field(
        description="Outras observações que o solicitante tenha feito durante a transcrição, mas que não estão presentes na descrição breve."
    )
    ponto_de_referencia: str = Field(
        description="Ponto de referência (rua, logradouro, etc.). Máximo de 100 caracteres."
    )
    nome_do_solicitante: str = Field(
        description="Nome do solicitante na chamada. Pessoa que fez a chamada. Máximo de 150 caracteres."
    )


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


def vllm_interpret_call_transcript(
    transcript: str,
    fmt_class,
    model_name: str,
    host: str = "localhost",
    port: str = "8000",
):
    """
    Interpret a call transcript and extract relevant information.

    Args:
        transcript (str): The call transcript to interpret.

    Returns:
        dict: A dictionary containing the extracted information.
    """
    vllm_client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="EMPTY")
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
                timeout=12,
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

            result_data = {
                "entities": recognized_entities,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency": latency,
            }

            return result_data

        except (APITimeoutError, APIConnectionError) as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {e}")
        except Exception as e:
            # Immediate fail for non-network errors (e.g., bad request)
            raise e


from openai import OpenAI, APITimeoutError, APIConnectionError
from typing import Tuple, Any, Optional
from abc import ABC, abstractmethod

emergencyInterpretationA_schema_str = json.dumps(
    EmergencyInterpretationA.model_json_schema(), ensure_ascii=False, indent=2
)

user_template_with_schema = f"""
Você é um assistente especializado em extrair informações de transcrições de chamadas de emergência.
Sua tarefa é analisar uma transcrição e produzir um JSON estritamente válido de acordo com o seguinte schema:
{emergencyInterpretationA_schema_str}

Você não deve discursar sobre o significado do schema, nem incluir elementos pré-textuais antes das informações dele. 
Apenas dê a saída em JSON puro e estritamente válido.

Você deve analisar a seguinte transcrição:
[[[[transcript_content]]]]
"""


class InterpretationClient(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.context_len: Optional[int] = 1001

    @abstractmethod
    def extract_structured(
        self, input_dict: dict
    ) -> Tuple[Any, Optional[int], Optional[int], float, Any]:
        """
        Deve executar uma única chamada e retornar:
        (structured_resp, input_tokens, output_tokens, processing_time, raw_response)
        """
        raise NotImplementedError


class VLLMTextAPIRunner(InterpretationClient):
    """
    Runner that encapsulates vLLM OpenAI-compatible API calls.
    """

    system_prompt = "Você é um assistente que sempre responde estritamente no formato JSON especificado."

    def __init__(self, model_name: str, host: str = "localhost", port: str = "8000"):
        super().__init__(model_name)
        self.model_name = model_name
        self.base_url = f"http://{host}:{port}/v1"
        self.client = OpenAI(base_url=self.base_url, api_key="EMPTY")

        # vLLM does not expose context length easily → fallback
        self.context_len = 1000000

    def extract_structured(
        self, input_dict: dict
    ) -> Tuple[Any, Optional[int], Optional[int], float, Any]:
        """
        Executes a single structured extraction call.

        Expected input_dict format:
        {
            "prompt": str,
            "format": Pydantic BaseModel class
        }
        """

        prompt = input_dict["prompt"]
        format_schema = input_dict["format"]

        max_retries = 2
        raw_response = None

        for attempt in range(1, max_retries + 1):
            start_time = time.time()

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "schema": format_schema.model_json_schema(),
                        },
                    },
                    temperature=0,
                    timeout=12,
                    extra_body={"enable_thinking": False},
                )

                latency = time.time() - start_time
                raw_response = response

                # --- Token usage ---
                input_tokens = None
                output_tokens = None
                if response.usage:
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens

                # --- Parse JSON ---
                content_str = response.choices[0].message.content

                try:
                    parsed_output = format_schema.model_validate_json(content_str)
                except Exception:
                    # fallback if invalid JSON
                    try:
                        parsed_dict = json.loads(content_str)
                        parsed_output = format_schema.model_validate(parsed_dict)
                    except Exception as err:
                        print("Validation error for prompt:\n", prompt, file=sys.stderr)
                        print(err, file=sys.stderr)
                        return None, input_tokens, output_tokens, latency, content_str

                return parsed_output, input_tokens, output_tokens, latency, raw_response

            except (APITimeoutError, APIConnectionError) as e:
                print(f"Attempt {attempt} failed: {e}", file=sys.stderr)
                if attempt == max_retries:
                    return None, None, None, time.time() - start_time, e

            except Exception as e:
                # Non-retryable error
                print(f"Unexpected error: {e}", file=sys.stderr)
                return None, None, None, time.time() - start_time, e


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

    """response_dict = vllm_interpret_call_transcript(
        transcript_example,
        EmergencyInterpretationA,
        model_name,
        host="localhost",
        port="8000",
    )

    print(json.dumps(response_dict, indent=4, ensure_ascii=False))"""

    runner = VLLMTextAPIRunner(model_name=model_name, host="localhost", port="8000")

    prompt = user_template_with_schema.replace(
        "[[[[transcript_content]]]]", transcript_example
    )

    result = runner.extract_structured(
        {"prompt": prompt, "format": EmergencyInterpretationA}
    )

    structured, in_tokens, out_tokens, latency, raw = result

    print(f"Structured: {structured}")
    print(f"Input tokens: {in_tokens}")
    print(f"Output tokens: {out_tokens}")
    print(f"Latency: {latency}")
