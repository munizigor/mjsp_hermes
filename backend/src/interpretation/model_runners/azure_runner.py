import json
import os
import sys
import time
from typing import List

from openai import AzureOpenAI
from pydantic import ValidationError

from model_runners.templates import (
    user_template_with_schema,
    EmergencyInterpretationA,
    NaturezaOcorrenciaA,
    prompt_b_template,
    InterpretationClient,
)

env_vals = {
    rawline.split("=")[0]: rawline.split("=")[1].rstrip("\n")
    for rawline in open(".env", "r").read().split("\n")
    if "=" in rawline
}
for k, v in env_vals.items():
    os.environ[k] = v

azure_context_lengths = {"gpt-5-nano": 400000}


class AzureAPIRunner(InterpretationClient):
    """
    Runner que encapsula chamadas á API Azure OpenAI via `AzureOpenAI`.

    Responsabilidades:
    - Construir e manter um cliente `AzureOpenAI`.
    - Fornecer métodos para executar prompts que retornam saída estruturada (parse).

    Variáveis de ambiente necessárias:
    - AZURE_AI_RESOURCE_KEY_{model_name}
    - AZURE_AI_RESOURCE_VERSION_{model_name}
    - AZURE_AI_RESOURCE_ENDPOINT_{model_name}

    Nota sobre erros:
    - `extract_structured` captura `pydantic.ValidationError` e retorna o erro no quinto campo da tupla de retorno.
    - Falhas de chave de ambiente resultam em KeyError durante inicialização.
    """

    system_prompt = "Você é um assistente que sempre responde estritamente no formato JSON especificado."

    def __init__(self, model_name):
        """
        Inicializa o runner para o `model_name` especificado.

        Args:
            model_name (str): nome do modelo conforme configurado nas variáveis de ambiente
        Raises:
            KeyError: se qualquer variável de ambiente esperada não existir
            KeyError: se `model_name` não estiver presente em `azure_context_lengths`
        """
        super().__init__(model_name)
        # Leitura de credenciais / endpoint a partir de variáveis de ambiente montadas
        self.api_key = os.environ["AZURE_AI_RESOURCE_KEY_" + model_name]
        try:
            self.api_version = os.environ["AZURE_AI_RESOURCE_VERSION_" + model_name]
        except KeyError:
            self.api_version = os.environ["AZURE_AI_RESOURCE_VERSION"]
        try:
            self.endpoint = os.environ["AZURE_AI_RESOURCE_ENDPOINT_" + model_name]
        except KeyError:
            self.endpoint = os.environ["AZURE_AI_RESOURCE_ENDPOINT"]
        self.model_name = model_name

        # Cria o cliente AzureOpenAI. O cliente espera os parâmetros mostrados.
        # print(self.api_key)
        # print(self.api_version)
        # print(self.endpoint)
        # print(self.model_name)
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
        )

        # Comprimento de contexto estimado para o modelo (usado como referência/internamente).
        self.context_len = azure_context_lengths.get(model_name, 10000)

    def extract_structured(self, input_dict):
        """
        Envia o prompt para o Azure e tenta obter saída parseada (estrutura definida via Pydantic).

        Args:
            input_dict (dict): deve conter:
                - 'prompt' (str): texto a ser enviado
                - 'format' (str): especificador de formatação/text_format esperado pela API

        Returns:
            tuple: (parsed_output_or_None, input_tokens_or_None, output_tokens_or_None, inference_time_seconds, raw_response_or_exception)
                - parsed_output_or_None: objeto parseado conforme o schema (ou None)
                - input_tokens_or_None, output_tokens_or_None: contagem de tokens quando disponível
                - inference_time_seconds: tempo de inferência em segundos
                - raw_response_or_exception: objeto de resposta cru do client ou a exceção capturada

        Tratamento de erros:
            - Captura ValidationError do pydantic e retorna o erro no lugar do raw_response.
        """

        prompt = input_dict["prompt"]
        format = input_dict["format"]
        # print(type(prompt))
        # print(type(format))
        # print(self.model_name)
        inf_start = time.time()
        try:
            response = self.client.responses.parse(
                model=self.model_name,
                input=prompt,
                text_format=format,
                reasoning={"effort": "minimal"},
            )
        except ValidationError as err:
            # Erro de validação do schema: retorna metadados de erro (sem levantar)
            print("Pydantic Validation error for prompt:\n", prompt, file=sys.stderr)
            print(type(err))
            print(err)
            print(err.args)
            print(err.__traceback__)
            print(err.__context__)
            return None, None, None, time.time() - inf_start, err
        except Exception as err:
            inf_time = time.time() - inf_start
            print(
                f"\n🚨 [AzureAPIRunner] API Connection/Execution Error!",
                file=sys.stderr,
            )
            print(f"   Model: {self.model_name}", file=sys.stderr)
            print(f"   Endpoint: '{self.endpoint}'", file=sys.stderr)
            print(f"   API Version: {self.api_version}", file=sys.stderr)
            print(f"   Error Type: {type(err).__name__}", file=sys.stderr)
            print(f"   Message: {err}\n", file=sys.stderr)

            # Return the error so the redo_queue in agent.py can safely handle it
            # instead of crashing the thread.
            return None, None, None, inf_time, err
        inf_time = time.time() - inf_start

        resp0 = response.output_parsed
        if resp0 is not None:
            # print(response.usage)
            output_tokens = response.usage.output_tokens
            input_tokens = response.usage.input_tokens
            return resp0, input_tokens, output_tokens, inf_time, response
        else:
            # Parse falhou, mas retorno cru é preservado para diagnóstico
            return None, None, None, inf_time, response
