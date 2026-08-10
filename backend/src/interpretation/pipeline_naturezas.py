import json
import os
import sys
import time
from typing import List

from model_runners.templates import (
    user_template_with_schema,
    EmergencyInterpretationA,
    NaturezaOcorrenciaA,
    prompt_b_template,
    BaseAPIRunner,
    EnvolvimentoPessoa,
    EnvolvimentosEmergencia,
    chance_de_classificacao_to_float,
)

env_vals = {
    rawline.split("=")[0]: rawline.split("=")[1].rstrip("\n")
    for rawline in open(".env", "r").read().split("\n")
    if "=" in rawline
}
for k, v in env_vals.items():
    os.environ[k] = v


class PipelineNaturezasAPI(BaseAPIRunner):
    """
    Runner que encapsula chamadas á API de geração de texto para determinar naturezas de ocorrencia.

    Responsabilidades:
    - Expor métodos de alto nível usados pelo fluxo de interpretação do projeto:
        - create_interpretation_a: extrai descrição breve e observações de transcrições
        - create_interpretation_b: escolhe a natureza da ocorrência baseado em candidatos

    Observação sobre paralelismo:
    - `supports_parallel` retorna True (o runner foi projetado para permitir chamadas paralelas).
    """

    system_prompt = "Você é um assistente que sempre responde estritamente no formato JSON especificado."

    def __init__(self, model_name, client_type="azure-api"):
        """
        Inicializa o runner para o `model_name` especificado.

        Args:
            model_name (str): nome do modelo conforme configurado nas variáveis de ambiente
        Raises:
            KeyError: se qualquer variável de ambiente esperada não existir
            KeyError: se `model_name` não estiver presente em `azure_context_lengths`
        """
        super().__init__(model_name)
        self.client_type = client_type
        if self.client_type == "azure-api":
            from model_runners.azure_runner import AzureAPIRunner

            self.client = AzureAPIRunner(model_name)
        elif self.client_type == "gcp-api":
            from model_runners.gcp_runner import GCPTextAPIRunner

            self.client = GCPTextAPIRunner(model_name)
        elif self.client_type == "triton-server":
            from model_runners.triton_runner import TritonCustomHTTPClient

            self.client = TritonCustomHTTPClient(model_name)
        elif self.client_type == "vllm-api":
            from model_runners.vllm_runner import VLLMTextAPIRunner

            self.client = VLLMTextAPIRunner(model_name)
        self.context_len = self.client.context_len

    def supports_parallel(self):
        """
        Indica se o runner suporta execução em paralelo.
        Retorno:
            bool: True se suportado.
        """
        return True

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

        out = self.client.extract_structured(input_dict)
        struct_resp, input_tokens, output_tokens, inf_time, raw_resp = out
        return struct_resp, input_tokens, output_tokens, inf_time, raw_resp

    def create_interpretation_a(self, transcriptions: List[str]):
        """
        Fluxo de alto-nível que cria interpretações (tipo A) para uma lista de transcrições.

        Args:
            transcriptions (List[str]): lista de strings com transcrições

        Retorna:
            List[tuple]: cada item é (resultado_dict_or_None, meta)
                - resultado_dict (se não None) deve conter:
                    'descricao_breve', 'outras_observacoes'
                - meta contém campos:
                    processing_time, no_gpu_time, input_tokens, output_tokens, model_name, etc.
        """
        cpu_start = time.time()
        user_prompts = [
            user_template_with_schema.replace("[[[[transcript_content]]]]", t)
            for t in transcriptions
        ]

        results = self.extract_multiple_structured(
            user_prompts, EmergencyInterpretationA
        )
        results2 = []
        max_inf_time = max([tp[-2] for tp in results])
        for structured_resp, tokens1, tokens2, inf_time, raw in results:
            no_gpu_time = time.time() - cpu_start - max_inf_time
            if structured_resp is not None:
                r_dict = {
                    "descricao_breve": structured_resp.descricao_breve,
                    "outras_observacoes": structured_resp.outras_observacoes,
                    # "ponto_de_referencia": structured_resp.ponto_de_referencia,
                    # "nome_do_solicitante": structured_resp.nome_do_solicitante,
                }

                try:
                    r_dict["ponto_de_referencia"] = structured_resp.ponto_de_referencia
                except Exception as err:
                    r_dict["ponto_de_referencia"] = None

                try:
                    r_dict["nome_do_solicitante"] = structured_resp.nome_do_solicitante
                except Exception as err:
                    r_dict["nome_do_solicitante"] = None

                # print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
                meta = {
                    "meta": {
                        "processing_time": inf_time,
                        "no_gpu_time": no_gpu_time,
                        "input_tokens": tokens1,
                        "output_tokens": tokens2,
                        "model_name": self.model_name,
                    }
                }
                results2.append((r_dict, meta))
            else:
                meta = {
                    "meta": {
                        "message": str(raw),
                        "processing_time": inf_time,
                        "no_gpu_time": no_gpu_time,
                        "error": """Error at extract_multiple_structured from 
                        PipelineNaturezasAPI.create_interpretation_a. """,
                        "input_tokens": tokens1,
                        "output_tokens": tokens2,
                        "model_name": self.model_name,
                    }
                }
                print("Error metadata:", file=sys.stderr)
                print(meta, file=sys.stderr)
                results2.append((None, meta))

        return results2

    def create_interpretation_b(self, transcriptions: List[str], classification_lists):
        """
        Fluxo que decide a 'natureza da ocorrencia' baseado em uma lista de candidatos.

        Args:
            transcriptions (List[str]): lista de transcrições
            classification_lists: lista paralela com candidatos (formato esperado pelo template)

        Retorna:
            List[tuple]: cada item é (natureza_nome_or_None, meta)
                - natureza_nome_or_None: string com nome da natureza ou None
                - meta: dicionário com métricas, tokens e mensagens de erro (se houver)
        """

        cpu_start = time.time()
        user_prompts = []
        for transcription, classifications in zip(transcriptions, classification_lists):
            classifications_indexed = {
                n + 1: c for n, c in enumerate([x for s, x in classifications])
            }
            classifications_json = json.dumps(
                classifications_indexed, ensure_ascii=False
            )

            user_prompt = prompt_b_template.replace(
                "naturezas_similares", classifications_json
            ).replace("[[[[transcript_content]]]]", transcription)
            user_prompts.append(user_prompt)

        results = self.extract_multiple_structured(user_prompts, NaturezaOcorrenciaA)
        max_inf_time = max([tp[-2] for tp in results])
        results2 = []
        for structured_resp, tokens1, tokens2, inf_time, raw in results:
            no_gpu_time = time.time() - cpu_start - max_inf_time
            meta = {
                "meta": {
                    "processing_time": inf_time,
                    "no_gpu_time": no_gpu_time,
                    "input_tokens": tokens1,
                    "output_tokens": tokens2,
                    "model_name": self.model_name,
                }
            }
            if structured_resp is not None:
                try:
                    nat_id = structured_resp.natureza_da_ocorrencia_indice
                except Exception as err:
                    print("Natureza ocorrencia inválida:" + str(err), file=sys.stderr)
                    meta["meta"]["message"] = str(raw)
                    meta["meta"]["error"] = err
                    meta["meta"][
                        "where"
                    ] = "Error at nat_id = structured_resp.natureza_da_ocorrencia_indice"
                    results2.append((raw, meta))

                # print('natureza_correta', nat_id)
                if nat_id in classifications_indexed:
                    nat_name = classifications_indexed[nat_id]
                else:
                    nat_name = None

                results2.append((nat_name, meta))
            else:
                meta["meta"]["message"] = str(raw)
                meta["meta"][
                    "error"
                ] = """Error at extract_multiple_structured from 
                    PipelineNaturezasAPI.create_interpretation_a."""
                results2.append((None, meta))
        return results2

    def create_envolvimentos(self, transcriptions: List[str]):
        """
        Fluxo de alto-nível que cria envolvimentos para uma lista de transcrições.

        Args:
            transcriptions (List[str]): lista de strings com transcrições

        Retorna:
            List[tuple]: cada item é (resultado_dict_or_None, meta)
                - resultado_dict (se não None) deve conter:
                    'envolvimentos_emergencia' (dict)
                - meta contém campos:
                    processing_time, no_gpu_time, input_tokens, output_tokens, model_name, etc.
        """
        cpu_start = time.time()
        user_prompts = [
            user_template_with_schema.replace("[[[[transcript_content]]]]", t)
            for t in transcriptions
        ]

        results = self.extract_multiple_structured(
            user_prompts, EnvolvimentosEmergencia
        )
        results2 = []
        max_inf_time = max([tp[-2] for tp in results])
        for structured_resp, tokens1, tokens2, inf_time, raw in results:
            no_gpu_time = time.time() - cpu_start - max_inf_time
            if structured_resp is not None:
                envolvimentos = structured_resp.envolvimentos
                # convert envolvimentos to json-compat dict
                envolvimentos_dicts = [e.model_dump() for e in envolvimentos]
                for e in envolvimentos_dicts:
                    print(e, file=sys.stderr)

                r_dict = {
                    "envolvimentos_emergencia": envolvimentos_dicts,
                }

                # print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
                meta = {
                    "meta": {
                        "processing_time": inf_time,
                        "no_gpu_time": no_gpu_time,
                        "input_tokens": tokens1,
                        "output_tokens": tokens2,
                        "model_name": self.model_name,
                    }
                }
                print("Nova interpretação de envolvimentos:", r_dict, file=sys.stderr)
                print(meta, file=sys.stderr)
                results2.append((r_dict, meta))
            else:
                meta = {
                    "meta": {
                        "message": str(raw),
                        "processing_time": inf_time,
                        "no_gpu_time": no_gpu_time,
                        "error": """Error at extract_multiple_structured from 
                        PipelineNaturezasAPI.create_envolvimentos. """,
                        "input_tokens": tokens1,
                        "output_tokens": tokens2,
                        "model_name": self.model_name,
                    }
                }
                print("Error metadata:", file=sys.stderr)
                print(meta, file=sys.stderr)
                results2.append((None, meta))

        return results2


if __name__ == "__main__":
    # Bloco de demonstração: executa interpretações de exemplo quando o arquivo é executado diretamente.
    transcript_text = "Operador: Corpo de Bombeiros, em que posso ajudar?\nSolicitante: Moço, eu preciso de ajuda. Um rapaz aqui, ele bateu a cabeça, tá no chão e parece que não consegue levantar.\nOperador: Ele está consciente?\nSolicitante: Não sei ao certo, ele tá falando muito baixo, tá tonto, meio zonzo. E sangrando um pouco na testa.\nOperador: Entendi. Ele chegou a cair de algum lugar alto?\nSolicitante: Não, acho que ele escorregou e bateu a cabeça na parede. Mas foi uma pancada forte.\nOperador: Você consegue dizer onde estão?\nSolicitante: Aqui na BR-369, número 456, é numa loja 1445. Perto da parada, Parada Brasília.\nOperador: Certo, entendi. Ele se queixa de dor em mais algum lugar?\nSolicitante: Ele só reclama muito de dor na cabeça. Tá com a mão no rosto, dizendo que tá tonto.\nOperador: Tudo bem, peço que evite mexer muito nele. Deixe-o deitado até chegarmos. Algum ponto de referência?\nSolicitante: É bem ao lado da parada. Tô aqui fora da loja 1445. Dá pra ver de longe.\nOperador: Vamos a caminho. Se ele parar de responder ou se o sangramento aumentar, me ligue de volta, tá bom? Fique ao lado dele e tente acalmá-lo.\nSolicitante: Tá certo. Obrigado, tô aguardando vocês."
    transcript_text2 = "Operador: Corpo de Bombeiros, em que posso ajudar?\nSolicitante: Moço, eu preciso de ajuda. Um rapaz aqui, ele bateu a cabeça, tá no chão e  e MORREU!\nOperador: Entendi. Solicitante: Aqui na BR-369, número 456, é numa loja 1445"
    client_types = [
        ("gaia", "triton-server"),
        ("gpt-5-nano", "azure-api"),
    ]
    for model_name, client_type in client_types:
        print(
            f"\n\n--- Testing PipelineNaturezasAPI with client_type={client_type} and model={model_name} ---\n"
        )
        interpreter = PipelineNaturezasAPI(model_name, client_type=client_type)
        user_prompts = [
            user_template_with_schema.replace(
                "[[[[transcript_content]]]]", transcript_text
            ),
            user_template_with_schema.replace(
                "[[[[transcript_content]]]]", transcript_text2
            ),
        ]
        resps = interpreter.extract_multiple_structured(
            user_prompts, EmergencyInterpretationA
        )
        for structured_resp, tokens1, tokens2, inf_time, raw in resps:
            print(structured_resp)
            print(inf_time, tokens1, tokens2)
