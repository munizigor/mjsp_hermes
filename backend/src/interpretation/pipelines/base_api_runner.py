from typing import List, Tuple, Any, Optional
from multiprocessing.pool import ThreadPool

from abc import ABC, abstractmethod

from model_runners.factories import load_llm_client

class BaseAPIRunner(ABC):
    """
    Interface/implementação mínima para runners que retornam respostas estruturadas.
    Subclasses devem implementar `extract_structured`, `create_interpretation_a` e
    `create_interpretation_b` conforme necessidade.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        # subclasses podem definir self.context_len e self.client
        self.context_len: Optional[int] = 500

    @abstractmethod
    def supports_parallel(self) -> bool:
        """Retorna True se a implementação suporta chamadas paralelas."""
        raise NotImplementedError

    @abstractmethod
    def extract_structured(
        self, input_dict: dict
    ) -> Tuple[Any, Optional[int], Optional[int], float, Any]:
        """
        Deve executar uma única chamada e retornar:
        (structured_resp, input_tokens, output_tokens, processing_time, raw_response)
        """
        raise NotImplementedError

    def extract_multiple_structured(
        self, prompts: List[str], fmt: Any, max_concurrent: int = 10
    ):
        """
        Implementação padrão: usa ThreadPool para mapear `extract_structured`.
        Subclasses podem sobrescrever para usar async nativo ou clients específicos.
        """
        pool = ThreadPool(processes=max_concurrent)
        responses = pool.map(
            self.extract_structured, [{"prompt": p, "format": fmt} for p in prompts]
        )
        return responses

    @abstractmethod
    def create_interpretation_a(self, transcriptions: List[str]):
        """Wrapper que converte transcriptions em interpretações A (schema definido)."""
        raise NotImplementedError

    @abstractmethod
    def find_similar_natures(self, descs: List[str], max_n: int, transcripts: List[str] = []):
        """Wrapper que encontra naturezas similares."""
        raise NotImplementedError

    @abstractmethod
    def create_interpretation_b(self, transcriptions: List[str], classification_lists):
        """Wrapper que decide a natureza da ocorrência."""
        raise NotImplementedError

    @abstractmethod
    def create_envolvimentos(self, transcriptions: List[str]):
        """Wrapper que decide os envolvimentos das pessoas na ocorrência."""
        raise NotImplementedError

class RemoteInferenceAPIRunner(BaseAPIRunner):

    def __init__(self, model_name: str, client_type: str):
        super().__init__(model_name)
        self.client_type = client_type
        self.client = load_llm_client(client_type, model_name)
        self.context_len = self.client.context_len
    
    def extract_structured(self, input_dict):
        """
        Envia o prompt para o client e tenta obter saída parseada (estrutura definida via Pydantic).

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