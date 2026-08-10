import json
import os
from enum import Enum, IntEnum
import re
from typing import List, Tuple, Any, Optional
import time
from multiprocessing.pool import ThreadPool

from pydantic import BaseModel, Field
import pandas as pd
from abc import ABC, abstractmethod


def extrair_json_do_markdown(texto: str):
    # Extrai o bloco JSON
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if not match:
        raise ValueError("Não foi encontrado JSON válido no texto")
    json_str = match.group(0)

    # Corrige aspas duplas internas que não estejam escapadas
    # Substitui aspas duplas repetidas por aspas simples (ou remove extras)
    # Cuidado: essa regex pode precisar ser ajustada conforme o padrão do seu texto
    json_str = re.sub(r'""', '"', json_str)

    # Também pode tentar escapar aspas internas entre aspas:
    # json_str = re.sub(r'(?<=: )"([^"]*)"', lambda m: '"' + m.group(1).replace('"', '\\"') + '"', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Erro ao decodificar JSON: {e}\nJSON extraído:\n{json_str}")


class TipoDeParticipacao(str, Enum):
    VITIMA = "Vítima"
    VITIMA_FATAL = "Vítima Fatal"
    TESTEMUNHA = "Testemunha"
    COMUNICANTE = "Comunicante"
    OUTRO = "Outro"
    FORAGIDO = "Foragido"
    DESAPARECIDO = "Desaparecido"
    CONSELHEIRO_TUTELAR = "Conselheiro Tutelar"
    CONDUTOR_VEICULAR = "Condutor Veicular"
    AUTOR_VITIMA = "Autor/Vítima"
    AUTOR = "Autor"
    AVERIGUADO = "Averiguado"
    AUSENTE = "Ausente"
    ADVOGADO = "Advogado"
    INFRATOR = "Infrator"


class ChanceDeClassificacao(str, Enum):
    Complete_Certainty_100 = "Certamente Sim"
    High_Probability_85 = "Sim"
    Moderate_Probability_75 = "Provavelmente Sim"
    Chances_Consideraveis_70 = "Chances Consideráveis"
    Talvez_65 = "Talvez"
    Complete_Uncertainty_50 = "Não sei"
    Low_Probability_35 = "Provavelmente Não"
    Very_Unlikely_10 = "Não"
    Impossible_0 = "Certamente Não"


chance_de_classificacao_to_float = {
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

"""class Participacao(BaseModel):
    chance: ChanceDeClassificacao
    envolvimento: TipoDeParticipacao
"""


class EnvolvimentoPessoa(BaseModel):
    nome_pessoa: str = Field(description="Nome da pessoa envolvida na ocorrência.")
    participacao: TipoDeParticipacao = Field(
        description="Tipo de participação da pessoa na ocorrência."
    )
    probabilidade: float = Field(
        description="Número entre 0.0 e 1.0. Probabilidade do tipo de participação ser adequado a essa pessoa."
    )


class EnvolvimentosEmergencia(BaseModel):
    envolvimentos: List[EnvolvimentoPessoa] = Field(
        description="List de envolvimentos de pessoas na ocorrência."
    )


class EmergencyInterpretationABasic(BaseModel):
    descricao_breve: str = Field(
        description="Breve descrição da ocorrência da chamada de emergência (máximo de 200 caracteres)."
    )
    outras_observacoes: str = Field(
        description="Outras observações que o solicitante tenha feito durante a transcrição, mas que não estão presentes na descrição breve."
    )
    # tipo_chamada: str = Field(description="Tipo de ligação recebida.",
    #    enum=["Ocorrência", 'Ligação Muda', 'Trote', 'Queda de Ligação', 'Informação', 'Agradecimento', 'Denúncia'])


class EmergencyInterpretationA(BaseModel):
    descricao_breve: str = Field(
        description="Breve descrição da ocorrência da chamada de emergência (máximo de 200 caracteres)."
    )
    outras_observacoes: str = Field(
        description="Outras observações que o solicitante tenha feito durante a transcrição, mas que não estão presentes na descrição breve."
    )
    """ponto_de_referencia: Optional[str] = Field(
        None,
        description="Ponto de referência (rua, logradouro, etc.). Máximo de 100 caracteres.",
    )
    nome_do_solicitante: Optional[str] = Field(
        None,
        description="Nome do solicitante na chamada. Pessoa que fez a chamada. Máximo de 150 caracteres.",
    )"""


emergencyInterpretationA_schema_str = json.dumps(
    EmergencyInterpretationA.model_json_schema(), ensure_ascii=False, indent=2
)


class NaturezaOcorrenciaA(BaseModel):
    natureza_da_ocorrencia_indice: int = Field(
        description="Indice da natureza de ocorrência mais adequada. Usar 0 (zero) quando não for nenhuma das opções.",
        default=0,
    )


emergencyInterpretationB_schema_str = json.dumps(
    NaturezaOcorrenciaA.model_json_schema(), ensure_ascii=False, indent=2
)

prompt_a_template = """
You are an expert in interpreting call transcripts for emergency services. Your task is to extract relevant information
from the provided call transcript.
Please analyze the following call transcript and extract the required information:
{[[[[transcript_content]]]]}
"""

prompt_b_template = f"""
Você receberá a transcrição de uma chamada de emergência de ocorrência. Deve decidir qual, 
dentre uma série de naturezas de ocorrência padronizadas, é a mais adequada. 
Cada natureza tem um índice e você deverá responder com o indice da opção correta.
Usar 0 (zero) quando não for nenhuma das opções.
Sua tarefa é  produzir um JSON estritamente válido de acordo com o seguinte schema:
{emergencyInterpretationB_schema_str}

Lista de naturezas:
\nnaturezas_similares

\nTranscrição da chamada:
[[[[transcript_content]]]]
"""

user_template_with_schema = f"""
Você é um assistente especializado em extrair informações de transcrições de chamadas de emergência.
Sua tarefa é analisar uma transcrição e produzir um JSON estritamente válido de acordo com o seguinte schema:
{emergencyInterpretationA_schema_str}

Você não deve discursar sobre o significado do schema, nem incluir elementos pré-textuais antes das informações dele. 
Apenas dê a saída em JSON puro e estritamente válido.

Você deve analisar a seguinte transcrição:
[[[[transcript_content]]]]
"""

naturezas_path = "/datasets/naturezas.csv"
if not os.path.exists(naturezas_path):
    naturezas_path = naturezas_path.lstrip("/")
naturezas_vec = pd.read_csv(naturezas_path, sep=";")["NO_NATUREZA_INICIAL"].to_list()


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
    def create_interpretation_b(self, transcriptions: List[str], classification_lists):
        """Wrapper que decide a natureza da ocorrência."""
        raise NotImplementedError

    @abstractmethod
    def create_envolvimentos(self, transcriptions: List[str]):
        """Wrapper que decide os envolvimentos das pessoas na ocorrência."""
        raise NotImplementedError
