import json
import time
from typing import List

from interpretation.model_runners.llama_cpp_runner import Llama

from model_runners.templates import (extrair_json_do_markdown,
    user_template_with_schema,
    emergencyInterpretationA_schema_str,
    prompt_b_template, BaseAPIRunner)

class LlamaCppRunner(BaseAPIRunner):
    """
    Runner que utiliza um backend Llama local (através do wrapper `Llama`, do modulo llama_cpp).

    Atributos de classe:
        system_prompt (str): prompt de sistema usado nas chamadas de chat.
        low_context (int): tamanho de contexto menor (para GPUs de baixo recurso).
        high_context (int): tamanho de contexto maior (para setups com mais memória).
    """
    system_prompt = "Você é um assistente que sempre responde estritamente no formato JSON especificado."
    low_context = 1500
    high_context = 2400

    def __init__(self, repo_id="cnmoro/Gemma-3-Gaia-PT-BR-4b-it-Q8_0-GGUF",
            filename="gemma-3-gaia-pt-br-4b-it-q8_0.gguf",
            low_end_gpu = True,
            full_cpu = False,
            n_cpus=6):
        """
        Inicializa o runner carregando o modelo local via `Llama.from_pretrained`.

        Args:
            repo_id (str): identificador/URL do repositório do modelo (ou prefixo).
            filename (str): nome do arquivo do modelo dentro do repo/local.
            low_end_gpu (bool): se True, configura n_ctx e n_gpu_layers para GPUs com menos VRAM.
            full_cpu (bool): se True, força modo somente-CPU.
            n_cpus (int): número de threads/cpus a expor ao backend.
        """
        model_name = repo_id + '/' + filename
        super().__init__(model_name)
        self.model_name = model_name
        self.repo_id = repo_id
        self.filename = filename

        # Seleciona comprimento de contexto de acordo com recursos esperados
        self.context_len = LlamaCppRunner.low_context if (low_end_gpu and not full_cpu) else LlamaCppRunner.high_context

        # Carrega o modelo via wrapper Llama; parâmetros como n_gpu_layers e n_ctx
        # são configurados conforme as flags passadas.
        if full_cpu:
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_threads=n_cpus,
                n_gpu_layers=0,     # 0 = só CPU; use -1 para tudo na GPU (se houver VRAM),
                max_tokens=2048,
                verbose=False,
                rope_scaling_type=1,   # Enable RoPE scaling
                n_ctx=LlamaCppRunner.high_context  # Defina o tamanho do contexto explicitamente
            )
        else:
            if low_end_gpu:
                self.llm = Llama.from_pretrained(
                    repo_id=repo_id,
                    filename=filename,
                    n_threads=n_cpus,
                    n_gpu_layers=15,     # 0 = só CPU; use -1 para tudo na GPU (se houver VRAM),
                    max_tokens=2048,
                    verbose=False,
                    rope_scaling_type=1,   # Enable RoPE scaling
                    n_ctx=LlamaCppRunner.low_context  # Defina o tamanho do contexto explicitamente
                )
            else:
                self.llm = Llama.from_pretrained(
                    repo_id=repo_id,
                    filename=filename,
                    n_gpu_layers=-1,     # 0 = só CPU; use -1 para tudo na GPU (se houver VRAM),
                    max_tokens=2048,
                    verbose=False,
                    rope_scaling_type=1,   # Enable RoPE scaling
                    n_ctx=LlamaCppRunner.high_context  # Defina o tamanho do contexto explicitamente
                )
    
    def supports_parallel(self):
        """
        Indica se o runner suporta execução paralela.
        Retorna:
            bool: False (não seguro para paralelismo neste runner).
        """
        return False
    
    def make_one_a(self, transcription: str):
        """
        Gera uma interpretação do tipo A para uma única transcrição.

        Args:
            transcription (str): texto da transcrição.

        Retorna:
            tuple: (parsed_json_or_exception, meta_dict)
                - parsed_json_or_exception: dicionário parseado extraído do markdown (ou resposta bruta em caso de erro)
                - meta_dict: dicionário com métricas (processing_time, no_gpu_time, tokens, model_name)
        """
        cpu_start = time.time()
        user_prompt = user_template_with_schema.replace('[[[[transcript_content]]]]', transcription)
        s = time.time()
        resp = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": LlamaCppRunner.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": emergencyInterpretationA_schema_str
            },
        )
        interval = time.time() - s

        try:
            resp2 = extrair_json_do_markdown(resp['choices'][0]['message']['content'])
        except Exception as err:
            print(err)
            return resp, err
        
        input_tokens = resp['usage']['prompt_tokens']
        output_tokens = resp['usage']['completion_tokens']
        #print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        return resp2, {'meta': {'processing_time': interval, 
            'no_gpu_time': time.time() - cpu_start - interval,
            'input_tokens': input_tokens, 
            'output_tokens': output_tokens,
            'model_name': self.model_name}}

    def create_interpretation_a(self, transcriptions: List[str]):
        return [self.make_one_a(t) for t in transcriptions]
    
    def make_one_b(self, transcription: str, classifications: list):
        """
        Decide a 'natureza da ocorrência' para uma transcrição entre candidatos.

        Args:
            transcription (str): texto da transcrição.
            classifications (list): lista de candidatos (cada item é uma tupla/estrutura esperada pelo template).

        Retorna:
            tuple: (natureza_name_or_None, meta_dict)
        """
        cpu_start = time.time()
        classifications_indexed = {n+1: c for n, c in enumerate([x for s, x in classifications])}
        classifications_json = json.dumps(classifications_indexed, ensure_ascii=False)
        #print(classifications_json)
        user_prompt = prompt_b_template.replace(
            'naturezas_similares', classifications_json).replace(
                '[[[[transcript_content]]]]', transcription
            )
        
        s = time.time()
        resp = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Você é um assistente que sempre responde estritamente no formato especificado."},
                {"role": "user", "content": user_prompt}
            ]
        )
        interval = time.time()-s

        input_tokens = resp['usage']['prompt_tokens']
        output_tokens = resp['usage']['completion_tokens']

        nat_id = resp['choices'][0]['message']['content']
        try:
            nat_id = int(nat_id.rstrip('\n').strip())
        except Exception as err:
            print(err)
            return resp, {'meta': {'message': resp['choices'][0]['message'],
                'error': err,
                'input_tokens': input_tokens, 
                'output_tokens': output_tokens}}
        #print('natureza_correta', nat_id)
        if nat_id in classifications_indexed:
            nat_name = classifications_indexed[nat_id]
        else:
            nat_name = None
        #print(nat_name)
        return nat_name, {'meta': 
                {
                'message': resp['choices'][0]['message'], 
                'processing_time': interval,
                'no_gpu_time': time.time() - cpu_start - interval,
                'input_tokens': input_tokens, 
                'output_tokens': output_tokens,
                'model_name': self.model_name
                }
            }
    
    def create_interpretation_b(self, transcriptions: List[str], classification_lists):
        return [self.make_one_b(t, cls) for t, cls in zip(transcriptions, classification_lists)]