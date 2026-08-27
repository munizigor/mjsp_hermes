import time
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    """
    Classe do modelo Python para o Triton.
    """

    def initialize(self, args):
        """
        Chamado uma vez quando o modelo é carregado.
        Carrega o tokenizador e o modelo da Hugging Face.
        """
        self.model_id = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"
        
        print(f"Carregando tokenizador {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        print(f"Carregando modelo {self.model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16, # Muito mais estável que float16
            device_map="auto"           # Deixa o accelerate gerenciar a GPU
        )

    def execute(self, requests):
        """
        Chamado para cada BATCH de inferência.
        """
        responses = []
        
        # 1. Coletar todos os prompts do lote
        prompt_strings = []
        for request in requests:
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            prompt_bytes = prompt_tensor.as_numpy().flatten()[0]
            prompt_strings.append(prompt_bytes.decode("utf-8"))

        # 2. Formatar o prompt de chat para todos os itens do lote
        all_messages = [
            [{"role": "user", "content": p}] for p in prompt_strings
        ]
        chat_prompts = self.tokenizer.apply_chat_template(
            all_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 3. Tokenizar o LOTE de prompts
        #    padding=True é essencial para processamento em lote!
        inputs = self.tokenizer(
            chat_prompts, 
            return_tensors="pt",
            padding=True,  # <--- ESSENCIAL
            truncation=True
        ).to(self.model.device)

        # 4. Gerar texto para o LOTE inteiro de uma vez
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        
        # 5. Decodificar a saída do lote
        input_length = inputs["input_ids"].shape[1]
        # Use batch_decode para decodificar todos de uma vez
        generated_texts = self.tokenizer.batch_decode(
            outputs[:, input_length:], 
            skip_special_tokens=True
        )

        # 6. Criar e enviar as respostas individuais
        for text in generated_texts:
            output_tensor = pb_utils.Tensor(
                "GENERATED_TEXT",
                np.array([text], dtype=object)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """
        Chamado quando o modelo é descarregado.
        """
        print("Limpando o modelo...")
        self.model = None
        self.tokenizer = None
        print("Finalizado.")