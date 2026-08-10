import sys
import time
import json
import requests
import os

from pydantic import ValidationError

from model_runners.templates import extrair_json_do_markdown, InterpretationClient

class TritonCustomHTTPClient(InterpretationClient):
    
    system_prompt = "Você é um assistente que sempre responde estritamente no formato JSON especificado."
    
    def __init__(self, model_name):
        self.host = os.environ.get(f'TRITON_SERVER_HOST_{model_name.upper()}', 'localhost')
        self.port = int(os.environ.get(f'TRITON_SERVER_PORT_{model_name.upper()}', '8000'))
        self.base_url = f"http://{self.host}:{self.port}"
        self.health_url = f"{self.base_url}/v2/health/ready"
        self.model_metadata_url = f"{self.base_url}/v2/models/{model_name}/metadata"
        self.generate_url = f"{self.base_url}/v2/models/{model_name}/generate"
        self.context_len = 10000
    
    # --- Function to run a request ---
    def run_http_generate(self, prompt: str, max_n_tokens: int = 2000):
        """
        Sends a POST request to the /generate endpoint and prints the result.
        """
        payload = {
            "text_input": f"<start_of_turn>user\n${prompt}<end_of_turn>\n",
            "parameters": {
                "return_num_output_tokens": True,
                "return_num_input_tokens": True,
                "max_tokens": max_n_tokens,
                "exclude_input_in_output": True,
                "stream": False # explicitly disable streaming
            }
        }
        print(f"--- Running inference for: '{payload['text_input'][:50]}...' ---")
        
        # Use 'time.time()' to mimic the 'time' shell command
        start_time = time.time()
        # Send the POST request with the payload as JSON
        response = requests.post(self.generate_url, json=payload)

        # Calculate and print the time
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"\nTime taken: {time_taken:.4f} seconds")

        # Check if the request was successful
        response.raise_for_status() 
        
        # Print the JSON response from the server
        print("Server Response:")
        # Use json.dumps for pretty-printing the output dict
        data = response.json()
        n_tokens = data.get("num_output_tokens", 1)
        tokens_per_second = n_tokens / time_taken if time_taken > 0 else 0
        print(f"Generated {n_tokens} tokens in {time_taken:.4f} seconds "
            f"({tokens_per_second:.2f} tokens/second)")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        return {
            "generated_text": data.get("text_output", ""),
            "num_output_tokens": data.get("num_output_tokens", 0),
            "num_input_tokens": data.get("num_input_tokens", 0),
            "inference_time": time_taken
        }
        
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

        prompt = input_dict['prompt']
        format = input_dict['format']
        #print(type(prompt))
        #print(type(format))
        #print(self.model_name)
        
        try:
            response_dict = self.run_http_generate(
                prompt
            )

            output_text = response_dict['generated_text']
            output_tokens = response_dict['num_output_tokens']
            input_tokens = response_dict['num_input_tokens']
            inf_time = response_dict['inference_time']
        
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            return None, None, None, None, e
        except requests.exceptions.ConnectionError as e:
            print(f"Connection Error: Could not connect to {self.generate_url}")
            return None, None, None, None, e
        except KeyError as e:
            print(f"JSON Key error:", e, response_dict)
            return None, None, None, None, e
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None, None, None, e

        resp0 = None
        try:
            #Use json custom json parser extract dict from response text
            resp_dict = extrair_json_do_markdown(output_text)
            #Use pydantic to parse the dict into the expected schema
            resp0 = format(**resp_dict)
        except ValidationError as err:
            try:
                resp0 = format.__pydantic_model__.parse_raw(output_text)
            except ValidationError as err2:
                # Erro de validação do schema: retorna metadados de erro (sem levantar)
                print('Pydantic Validation error for prompt:\n', prompt, file=sys.stderr)
                print(type(err))
                print(err)
                print(err.args)
                print(err.__traceback__)
                print(err.__context__)
                return None, input_tokens, output_tokens, inf_time, err

        if resp0 is not None:
            #print(response.usage)
            return resp0, input_tokens, output_tokens, inf_time, response_dict
        else:
            # Parse falhou, mas retorno cru é preservado para diagnóstico
            return None, input_tokens, output_tokens, inf_time, response_dict