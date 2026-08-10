import json
import os
import sys
import time
from typing import List, Optional, Tuple, Any

from pydantic import ValidationError, BaseModel
from openai import OpenAI, APITimeoutError, APIConnectionError

from model_runners.templates import InterpretationClient

# --- Environment Loading (Kept as is) ---
if os.path.exists(".env"):
    env_vals = {
        rawline.split("=")[0]: rawline.split("=")[1].rstrip("\n")
        for rawline in open(".env", "r").read().split("\n")
        if "=" in rawline
    }
    for k, v in env_vals.items():
        os.environ[k] = v

INTERPRETATION_VLLM_HOST = os.environ.get("INTERPRETATION_VLLM_HOST", "localhost")
INTERPRETATION_VLLM_PORT = os.environ.get("INTERPRETATION_VLLM_PORT", "8000")


class VLLMTextAPIRunner(InterpretationClient):
    """
    Runner that encapsulates vLLM OpenAI-compatible API calls.
    """

    system_prompt = "Você é um assistente que sempre responde estritamente no formato JSON especificado."

    def __init__(
        self,
        model_name: str,
        host: str = INTERPRETATION_VLLM_HOST,
        port: str = INTERPRETATION_VLLM_PORT,
    ):
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
