import json
import os
import sys
import time
from typing import List, Optional

from google import genai
from google.genai import types  # Import types for Config
from pydantic import ValidationError, BaseModel

# Assuming this import exists in your project structure
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

gemini_context_lengths = {
    # --- Gemini 3.0 Family (Newest / Preview) ---
    "gemini-3.0-pro-preview": 1_048_576,
    # --- Gemini 2.5 Family (Stable / Current) ---
    "gemini-2.5-pro": 2_097_152,  # Often 2M in paid tiers, 1M in standard
    "gemini-2.5-flash": 1_048_576,  # High efficiency, 1M context
    "gemini-2.5-flash-lite": 1_048_576,  # Cost optimized
    # --- Gemini 2.0 Family ---
    "gemini-2.0-pro-exp": 2_097_152,  # Experimental 2.0 Pro often had 2M context
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.0-flash-lite-preview": 1_048_576,
    # --- Gemini 1.5 Family (Legacy / LTS) ---
    "gemini-1.5-pro": 2_097_152,  # 2M context
    "gemini-1.5-flash": 1_048_576,  # 1M context
    "gemini-1.5-flash-8b": 1_048_576,
}


class GCPTextAPIRunner(InterpretationClient):
    """
    Runner that encapsulates Google GenAI SDK (v1.0+) calls.
    """

    system_prompt = "Você é um assistente que sempre responde estritamente no formato JSON especificado."

    def __init__(self, model_name):
        super().__init__(model_name)
        self.model_name = model_name
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        try:
            # Fetch model metadata from Google directly
            model_info = self.client.models.get(model=model_name)

            # Dynamically set context length from the API metadata
            # This ensures you always have the correct limit (e.g. if Pro gets upgraded to 4M)
            self.context_len = model_info.input_token_limit
        except Exception as e:
            print(
                f"Warning: Could not fetch metadata for {model_name}. Defaulting to local model info.",
                file=sys.stderr,
            )
            try:
                self.context_len = gemini_context_lengths[self.model_name]
            except Exception as e:
                self.context_len = 1000000

    def extract_structured(self, input_dict):
        prompt = input_dict["prompt"]
        format_schema = input_dict["format"]  # Pydantic BaseModel class

        # print('Prompt:', prompt, file=sys.stderr)
        # print('Format Schema:', format_schema.schema(), file=sys.stderr)

        try:
            inf_start = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    # You can pass the Pydantic class directly in the new SDK!
                    response_schema=format_schema,
                    system_instruction=self.system_prompt,
                ),
            )
            inf_time = time.time() - inf_start

            # The SDK can often parse automatically into the Pydantic object:
            # parsed_output = response.parsed
            # However, sticking to manual validation is safe and robust:
            parsed_output = format_schema.model_validate_json(response.text)
            # print('Parsed Output:', parsed_output, file=sys.stderr)
            input_tokens = None
            output_tokens = None
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count

        except ValidationError as err:
            print("Pydantic Validation error for prompt:\n", prompt, file=sys.stderr)
            print(err, file=sys.stderr)
            return None, None, None, time.time() - inf_start, err
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            return None, None, None, time.time() - inf_start, e

        return parsed_output, input_tokens, output_tokens, inf_time, response
