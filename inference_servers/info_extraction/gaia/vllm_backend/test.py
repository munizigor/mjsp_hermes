import os
import time
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

# --- Configuration ---
# Get server URL and model name from environment variables, similar to your curl script
TRITON_SERVER_URL = os.environ.get("TRITON_SERVER_URL", "0.0.0.0:8000")
LLM_NAME = os.environ.get("LLM_NAME", "text_llm3")

# --- Function to Run Inference ---
def run_llm_inference(client: InferenceServerClient, model_name: str, text_input: str, max_tokens: int):
    """
    Constructs the request and sends it to the Triton server for LLM inference.
    """
    print(f"\n--- Running inference for: '{text_input[:50]}...' ---")
    start_time = time.time()

    # 1. Prepare Input Tensor for 'text_input'
    # The API doesn't specify the expected shape/datatype for 'text_input',
    # but for string input to a common text model, we treat it as a string
    # with shape [1] (one batch item).
    # NOTE: The LLM model may expect 'text_input' to be a string type.
    # We convert the text_input string into a numpy array of objects/strings.
    # The name 'text_input' must match the model's expected input name.
    input_text = InferInput('text_input', [1], "BYTES")
    input_data = np.array([text_input.encode('utf-8')], dtype=object)
    input_text.set_data_from_numpy(input_data, binary_data=True)

    # 2. Prepare Output
    # The output from a 'generate' endpoint is typically 'text_output'.
    # We use InferRequestedOutput to specify which output we want.
    output_text = InferRequestedOutput('text_output', binary_data=True)
    # The model may also have a 'stream' output or other outputs,
    # but we'll focus on the main text output for this example.

    # 3. Define Custom Parameters
    # These are passed in the 'parameters' field of the HTTP request,
    # and the tritonclient's `infer` method handles this via the `parameters` argument.
    custom_parameters = {
        "max_tokens": max_tokens,
        #"stream": False
    }

    # 4. Perform Synchronous Inference
    # The tritonclient handles marshalling all components into the correct HTTP request structure.
    client.get_inference_statistics
    try:
        response = client.infer(
            model_name=model_name,
            inputs=[input_text],
            outputs=[output_text],
            parameters=custom_parameters,
            # For the /generate endpoint, the model_version is often left as default
            # model_version='',
        )

        # 5. Process the Result
        # Use as_numpy() to get the output data as a numpy array.
        output_data = response.as_numpy('text_output')

        # Decode and print the result
        if output_data is not None and len(output_data) > 0:
            # The result is likely a numpy array of bytes (b'...')
            decoded_text = output_data[0].decode('utf-8')
            print(f"Generated Text:\n{decoded_text}")
        else:
            print("ERROR: Could not retrieve 'text_output' from the response.")

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return

    end_time = time.time()
    print(f"\nTime taken: {end_time - start_time:.4f} seconds")
    print("-" * 60)


# --- Main Execution ---
if __name__ == '__main__':
    # Initialize the client
    # concurrency=1 is the default and sufficient for sequential calls
    # verbose=True can be helpful for debugging
    client = InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)
    
    # 1. Health Checks (Good practice before inference)
    try:
        print(f"Server Ready: {client.is_server_ready()}")
        print(f"Model '{LLM_NAME}' Ready: {client.is_model_ready(model_name=LLM_NAME)}")
        if not client.is_model_ready(model_name=LLM_NAME):
             print(f"FATAL: Model '{LLM_NAME}' is not ready. Please check the server logs.")
             exit(1)
    except Exception as e:
        print(f"Could not connect to the server at {TRITON_SERVER_URL}. Is it running? Error: {e}")
        exit(1)
    
    print("Starting inference requests...")
    
    # 2. First Request
    run_llm_inference(
        client=client,
        model_name=LLM_NAME,
        text_input="Dê uma descrição curta sobre a cidade de São Paulo.",
        max_tokens=128
    )

    # 3. Second Request
    run_llm_inference(
        client=client,
        model_name=LLM_NAME,
        text_input="Quais são os benefícios da energia solar?",
        max_tokens=256
    )

    # Clean up the client resources (optional but good practice)
    client.close()