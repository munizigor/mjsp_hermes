import os
import time
import json
import requests # Use the 'requests' library to make simple HTTP calls

# --- Configuration ---
# Get server URL and model name from environment variables
# Note: Add 'http://' prefix if not already present
TRITON_SERVER_URL = os.environ.get("TRITON_SERVER_URL", "0.0.0.0:8000")
LLM_NAME = os.environ.get("LLM_NAME", "text_llm3")

# Ensure the URL has the http prefix
if not TRITON_SERVER_URL.startswith("http"):
    TRITON_SERVER_URL = f"http://{TRITON_SERVER_URL}"

# Construct the specific '/generate' endpoint URL
GENERATE_URL = f"{TRITON_SERVER_URL}/v2/models/{LLM_NAME}/generate"

print(f"Targeting Triton endpoint: {GENERATE_URL}\n")

prompts = [
    ("Dê uma descrição curta sobre a cidade de São Paulo.",
        128),
    ("Quais são os benefícios da energia solar?",
        256)
]

# --- Function to run a request ---
def run_http_generate(prompt, n_tokens):
    """
    Sends a POST request to the /generate endpoint and prints the result.
    """
    payload = {
        "text_input": f"<start_of_turn>user\n${prompt}<end_of_turn>\n",
        "parameters": {
            "return_num_output_tokens": True,
            "return_num_input_tokens": True,
            "max_tokens": n_tokens,
            "exclude_input_in_output": True,
            "stream": False # The key: explicitly disable streaming
        }
    }
    print(f"--- Running inference for: '{payload['text_input'][:50]}...' ---")
    
    # Use 'time.time()' to mimic the 'time' shell command
    start_time = time.time()
    
    try:
        # Send the POST request with the payload as JSON
        response = requests.post(GENERATE_URL, json=payload)
        
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

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: Could not connect to {GENERATE_URL}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    print("-" * 60)

# --- Main Execution ---
if __name__ == '__main__':
    for prompt, n_tokens in prompts:
        run_http_generate(prompt, n_tokens)