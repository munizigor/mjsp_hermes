
def load_llm_client(client_type, model_name):
    if client_type == "azure-api":
        from model_runners.azure_runner import AzureAPIRunner
        client = AzureAPIRunner(model_name)
    elif client_type == "gcp-api":
        from model_runners.gcp_runner import GCPTextAPIRunner
        client = GCPTextAPIRunner(model_name)
    elif client_type == "triton-server":
        from model_runners.triton_runner import TritonCustomHTTPClient
        client = TritonCustomHTTPClient(model_name)
    elif client_type == "vllm-api":
        from model_runners.vllm_runner import VLLMTextAPIRunner
        client = VLLMTextAPIRunner(model_name)
    elif client_type == "serpro-llm":
        from model_runners.serpro_runner import SerproLLMTextAPIRunner
        client = SerproLLMTextAPIRunner(model_name)
    else:
        raise ValueError(f"Unknown client type: {client_type}")
    
    return client
    
    