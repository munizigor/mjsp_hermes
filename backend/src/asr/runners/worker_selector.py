
def select_worker(hardware_config):
    if hardware_config in ["cpu", "cuda"]:
        from runners.local_whisper import whisper_worker_process

        worker_func = whisper_worker_process
    elif hardware_config == "azure-api":
        from runners.azure import azure_caller_process

        worker_func = azure_caller_process
    elif hardware_config == "triton-server":
        from runners.triton import triton_caller_process

        worker_func = triton_caller_process
    elif hardware_config == "gcp-api":
        from runners.chirp import google_caller_process

        worker_func = google_caller_process
    elif hardware_config == "flask-server":
        from runners.flask_server import flask_caller_process

        worker_func = flask_caller_process
    elif hardware_config == "vllm-api":
        from runners.vllm_server import vllm_caller_process

        worker_func = vllm_caller_process
    elif hardware_config == "whisper-cpp-api":
        from runners.whisper_cpp import whisper_cpp_caller_process

        worker_func = whisper_cpp_caller_process
    else:
        raise Exception("Invalid hardware-config for ASR")
    
    return worker_func