from time import time, sleep

from transformers import pipeline
import torch

from runners.asr_utils import (
    remove_duplicates_regex,
    remove_duplicates_regex_simple,
    normalize_smart_light,
)


def load_whisper_cpu(model_name_str, lang):
    model = pipeline(
        "automatic-speech-recognition",
        model=model_name_str,
        return_timestamps=True,
        generate_kwargs={"language": lang},
        device="cpu",
    )
    return model


def load_whisper_cuda(model_name_str, lang):
    print(f"Loading whisper model {model_name_str}...")

    try:
        model = pipeline(
            "automatic-speech-recognition",
            model=model_name_str,
            return_timestamps=True,
            generate_kwargs={"language": lang},
            device="cuda",
        )
    except Exception as e:
        print(f"Error loading Whisper model {model_name_str}: {e}")
        print("Falling back to CPU...")
        # If CUDA is not available, load the model on CPU
        model = load_whisper_cpu(model_name_str, lang)
    print("Loaded whisper model...")
    return model


def whisper_worker_process(
    whisper_mname, audio_queue, result_queue, hardware_config, language, n_cpus
):
    # Cada worker carrega seu próprio modelo
    # calls_being_processed_lockset[worker_index] = -1
    print("whisper_worker_process: Starting model loading")
    torch.set_num_threads(n_cpus)
    if hardware_config == "cuda":
        whisper_model = load_whisper_cuda(whisper_mname, language)
    elif hardware_config == "cpu":
        whisper_model = load_whisper_cpu(whisper_mname, language)
    torch.set_num_threads(n_cpus)

    print("Loaded model!")
    while True:
        audio_id, id_emergencia, start_time, sampling_rate, audio_path = (
            audio_queue.get()
        )
        try:
            if audio_path == "STOP":
                break
            start_time = time()
            result = whisper_model(audio_path)
            time_spent = time() - start_time
            transcript = result["text"]
            transcript = normalize_smart_light(transcript)
            transcript = remove_duplicates_regex_simple(transcript)
            result = {
                "audio_id": audio_id,
                "id_emergencia": id_emergencia,
                "part": transcript,
                "start_time": start_time,
                "transcription_seconds": time_spent,
                "transcription_model": whisper_mname,
            }
            # print(f'result: {result}')
            result_queue.put(result)
        except Exception as err:
            print(
                f"Error processing audio {audio_path}, removing processing_running flag"
            )
            print(f"transcript: {transcript}")
            print(f"Erro: {err}")
            print(err)
            print(err.with_traceback(None))
            result_queue.put(
                {
                    "audio_id": audio_id,
                    "id_emergencia": id_emergencia,
                    "part": "",
                    "start_time": None,
                    "transcription_seconds": 0.0,
                    "transcription_model": whisper_mname,
                    "error": str(err),
                }
            )
            sleep(5)

        sleep(0.05)
