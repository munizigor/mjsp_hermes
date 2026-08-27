
#Test GCP and Azure Speech ASR endpoints

import io
import sys
import traceback
import os
import json
import concurrent.futures
from time import time, sleep
import hashlib

import requests
from google.api_core import client_options
from google.cloud import speech_v2
from google.cloud.speech_v2 import types as speech_types
import soundfile as sf
from tqdm import tqdm
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.transcription import TranscriptionClient
from azure.ai.transcription.models import TranscriptionContent, TranscriptionOptions, EnhancedModeProperties

# Força o carregamento do arquivo .env
load_dotenv()

from asr_lib.utils import get_inputs, save_results, clean_reference_text
from asr_lib.audio_processing import prepare_audio_for_apis

# Make sure you have these environment variables set
GCP_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")
GCP_REGION = os.environ.get("GOOGLE_RECOGNIZER_REGION")
GCP_RECOGNIZER_ID = os.environ.get("GOOGLE_RECOGNIZER_ID") # e.g., 'chirp-recognizer'

AZURE_REGION = os.environ.get('AZURE_SERVICE_REGION')
AZURE_KEY = os.environ.get('AZURE_AI_RESOURCE_KEY')

AZURE_LLM_ENDPOINT = os.environ.get('AZURE_LLM_SPEECH_ENDPOINT')
AZURE_LLM_KEY = os.environ.get('AZURE_LLM_SPEECH_API_KEY')

def get_audio_id(item):
    """
    Extrai um ID estável do item do dataset. 
    Tenta buscar colunas conhecidas ou gera um hash MD5 do áudio real como fallback infalível.
    
    correct usage of dataset audio items:
    id_str = item["ID"]
    modelo_audio = item["audio"]
    audio_array = audio["array"]
    sr = audio["sampling_rate"]
    """


    # 1. Tenta buscar identificadores comuns nas chamadas de emergência do dataset
    audio_name = item["ID"] + "_" + item["modelo_audio"]
    return audio_name.replace('/', '_').replace('\\', '_')

def array_to_wav_bytes(audio_array, sr):
    """Converts a numpy array to a WAV byte stream in memory."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sr, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer.read()

def call_azure_api(audio_bytes):
    url = f"https://{AZURE_REGION}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-11-15"
    headers = {"Ocp-Apim-Subscription-Key": AZURE_KEY}
    definition = {"locales": ["pt-br"]}
    files = {"audio": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"definition": json.dumps(definition, ensure_ascii=False)}

    request_start = time()
    resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    request_time = time() - request_start
    resp.raise_for_status()
    
    if resp.status_code in (200, 201):
        result_json = resp.json()
        
        # Parse based on Azure's Fast Transcription response structure
        if 'combinedPhrases' in result_json:
            transcription = '\n'.join([x['text'] for x in result_json['combinedPhrases']])
        else:
            transcription = result_json.get('text', '')
            
        return transcription, request_time
    else:
        raise Exception(f"Azure API returned status {resp.status_code}")

def call_azure_llm_speech_api(audio_bytes):
    if not AZURE_LLM_ENDPOINT or not AZURE_LLM_KEY:
        raise ValueError("Credenciais do AZURE_LLM_SPEECH não encontradas no .env!")

    client = TranscriptionClient(
        endpoint=AZURE_LLM_ENDPOINT, 
        credential=AzureKeyCredential(AZURE_LLM_KEY)
    )
    
    # Implementação de Exponential Backoff conforme recomendado pela documentação (preview)
    backoff_times = [2, 4, 8, 16, 32]
    
    for attempt, delay in enumerate(backoff_times + [0]):
        try:
            # Envolve os bytes em um objeto similar a arquivo para o SDK
            audio_stream = io.BytesIO(audio_bytes)
            audio_stream.name = "audio.wav" # O SDK exige um nome para inferir o formato
            
            # Utiliza o "prompt-tuning" sugerido na documentação para guiar o modelo
            enhanced_mode = EnhancedModeProperties(
                task="transcribe",
                prompt=[
                    "Transcreva o áudio em português do Brasil.",
                    "O áudio é uma chamada de emergência, capture tudo com precisão."
                ]
            )
            
            options = TranscriptionOptions(
                locales=["pt-BR"], 
                enhanced_mode=enhanced_mode
            )
            
            request_content = TranscriptionContent(
                definition=options, 
                audio=audio_stream
            )

            request_start = time()
            result = client.transcribe(request_content)
            request_time = time() - request_start
            
            if result and result.combined_phrases:
                text = " ".join([phrase.text for phrase in result.combined_phrases])
                return text, request_time
            return "", request_time
            
        except Exception as e:
            # Se for a última tentativa, repassa o erro para travar a thread
            if attempt == len(backoff_times):
                raise e
            print(f"[WARN] Azure LLM API Rate Limit ou Erro: {e}. Tentando novamente em {delay}s...")
            sleep(delay)

def call_google_chirp_api(model_id, audio_bytes):
    # Proteção caso o .env falhe
    if not GCP_PROJECT_ID:
        raise ValueError("GOOGLE_PROJECT_ID não encontrado! Verifique o seu arquivo .env.")

    # FIX: Roteamento dinâmico de região baseado na sua lista do GCP
    recognizer_to_regions = {
        "hermes-chirp3-us": "us",
        "hermes-telephony-us": "us-central1",
        "hermes-chirp2-us-central1": "us-central1",
    }
    if model_id in recognizer_to_regions:
        region = recognizer_to_regions[model_id]
    else:
        region = "us"
        
    api_endpoint = f"{region}-speech.googleapis.com"
    
    client_opts = client_options.ClientOptions(api_endpoint=api_endpoint)
    client = speech_v2.SpeechClient(client_options=client_opts)
    
    recognizer_path = f"projects/{GCP_PROJECT_ID}/locations/{region}/recognizers/{model_id}"
    
    
    request = speech_types.RecognizeRequest(
        recognizer=recognizer_path,
        config=speech_types.RecognitionConfig(
            auto_decoding_config=speech_types.AutoDetectDecodingConfig(),
            language_codes=["pt-BR"],
        ),
        content=audio_bytes,
    )
    
    request_start = time()
    response = client.recognize(request=request, timeout=60)
    request_time = time() - request_start
    
    full_transcription_parts = []
    for result in response.results:
        if len(result.alternatives) > 0:
            full_transcription_parts.append(result.alternatives[0].transcript)
    
    #print(full_transcription_parts)
            
    return "\n".join(full_transcription_parts), request_time

def process_single_api_sample(idx, item, model_id, cache_dir):
    """Processa um único áudio na API, verificando o cache de forma determinística primeiro."""
    audio = item["audio"]
    audio_array = audio["array"]
    sr = audio["sampling_rate"]
    audio_len = len(audio_array) / sr

    # Converte o array para bytes WAV em memória
    audio_bytes = array_to_wav_bytes(audio_array, sr)
    
    # Gera o ID inteligente e monta o caminho do cache
    stable_id = get_audio_id(item)
    cached_path = os.path.join(cache_dir, f"{stable_id}.json")

    # Verifica o cache
    if os.path.exists(cached_path):
        with open(cached_path, "r", encoding="utf-8") as f:
            try:
                cached_data = json.load(f)
                return idx, cached_data["text"], cached_data["processing_time"], audio_len, True
            except:
                pass
    #split audio if longer than 30 seconds
    if audio_len > 30:
        audio_segments = prepare_audio_for_apis(audio_array, sr)
    else:
        audio_segments = [audio_bytes]

    if model_id == "azure-fast-transcribe":
        sleep(3)
        
    # Se não está no cache, chama a API correspondente
    total_latency = 0.0
    out_text = ""
    for audio_bytes_chunk in audio_segments:
        if model_id == "azure-fast-transcribe":
            sleep(1.5)
            transcription, request_time = call_azure_api(audio_bytes_chunk)
            out_text += transcription + "\n"
            total_latency += request_time
        elif model_id == "azure-llm-speech":
            text_part, request_time = call_azure_llm_speech_api(audio_bytes_chunk)
            out_text += text_part + "\n"
            total_latency += request_time
        elif "chirp" in model_id or "telephony" in model_id:
            transcription, request_time = call_google_chirp_api(model_id, audio_bytes_chunk)
            out_text += transcription + "\n"
            total_latency += request_time
        else:
            raise ValueError(f"Modelo de API desconhecido: {model_id}")
    processing_time = total_latency

    # Salva no cache com o ID estável
    with open(cached_path, "w", encoding="utf-8") as f:
        json.dump({
            "text": out_text.strip(), 
            "processing_time": processing_time
        }, f, ensure_ascii=False)

    return idx, out_text.strip(), processing_time, audio_len, False

def test_api_model(model_id, n_samples, use_concurrency=True):
    print(f"Testing API model {model_id} on {n_samples} samples concurrently...")
    
    ds = get_inputs(max_n=n_samples)
    original_scripts = ds["roteiro_segmentado"]
    
    cache_dir = os.path.join(".asr_cache", model_id.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)

    # Use a dictionary to keep results sorted by their original dataset index
    results_by_idx = {}
    
    # Configure max_workers carefully to avoid hitting HTTP 429 Too Many Requests
    max_concurrent_requests = 4 
    
    
    try:
        if use_concurrency:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_single_api_sample, i, item, model_id, cache_dir): i 
                    for i, item in enumerate(ds)
                }
                
                # Process as they complete
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    idx, out_text, proc_time, audio_len, is_cached = future.result()
                    results_by_idx[idx] = {
                        "out_text": out_text,
                        "proc_time": proc_time,
                        "audio_len": audio_len
                    }
        else:
            for i, item in tqdm(enumerate(ds), total=len(ds)):
                idx, out_text, proc_time, audio_len, is_cached = process_single_api_sample(i, item, model_id, cache_dir)
                results_by_idx[idx] = {
                    "out_text": out_text,
                    "proc_time": proc_time,
                    "audio_len": audio_len
                }
                
    except KeyboardInterrupt:
        print("\n[!] Ctrl-C detected! Canceling pending API requests...")
        for f in futures:
            f.cancel() # Stops threads that haven't started yet
        raise # Reraise to stop the main script execution

    # Assemble results in the correct order
    transcriptions = []
    processing_times = []
    audio_lens = []
    texts_segmented = []
    
    for i in range(len(ds)):
        res = results_by_idx[i]
        transcriptions.append(res["out_text"])
        processing_times.append(res["proc_time"])
        audio_lens.append(res["audio_len"])
        texts_segmented.append(original_scripts[i])

    # ---------------------------------------------------------------------
    # From here, apply your existing evaluation logic (texts_joined, normalize_light, etc.)
    # ---------------------------------------------------------------------
    
    texts_joined = []
    for text in texts_segmented:
        lines = [line[-1] if len(line) == 2 else line[0] for line in text]
        ref_raw = ' . '.join(lines)
        texts_joined.append(clean_reference_text(ref_raw))

    raw_result_lines = []
    for i in range(len(ds)):
        raw_result_lines.append({
            "audio_len": audio_lens[i],
            "processing_time": processing_times[i],
            "original": texts_joined[i],
            "transcribed": transcriptions[i],
        })
        
    return raw_result_lines

if __name__ == "__main__":
    n_samples_to_test = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    results_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    os.makedirs(results_dir, exist_ok=True)
    api_models = [
        "azure-fast-transcribe", 
        "hermes-chirp3-us", 
        "hermes-telephony-us", 
        "hermes-chirp2-us-central1", 
        "azure-llm-speech",

    ]
    all_raw_results = []
    for model_id in api_models:
        try:
            raw_result_lines = test_api_model(model_id, n_samples_to_test,
                use_concurrency="azure" not in model_id)
            for line in raw_result_lines:
                line['model_id'] = model_id
            
            all_raw_results.extend(raw_result_lines)
            save_results(all_raw_results, results_dir)
            
        except KeyboardInterrupt:
            print("Execution stopped by user.")
            break
        except Exception as e:
            print(f"Error testing model {model_id}: {e}")
            print(e)
            print(traceback.format_exc())
            quit(1)