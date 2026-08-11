import json
from time import sleep
import re

#mp.set_start_method('spawn', force=True)
#print("spawned")

from time import time

#import logging

#import requests
import numpy as np
import triton_python_backend_utils as pb_utils


# The following functions are no longer needed after refactoring
def remove_duplicates_regex(text: str, max_ngram: int = 5) -> str:
    """
    Remove repetições consecutivas de n-gramas (1..max_ngram).
    Ex.: "fez a de fez a de fez a de" -> "fez a de"
    Mantém separadores (pontuação/espacos) na medida do possível.
    """
    import re

    if not text or text.strip() == "":
        return text

    # Tokeniza em palavras (\w+) e não-palavras (separadores)
    tokens = re.findall(r'\w+|\W+', text, flags=re.UNICODE)
    word_indices = [i for i, tok in enumerate(tokens) if re.match(r'\w+', tok, flags=re.UNICODE)]
    words = [tokens[i] for i in word_indices]
    lower_words = [w.lower() for w in words]

    if not words:
        return text

    keep_word = [True] * len(words)
    i = 0
    L = len(words)

    while i < L:
        matched = False
        # tenta maiores n-grams primeiro
        max_n = min(max_ngram, L - i)
        for n in range(max_n, 0, -1):
            seq = tuple(lower_words[i:i + n])
            j = i + n
            # conta quantas vezes a seq se repete consecutivamente
            while j + n <= L and tuple(lower_words[j:j + n]) == seq:
                j += n
            if j > i + n:
                # houve repetição: marca palavras repetidas para remoção
                for k in range(i + n, j):
                    keep_word[k] = False
                i = j  # pula bloco repetido
                matched = True
                break
        if not matched:
            i += 1

    # Reconstrói texto: mantém separadores e apenas palavras marcadas
    out = []
    widx = 0
    for idx, tok in enumerate(tokens):
        if re.match(r'\w+', tok, flags=re.UNICODE):
            if keep_word[widx]:
                out.append(tok)
            # se palavra removida, não append; mantemos separadores seguintes normalmente
            widx += 1
        else:
            out.append(tok)

    result = ''.join(out)
    # Normaliza espaços extras introduzidos pela remoção
    result = re.sub(r'\s{2,}', ' ', result)
    # Remove espaço antes de pontuação (opcional, melhora saída)
    result = re.sub(r'\s+([,.;:!?])', r'\1', result)
    return result.strip()

def remove_duplicates_regex_simple(seq):
    my_output = re.sub(r'\b(\w+)(?:\W+\1\b)+', r'\1', seq, flags=re.IGNORECASE)
    return my_output

def load_whisper_cpu(model_name_str, lang):
    from transformers import pipeline
    model = pipeline("automatic-speech-recognition", 
            model=model_name_str,
            return_timestamps=True,
            generate_kwargs={"language": lang},
            device='cpu')
    return model

def load_whisper_cuda(model_name_str, lang):
    print(f'Loading whisper model {model_name_str}...')
    from transformers import pipeline

    try:
        model = pipeline("automatic-speech-recognition", 
            model=model_name_str,
            return_timestamps=True,
            generate_kwargs={"language": lang},
            device='cuda')
    except Exception as e:
        print(f"Error loading Whisper model {model_name_str}: {e}")
        print('Falling back to CPU...')
        # If CUDA is not available, load the model on CPU
        model = load_whisper_cpu(model_name_str, lang)
    print('Loaded whisper model...')
    return model

def whisper_worker_process(whisper_mname, audio_queue, result_queue, 
        hardware_config, language, n_cpus,):
    # Cada worker carrega seu próprio modelo
    #calls_being_processed_lockset[worker_index] = -1
    print('whisper_worker_process: Starting model loading')
    import torch
    torch.set_num_threads(n_cpus)
    if hardware_config == "cuda":
        whisper_model = load_whisper_cuda(whisper_mname, language)
    elif hardware_config == "cpu":
        whisper_model = load_whisper_cpu(whisper_mname, language)
    torch.set_num_threads(n_cpus)

    print('Loaded model!')
    while True:
        audio_id, id_emergencia, start_time, sampling_rate, audio_path = audio_queue.get()
        try:
            if audio_path == "STOP":
                break
            start_time = time()
            result = whisper_model(audio_path)
            time_spent = time() - start_time
            transcript = result['text']
            transcript = remove_duplicates_regex(transcript)
            transcript = remove_duplicates_regex_simple(transcript)
            result = {
                'audio_id': audio_id,
                'id_emergencia': id_emergencia,
                'part': transcript,
                'start_time': start_time,
                'transcription_seconds': time_spent,
                'transcription_model': whisper_mname
            }
            #print(f'result: {result}')
            result_queue.put(result)
        except Exception as err:
            print(f'Error processing audio {audio_path}, removing processing_running flag')
            print(f'transcript: {transcript}')
            print(f'Erro: {err}')
            print(err)
            print(err.with_traceback(None))
            result_queue.put({
                'audio_id': audio_id,
                'id_emergencia': id_emergencia,
                'part': '',
                'start_time': None,
                'transcription_seconds': 0.0,
                'transcription_model': whisper_mname,
                "error": str(err)
            })
            sleep(5)

        sleep(0.05)


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_0"
        )
        self.output_dtype = pb_utils.triton_to_numpy_type(
            output_config['data_type']
        )

        # Determine model name and language from PYTHON_BACKEND_ARGS
        python_backend_args = args.get("parameters", {}).get("PYTHON_BACKEND_ARGS", {}).get("string_value", "")
        model_name_match = re.search(r"--model_name\s+(\S+)", python_backend_args)
        language_match = re.search(r"--language\s+(\S+)", python_backend_args)

        self.whisper_mname = model_name_match.group(1) if model_name_match else "openai/whisper-tiny"
        self.language = language_match.group(1) if language_match else "pt"

        print(f"Loading whisper model {self.whisper_mname} with language {self.language}...")
        from transformers import pipeline
        self.whisper_model = pipeline("automatic-speech-recognition",
                                      model=self.whisper_mname,
                                      return_timestamps=True,
                                      generate_kwargs={"language": self.language},
                                      device='cpu')
        print("Loaded whisper model!")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_audio_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            audio_input_data = input_audio_tensor.as_numpy()

            start_time = time()
            print("Starting transcription...")
            result = self.whisper_model(audio_input_data)
            time_spent = time() - start_time
            print(f"Transcription took {time_spent} seconds")
            transcript = result['text']
            transcript = self._remove_duplicates_regex(transcript)
            transcript = self._remove_duplicates_regex_simple(transcript)

            # Encode transcript to bytes for Triton output
            output_transcript = np.array([transcript.encode('utf-8')], dtype=self.output_dtype)

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("OUTPUT_0", output_transcript)
            ])
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Cleaning up...')

    def _remove_duplicates_regex(self, text: str, max_ngram: int = 5) -> str:
        """
        Remove repetições consecutivas de n-gramas (1..max_ngram).
        Ex.: "fez a de fez a de fez a de" -> "fez a de"
        Mantém separadores (pontuação/espacos) na medida do possível.
        """
        if not text or text.strip() == "":
            return text

        # Tokeniza em palavras (\w+) e não-palavras (separadores)
        tokens = re.findall(r'\w+|\W+', text, flags=re.UNICODE)
        word_indices = [i for i, tok in enumerate(tokens) if re.match(r'\w+', tok, flags=re.UNICODE)]
        words = [tokens[i] for i in word_indices]
        lower_words = [w.lower() for w in words]

        if not words:
            return text

        keep_word = [True] * len(words)
        i = 0
        L = len(words)

        while i < L:
            matched = False
            # tenta maiores n-grams primeiro
            max_n = min(max_ngram, L - i)
            for n in range(max_n, 0, -1):
                seq = tuple(lower_words[i:i + n])
                j = i + n
                # conta quantas vezes a seq se repete consecutivamente
                while j + n <= L and tuple(lower_words[j:j + n]) == seq:
                    j += n
                if j > i + n:
                    # houve repetição: marca palavras repetidas para remoção
                    for k in range(i + n, j):
                        keep_word[k] = False
                    i = j  # pula bloco repetido
                    matched = True
                    break
            if not matched:
                i += 1

        # Reconstrói texto: mantém separadores e apenas palavras marcadas
        out = []
        widx = 0
        for idx, tok in enumerate(tokens):
            if re.match(r'\w+', tok, flags=re.UNICODE):
                if keep_word[widx]:
                    out.append(tok)
                # se palavra removida, não append; mantemos separadores seguintes normalmente
                widx += 1
            else:
                out.append(tok)

        result = ''.join(out)
        # Normaliza espaços extras introduzidos pela remoção
        result = re.sub(r'\s{2,}', ' ', result)
        # Remove espaço antes de pontuação (opcional, melhora saída)
        result = re.sub(r'\s+([,.;:!?])', r'\1', result)
        return result.strip()

    def _remove_duplicates_regex_simple(self, seq):
        my_output = re.sub(r'\b(\w+)(?:\W+\1\b)+', r'\1', seq, flags=re.IGNORECASE)
        return my_output