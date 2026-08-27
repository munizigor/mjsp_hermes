import time
import re
import os
import sys
import torch
import unidecode
from glob import glob
import gc
import traceback
import json

from tqdm import tqdm
import string

import pandas as pd

from asr_lib.utils import (get_inputs, eval_transcriptions, intelligent_chunk_merge, 
    clean_reference_text, 
    normalize_light, normalize_special, normalize_punct, normalize_repeats,
    save_results)

from asr_lib.torch_asr_utils import (fix_whisper_generation_config, process_with_pipeline, 
    load_pipeline, load_asr_generator)

'''"ibm-granite/granite-4.0-1b-speech",#cannot use HF pipeline, not implemented
    "nvidia/canary-1b-flash",
    "nvidia/canary-qwen-2.5b",
    "nvidia/parakeet-tdt-0.6b-v3",
    "nvidia/parakeet-tdt-0.6b-v2",
    "microsoft/Phi-4-multimodal-instruct",'''

hf_models = [
    "Qwen/Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ASR-1.7B",
    "facebook/mms-1b-all",
    "pierreguillou/whisper-medium-portuguese",
    "nilc-nlp/distil-whisper-coraa-mupe-asr",
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    #"remynd/whisper-large-v3-pt", # not enough memory
    "openai/whisper-base",
    "dominguesm/whisper-tiny-pt",
    "openai/whisper-small",
    "remynd/whisper-small-pt",
    "openai/whisper-medium",
    "remynd/whisper-medium-pt",
]

def process_low_level(model, processor, audio_array, sr, make_chunks=False):
    from transformers import Wav2Vec2ForCTC
    import torch
    
    is_mms = isinstance(model, Wav2Vec2ForCTC)
    
    if is_mms:
        processor.tokenizer.set_target_lang("por")
        model.load_adapter("por")

    # Identifica o modelo para ajustar as configurações da geração
    try:
        model_name = model.config._name_or_path.lower()
    except:
        model_name = ""
        
    # CrisperWhisper obriga o uso de word-timestamps
    timestamps_param = "word" if "crisper" in model_name else True

    gpu_time_sum = 0.0

    if not make_chunks:
        start_time = time.time()
        inputs = processor(
            audio_array, 
            sampling_rate=sr, 
            return_tensors="pt",
            return_attention_mask=True 
        ).to(model.device) 
        gpu_time_sum += time.time() - start_time
        
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(model.dtype)
        if "input_values" in inputs:
            inputs["input_values"] = inputs["input_values"].to(model.dtype)
        
        with torch.no_grad():
            start_time = time.time()
            if is_mms:
                logits = model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
            else:
                predicted_ids = model.generate(
                    **inputs,
                    return_timestamps=timestamps_param,
                    language="portuguese", task="transcribe"
                )
            gpu_time_sum += time.time() - start_time
        
        text_chunks = []
        for part in processor.batch_decode(predicted_ids, skip_special_tokens=True):
            text_chunks.append(part.strip())
        chunk_text = intelligent_chunk_merge(text_chunks)
        out_text = chunk_text
        
    else:
        chunk_length_s = 30
        chunk_samples = chunk_length_s * sr
        gpu_time_sum = 0.0
        transcribed_chunks = []
        
        if processor is not None:
            for i in range(0, len(audio_array), chunk_samples):
                chunk = audio_array[i : i + chunk_samples]
                
                # FIX: Skip extremely short trailing chunks that break Wav2Vec2 convolutions
                if len(chunk) < (0.1 * sr): 
                    continue
                start_time = time.time()
                inputs = processor(
                    chunk, 
                    sampling_rate=sr, 
                    return_tensors="pt",
                    return_attention_mask=True 
                ).to(model.device)
                gpu_time_sum += time.time() - start_time

                if "input_features" in inputs:
                    inputs["input_features"] = inputs["input_features"].to(model.dtype)
                if "input_values" in inputs:
                    inputs["input_values"] = inputs["input_values"].to(model.dtype)
            
                with torch.no_grad():
                    start_time = time.time()
                    if is_mms:
                        logits = model(**inputs).logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                    else:
                        predicted_ids = model.generate(
                            **inputs,
                            return_timestamps=timestamps_param,
                            language="portuguese", task="transcribe"
                        )
                    gpu_time_sum += time.time() - start_time

                    chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    transcribed_chunks.append(chunk_text.strip())
        else:
            #TODO: test multiple chunks
            chunks = [(audio_array[i : i + chunk_samples], sr) 
                      for i in range(0, len(audio_array), chunk_samples)]
            start_time = time.time()
            results = model.transcribe(
                audio=chunks,
                language="Portuguese",
            )
            gpu_time_sum += time.time() - start_time
            out_texts = [r.text.strip() for r in results]
            transcribed_chunks.extend(out_texts)
            
        out_text = intelligent_chunk_merge(transcribed_chunks)
        print(out_text)
        
    return out_text, gpu_time_sum

def test_model(model_id, n_samples, device, cache_dir=".asr_cache"):
    model_cache = os.path.join(cache_dir, model_id.replace('/', '_'))
    os.makedirs(model_cache, exist_ok=True)
    print(f"Testing model {model_id} on {n_samples} samples...")
    n_for_speedup_test = int(n_samples*0.05)
    if n_for_speedup_test < 5:
        n_for_speedup_test = 5
    model = load_pipeline(model_id, device)

    ds = get_inputs(max_n=n_samples)
    original_scripts = ds["roteiro_segmentado"]

    transcriptions = []
    texts_segmented = []
    time_start = time.time()
    audio_lens = []

    line_n = 0
    processing_times = []

    lines_for_low_level_eval = []
    for item in tqdm(ds):
        id_str = item["ID"]
        audio = item["audio"]
        audio_array = audio["array"]
        sr = audio["sampling_rate"]
        audio_len_secs = len(audio_array) / sr

        audio_name = item["ID"] + "_" + item["modelo_audio"]

        cached_path = os.path.join(model_cache, audio_name + ".json")
        loaded_cache = False
        if os.path.exists(cached_path):
            try:
                with open(cached_path, "r") as f:
                    result = json.load(f)
                    out_text = result["text"]
                    processing_time = result["processing_time"]
                    loaded_cache = True
            except json.JSONDecodeError as e:
                print(f"Error loading {cached_path}: {e}")
                os.remove(cached_path)
                out_text = None
                processing_time = None
        
        if not loaded_cache:
            time_start = time.time()
            out_text = process_with_pipeline(model_id, model, audio_array)
            if len(out_text) == 0:
                print("Empty transcription")
                raise ValueError("Empty transcription")
            processing_time = time.time() - time_start

            if "qwen" in model_id.lower() or "granite" in model_id.lower():
                print("Original:")
                print(original_scripts[line_n])
                print("Transcribed:")
                print(out_text)
                

            obj = {"text": out_text, "processing_time": processing_time}
            #print(f"Saving {cached_path} -> {obj}")
            json.dump(obj, open(cached_path, "w"), ensure_ascii=False)
        processing_times.append(processing_time)
        
        transcriptions.append(out_text)
        
        original_text = original_scripts[line_n]
        texts_segmented.append(original_text)
        audio_lens.append(audio_len_secs)

        '''print("Original:")
        print(original_text)
        print("Transcribed:")
        print(out_text)
        print()'''

        if line_n < n_for_speedup_test:
            lines_for_low_level_eval.append(
                [original_text, audio_array, processing_times[-1]])
        line_n += 1
    
    total_audio_len = sum(audio_lens)
    processing_time = sum(processing_times)
    print(f"Total time: {processing_time}")
    speed = total_audio_len / processing_time
    print(f"Speed: {speed:.2f}x")

    texts_joined = []
    for text in texts_segmented:
        lines = []
        for line in text:
            if len(line) == 2:
                speaker, txt = line
                lines.append(txt)
            elif len(line) == 1:
                lines.append(line[0])
            else:
                print("Error: more than 2 elements in line")
                quit(1)
            
        ref_raw = ' . '.join(lines)
        ref_clean = clean_reference_text(ref_raw)

        texts_joined.append(ref_clean)
    
    texts_original_norm = [normalize_light(t) for t in texts_joined]
    texts_original_special = [normalize_special(t) for t in texts_original_norm]
    texts_original_punct = [normalize_punct(t) for t in texts_original_norm]
    texts_original_repeats = [normalize_repeats(t) for t in texts_original_norm]

    transcribed_norms = [normalize_light(t) for t in transcriptions]
    transcribed_specials = [normalize_special(t) for t in transcribed_norms]
    transcribed_puncts = [normalize_punct(t) for t in transcribed_norms]
    transcribed_repeats = [normalize_repeats(t) for t in transcribed_norms]

    raw_result_lines = []
    for index in range(len(texts_joined)):
        text_joined = texts_joined[index]
        transcription = transcriptions[index]
        audio_len = audio_lens[index]
        original_norm = texts_original_norm[index]
        original_special = texts_original_special[index]
        original_punct = texts_original_punct[index]
        original_repeats = texts_original_repeats[index]

        transcribed_norm = transcribed_norms[index]
        transcribed_special = transcribed_specials[index]
        transcribed_punct = transcribed_puncts[index]
        transcribed_repeat = transcribed_repeats[index]

        processing_time = processing_times[index]
        weight = len(original_norm) / 100
        '''print("Original (norm):")
        print(original_norm)
        print("Original (simple):")
        print(original_simple)
        print("Transcribed (norm):")
        print(transcribed_norm)
        print("Transcribed (simple):")
        print(transcribed_simple)
        print()'''

        raw_result_lines.append({
            "audio_len": audio_len,
            "processing_time": processing_time,
            "weight": weight,
            "original": text_joined,
            "original_norm": original_norm,
            "original_special": original_special,
            "original_punct": original_punct,
            "original_repeats": original_repeats,
            "transcribed": transcription,
            "transcribed_norm": transcribed_norm,
            "transcribed_special": transcribed_special,
            "transcribed_punct": transcribed_punct,
            "transcribed_repeats": transcribed_repeat,
        })
    
    print("Evaluating speed on splitted audio chunks...")

    #if not("qwen" in model_id.lower()):
        
    inf_speed_test_cache_path = os.path.join(model_cache, f"inf_speed_test_{n_samples}.json")
    if os.path.exists(inf_speed_test_cache_path):
        with open(inf_speed_test_cache_path, "r") as f:
            inf_speed_test_cache = json.load(f)
    else:
        if "qwen" not in model_id.lower():
            del model
            gc.collect()
            torch.cuda.empty_cache()
            processor, model = load_asr_generator(model_id, device)
        else:
            processor = None
        inf_speed_test_cache = []
        for original_text, audio_array, _ in tqdm(lines_for_low_level_eval):
            out_text, gpu_time = process_low_level(model, processor, audio_array, 
                sr, make_chunks=True)
            inf_speed_test_cache.append({
                "audio_len": audio_len,
                "original": original_text,
                "transcribed": out_text,
                "processing_time": gpu_time,
            })
        with open(inf_speed_test_cache_path, "w") as f:
            json.dump(inf_speed_test_cache, f)
        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()
    '''else:
        inf_speed_test_cache = [
            {"processing_time": t} for _, _, t in lines_for_low_level_eval
        ]'''
    
    full_audio_times = sum([line[-1] for line in lines_for_low_level_eval])
    new_proc_times = [line["processing_time"] for line in inf_speed_test_cache]
    chunked_audio_times = sum(new_proc_times)
    speedup = full_audio_times / chunked_audio_times
    print(f"Speedup: {speedup:.2f}x")

    return raw_result_lines, speedup



if __name__ == "__main__":
    # 1. Load the generic processor and model
    n_samples_to_test = int(sys.argv[1]) if len(sys.argv) > 1 else 429
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    os.makedirs(results_dir, exist_ok=True)

    eval_csv_paths = glob(os.path.join(results_dir, "eval_results.*.csv"))
    previous_n_samples_of_models = {}
    for prev_csv in eval_csv_paths:
        eval_df = pd.read_csv(prev_csv)
        for _, row in eval_df.iterrows():
            if "wer_basic_norm" not in row:
                continue
            wer_norm = row['wer_basic_norm']
            if wer_norm > 0.75:
                continue
            else:
                if row['model_id'] not in previous_n_samples_of_models:
                    previous_n_samples_of_models[row['model_id']] = row['n_samples']
                else:
                    previous_n_samples_of_models[row['model_id']] = max(previous_n_samples_of_models[row['model_id']], row['n_samples'])

    hf_models_to_test = []
    #only test models with fewer samples in the previous evaluation
    for model_id in hf_models:
        if model_id not in previous_n_samples_of_models:
            hf_models_to_test.append(model_id)
        elif previous_n_samples_of_models[model_id] < n_samples_to_test:
            hf_models_to_test.append(model_id)
    hf_models_to_load_from_previous = [m for m in hf_models if m not in hf_models_to_test]
    if len(hf_models_to_test) == 0:
        print("All models have been tested with the specified number of samples.")
        quit(0)
    
    all_raw_results = []

    for model_id in hf_models:
        try:
            raw_result_lines, speedup = test_model(model_id, n_samples_to_test, device)
            for line in raw_result_lines:
                line['model_id'] = model_id
                line['chunking_speedup'] = speedup
            all_raw_results.extend(raw_result_lines)
            save_results(all_raw_results, results_dir)
        except Exception as e:
            print(f"Error testing model {model_id}: {e}")
            print(e)
            print(traceback.format_exc())
            quit(1)
    
    