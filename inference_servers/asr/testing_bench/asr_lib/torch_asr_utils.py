import torch

def fix_whisper_generation_config(model, model_name_str):
    """Injects missing generation_config parameters into community models."""
    from transformers import GenerationConfig
    
    # Verifica se a configuração atual está quebrada/incompleta
    if getattr(model.generation_config, "no_timestamps_token_id", None) is None:
        # Tenta adivinhar o modelo base pelo nome
        name_lower = model_name_str.lower()
        base_id = "openai/whisper-base" # fallback padrão
        
        if "tiny" in name_lower: base_id = "openai/whisper-tiny"
        elif "small" in name_lower: base_id = "openai/whisper-small"
        elif "medium" in name_lower: base_id = "openai/whisper-medium"
        elif "large" in name_lower: base_id = "openai/whisper-large-v2"
        
        # Puxa a configuração imaculada da OpenAI e sobrescreve a do modelo problemático
        print(f"Fixing generation_config for {model_name_str} using {base_id} as template...")
        model.generation_config = GenerationConfig.from_pretrained(base_id)
        
    return model

'''def load_asr_generator(model_name_str, device):
    model_family = "whisper"
    if "mms" in model_name_str:
        model_family = "mms"
    
    if model_family == "whisper":
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        processor = WhisperProcessor.from_pretrained(model_name_str)
        model = WhisperForConditionalGeneration.from_pretrained(model_name_str).to(device)
        
        # Aplica a correção da configuração
        model = fix_whisper_generation_config(model, model_name_str)
        
    elif model_family == "mms":
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name_str)
        model = Wav2Vec2ForCTC.from_pretrained(model_name_str).to(device)
    else:
        from transformers import AutoProcessor, AutoModelForSeq2SeqLM
        processor = AutoProcessor.from_pretrained(model_name_str)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_str).to(device)
        
    return processor, model

def load_whisper(model_name_str, device, lang="portuguese"):
    from transformers import pipeline
    
    if "whisper" in model_name_str.lower():
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        # Para consertar o pipeline, precisamos carregar o modelo explicitamente primeiro
        processor = WhisperProcessor.from_pretrained(model_name_str)
        model_obj = WhisperForConditionalGeneration.from_pretrained(model_name_str)
        
        # Aplica a correção da configuração
        model_obj = fix_whisper_generation_config(model_obj, model_name_str)
        
        # Passa os objetos corrigidos para o pipeline ao invés da string
        model = pipeline(
            "automatic-speech-recognition", 
            model=model_obj,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps=True,
            generate_kwargs={"language": lang},
            device=device
        )
    elif "mms" in model_name_str.lower():
        model = pipeline(
            "automatic-speech-recognition", 
            model=model_name_str,
            return_timestamps='word',
            generate_kwargs={"language": lang},
            device=device
        )
        
    else:
        model = pipeline(
            "automatic-speech-recognition", 
            model=model_name_str,
            return_timestamps=True,
            generate_kwargs={"language": lang},
            device=device
        )
        
    return model'''


def process_with_pipeline(model_id, model, audio_array, sr=16000):
    if "qwen" in model_id.lower():
        results = model.transcribe(
            audio=(audio_array, sr),
            language="Portuguese",
        )
        out_text = results[0].text
    else:
        result = model(audio_array)
        out_text = result["text"]
    return out_text

def load_asr_generator(model_name_str, device):
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    
    # FIX: Carregar sempre em float16 na GPU para evitar OOM
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_family = "whisper"
    if "mms" in model_name_str.lower():
        model_family = "mms"
    
    if model_family == "whisper":
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        processor = AutoProcessor.from_pretrained(model_name_str)
        # Injeta o torch_dtype e low_cpu_mem_usage
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name_str, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True
        ).to(device)
        model = fix_whisper_generation_config(model, model_name_str)
        
    elif model_family == "mms":
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name_str)
        model = Wav2Vec2ForCTC.from_pretrained(
            model_name_str, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True
        ).to(device)
    else:
        from transformers import AutoProcessor, AutoModelForSeq2SeqLM
        processor = AutoProcessor.from_pretrained(model_name_str)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_str, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True
        ).to(device)
        
    hf_logging.set_verbosity_warning()
    return processor, model

def load_pipeline(model_name_str, device, lang="portuguese"):
    from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
    from transformers.utils import logging as hf_logging
    
    hf_logging.set_verbosity_error()
    name_lower = model_name_str.lower()
    
    # FIX: Precisão para não estourar a VRAM
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    batch_size = 8
    
    if "whisper" in name_lower:
        processor = AutoProcessor.from_pretrained(model_name_str)
        model_obj = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name_str, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True
        ).to(device)
        model_obj = fix_whisper_generation_config(model_obj, model_name_str)
        
        timestamps_param = "word" if "crisper" in name_lower else True
        
        model = pipeline(
            "automatic-speech-recognition", 
            model=model_obj,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps=timestamps_param,
            chunk_length_s=30,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            # FIX: Adicionado "task": "transcribe" para evitar traduções fantasmas!
            generate_kwargs={"language": lang, "task": "transcribe"},
            device=device,
            language=lang,
            #task="transcribe"
        )
    elif "mms" in name_lower:
        model = pipeline(
            "automatic-speech-recognition", 
            model=model_name_str,
            return_timestamps='word',
            chunk_length_s=30,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            generate_kwargs={"language": lang}, # MMS não usa "task"
            device=device
        )
    elif "qwen" in name_lower:
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(
            model_name_str,
            dtype=torch_dtype,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2",
            max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=768, # Maximum number of tokens to generate. Set a larger value for long audio input.
        )

        '''
        #usage:
        #audio:
        #        Audio input(s). Supported:
        #          - str: local path / URL / base64 data url
        #          - (np.ndarray, sr)
        #          - list of above
        results = model.transcribe(
            audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
            language=None, # set "English" to force the language
        )

        print(results[0].language)
        print(results[0].text)'''
    else:
        model = pipeline(
            "automatic-speech-recognition", 
            model=model_name_str,
            #return_timestamps=True,
            chunk_length_s=30,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            generate_kwargs={"language": lang},
            device=device
        )
        
    hf_logging.set_verbosity_warning()
    return model