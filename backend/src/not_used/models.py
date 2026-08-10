
import logging
logging.basicConfig(filename='/logs/main.log', level=logging.INFO)
logger = logging.getLogger(__name__)

import regex as re

import torch
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
print(TO_LANGUAGE_CODE)
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio
from tqdm import tqdm

asr_models = {
    #'base': 'thiagobarbosa/whisper-base-common-voice-16-pt-v6',
    'small': 'openai/whisper-small',
    'medium': 'my-north-ai/whisper-medium-pt',
    'large': 'nilc-nlp/distil-whisper-coraa-mupe-asr',
    'turbo': 'openai/whisper-large-v3-turbo'
}

MAX_INPUT_LENGTH = 30.0

def is_audio_in_length_range(length):
    return length < MAX_INPUT_LENGTH

# Text normalization function
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\sàáâãçéêíóôõúü]", "", text)  # Keep Portuguese chars
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_dataset(example, processor):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=processor.feature_extractor.sampling_rate, # Use the correct sampling rate
        text=example["sentence"],
    )

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example

class WhisperInstance:

    def get_processor(model_name, lang='portuguese'):
        processor = WhisperProcessor.from_pretrained(
            model_name, language=lang, task="transcribe"
        )
        sampling_rate = processor.feature_extractor.sampling_rate
        return processor, sampling_rate

    def load_model(model_name, device='cuda'):
        print(f'Loading ASR model {model_name}...')
        processor, sampling_rate = WhisperInstance.get_processor(model_name)
        try:
            model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
            used_device = device
        except Exception as err:
            print(f"Error loading ASR model: {err}")
            print('Falling back to CPU...')
            # If CUDA is not available, load the model on CPU
            model = WhisperForConditionalGeneration.from_pretrained(model_name).to('cpu')
            used_device = 'cpu'
        
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="pt",
            task="transcribe"
        )

        return model, processor, sampling_rate, used_device

    def __init__(self, model_size, device='cuda'):
        self.model_name = asr_models[model_size]
        self.model, self.processor, self.sp_rate, self.used_device = WhisperInstance.load_model(
            self.model_name, device='cuda')

    def transcribe_audio(self, audio):
        # Process audio and generate prediction
        inputs = self.processor(
            audio,
            sampling_rate=self.sp_rate,
            return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(inputs, max_length=225)

        prediction = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            normalize=False
        )[0]

        return prediction