import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import io
from pydub import AudioSegment
import numpy as np

import safety_module


# Setup device and data types
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the model and processor
model_id = "openai/whisper-medium"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Create a pipeline for automatic speech recognition with translation
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if torch.cuda.is_available() else -1,)

def translate_audio(audio_file):
    # Translate using the pipeline
    result = pipe(audio_file, generate_kwargs={"task": "translate"})
    translated_text = result["text"]
    return translated_text

#     # Check for safety in the translated speech
#     safety_status = safety_module.check_safety(translated_text)
#     if safety_status.startswith("Unsafe"):
#         return None, safety_status

#     else:
#         return translated_text, safety_status
    

# def handle_audio_translation(audio_file):
#     translated_text, safety_status = translate_audio(audio_file)
#     if translated_text is None:  # Unsafe content detected
#         raise "Translation blocked due to unsafe content."
#     return translated_text, safety_status
