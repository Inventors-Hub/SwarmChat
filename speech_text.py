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
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Create a pipeline for automatic speech recognition with translation
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if torch.cuda.is_available() else -1,)


def translate_audio(audio_file):
    if isinstance(audio_file, io.BytesIO):
        audio_file.seek(0)
        audio = AudioSegment.from_file(audio_file, format="mp3")
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio_array = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        audio_file = {"array": audio_array, "sampling_rate": 16000}

        # Translate using the pipeline
        result = pipe(audio_file, generate_kwargs={"task": "translate"})
        translated_text = result["text"]
        return translated_text, "Safe"

    elif isinstance(audio_file, str):
        # Translate using the pipeline
        result = pipe(audio_file, generate_kwargs={"task": "translate"})
        translated_text = result["text"]

        # Check for safety in the translated speech
        safety_status = safety_module.check_safety(translated_text)
        if safety_status.startswith("Unsafe"):
            return "Sorry, I can't translate this.", safety_status
        else:
            return translated_text, safety_status
    
    else:
        return "Invalid audio input format. Please provide a valid file or buffer.", "Unknown"


