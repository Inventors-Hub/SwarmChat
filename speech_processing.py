from transformers import SeamlessM4Tv2Model, AutoProcessor
import numpy as np
import torch
from pydub import AudioSegment

# Load processor and model
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

def translate_audio(audio_file):
    if audio_file is None:
        return "No audio file detected. Please try again."
    
    try:
        # Set the device (use GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Reset audio file pointer and load audio
        audio = AudioSegment.from_file(audio_file, format="wav")
        audio = audio.set_frame_rate(16000).set_channels(1)

        # Convert audio to float32 NumPy array
        audio_array = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

        # Process input
        audio_inputs = processor(audios=audio_array, sampling_rate=16000, return_tensors="pt")
        audio_inputs = {key: val.to(device) for key, val in audio_inputs.items()}  # Ensure tensors are on the correct device

        # Generate translation
        output_tokens = model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)

        # Extract token IDs from the generated output
        token_ids = output_tokens.sequences
        # Decode token IDs to text
        translated_text_from_audio = processor.batch_decode(token_ids, skip_special_tokens=True)[0]

        return translated_text_from_audio
    except Exception as e:
        return f"Error during audio translation: {e}"
