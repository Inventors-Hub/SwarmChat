from langdetect import detect
from gtts import gTTS
import io
import safety_module
import speech_text


def process_text_input(text):

    safety_status = safety_module.check_safety(text)  # Check safety first
    if safety_status.startswith("Unsafe"):
        return "Sorry, I can't translate this.", safety_status  # Stop processing if unsafe

    try:
        # Detect language of the text
        language = detect(text)
        # print(f"Detected language: {language}")

        # Convert text to speech and save to an in-memory buffer
        tts = gTTS(text=text, lang=language)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)  # Reset the buffer's position

        # Pass the in-memory audio buffer to the translation pipeline
        translated_text,_ = speech_text.translate_audio(audio_buffer)
        return translated_text, safety_status

    except Exception as e:
        return f"Error: {e}", "Unknown"

