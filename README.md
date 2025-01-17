# SwarmChat: Unified Audio and Text Processing for Human-Swarm Interaction

SwarmChat is an innovative project that enables intuitive communication with swarm robotics through natural language. This system integrates advanced audio transcription, text processing, and safety mechanisms to ensure reliable and secure interactions between humans and robotic swarms.

## Features

- **Audio Input Processing**:
  - Record commands via a microphone.
  - Translate speech into English for swarm control.
  - Perform a safety check on the translated text.
- **Text Input Processing**:

  - Enter text commands for swarm control.
  - Detect unsafe or inappropriate text and flag it.
  - Translate text commands into instructions for robotic swarms.

- **Safety Module**:

  - Utilizes a fine-tuned **LLaMA model** for safety classification.
  - Identifies unsafe content across predefined categories (e.g., violent crimes, privacy violations, hate speech).
  - Ensures commands comply with safety standards.

- **Speech and Text Translation**:
  - Leverages OpenAI's Whisper model for high-accuracy audio transcription and translation.
  - Utilizes Google Text-to-Speech (gTTS) for seamless language processing.

## Technology Stack

- **Backend**:

  - Python
  - Transformers (Hugging Face)
  - PyTorch
  - Google Text-to-Speech (gTTS)

- **Frontend**:

  - [Gradio](https://gradio.app/) for an interactive web-based interface.

- **Safety and AI Models**:
  - LLaMA for safety classification.
  - Whisper for speech-to-text and translation.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Inventors-Hub/SwarmChat.git
   cd SwarmChat
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Setup the LLaMA Guard model**:

   Place your LLaMA Guard model file at the specified path in safety_module.py

   ```bash
   model_path = "path_to_llama_model_file"
   ```

4. **Setup the FFmpeg**:

   Follow the official instructions to install FFmpeg for your operating system.
   [Link](https://www.ffmpeg.org/download.html)

5. **Run the Application**:
   ```bash
   python app.py
   ```
6. **Access the Interface**:

   Open your browser and navigate to http://127.0.0.1:7860 to start using SwarmChat.
