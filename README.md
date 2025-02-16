# SwarmChat: Unified Audio, Text, and Simulation Environment for Human-Swarm Interaction

SwarmChat is an innovative project that enables intuitive communication with swarm robotics through natural language. This system integrates advanced audio transcription, text processing, and safety mechanisms with a live simulation environment that visualizes a swarm of agents executing behavior trees.

## Features

- **Audio Input Processing**:

  - Record commands via a microphone.
  - Translate speech into English using the `facebook/seamless-m4t-v2-large` model.
  - Perform a safety check on the translated text before execution.

- **Text Input Processing**:

  - Enter text commands for swarm control.
  - Translate text using EuroLLM (EuroLLM-9B-Instruct-Q4_K_M.gguf).
  - Detect unsafe or inappropriate content with an integrated safety module.

- **Safety Module**:

  - Utilizes a fine-tuned LLaMA-based model (llama-guard-3-8b-q4_k_m.gguf) for safety classification.
  - Identifies unsafe content across predefined categories (e.g., violent crimes, privacy violations, hate speech).
  - Ensures commands comply with safety standards.

- **Swarm Simulation**:

  - Visualize a swarm of agents in a live simulation powered by Violet simulator and Pygame.
  - Agents are controlled by behavior trees defined in an XML file (`tree.xml`), using the `py_trees` framework.
  - Real-time simulation updates streamed via a Gradio web interface.

- **Integrated Interface**:
  - A unified Gradio web interface for both audio and text inputs.
  - Live streaming of the simulation environment.
  - Seamless switching between different input modalities.

## Technology Stack

- **Backend**:

  - Python
  - [Transformers](https://huggingface.co/transformers/) (Hugging Face)
  - PyTorch
  - Pygame
  - Threading and Queue modules for simulation management

- **Frontend**:

  - [Gradio](https://gradio.app/) for an interactive web-based interface.

- **AI Models**:

  - **Speech Processing**: `facebook/seamless-m4t-v2-large` for audio transcription and translation.
  - **Text Processing**: EuroLLM (EuroLLM-9B-Instruct-Q4_K_M.gguf) for text translation.
  - **Safety Classification**: LLaMA Guard (llama-guard-3-8b-q4_k_m.gguf) for content safety assessment.

- **Behavior Trees**:
  - Agents utilize behavior trees—parsed from XML and built with `py_trees`—to dictate their actions within the simulation.

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
3. **Setup AI Models**:

- Place the EuroLLM model file (`EuroLLM-9B-Instruct-Q4_K_M.gguf`) at the specified path in `text_processing.py`.
- Place the LLaMA Guard model file (`llama-guard-3-8b-q4_k_m.gguf`) at the specified path in `safety_module.py`.

4. **Run the Application**:
   ```bash
   python app.py
   ```
5. **Access the Interface**:

   Open your browser and navigate to http://127.0.0.1:7860 to start using SwarmChat.
