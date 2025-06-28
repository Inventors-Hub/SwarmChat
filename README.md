# SwarmChat: Enabling Intuitive Swarm Robotics with Natural Language

SwarmChat is an innovative project that enables intuitive communication with swarm robotics through natural language. This system integrates advanced audio transcription, text processing, and safety mechanisms with a live simulation environment that visualizes a swarm of agents executing behavior trees.

🚀 This project is Funded by the European Union’s [UTTER programme](https://he-utter.eu/), in collaboration with the UTTER consortium.

## Features

- **Audio Input Processing**:

  - Record commands via a microphone.
  - Translate speech into English using the `facebook/seamless-m4t-v2-large` model.
  - Perform a safety check on the translated text before execution.

- **Text Input Processing**:

  - Enter text commands for swarm control.
  - Translate text using `EuroLLM` (EuroLLM-9B-Instruct).
  - Detect unsafe or inappropriate content with an integrated safety module.

- **Safety Module**:

  - Utilizes `Llama-Guard` model (Llama-Guard-3-8B) for safety classification.
  - Identifies unsafe content across predefined categories (e.g., violent crimes, privacy violations, hate speech).
  - Ensures commands comply with safety standards.

- **Swarm Simulation**:

  - Visualize a swarm of agents in a live simulation powered by Violet simulator and Pygame.
  - Agents are controlled by behavior trees defined in an XML file (`tree.xml`), using the `py_trees` framework.
  - Real-time simulation updates streamed via a Gradio web interface.

- **Behavior Tree Generator**:

  - `Falcon3-10B-Instruct-BehaviorTree` model to dynamically generate behavior trees in XML format.
  - Automatically extracts available behaviors from the SwarmAgent class and constructs a detailed prompt using a predefined XML template.
  - Generates and saves new behavior tree configurations (updating tree.xml) based on user-specified tasks.

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

  - **Speech Processing**: [Seamless-m4t-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large) for audio transcription and translation.
  - **Text Processing**: [EuroLLM-9B-Instruct](https://huggingface.co/utter-project/EuroLLM-9B-Instruct) for text translation.
  - **Safety Classification**: [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) for content safety assessment.
  - **Behavior Tree Generation**: [Falcon3-10B-Instruct-BehaviorTree](https://huggingface.co/Inventors-Hub/Falcon3-10B-Instruct-BehaviorTree-3-epochs) for creating and updating behavior trees.

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
- Place the DeepSeek model file (`Falcon3-10B-Instruct-BehaviorTree-3-epochs-GGUF`) at the specified path in `bt_generator.py`.

4. **Run the Application**:
   ```bash
   python app.py
   ```
5. **Access the Interface**:

   Open your browser and navigate to http://127.0.0.1:7860 to start using SwarmChat.

## Overview of Modules

- **app.py**  
  The main application integrates audio/text processing, behavior tree generation, and the live simulation. It sets up the Gradio interface, handles simulation streaming, and routes user inputs to the appropriate processing modules.

- **speech_processing.py**  
  Implements audio transcription and translation using the `facebook/seamless-m4t-v2-large` model.

- **text_processing.py**  
  Translates text commands using `EuroLLM` (EuroLLM-9B-Instruct).

- **safety_module.py**  
  Utilizes `LLaMA Guard` to assess the safety of incoming commands, ensuring compliance with safety policies.

- **bt_generator.py**  
  Dynamically generates behavior trees in XML format by extracting behaviors from the SwarmAgent class, constructing a prompt, and querying `Falcon3-10B-Instruct-BehaviorTree` model. The generated XML is saved to `tree.xml` for simulation use.

- **simulator_env.py**  
  Powers the simulation environment, manages agent behaviors using XML-defined behavior trees, and handles real-time simulation updates.

## Acknowledgments

This work was funded by the European Union under the [UTTER programme](https://he-utter.eu/).  
We gratefully acknowledge the support of the entire UTTER consortium.
