"""
Wraps EuroLLM-9B Llama for text→English translation.

Functions:
    translate_text(str) -> str:  
        Translates arbitrary text into English via a llama_cpp model.
"""

from llama_cpp import Llama
import os
import struct
print(struct.calcsize("P") * 8, "bit")


model_path = r"G:\Inventors Hub Projects\SwarmChat\models\EuroLLM-9B-Instruct-Q4_K_M.gguf"
print("Model exists?", os.path.exists(model_path))

# llm = Llama(model_path=model_path, n_ctx=1024)
print("Llama backend initialized successfully!")



# Function to process text using EuroLLM
def translate_text(text):
    llm = Llama(model_path=model_path, n_ctx=1024)#, verbose=True)
    input_prompt = f"""
    <|im_start|>system
    <|im_end|>
    <|im_start|>user
    Translate the following text to English:
    Text: {text}
    English: 
    <|im_end|>
    <|im_start|>assistant
    """
    output = llm(input_prompt, max_tokens=1024, temperature=0)
    # print('\n\nOutput: ',output,'\n\n')
    translated_text = output.get("choices", [{}])[0].get("text", "").strip()

    return translated_text