from llama_cpp import Llama
import safety_module

model_path_9b = r"G:\SwarmChat\model\EuroLLM-9B-Instruct-Q4_K_M.gguf"

llm_9b = Llama(model_path=model_path_9b, n_ctx=1024)


# Function to process text using EuroLLM
def translate_text(text):
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
    output = llm_9b(input_prompt, max_tokens=1024, temperature=0)
    # print('\n\nOutput: ',output,'\n\n')
    translated_text = output.get("choices", [{}])[0].get("text", "").strip()

    return translated_text

#     # Check for safety in the translated speech
#     safety_status = safety_module.check_safety(translated_text)
#     if safety_status.startswith("Unsafe"):
#         return None, safety_status

#     else:
#         return translated_text, safety_status
    

# def handle_text_translation(text):
#     translated_text, safety_status = translate_text(text)
#     if translated_text is None:  # Unsafe content detected
#         return "Translation blocked due to unsafe content.", safety_status  
#     return translated_text, safety_status