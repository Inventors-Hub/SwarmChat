from llama_cpp import Llama


# Load the Llama model for safety classification
model_path = r"G:\SwarmChat\model\llama-guard-3-8b-q4_k_m.gguf"
llm = Llama(model_path=model_path, n_ctx=1024)


def check_safety(text, checkbox):

    if checkbox:
        return "Safe"

    categories = {
        "S1": "Violent Crimes.",
        "S2": "Non-Violent Crimes.",
        "S3": "Sex-Related Crimes.",
        "S4": "Child Sexual Exploitation.",
        "S5": "Defamation.",
        "S6": "Specialized Advice.",
        "S7": "Privacy.",
        "S8": "Intellectual Property.",
        "S9": "Indiscriminate Weapons.",
        "S10": "Hate.",
        "S11": "Suicide & Self-Harm.",
        "S12": "Sexual Content.",
        "S13": "Elections.",
        "S14": "Code Interpreter Abuse.",
    }

    input_prompt = f"""
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>

    Task: Determine if the following text contains unsafe content according to our safety policy. Classify the content into one or more of the categories below if unsafe.

    <BEGIN UNSAFE CONTENT CATEGORIES>
    {', '.join([f'{key}: {value}' for key, value in categories.items()])}
    <END UNSAFE CONTENT CATEGORIES>

    Text: {text}

    Provide your safety assessment:
    - Respond with "safe" if the text is appropriate.
    - If unsafe, respond with: "unsafe" followed by a comma-separated list of the violated categories (e.g., S1, S2).

    <|eot_id|>
    """

    # Generate a response from the model
    output = llm(input_prompt, temperature=0)
    # Extract the model's response
    response = output.get("choices", [{}])[0].get("text", "").strip()
    print("\n\n", response, "\n\n")
    # Parse the response
    if response.lower().startswith("safe"):
        return "Safe"
    else:
        unsafe_categories = categories[response.split("unsafe", 1)[-1].strip()]
        return f"Unsafe: This prompt is categorized as '{unsafe_categories}'"

        # unsafe_categories = categories[response.split("unsafe", 1)[-1].strip()]
        # return f"Unsafe: This prompt categorized as '{unsafe_categories}'"