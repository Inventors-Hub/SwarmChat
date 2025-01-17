import speech_text
import text_speech
import gradio as gr




# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ SwarmChat: Unified Audio and Text Processing for Human-Swarm Interaction")
    gr.Markdown("""
    SwarmChat enables intuitive communication with swarm robotics through natural language. 
    Utilize this interface to translate audio and process text for effective human-swarm interaction.
    """)    
    with gr.Tabs():
        # Tab for microphone input
        with gr.Tab("Microphone Input"):
            gr.Markdown("## Record and Translate Audio")
            gr.Markdown("""
            Use your microphone to record audio instructions for swarms. The system will translate your 
            natural language commands into English while ensuring content safety.
            """)
            with gr.Row():
                with gr.Column():
                    microphone_input = gr.Audio(sources="microphone", type="filepath", label="🎙️ Record Audio")
                with gr.Column():
                    output_text_audio = gr.Textbox(label="📄 Translated Instructions")
                    safty_check_audio = gr.Textbox(label="✅ Safety Check")

            translate_button_audio = gr.Button("Send Audio")
            translate_button_audio.click(
                fn=speech_text.translate_audio,
                inputs=microphone_input,
                outputs=[output_text_audio, safty_check_audio]
            )

        # Tab for text input
        with gr.Tab("📝 Text Input"):
            gr.Markdown("## Process Text Commands")
            gr.Markdown("""
            Enter text instructions for swarm control or natural language processing. 
            This tool translates and processes your text for safe execution by swarms.
            """)
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        lines=5, placeholder="Enter your instructions here...", label="📝 Input Text")
                with gr.Column():
                    output_text_text = gr.Textbox(label="📄 Processed Commands", lines=5)
                    safty_check_text = gr.Textbox(label="✅ Safety Check")

            process_button_text = gr.Button("Send Text")
            process_button_text.click(
                fn=text_speech.process_text_input,
                inputs=text_input,
                outputs=[output_text_text, safty_check_text]
            )

demo.launch()
