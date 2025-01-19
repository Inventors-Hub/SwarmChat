import speech_text
import text_speech
import gradio as gr
from simulator_env import StreamableSimulation, MyAgent, MyConfig, MyWindow
import time
import threading


class GradioStreamer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GradioStreamer, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self.latest_frame = None
            self.running = True
            self.sim = None
            self.sim_thread = None
            self.initialized = True
    
    def update_frame(self, frame):
        self.latest_frame = frame
    
    def run_simulation(self):
        config = MyConfig(
            radius=25,
            visualise_chunks=True,
            movement_speed=2
        )
        self.sim = StreamableSimulation(config=config)

        # Use MyAgent with the correct parameters
        self.sim.batch_spawn_agents(100, MyAgent, ["images/white.png", "images/red.png"])

        while self.sim.running and self.running:
            self.sim.tick()
            if not self.sim.frame_queue.empty():
                frame = self.sim.frame_queue.get()
                self.update_frame(frame)
            time.sleep(1 / 30)
    
    def stream(self):
        while self.running:
            if self.latest_frame is not None:
                yield self.latest_frame
            time.sleep(1/30)
    
    def start_simulation(self):
        if not self.sim_thread or not self.sim_thread.is_alive():
            self.running = True
            self.sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
            self.sim_thread.start()
    
    def stop_simulation(self):
        self.running = False
        if self.sim:
            self.sim.stop()

def create_gradio_interface():
    streamer = GradioStreamer()
    
    def on_translate_or_process():
        streamer.start_simulation()
        return gr.update(visible=True)
    
    def on_stop():
        streamer.stop_simulation()
        return gr.update(visible=False)
    



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

                simulation_output = gr.Image(label="Live Stream", streaming=True, visible=False)
                stop_button = gr.Button("Stop Simulation")

                translate_button_audio.click(
                    fn=speech_text.translate_audio,
                    inputs=microphone_input,
                    outputs=[output_text_audio, safty_check_audio]
                ).then(
                    fn=on_translate_or_process,
                    outputs=simulation_output
                )
                stop_button.click(fn=on_stop, outputs=simulation_output)

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

                simulation_output = gr.Image(label="Live Stream", streaming=True, visible=False)
                stop_button = gr.Button("Stop Simulation")

                process_button_text.click(
                    fn=text_speech.process_text_input,
                    inputs=text_input,
                    outputs=[output_text_text, safty_check_text]
                ).then(
                    fn=on_translate_or_process,
                    outputs=simulation_output
                )
                stop_button.click(fn=on_stop, outputs=simulation_output)
        
        demo.load(fn=streamer.stream, outputs=simulation_output)
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    try:
        demo.launch()
    finally:
        streamer = GradioStreamer()
        streamer.stop_simulation()



# # Gradio Interface
# with gr.Blocks() as demo:
#     gr.Markdown("# 🎙️ SwarmChat: Unified Audio and Text Processing for Human-Swarm Interaction")
#     gr.Markdown("""
#     SwarmChat enables intuitive communication with swarm robotics through natural language. 
#     Utilize this interface to translate audio and process text for effective human-swarm interaction.
#     """)    
#     with gr.Tabs():
#         # Tab for microphone input
#         with gr.Tab("Microphone Input"):
#             gr.Markdown("## Record and Translate Audio")
#             gr.Markdown("""
#             Use your microphone to record audio instructions for swarms. The system will translate your 
#             natural language commands into English while ensuring content safety.
#             """)
#             with gr.Row():
#                 with gr.Column():
#                     microphone_input = gr.Audio(sources="microphone", type="filepath", label="🎙️ Record Audio")
#                 with gr.Column():
#                     output_text_audio = gr.Textbox(label="📄 Translated Instructions")
#                     safty_check_audio = gr.Textbox(label="✅ Safety Check")

#             translate_button_audio = gr.Button("Send Audio")
#             translate_button_audio.click(
#                 fn=speech_text.translate_audio,
#                 inputs=microphone_input,
#                 outputs=[output_text_audio, safty_check_audio]
#             )

#         # Tab for text input
#         with gr.Tab("📝 Text Input"):
#             gr.Markdown("## Process Text Commands")
#             gr.Markdown("""
#             Enter text instructions for swarm control or natural language processing. 
#             This tool translates and processes your text for safe execution by swarms.
#             """)
#             with gr.Row():
#                 with gr.Column():
#                     text_input = gr.Textbox(
#                         lines=5, placeholder="Enter your instructions here...", label="📝 Input Text")
#                 with gr.Column():
#                     output_text_text = gr.Textbox(label="📄 Processed Commands", lines=5)
#                     safty_check_text = gr.Textbox(label="✅ Safety Check")

#             process_button_text = gr.Button("Send Text")
#             process_button_text.click(
#                 fn=text_speech.process_text_input,
#                 inputs=text_input,
#                 outputs=[output_text_text, safty_check_text]
#             )

# demo.launch()
