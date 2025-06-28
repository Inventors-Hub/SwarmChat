"""SwarmChat front-end interface.

Defines a Gradio GUI for:
  - recording/translating audio
  - translating text
  - running safety checks
  - generating BT XML
  - streaming the swarm simulation

Classes:
    GradioStreamer: Manages the background simulator thread and frame streaming.
Functions:
    create_gradio_interface: Build and configure the Gradio Blocks interface.
    stop_gradio_interface: Interrupt the Gradio flow when safety fails.
"""
import gradio as gr
from pygame import Vector2
import time
import threading
import queue
from simulator_env import StreamableSimulation, SwarmAgent, MyConfig, MyWindow

import speech_processing
import text_processing
import safety_module
import bt_generator



class GradioStreamer:
    """
    Singleton that drives the SwarmChat simulation in a background thread
    and provides a 30 FPS image stream to Gradio.

    Attributes:
        latest_frame (np.ndarray | None): Last rendered frame.
        running (bool): Controls the simulation loop.
        sim (StreamableSimulation | None): The Pygame/py-trees sim instance.
        sim_thread (Thread | None): Background thread reference.
    """
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
            self.quit = False
    
    def update_frame(self, frame):
        self.latest_frame = frame

    def run_simulation(self):
        """Main loop: steps the sim, grabs frames, enqueues them, stops after 2 minutes."""
        
        nest_pos = Vector2(450, 400)
        target_pos = Vector2(300, 200)
        agent_images_paths = ["./images/white.png", "./images/green.png", "./images/red circle.png"]
        config = MyConfig(radius=250, visualise_chunks=True, movement_speed=2)
        self.sim = StreamableSimulation(config=config)
        loaded_agent_images = self.sim._load_image(agent_images_paths)

        # Create agents (each agent builds its own BT in its __init__)
        for _ in range(50):
            agents_pos = Vector2(450, 400)
            agent = SwarmAgent(
                images=loaded_agent_images,
                simulation=self.sim,
                pos=agents_pos,
                nest_pos=nest_pos,
                target_pos=target_pos
            )
            self.sim._agents.add(agent)
            self.sim._all.add(agent)
        # (Optionally spawn obstacles and sites.)
        self.sim.spawn_obstacle("./images/rect_obst.png", 350, 50)
        self.sim.spawn_obstacle("./images/rect_obst (1).png", 100, 350)

        self.sim.spawn_site("./images/rect.png", 300, 200)
        self.sim.spawn_site("./images/nest.png", 450, 400)


        start_time = time.time()  
        
        while self.running:
            self.sim.tick()
            
            if not self.sim.frame_queue.empty():
                frame = self.sim.frame_queue.get()
                self.update_frame(frame)    

            time.sleep(1/30)  # Maintain a frame rate of ~30 FPS
            # Stop after 1 minute
            if time.time() - start_time >= 120:
                print("Simulation stopped after 1 minute.")
                break

            

    def stream(self):
        """
        Generator that yields the latest frame at ~30 Hz for Gradio’s `streaming=True`.
        Yields None until the sim produces frames.
        """
        while True:
            if self.sim is not None and self.latest_frame is not None:
                yield self.latest_frame
            else:
                # Optionally, yield a blank image or None.
                yield None
            time.sleep(1/30)


    def start_simulation(self):
        """Start the simulation, creating a new thread if necessary."""
        if not self.sim_thread or not self.sim_thread.is_alive():
            self.running = True      
            self.quit = False        
            self.latest_frame = None 
            self.sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
            self.sim_thread.start()


    def clear_frame_queue(self):
        if self.sim:
            try:
                while True:
                    self.sim.frame_queue.get_nowait()
            except queue.Empty:
                pass



    def stop_simulation(self):
        """
        Gracefully halts the simulation thread, clears queues, and resets state.
        Called when the user presses “Stop” or a safety violation occurs.
        """
        self.running = False
        self.quit = True
        if self.sim:
            for agent in self.sim._agents:
                agent.bt_active = False
            self.sim.running = False
            self.sim.stop()
            self.clear_frame_queue()
            self.sim = None
        if self.sim_thread and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=2)
            print("Simulation thread terminated.")
        self.latest_frame = None  # Clear the displayed frame
        print("Simulation stopped successfully.")




def stop_gradio_interface():
    raise Exception("Simulation stopped!")


def create_gradio_interface():
    streamer = GradioStreamer()
    
    def on_translate_or_process():
        streamer.start_simulation()
        return gr.update(visible=True)
    
    def on_stop():
        print("Simulation on_stop")
        streamer.stop_simulation()
        return gr.update(visible=False)
    
    behaviors = bt_generator.call_behaviors()       
    formatted_behaviors = "\n".join(
        f"- **{name}**: {doc.split('Returns:')[0].strip()}"
        for name, doc in sorted(
            behaviors.items(),
            key=lambda item: item[0].lower()
                )
            )


    # Gradio Interface
    with gr.Blocks() as demo:
        gr.Markdown(
            """
        # 🐝 **SwarmChat:** Enabling Human–Swarm Interaction and Robot Control via Natural Language
        Easily talk to virtual robots, and see the result live.  
        """
        )
        gr.Markdown(
            """
        **How it works**

        1.  Speak or type a task in *any EU language* (e.g. “Find the target, then line up by colour”).
        2.  Press **Start** to launch the simulator. Use **Stop** to halt & reset.
        3.  SwarmChat translates your command, runs a safety check, and auto-builds a behaviour tree (BT).          

        > The BT XML is shown on the right so you can copy / save it for real robots.
        """
        )  
        with gr.Tabs():
            # Tab for microphone input
            with gr.Tab("Microphone Input"):
                gr.Markdown("## 🎙️ Voice mode")
                gr.Markdown("""
                Use your microphone to record audio instructions for the swarm. The system translates them into a robot-executable BT.
                """)
                with gr.Row():
                    with gr.Column():
                        microphone_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Record Audio")
                        safety_checkbox = gr.Checkbox(label="Turn off Safety Model")                        
                    with gr.Column():
                        output_text_audio = gr.Textbox(label="📄 Translated Instructions to English" )
                        safty_check_audio = gr.Textbox(label="✅ Safety Check")
                        
                
                translate_button_audio = gr.Button("Start")

                simulation_output = gr.Image(label="Live Stream", streaming=True, visible=False)
                stop_button = gr.Button("Stop")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"""**🛠 The available behaviours.**\n{formatted_behaviors}\n\nThese are the only low-level actions/conditions the model is allowed to use yet.""")

                    with gr.Column():
                        generated_BT_audio = gr.Textbox(label="Generated behavior tree")
                
                translate_button_audio.click(
                    fn=speech_processing.translate_audio,
                    inputs=microphone_input,
                    outputs=output_text_audio
                ).then(
                    fn=safety_module.check_safety, 
                    inputs=[output_text_audio,safety_checkbox], 
                    outputs=safty_check_audio
                ).then(
                    fn=lambda x: x if x == "Safe" else stop_gradio_interface(),
                    inputs=safty_check_audio, 
                    outputs=None
                ).success(
                    fn=bt_generator.generate_behavior_tree,
                    inputs=output_text_audio, 
                    outputs=generated_BT_audio
                ).success(                    
                    fn=on_translate_or_process,
                    outputs=simulation_output
                )


            stop_button.click(fn=on_stop,outputs=simulation_output)
            demo.load(fn=streamer.stream, outputs=simulation_output)

            # Tab for text input
            with gr.Tab("📝 Text Input"):
                gr.Markdown("## 📝 Text mode")
                gr.Markdown("""
                Enter text-based instructions for the swarm. The system translates them into a robot-executable BT.
                """)
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(lines=4, placeholder="Enter your instructions here...", label="📝 Input Text")
                        safety_checkbox_text = gr.Checkbox(label="Turn off Safety Model")                        
                    with gr.Column():
                        output_text_text = gr.Textbox(label="📄 Translated Instructions to English", lines=2)
                        safty_check_text = gr.Textbox(label="✅ Safety Check")

                process_button_text = gr.Button("Start")

                simulation_output = gr.Image(label="Live Stream", streaming=True, visible=False)
                stop_button = gr.Button("Stop")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"""**🛠 The available behaviours.**\n{formatted_behaviors}\n\nThese are the only low-level actions/conditions the model is allowed to use yet.""")

                    with gr.Column():
                        generated_BT_text = gr.Textbox(label="Generated behavior tree")

                process_button_text.click(
                    fn=text_processing.translate_text,
                    inputs=text_input,
                    outputs=output_text_text
                ).then(
                    fn=safety_module.check_safety, 
                    inputs=[output_text_text,safety_checkbox_text], 
                    outputs=safty_check_text
                ).then(
                    fn=lambda x: x if x == "Safe" else stop_gradio_interface(), 
                    inputs=safty_check_text, 
                    outputs=None
                ).success(
                    fn=bt_generator.generate_behavior_tree,
                    inputs=output_text_text, 
                    outputs=generated_BT_text
                ).success(                    
                    fn=on_translate_or_process,
                    outputs=simulation_output
                )
                stop_button.click(fn=on_stop,outputs=simulation_output)
                demo.load(fn=streamer.stream, outputs=simulation_output)
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    try:
        demo.launch()
    finally:
        streamer = GradioStreamer()
        streamer.stop_simulation()

