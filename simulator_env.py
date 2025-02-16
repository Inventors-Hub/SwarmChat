import math
import time
import threading
import pygame as pg
from vi import Agent, Config, Window, HeadlessSimulation
from typing import Optional
from queue import Queue
import numpy as np
from pygame.math import Vector2
import py_trees as pt
import parser
import xml.etree.ElementTree as ET

class MyWindow(Window):
    """Custom window class for simulation."""
    def __init__(self, width=800, height=600):
        super().__init__(width, height)




class MyConfig(Config):
    """Custom configuration for simulation."""
    def __init__(self, radius=25, visualise_chunks=True, window=None, movement_speed=2):
        super().__init__(
            radius=radius,
            visualise_chunks=visualise_chunks,
            window=window or MyWindow(800, 600),
            movement_speed=movement_speed
        )




# class MyAgent(Agent):
#     """Custom agent with adjustable movement."""
#     def __init__(self, images, simulation, pos=None, move=None):
#         # Call the parent class's constructor
#         super().__init__(images=images, simulation=simulation, pos=pos, move=move)

#         # Add any custom initialization logic here
#         self.custom_property = "I am a custom agent"

#     def update(self):
#         # Add custom update logic for the agent here
#         # Example: Change image when in proximity to other agents
#         in_proximity = self.in_proximity_accuracy().count()
#         if in_proximity > 0:
#             self.change_image(1)
#         else:
#             self.change_image(0)




class SwarmAgent(Agent):
    def __init__(self, images, simulation, pos, nest_pos, target_pos):
        super().__init__(images=images, simulation=simulation)
        # Ensure the agent gets the configuration from the simulation.
        self.config = simulation.config  
        
        self.pos = pos
        self.nest_pos = nest_pos
        self.target_pos = target_pos  
        self.target_detected_flag = False
        self.target_reached_flag = False
        self.is_agent_in_nest_flag = False
        self.obstacle_radius = 5
        self.state = "seeking"
        self.bt_active = True  # Add a flag

        # Load and build the BT from XML:
        file_path = "tree.xml"
        trees = parser.parse_behavior_trees(file_path)
        subtree_mapping = { tree.attributes.get("ID"): tree for tree in trees }
        
        xml_tree = ET.parse(file_path)
        xml_root = xml_tree.getroot()
        main_tree_id = xml_root.attrib.get("main_tree_to_execute")
        if not main_tree_id or main_tree_id not in subtree_mapping:
            raise ValueError("Main tree not found in the XML!")
        main_tree_node = subtree_mapping[main_tree_id]
        
        # Build the py_trees tree:
        self.bt = parser.build_behavior(main_tree_node, subtree_mapping)
        
        # Inject the agent instance into all leaf behaviors.
        self._inject_agent(self.bt)
        
    def _inject_agent(self, node):
        """Recursively set the agent for any custom BT nodes."""
        if hasattr(node, "agent"):
            node.agent = self
        if hasattr(node, "children"):
            for child in node.children:
                self._inject_agent(child)

    def update(self):
        if self.bt_active:
            self.bt.tick_once()


    # def update(self):
    #     self.bt.tick_once()
    #     # self.root_node.run(self)

    def say(self, message: str):
        # print(message)
        return pt.common.Status.SUCCESS
 
    def obstacle(self):
        """
        Check for obstacle intersections within a predefined radius.        
        Returns: True if an obstacle is detected within the radius, False otherwise.
        """
        for intersection in self.obstacle_intersections(scale=self.obstacle_radius):
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE

    def is_obstacle_detected(self):
        """
        Condition node: Determine if any obstacles are detected in the vicinity of the agent. Returns: True if an obstacle is detected, False otherwise.
        """
        if self.obstacle():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE

    # Built-in behavior
    def avoid_obstacle(self):
        """
        Action node: Execute an action to avoid detected obstacles. Returns: Always returns True, indicating the action was executed.
        """
        return pt.common.Status.SUCCESS
    
    def is_target_detected(self):
        """
        Action node: Check if the target is within a detectable distance from the agent's position. Returns: True if the target is within 20 units of distance, False otherwise.
        """
        distance = math.dist(self.target_pos, self.pos)
        if distance <= 20:
            self.target_detected_flag = True        
        if self.target_detected_flag:
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE
        
    
    def is_target_reached(self):
        """
        Condition node: Check if the agent has reached the target. Returns: True if the target is within 15 units of distance, False otherwise.
        """
        distance = math.dist(self.target_pos, self.pos)
        if distance <= 15:
            self.target_reached_flag = True        
        if self.target_reached_flag:            
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE
        

    def change_color_to_green(self):
        """
        Action node: Change the agent's color to green. Returns: Always returns True, indicating the action was executed.
        """
        self.change_image(1)
        return pt.common.Status.SUCCESS
    
    def change_color_to_white(self):
        """
        Action node: Change the agent's color to white. Returns: Always returns True, indicating the action was executed.
        """
        self.change_image(0)  
        return pt.common.Status.SUCCESS
    
    def change_color_to_red(self):
        """
        Action node: Change the agent's color to red. Returns: Always returns True, indicating the action was executed.
        """
        self.change_image(2)  
        return pt.common.Status.SUCCESS
    

    def is_agent_in_nest(self):
        """
        Condition node: Determine if the agent is in the nest. Returns: True if the agent is in the nest, False otherwise.
        """
        distance = math.dist(self.nest_pos, self.pos)
        if distance <= 17 and (self.target_reached_flag==True or self.target_detected_flag == True or self.state == "completed" ) :
            self.state = "seeking"
            # self.target_detected_flag = False
            # self.target_reached_flag = False   
            self.is_agent_in_nest_flag = True

        if self.is_agent_in_nest_flag:
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE


    def agent_movement_freeze(self):
        """
        Action node: Freeze the agent's movement, typically to indicate a stop in activity. Returns: Always returns True, indicating the action was executed.
        """
        self.freeze_movement()
        return pt.common.Status.SUCCESS
    
    def continue_movement_agent(self):
        """
        Action node: Continue the agent's movement after it has been previously frozen. Returns: Always returns True, indicating the action was executed.
        """
        self.continue_movement()
        return pt.common.Status.SUCCESS

    def wander(self):
        """
        Action node: Perform a wandering action where the agent moves randomly within the environment. Returns: Always returns True, indicating the action was executed.
        """
        Agent.change_position(self)
        return pt.common.Status.SUCCESS

    def is_path_clear(self):
        """
        Condition node: Check if the path ahead of the agent is clear of obstacles. Returns: True if no obstacles are detected ahead, False if obstacles are present.
        """
        return not self.obstacle()
    
    def is_line_formed(self):
        """
        Condition node: Determine if the agent has formed a line with a reference point at the center of the window. Returns: True if the line is formed with the center, False otherwise.
        """
        center_x = self.config.window.width / 2
        direction = Vector2(center_x, self.pos.y) - self.pos
        if direction.length() > 0.5:
            return pt.common.Status.FAILURE        
        return pt.common.Status.SUCCESS

    def form_line(self):
        """
        Action node: Direct the agent to form a line towards the center of the window. This function adjuststhe agent's position to align it with the center. Returns: Always returns True, indicating the action was executed.
        """
        # print("form_line")
        center_x = self.config.window.width / 2
        direction = Vector2(center_x, self.pos.y) - self.pos
        if direction.length() > 0.5:
            direction.scale_to_length(self.config.movement_speed)
            self.pos += direction     
        return pt.common.Status.SUCCESS
    
    def task_completed(self):
        """
        Action node: Signal that the agent has completed its designated task. Returns: Always returns True, indicating that the task completion action was executed.
        """
        self.state = "completed"
        return pt.common.Status.SUCCESS
    

    def check_for_distance(self, x: float, y: float, z: float):
        # Implement a check or simply return a default value.
        # print("Checking distance:", abs(x - y))
        return pt.common.Status.FAILURE

    def test1(self, x: int, y: int, z: int):
        # Example implementation.
        # print(f"Control operation in test1: x={x}, y={y}, z={x+y}")
        return pt.common.Status.SUCCESS

    def testDecorator(self, a: int, b: int, c: int):
        # Example implementation.
        # print(f"Decorator: a={a}, b={a*2}, c={(a*2)+10}")
        return pt.common.Status.SUCCESS



    



class StreamableSimulation(HeadlessSimulation):
    """Modified Simulation class that captures frames for streaming."""
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        pg.init()

        size = self.config.window.as_tuple()
        self._screen = pg.Surface(size, pg.SRCALPHA)
        self._background = pg.Surface(size, pg.SRCALPHA)
        self._background.fill((0, 0, 0))

        self.frame_queue = Queue(maxsize=30)
        # self.running = True
        self._frame_lock = threading.Lock()

    
    def get_frame(self):
        with self._frame_lock:
            surf_copy = self._screen.copy()
            frame = np.array(pg.surfarray.pixels3d(surf_copy))
            return np.transpose(frame, (1, 0, 2))

    def tick(self):
        """Run a simulation step and capture frames."""
        super().tick()

        with self._frame_lock:
            self._screen.blit(self._background, (0, 0))
            for sprite in self._all.sprites():
                self._screen.blit(sprite.image, sprite.rect)

        try:
            frame = self.get_frame()
            self.frame_queue.put(frame, block=False)
        except Queue.Full:
            print("Frame queue is full. Dropping frame.")

    # def _load_image(self, path: str) -> pg.surface.Surface:
    #     """Load an image from the given path."""
    #     return pg.image.load(path)
    
    def _load_image(self, paths):
        """Load one or more images from given paths."""
        if isinstance(paths, str):  # If it's a single string, load normally
            return pg.image.load(paths)
        elif isinstance(paths, list):  # If it's a list, load all images
            return [pg.image.load(path) for path in paths]
        raise TypeError("Expected a string (file path) or a list of file paths")

    def stop(self):
        """Stop the simulation."""
        # Do not try to call self.bt.stop() because simulation does not own a BT.
        # self.running = False
        super().stop()
        pg.quit()       # Quit the Pygame environment
        




if __name__=="__main__":

    # Define nest and target positions
    nest_x, nest_y = 450, 400
    target_x, target_y = 200, 100
    nest_pos = Vector2(nest_x, nest_y)
    target_pos = Vector2(target_x, target_y)

    # Load images for agents
    agent_images_paths = ["./images/white.png", "./images/green.png", "./images/red circle.png"]

    config = MyConfig(radius=250, visualise_chunks=True, movement_speed=2)
    sim = StreamableSimulation(config=config)

    # Load images
    loaded_agent_images = sim._load_image(agent_images_paths)



    # Initialize agents with behavior tree parsing
    for _ in range(50):
        agent = SwarmAgent(
            images=loaded_agent_images,
            simulation=sim,
            pos=Vector2(nest_x, nest_y),
            nest_pos=nest_pos,
            target_pos=target_pos,
        )
        sim._agents.add(agent)
        sim._all.add(agent)
    # Draw environment elements
    sim.spawn_obstacle("./images/rect_obst.png", 350, 100)
    sim.spawn_obstacle("./images/rect_obst (1).png", 100, 350)
    sim.spawn_site("./images/rect.png", target_x, target_y)
    sim.spawn_site("./images/nest.png", nest_x, nest_y)

    for agent in sim._agents:
        agent.bt.tick_once()
    
    # Then run your simulation loop without ticking the BT further.
    while sim.running:
        sim.tick()
        if not sim.frame_queue.empty():
            frame = sim.frame_queue.get()
            # update_frame(frame) or display the frame as needed.
        time.sleep(1/30)