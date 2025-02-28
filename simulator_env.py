import math
import time
import pygame as pg
from vi import Agent, Config, Window, HeadlessSimulation
from typing import Optional
from queue import Queue
import numpy as np
from pygame.math import Vector2
import py_trees as pt
import parser
import xml.etree.ElementTree as ET

import threading
import pyttsx3

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
        self.tts_engine = pyttsx3.init()  # Initialize text-to-speech engine


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
        # Initialize lists if they don't already exist
        if not hasattr(self, 'old_message'):
            self.old_message = []
                
        # Only speak the message if it has not been spoken before (i.e. not in old_message)
        if message not in self.old_message:
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
            self.old_message.append(message)
            
        return pt.common.Status.SUCCESS

    
    # def flocking(self):
    #     """
    #     Compute a flocking steering vector using:
    #      - Separation: steer away from nearby agents.
    #      - Alignment: match the average heading of nearby agents.
    #      - Cohesion: move toward the average position of neighbors.
    #     Adjusts the agent’s move vector accordingly.
    #     """
    #     nearby_agents = list(self.in_proximity_accuracy().without_distance())
    #     if not nearby_agents:
    #         return pt.common.Status.SUCCESS
        
    #     separation = Vector2(0, 0)
    #     alignment = Vector2(0, 0)
    #     cohesion = Vector2(0, 0)
        
    #     for other in nearby_agents:
    #         separation += (self.pos - other.pos)
    #         alignment += other.move
    #         cohesion += other.pos
        
    #     n = len(nearby_agents)
    #     alignment /= n
    #     cohesion /= n
    #     cohesion = cohesion - self.pos
        
    #     # Weigh the contributions (tweak these weights as needed)
    #     flock_vector = 0.5 * separation + 0.3 * alignment + 0.2 * cohesion
        
    #     # Update the agent's movement vector
    #     self.move += flock_vector
    #     # Cap the velocity
    #     max_speed = self.config.movement_speed
    #     if self.move.length() > max_speed:
    #         self.move.scale_to_length(max_speed)
    #     return pt.common.Status.SUCCESS
    def flocking(self):
        """
        Adjust the agent's move vector to align with nearby agents while maintaining a minimum separation distance.
        The agent blends its current movement toward the average movement (alignment) and away from agents that are too close (separation).
        """
        nearby_agents = list(self.in_proximity_accuracy().without_distance())
        if not nearby_agents:
            return pt.common.Status.SUCCESS

        alignment = Vector2(0, 0)
        separation = Vector2(0, 0)
        separation_count = 0

        # Desired minimum separation distance (adjust as needed)
        separation_threshold = 3

        # Calculate alignment and separation contributions.
        for other in nearby_agents:
            alignment += other.move
            
            diff = self.pos - other.pos
            distance = diff.length()
            if 0 < distance < separation_threshold:
                # The closer the neighbor, the stronger the repulsive force.
                separation += diff.normalize() * (separation_threshold - distance)
                separation_count += 1

        # Average the alignment vector over all neighbors.
        alignment /= len(nearby_agents)
        
        # If any agents are too close, average the separation vector.
        if separation_count > 0:
            separation /= separation_count

        # Blend the two influences. Here, alignment has a stronger influence than separation.
        # Adjust the blend factor (e.g., 0.3) to control separation influence.
        blended_force = alignment.lerp(separation, 0.3)
        
        # Smoothly blend the current move with the blended force.
        self.move = self.move.lerp(blended_force, 0.5)
        
        # Normalize and scale to the configured movement speed.
        if self.move.length() > 0:
            self.move = self.move.normalize() * self.config.movement_speed

        # Update position and apply wrap-around if necessary.
        self.pos += self.move
        self.there_is_no_escape()
        
        return pt.common.Status.SUCCESS


    
    def align_with_swarm(self):
        """
        Adjust the agent's move vector to better align with the average movement of nearby agents.
        """
        nearby_agents = list(self.in_proximity_accuracy().without_distance())
        if not nearby_agents:
            return pt.common.Status.SUCCESS

        avg_direction = Vector2(0, 0)
        for other in nearby_agents:
            avg_direction += other.move
        avg_direction /= len(nearby_agents)
        
        # Blend current movement with average direction.
        self.move = self.move.lerp(avg_direction, 0.5)
        if self.move.length() > 0:
            self.move = self.move.normalize() * self.config.movement_speed

        # Update position and wrap-around if necessary.
        self.pos += self.move
        self.there_is_no_escape()
        
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
    
    def change_color(self, color):
        """
        Action node: Change the agent's color to the specified color. Returns: Always returns True, indicating the action was executed.
        """
        color = color.lower()
        if color == "white":
            self.change_image(0)
        elif color == "green":
            self.change_image(1)
        elif color == "red":
            self.change_image(2)

        return pt.common.Status.SUCCESS
        

    # def change_color_to_green(self):
    #     """
    #     Action node: Change the agent's color to green. Returns: Always returns True, indicating the action was executed.
    #     """
    #     self.change_image(1)
    #     return pt.common.Status.SUCCESS
    
    # def change_color_to_white(self):
    #     """
    #     Action node: Change the agent's color to white. Returns: Always returns True, indicating the action was executed.
    #     """
    #     self.change_image(0)  
    #     return pt.common.Status.SUCCESS
    
    # def change_color_to_red(self):
    #     """
    #     Action node: Change the agent's color to red. Returns: Always returns True, indicating the action was executed.
    #     """
    #     self.change_image(2)  
    #     return pt.common.Status.SUCCESS
    

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

        # return not self.obstacle()

        if not self.obstacle():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
    
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
    


    # # for testing
    # def check_for_distance(self, x: float, y: float, z: float):
    #     # Implement a check or simply return a default value.
    #     print("Checking distance:", abs(x - y))
    #     return pt.common.Status.FAILURE

    # def test1(self, x: int, y: int, z: int):
    #     # Example implementation.
    #     print(f"Control operation in test1: x={x}, y={y}, z={x+y}")
    #     return pt.common.Status.SUCCESS

    # def testDecorator(self, a: int, b: int, c: int):
    #     # Example implementation.
    #     print(f"Decorator: a={a}, b={a*2}, c={(a*2)+10}")
    #     return pt.common.Status.SUCCESS



    



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
        self.running = True
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