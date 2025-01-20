import time
import threading
import pygame as pg
from vi import Agent, Config, Window, HeadlessSimulation
from typing import Optional
from queue import Queue
import numpy as np


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




class MyAgent(Agent):
    """Custom agent with adjustable movement."""
    def __init__(self, images, simulation, pos=None, move=None):
        # Call the parent class's constructor
        super().__init__(images=images, simulation=simulation, pos=pos, move=move)

        # Add any custom initialization logic here
        self.custom_property = "I am a custom agent"

    def update(self):
        # Add custom update logic for the agent here
        # Example: Change image when in proximity to other agents
        in_proximity = self.in_proximity_accuracy().count()
        if in_proximity > 0:
            self.change_image(1)
        else:
            self.change_image(0)


    



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

    def _load_image(self, path: str) -> pg.surface.Surface:
        """Load an image from the given path."""
        return pg.image.load(path)

    def stop(self):
        """Stop the simulation."""
        self.running = False
        super().stop()
        pg.quit()       # Quit the Pygame environment