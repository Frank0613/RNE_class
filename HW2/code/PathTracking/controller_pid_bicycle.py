import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPIDBicycle(Controller):
    def __init__(self, model, 
                 # TODO 4.1.3: Tune PID Gains
                 kp=1.0, 
                 ki=0.01, 
                 kd=0.5):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0
        self.dt = model.dt
        self.current_idx = 0
    
    def set_path(self, path):
        super().set_path(path)
        self.acc_ep = 0
        self.last_ep = 0
        self.current_idx = 0
    
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State
        x, y, yaw = info["x"], info["y"], info["yaw"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        # Search Nearest Target Locally
        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        
        # TODO 4.1.3: PID Control for Bicycle Kinematic Model
        # 1. Get the target point from the path
        target = self.path[self.current_idx]

        # 2. Calculate Cross-Track Error (CTE)
        # We use a simple distance-based error or vector-based error [cite: 416]
        dx = target[0] - x
        dy = target[1] - y
        # Error is the perpendicular distance to the path heading
        error = dy * np.cos(np.deg2rad(yaw)) - dx * np.sin(np.deg2rad(yaw))

        # 3. PID calculation [cite: 416]
        self.acc_ep += error * self.dt
        diff_ep = (error - self.last_ep) / self.dt
        
        next_delta = self.kp * error + self.ki * self.acc_ep + self.kd * diff_ep
        self.last_ep = error
        
        # [end] TODO 4.1.3
        return next_delta
