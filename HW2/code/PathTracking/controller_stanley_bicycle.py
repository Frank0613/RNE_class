import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerStanleyBicycle(Controller):
    def __init__(self, model, 
                 # TODO 4.3.1: Tune Stanley Gain
                 kp=0.5):
        self.path = None
        self.kp = kp
        self.l = model.l
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    # State: [x, y, yaw, delta, v]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, delta, v = info["x"], info["y"], info["yaw"], info["delta"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        # Search Front Wheel Target Locally
        front_x = x + self.l*np.cos(np.deg2rad(yaw))
        front_y = y + self.l*np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta)) if np.cos(np.deg2rad(delta)) != 0 else v
        
        min_idx, min_dist = utils.search_nearest_local(self.path, (front_x,front_y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        target = self.path[min_idx]

        # TODO 4.3.1: Stanley Control for Bicycle Kinematic Model
        # 1. Heading error
        target_yaw = np.deg2rad(target[2])
        current_yaw = np.deg2rad(yaw)
        theta_e = utils.angle_norm(target_yaw - current_yaw)

        while theta_e > np.pi:
            theta_e -= 2 * np.pi
        while theta_e < -np.pi:
            theta_e += 2 * np.pi

        # 2. Cross-track error (e)
        error = np.sin(target_yaw) * (front_x - target[0]) - np.cos(target_yaw) * (front_y - target[1])

        # 3. Stanley formula
        ks = 5.0
        delta_e = np.arctan2(self.kp * error, vf + ks)
        next_delta = np.rad2deg(theta_e + delta_e)
        next_delta = np.clip(next_delta, -40.0, 40.0)



        # [end] TODO 4.3.1
        return next_delta
