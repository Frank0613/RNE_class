import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPurePursuitBicycle(Controller):
    def __init__(self, model, 
                 # TODO 4.3.1: Tune Pure Pursuit Gain
                 kp=0.35, Lfc=4.5):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc
        self.dt = model.dt
        self.l = model.l
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    # State: [x, y, yaw, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        
        Ld = self.kp*v + self.Lfc
        
        # TODO 4.3.1: Pure Pursuit Control for Bicycle Kinematic Model
        # 1. Find the lookahead point
        lookahead_idx = self.current_idx
        for i in range(self.current_idx, len(self.path)):
            tx, ty = self.path[i][0], self.path[i][1]
            distance = np.sqrt((tx - x)**2 + (ty - y)**2)
            if distance >= Ld:
                lookahead_idx = i
                break

        self.current_idx = lookahead_idx
        target = self.path[self.current_idx]
        tx, ty = target[0], target[1]

        # 2. Calculate the angle to the target relative to the world frame
        target_angle_world = np.arctan2(ty - y, tx - x)

        # 3. Calculate Alpha
        current_yaw_rad = np.deg2rad(yaw)
        alpha = utils.angle_norm(target_angle_world - current_yaw_rad)

        # 4. Calculate Steering Angle (delta)
        delta_rad = np.arctan2(2.0 * self.l * np.sin(alpha), Ld)

        # 5. Convert to degrees
        next_delta = np.rad2deg(delta_rad)

        # [end] TODO 4.3.1
        return next_delta
