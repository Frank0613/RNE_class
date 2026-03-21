import numpy as np
import sys
sys.path.append("..")
from Simulation.utils import State, ControlState
from Simulation.kinematic import KinematicModel

class KinematicModelBicycle(KinematicModel):
    def __init__(self,
            l = 30,     # distance between rear and front wheel
            dt = 0.05
        ):
        # Distance from center to wheel
        self.l = l
        # Simulation delta time
        self.dt = dt

    def step(self, state:State, cstate:ControlState) -> State:
        # TODO 2.3.1: Bicycle Kinematic Model
        # 1. Update velocity using 'a' (acceleration) from ControlState
        # v_next = v_current + a * dt
        v = state.v + cstate.a * self.dt
        
        # 2. Convert degrees to radians for trigonometric calculations
        yaw_rad = np.deg2rad(state.yaw)
        delta_rad = np.deg2rad(cstate.delta)

        # 3. Calculate derivatives based on kinematic equations
        # dx/dt = v * cos(theta)
        # dy/dt = v * sin(theta)
        # dyaw/dt = (v / L) * tan(delta)
        dx = v * np.cos(yaw_rad)
        dy = v * np.sin(yaw_rad)
        dyaw_rad = (v / self.l) * np.tan(delta_rad)

        # 4. Integrate to find the next state (Euler integration)
        x = state.x + dx * self.dt
        y = state.y + dy * self.dt
        # Convert angular change back to degrees for the State object
        yaw = state.yaw + np.rad2deg(dyaw_rad) * self.dt
        
        # 5. Set angular velocity (w) in degrees per second
        w = np.rad2deg(dyaw_rad)

        # [end] TODO 2.3.1
        state_next = State(x, y, yaw, v, w)
        return state_next
