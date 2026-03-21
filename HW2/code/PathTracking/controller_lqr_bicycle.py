import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBicycle(Controller):
    def __init__(self, model, Q=None, R=None, control_state='steering_angle'):
        self.path = None
        if control_state == 'steering_angle':
            self.Q = np.eye(2)
            self.R = np.eye(1)
            # TODO 4.4.1: Tune LQR Gains
            self.Q[0,0] = 1
            self.Q[1,1] = 10 
            self.R[0,0] = 1
        elif control_state == 'steering_angular_velocity':
            self.Q = np.eye(3)
            self.R = np.eye(1)
            # TODO 4.4.4: Tune LQR Gains
            self.Q[0,0] = 1
            self.Q[1,1] = 10
            self.Q[2,2] = 100
            self.R[0,0] = 1
        self.pe = 0
        self.pth_e = 0
        self.pdelta = 0
        self.dt = model.dt
        self.l = model.l
        self.control_state = control_state
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0
        self.pdelta = 0
        self.current_idx = 0

    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=0.01): # Discrete-time Algebra Riccati Equation (DARE)
        P = Q.copy()
        for i in range(max_iter):
            temp = np.linalg.inv(R + B.T @ P @ B)
            Pn = A.T @ P @ A - A.T @ P @ B @ temp @ B.T @ P @ A + Q
            if np.abs(Pn - P).max() < eps:
                break
            P = Pn
        return Pn

    # State: [x, y, yaw, delta, v]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, delta, v = info["x"], info["y"], info["yaw"], info["delta"], info["v"]
        yaw = utils.angle_norm(yaw)

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0.0
        
        # Search Nesrest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x,y))
        target = self.path[min_idx]
        target[2] = utils.angle_norm(target[2])
        
        if self.control_state == 'steering_angle':
            # TODO 4.4.1: LQR Control for Bicycle Kinematic Model with steering angle as control input
            target_yaw = np.deg2rad(target[2])
            current_yaw = np.deg2rad(yaw)
            
            error = (y - target[1]) * np.cos(target_yaw) - (x - target[0]) * np.sin(target_yaw)
            theta_e = utils.angle_norm(current_yaw - target_yaw)
            while theta_e > np.pi:
                theta_e -= 2 * np.pi
            while theta_e < -np.pi:
                theta_e += 2 * np.pi

            state_vector = np.array([[error], [theta_e]])

            A = np.array([[1.0, v * self.dt],
                        [0.0, 1.0]])
            B = np.array([[0.0],
                        [v / self.l * self.dt]])

            P = self._solve_DARE(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)

            u = -K @ state_vector
            next_delta = np.rad2deg(u[0, 0])
            # [end] TODO 4.4.1
        elif self.control_state == 'steering_angular_velocity':
            # TODO 4.4.4: LQR Control for Bicycle Kinematic Model with steering angular velocity as control input
            # 1. Extract target and current values
            target_yaw = np.deg2rad(target[2])
            current_yaw = np.deg2rad(yaw)
            current_delta = np.deg2rad(delta)

            # 2. Calculate errors
            error = (y - target[1]) * np.cos(target_yaw) - (x - target[0]) * np.sin(target_yaw)
            theta_e = utils.angle_norm(current_yaw - target_yaw)
            while theta_e > np.pi:
                theta_e -= 2 * np.pi
            while theta_e < -np.pi:
                theta_e += 2 * np.pi

            # 3. State vector: [e, theta_e, delta]
            state_vector = np.array([
                [error],
                [theta_e],
                [current_delta]
            ])

            # 4. System matrices

            A = np.eye(3)
            A[0, 1] = v * self.dt       
            A[1, 2] = v / self.l * self.dt  

            B = np.zeros((3, 1))
            B[2, 0] = self.dt

            # 5. Solve DARE
            P = self._solve_DARE(A, B, self.Q, self.R)

            # 6. Compute gain K
            K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)

            # 7. Control input
            u = -K @ state_vector
            delta_dot = u[0, 0]

            # 8. Integrate to get next steering angle
            next_delta = delta + np.rad2deg(delta_dot) * self.dt
            next_delta = np.clip(next_delta, -40.0, 40.0)
            # [end] TODO 4.4.4
        
        return next_delta
