import casadi as cs
import numpy as np
from example_robot_data.robots_loader import load
from adam.casadi.computations import KinDynComputations

class BaseSystem:
    def __init__(self):
        self.nx = 0
        self.nu = 0
        self.dt = 0.01

    def poly_cost(self, x):
        return (x - 1.9)*(x - 1.0)*(x - 0.6)*(x + 0.5)*(x + 1.2)*(x + 2.1)

    def running_cost_sym(self, x, u):
        raise NotImplementedError

    def dynamics_sym(self, x, u):
        raise NotImplementedError

class SingleIntegrator(BaseSystem):
    def __init__(self, dt=0.01):
        super().__init__()
        self.nx = 1
        self.nu = 1
        self.dt = dt

    def __name__(self):
        return "single_integrator"

    def dynamics_sym(self, x, u):
        return x + self.dt * u

    def running_cost_sym(self, x, u):
        # x is a scalar here
        return self.dt * (0.5 * u**2 + self.poly_cost(x))

class DoubleIntegrator(BaseSystem):
    def __init__(self, dt=0.01):
        super().__init__()
        self.nx = 2
        self.nu = 1
        self.dt = dt

    def __name__(self):
        return "double_integrator"

    def dynamics_sym(self, x, u):
        p, v = x[0], x[1]
        p_next = p + self.dt * v + 0.5 * self.dt**2 * u
        v_next = v + self.dt * u
        return cs.vertcat(p_next, v_next)

    def running_cost_sym(self, x, u):
        p, v = x[0], x[1]
        
        # Weighting
        Q_v = 1.0  # Weight on velocity
        R = 0.5    # Control weight
        
        # Instantaneous cost
        # Add v**2 to force the system to decelerate to 0
        instantaneous_cost = R * u**2 + self.poly_cost(p) + Q_v * v**2
        
        return instantaneous_cost * self.dt
        
class SinglePendulum(BaseSystem):
    def __init__(self, dt=0.01, g=9.81, l=1.0, m=1.0):
        super().__init__()
        self.nx = 2  # [theta, omega]
        self.nu = 1  # [tau]
        self.dt = dt
        self.g = g
        self.l = l
        self.m = m

    def __name__(self):
        return "single_pendulum"

    def dynamics_sym(self, x, u):
        theta = x[0]
        omega = x[1]
        
        # Equation: theta_dot = omega
        # Equation: omega_dot = - (g/l) * sin(theta) + u / (m * l^2)
        
        theta_next = theta + self.dt * omega
        omega_next = omega + self.dt * (-(self.g / self.l) * cs.sin(theta) + u / (self.m * self.l**2))
        
        return cs.vertcat(theta_next, omega_next)

    def running_cost_sym(self, x, u):
        theta = x[0]
        omega = x[1]

        # Quadratic cost - track upright position (theta = 0)
        Q_theta = 10.0  # weight on angle error
        Q_omega = 1.0   # weight on velocity
        R = 0.1         # control effort weight

        target_theta = 0.0  # Upright position
        cost = (Q_theta * (theta - target_theta)**2 + Q_omega * omega**2 + R * u**2) * self.dt
        return cost

class DoublePendulum(BaseSystem):
    def __init__(self, dt=0.05):
        super().__init__()
        self.nx = 4  # [theta1, theta2, omega1, omega2]
        self.nu = 2  # [tau1, tau2]
        self.dt = dt
        
        robot = load("double_pendulum")
        joints_name_list = [s for s in robot.model.names[1:]]
        kinDyn = KinDynComputations(robot.urdf, joints_name_list)
        
        q = cs.SX.sym("q", 2)
        dq = cs.SX.sym("dq", 2)
        ddq = cs.SX.sym("ddq", 2)
        
        state = cs.vertcat(q, dq)
        rhs = cs.vertcat(dq, ddq)
        self.f_dynamics = cs.Function("f", [state, ddq], [rhs])
        
    def __name__(self):
        return "double_pendulum"

    def dynamics_sym(self, x, u):
        # x_next = x + dt * f(x, u)
        return x + self.dt * self.f_dynamics(x, u)

    def running_cost_sym(self, x, u):
        # x = [theta1, theta2, omega1, omega2]
        theta1, theta2 = x[0], x[1]
        omega1, omega2 = x[2], x[3] # Angular velocities
        
        roots = [1.9, 1.0, 0.6, -0.5, -1.2, -2.1]
        poly1 = 1.0
        poly2 = 1.0
        for r in roots:
            poly1 *= (theta1 - r)
            poly2 *= (theta2 - r)
            
        # Weight on velocity: the higher it is, the more the system will try to stay still
        w_omega = 1.0 
        velocity_cost = w_omega * (omega1**2 + omega2**2)
        
        # Total cost: control + potential (polynomial) + velocity
        cost = (0.5 * cs.sumsqr(u) + poly1 + poly2 + velocity_cost) * self.dt
        return cost

def get_system(system_name, dt):
    systems = {
        "single_integrator": SingleIntegrator(dt),
        "double_integrator": DoubleIntegrator(dt),
        "single_pendulum": SinglePendulum(dt),
        "double_pendulum": DoublePendulum(dt)
    }
    return systems.get(system_name)
