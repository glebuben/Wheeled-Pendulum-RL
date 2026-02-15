import numpy as np
from scipy.integrate import odeint


class WheelPoleSystem:
    """
    A wheel-pole balancing system where a pole is attached to a wheel.
    The wheel can rotate (generalized coordinate: phi) and the pole can swing (theta).
    
    Equations of motion derived from Lagrangian mechanics:
    - phi: wheel rotation angle
    - theta: pole angle from vertical (0 = upright)
    
    Physics:
    The system dynamics are derived from the Lagrangian L = T - V where:
    - T includes kinetic energy of wheel rotation, wheel translation, and rod rotation
    - V is the gravitational potential energy of the rod's center of mass
    
    The resulting equations of motion are:
    (I_b + l²·m_b)·θ̈ + l·m_b·r·cos(θ)·φ̈ = g·l·m_b·sin(θ)
    [I_w + m_b·r² + m_w·r²]·φ̈ + m_b·r·l·cos(θ)·θ̈ = τ + m_b·r·l·sin(θ)·θ̇²
    
    where:
    - I_b = (1/3)·m_b·l² is the moment of inertia of the rod about its end
    - I_w = (1/2)·m_w·r² is the moment of inertia of the wheel (solid cylinder)
    """
    
    def __init__(self, rod_length=1.0, wheel_radius=0.2, 
                 wheel_mass=1.0, pole_mass=0.1, gravity=9.81, dt=0.02):
        """
        Initialize the wheel-pole system.
        
        Parameters:
        -----------
        rod_length : float
            Length of the pole (m)
        wheel_radius : float
            Radius of the wheel (m)
        wheel_mass : float
            Mass of the wheel (kg)
        pole_mass : float
            Mass of the pole/rod (kg)
        gravity : float
            Gravitational acceleration (m/s^2)
        dt : float
            Time step for simulation (s)
        """
        self.L = rod_length  # pole/rod length (l in your notation)
        self.R = wheel_radius  # wheel radius (r in your notation)
        self.m_wheel = wheel_mass  # m_w in your notation
        self.m_pole = pole_mass  # m_b in your notation
        self.g = gravity
        self.dt = dt
        
        # Moment of inertia of wheel (assuming solid cylinder)
        self.I_wheel = 0.5 * self.m_wheel * self.R**2  # I_w
        
        # Moment of inertia of rod about its end (I_b in your notation)
        # For a uniform rod rotating about one end: I = (1/3) * m * L^2
        self.I_rod = (1.0 / 3.0) * self.m_pole * self.L**2  # I_b
        
        # State: [phi, phi_dot, theta, theta_dot]
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.time = 0.0
        self.step_count = 0
        
    def set_initial_state(self, phi=0.0, phi_dot=0.0, theta=0.0, theta_dot=0.0):
        """
        Set the initial state of the system.
        
        Parameters:
        -----------
        phi : float
            Initial wheel angle (rad)
        phi_dot : float
            Initial wheel angular velocity (rad/s)
        theta : float
            Initial pole angle from vertical (rad)
        theta_dot : float
            Initial pole angular velocity (rad/s)
        """
        self.state = np.array([phi, phi_dot, theta, theta_dot])
        self.time = 0.0
        self.step_count = 0
        
    def _dynamics(self, state, t, torque):
        """
        Compute the derivatives of the state variables using the correct equations.
        
        Based on the Lagrangian-derived equations:
        Equation 1 (theta): I_b*theta_ddot + l^2*m_b*theta_ddot + l*m_b*r*cos(theta)*phi_ddot - g*l*m_b*sin(theta) = 0
        Equation 2 (phi): I_w*phi_ddot + m_b*r*(-l*sin(theta)*theta_dot^2 + l*cos(theta)*theta_ddot + r*phi_ddot) + m_w*r^2*phi_ddot = torque
        
        Returns: [phi_dot, phi_ddot, theta_dot, theta_ddot]
        """
        phi, phi_dot, theta, theta_dot = state
        
        # System parameters (using your notation)
        m_b = self.m_pole  # rod mass
        m_w = self.m_wheel  # wheel mass
        l = self.L  # rod length
        r = self.R  # wheel radius
        g = self.g
        I_b = self.I_rod  # rod moment of inertia
        I_w = self.I_wheel  # wheel moment of inertia
        
        # Useful terms
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # From your equations:
        # Equation 1: (I_b + l^2*m_b)*theta_ddot + l*m_b*r*cos(theta)*phi_ddot = g*l*m_b*sin(theta)
        # Equation 2: [I_w + m_b*r^2 + m_w*r^2]*phi_ddot + m_b*r*l*cos(theta)*theta_ddot = torque + m_b*r*l*sin(theta)*theta_dot^2
        
        # Coefficients for the matrix equation: A * [theta_ddot; phi_ddot] = b
        # a11*theta_ddot + a12*phi_ddot = b1
        # a21*theta_ddot + a22*phi_ddot = b2
        
        a11 = I_b + l**2 * m_b
        a12 = l * m_b * r * cos_theta
        b1 = g * l * m_b * sin_theta
        
        a21 = m_b * r * l * cos_theta
        a22 = I_w + m_b * r**2 + m_w * r**2
        b2 = torque + m_b * r * l * sin_theta * theta_dot**2
        
        # Solve for accelerations using Cramer's rule
        det = a11 * a22 - a12 * a21
        
        if abs(det) < 1e-10:
            det = 1e-10  # Avoid division by zero
            
        theta_ddot = (a22 * b1 - a12 * b2) / det
        phi_ddot = (a11 * b2 - a21 * b1) / det
        
        return [phi_dot, phi_ddot, theta_dot, theta_ddot]
    
    def step(self, action):
        """
        Apply an action (torque) and simulate one time step.
        
        Parameters:
        -----------
        action : float
            Torque applied to the wheel (N⋅m)
            
        Returns:
        --------
        state : np.ndarray
            Current state [phi, phi_dot, theta, theta_dot]
        """
        # Integrate the equations of motion
        t_span = [0, self.dt]
        solution = odeint(self._dynamics, self.state, t_span, args=(action,))
        
        self.state = solution[-1]
        self.time += self.dt
        self.step_count += 1
        
        return self.state.copy()
    
    def get_state(self):
        """
        Get the current state of the system.
        
        Returns:
        --------
        state : np.ndarray
            Current state [phi, phi_dot, theta, theta_dot]
        """
        return self.state.copy()
    
    def get_current_step(self):
        """
        Get the current simulation step number.
        
        Returns:
        --------
        step : int
            Current step count
        """
        return self.step_count
    
    def get_time(self):
        """
        Get the current simulation time.
        
        Returns:
        --------
        time : float
            Current time (s)
        """
        return self.time
    
    def get_cartesian_positions(self):
        """
        Get Cartesian positions for visualization.
        
        Returns:
        --------
        wheel_center : tuple
            (x, y) position of wheel center
        pole_end : tuple
            (x, y) position of pole end
        """
        phi, _, theta, _ = self.state
        
        # Wheel center moves in a circle (or we can track accumulated displacement)
        # For visualization, we'll track the displacement
        wheel_x = self.R * phi
        wheel_y = self.R  # Height of wheel center from ground
        
        # Pole end position relative to wheel center
        pole_end_x = wheel_x + self.L * np.sin(theta)
        pole_end_y = wheel_y + self.L * np.cos(theta)
        
        return (wheel_x, wheel_y), (pole_end_x, pole_end_y)
    
    def reset(self):
        """
        Reset the system to initial conditions.
        """
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.time = 0.0
        self.step_count = 0


if __name__ == "__main__":
    # Example usage
    system = WheelPoleSystem(rod_length=1.0, wheel_radius=0.2)
    
    # Set initial state with small perturbation
    system.set_initial_state(phi=0.0, theta=0.1)
    
    print("Initial state:", system.get_state())
    print("Step:", system.get_current_step())
    
    # Simulate with no control
    for i in range(100):
        state = system.step(action=0.0)
        if i % 20 == 0:
            print(f"Step {system.get_current_step()}: phi={state[0]:.3f}, theta={state[2]:.3f}")