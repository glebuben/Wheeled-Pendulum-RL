# Wheel-Pole Balancing System

A reinforcement learning environment featuring a wheel-pole balancing system - a variation of the classic cartpole where linear cart motion is replaced by wheel rotation.

## System Description

The system consists of:
- A **wheel** that can rotate (generalized coordinate: φ)
- A **pole** attached to the wheel that can swing (generalized coordinate: θ)
- The control input is a **torque** applied to the wheel

The dynamics are derived from Lagrangian mechanics, with the key difference from cartpole being that horizontal displacement x = R·φ (where R is wheel radius).

## Files

1. **wheel_pole_system.py** - Core physics engine
2. **wheel_pole_simulation.py** - Interactive pygame visualization
3. **README.md** - This file

## Installation

Required dependencies:
```bash
pip install numpy scipy pygame
```

## Usage

### 1. Physics Engine (wheel_pole_system.py)

```python
from wheel_pole_system import WheelPoleSystem

# Create system with custom parameters
system = WheelPoleSystem(
    rod_length=1.0,      # Pole length (m)
    wheel_radius=0.2,    # Wheel radius (m)
    wheel_mass=1.0,      # Wheel mass (kg)
    pole_mass=0.1,       # Pole mass (kg)
    gravity=9.81,        # Gravity (m/s²)
    dt=0.02              # Time step (s)
)

# Set initial conditions
system.set_initial_state(
    phi=0.0,        # Initial wheel angle (rad)
    phi_dot=0.0,    # Initial wheel angular velocity (rad/s)
    theta=0.1,      # Initial pole angle (rad)
    theta_dot=0.0   # Initial pole angular velocity (rad/s)
)

# Run simulation
for i in range(1000):
    torque = 0.0  # Control input (N⋅m)
    state = system.step(torque)
    
    # Get state: [phi, phi_dot, theta, theta_dot]
    phi, phi_dot, theta, theta_dot = state
    
    # Get current step number
    step = system.get_current_step()
    
    # Your RL algorithm here
    # ...
```

### 2. Interactive Simulation (wheel_pole_simulation.py)

Run the interactive simulation:
```bash
python wheel_pole_simulation.py
```

**Controls:**
- **LEFT/RIGHT arrows**: Apply torque to the wheel
- **SPACE**: Start/Stop simulation
- **R**: Reset system to initial conditions
- **T**: Toggle pole trajectory trace
- **Mouse drag sliders**: Adjust parameters and initial conditions (disabled while running)
- **Mouse drag wheel center (blue node)**: Directly change φ angle (disabled while running)
- **Mouse drag pole end (red node)**: Directly change θ angle (disabled while running)

**Features:**
- Sliders are frozen while simulation is running to prevent parameter changes during execution
- Direct manipulation of system state by dragging the wheel center or pole end nodes
- Real-time visual feedback with node highlighting during drag operations

**Adjustable Parameters:**
- Rod length (0.5 - 2.0 m)
- Wheel radius (0.1 - 0.5 m)
- Wheel mass (0.5 - 3.0 kg)
- Pole mass (0.05 - 0.5 kg)
- Initial φ (wheel angle)
- Initial θ (pole angle)

## API Reference

### WheelPoleSystem Class

#### Constructor
```python
WheelPoleSystem(rod_length=1.0, wheel_radius=0.2, wheel_mass=1.0, 
                pole_mass=0.1, gravity=9.81, dt=0.02)
```

#### Methods

- `set_initial_state(phi, phi_dot, theta, theta_dot)` - Set initial conditions
- `step(action)` - Apply torque and simulate one time step
  - **Parameters**: `action` (float) - Torque in N⋅m
  - **Returns**: Current state as numpy array [phi, phi_dot, theta, theta_dot]
  
- `get_state()` - Get current state
  - **Returns**: numpy array [phi, phi_dot, theta, theta_dot]
  
- `get_current_step()` - Get current simulation step number
  - **Returns**: int
  
- `get_time()` - Get current simulation time
  - **Returns**: float (seconds)
  
- `get_cartesian_positions()` - Get positions for visualization
  - **Returns**: ((wheel_x, wheel_y), (pole_end_x, pole_end_y))
  
- `reset()` - Reset system to zero state

## State Variables

- **φ (phi)**: Wheel rotation angle (rad)
- **φ_dot**: Wheel angular velocity (rad/s)
- **θ (theta)**: Pole angle from vertical, 0 = upright (rad)
- **θ_dot**: Pole angular velocity (rad/s)

## Physics Notes

The equations of motion are derived from the Lagrangian:

```
L = T - V

where:
T = Kinetic energy = (1/2)*I_w*φ_dot² + (1/2)*m_w*v_wheel² + (1/2)*I_b*θ_dot² + (1/2)*m_b*v_cm²
V = Potential energy = m_b*g*h_cm

with:
- I_w = (1/2)*m_w*r² (moment of inertia of wheel, solid cylinder)
- I_b = (1/3)*m_b*l² (moment of inertia of rod about its end)
- v_wheel: velocity of wheel center due to rotation
- v_cm: velocity of rod's center of mass
```

This results in the coupled equations of motion:

**Equation 1 (θ dynamics):**
```
(I_b + l²·m_b)·θ̈ + l·m_b·r·cos(θ)·φ̈ = g·l·m_b·sin(θ)
```

**Equation 2 (φ dynamics):**
```
[I_w + m_b·r² + m_w·r²]·φ̈ + m_b·r·l·cos(θ)·θ̈ = τ + m_b·r·l·sin(θ)·θ̇²
```

These are coupled second-order ODEs that are integrated using scipy's odeint.

## Use Cases

This system is ideal for:
- Reinforcement learning projects (policy gradient, Q-learning, etc.)
- Control theory experiments (PID, LQR, MPC)
- Understanding nonlinear dynamics
- Testing balance control algorithms

## Example RL Integration

```python
import numpy as np
from wheel_pole_system import WheelPoleSystem

# Create environment
env = WheelPoleSystem(rod_length=1.0, wheel_radius=0.2)

# RL training loop
for episode in range(num_episodes):
    env.set_initial_state(theta=np.random.uniform(-0.2, 0.2))
    
    for step in range(max_steps):
        state = env.get_state()
        
        # Your policy network
        action = policy(state)  # Returns torque
        
        # Step environment
        next_state = env.step(action)
        
        # Calculate reward (e.g., pole upright)
        reward = np.cos(next_state[2])  # Reward for keeping pole vertical
        
        # Store transition, update policy, etc.
        # ...
```

## License

Free to use for educational and research purposes.