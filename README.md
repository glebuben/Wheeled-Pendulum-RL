# Wheel-Pole Balancing System

A reinforcement learning environment featuring a wheel-pole balancing system - a variation of the classic cartpole where linear cart motion is replaced by wheel rotation.

## System Description

The system consists of:
- A **wheel** that can rotate (generalized coordinate: φ)
- A **pole** attached to the wheel that can swing (generalized coordinate: θ)
- The control input is a **torque** applied to the wheel

The dynamics are derived from Lagrangian mechanics,  with the key difference from CartPole being the horizontal displacement  
`x = R·φ`, where R is the wheel radius.



## Project Structure

```text
Wheeled-Pendulum-RL/
│
├── src/
│   ├── systems/
│   │   ├── wheel_pole_system.py        # Core dynamics model
│   │   └── numba_wheel_pole_system.py  # Accelerated (Numba) dynamics
│   │
│   ├── policy_network.py               # Neural network policy
│   ├── reinforce_vectorized.py         # Vectorized REINFORCE training
│   ├── policy_visualizer.py            # Policy behaviour visualization
│   └── wheel_pole_simulation.py        # Interactive simulation (pygame)
│
├── checkpoints/                        # Saved trained models
│   └── best_*.pt
│
├── plots/                              # Training curves & debug plots
│   └── tmp.py
│
├── environment.yml                     # Conda environment
└── README.md                           # Project documentation

```

## Installation

```bash
git clone https://github.com/glebuben/Wheeled-Pendulum-RL.git
cd Wheeled-Pendulum-RL

conda env create -f environment.yml
conda activate wheeled_pendulum

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
python src/wheel_pole_simulation.py
```

## Training

To train the policy using the vectorized REINFORCE algorithm:
```bash
python src/reinforce_vectorized.py
```


After training:

- trained models are saved to `checkpoints/`
- training curves are saved to `plots/`


## Evaluation

To visualize and evaluate a trained policy:
```bash
python src/policy_visualizer.py
```


This script loads the trained model and runs a rollout of the balancing policy.


### Controls

- **← / →** — apply torque to the wheel  
- **Space** — start / stop simulation  
- **R** — reset to initial conditions  
- **T** — toggle pole trajectory trace  

**Mouse interactions**

- Drag sliders — adjust parameters *(disabled while running)*  
- Drag wheel center *(blue node)* — change φ angle  
- Drag pole end *(red node)* — change θ angle  


### Features

- Parameters are locked during simulation to ensure stability  
- Direct manipulation of system state via draggable nodes  
- Real-time visual feedback with node highlighting


### Adjustable Parameters

**Physical parameters**

- Rod length: **0.5 – 2.0 m**
- Wheel radius: **0.1 – 0.5 m**
- Wheel mass: **0.5 – 3.0 kg**
- Pole mass: **0.05 – 0.5 kg**

**Initial conditions**

- Initial φ — wheel angle  
- Initial θ — pole angle


## API Reference

### WheelPoleSystem Class

#### Constructor
```python
WheelPoleSystem(rod_length=1.0, wheel_radius=0.2, wheel_mass=1.0, 
                pole_mass=0.1, gravity=9.81, dt=0.02)
```

#### Methods

- `set_initial_state(phi, phi_dot, theta, theta_dot)` - Set initial conditions
- `step(action, reward_func=None)` - Apply torque and simulate one time step
  - **Parameters**: 
    - `action` (float) - Torque in N⋅m
    - `reward_func` (callable, optional) - Function with signature `reward_func(prev_state, action, new_state) -> float`. If no reward function is provided, a default balancing reward is used.
  - **Returns**: Tuple of (state, reward)
    - `state` (np.ndarray) - Current state [phi, phi_dot, theta, theta_dot]
    - `reward` (float) - Computed reward value
  
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

| Variable | Description | Units |
|----------|------------|-------|
| φ | Wheel rotation angle | rad |
| φ̇ | Wheel angular velocity | rad/s |
| θ | Pole angle from vertical (0 = upright) | rad |
| θ̇ | Pole angular velocity | rad/s |

## Episode Termination

An episode terminates if:

- |θ| > 60° (pole falls)
- numerical instability occurs

## Reward Function

The default reward encourages upright balance:

r = cos(θ) − 0.001 τ²

where:

- cos(θ) rewards upright position
- torque penalty encourages energy efficiency

### Terminal penalty

- pole falls → reward = −100


## Episode Truncation

An episode is truncated if:

- max_steps reached (default: 1000)
- time limit exceeded


## Physics Model

The system dynamics are derived using the Lagrangian formulation.

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\color{white}\Large%20L=T-V" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\color{white}\Large%20T=\frac{1}{2}I_w\dot{\phi}^2+\frac{1}{2}m_wv_{wheel}^2+\frac{1}{2}I_b\dot{\theta}^2+\frac{1}{2}m_bv_{cm}^2" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\color{white}\Large%20V=m_bg\,h_{cm}" />
</p>

**Parameters**

- **I_w = ½ m_w r²** — wheel inertia (solid cylinder)  
- **I_b = ⅓ m_b l²** — rod inertia about its end  
- **v_wheel** — velocity of wheel center  
- **v_cm** — velocity of rod center of mass


---

These expressions lead to the coupled equations of motion:

### θ dynamics

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\color{white}\Large(I_b+m_bl^2)\ddot{\theta}+m_blr\cos(\theta)\ddot{\phi}=m_bgl\sin(\theta)" />
</p>

### φ dynamics

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\color{white}\Large(I_w+m_br^2+m_wr^2)\ddot{\phi}+m_brl\cos(\theta)\ddot{\theta}=\tau+m_brl\sin(\theta)\dot{\theta}^2" />
</p>


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
reward_func  = lambda prev_state, action, new_state: np.cos(new_state[2])

# RL training loop
for episode in range(num_episodes):
    env.set_initial_state(theta=np.random.uniform(-0.2, 0.2))
    
    for step in range(max_steps):
        state = env.get_state()
        
        # Your policy network
        action = policy(state)  # Returns torque
        
        # Step environment
        next_state, reward = env.step(action, reward_func)

        # Store transition, update policy, etc.
        # ...
```

## License

Free to use for educational and research purposes.
