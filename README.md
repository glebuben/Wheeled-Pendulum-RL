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
conda activate willed_pendulum

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

```

### 2. Interactive Simulation (wheel_pole_simulation.py)

Run the interactive simulation:
```bash
python -m src.wheel_pole_simulation
```

Run training:
```bash
python -m src.reinforce_vectorized -c CONFIG_PATH
```

Visualize policy:
```bash
python -m src.policy_visualizer CKPT_PATH
```

Plot performance:
```bash
python -m plots.make_plot --checkpoint CKPT_PATH
```

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

## State Variables

| Variable | Description | Units |
|----------|------------|-------|
| φ | Wheel rotation angle | rad |
| φ̇ | Wheel angular velocity | rad/s |
| θ | Pole angle from vertical (0 = upright) | rad |
| θ̇ | Pole angular velocity | rad/s |


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

