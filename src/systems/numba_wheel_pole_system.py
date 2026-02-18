import numpy as np
from numba import jit, prange
import time


@jit(nopython=True, fastmath=True)
def dynamics_kernel(phi, phi_dot, theta, theta_dot, torque, 
                   a11_const, a22_const, mlr, glm):
    """
    Compute accelerations for a single state (JIT-compiled).
    Runs at C speed with Numba.
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Mass matrix elements
    a11 = a11_const
    a12 = mlr * cos_theta
    a21 = a12
    a22 = a22_const
    
    # RHS
    b1 = glm * sin_theta
    b2 = torque + mlr * sin_theta * theta_dot**2
    
    # Solve 2x2 system
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-10:
        det = 1e-10
    
    theta_ddot = (a22 * b1 - a12 * b2) / det
    phi_ddot = (a11 * b2 - a21 * b1) / det
    
    return phi_dot, phi_ddot, theta_dot, theta_ddot


@jit(nopython=True, parallel=True, fastmath=True)
def step_vectorized_numba(states, actions, dt, a11_const, a22_const, mlr, glm):
    """
    Vectorized step function using Numba parallel execution.
    
    Parameters:
    -----------
    states : (n_envs, 4) array
    actions : (n_envs,) array
    dt : float
    
    Returns:
    --------
    new_states : (n_envs, 4) array
    """
    n_envs = states.shape[0]
    new_states = np.empty_like(states)
    
    # Parallel loop over environments (true parallelism with Numba)
    for i in prange(n_envs):
        phi = states[i, 0]
        phi_dot = states[i, 1]
        theta = states[i, 2]
        theta_dot = states[i, 3]
        torque = actions[i]
        
        # Compute derivatives
        d0, d1, d2, d3 = dynamics_kernel(
            phi, phi_dot, theta, theta_dot, torque,
            a11_const, a22_const, mlr, glm
        )
        
        # Forward Euler
        new_states[i, 0] = phi + dt * d0
        new_states[i, 1] = phi_dot + dt * d1
        new_states[i, 2] = theta + dt * d2
        new_states[i, 3] = theta_dot + dt * d3
    
    return new_states


class NumbaWheelPoleSystem:
    """
    Ultra-fast vectorized wheel-pole system using Numba JIT compilation.
    Achieves near-C performance with automatic parallelization.
    """
    
    def __init__(self, n_envs, rod_length=1.0, wheel_radius=0.2,
                 wheel_mass=1.0, pole_mass=0.1, gravity=9.81, dt=0.02):
        self.n_envs = n_envs
        self.L = rod_length
        self.R = wheel_radius
        self.m_wheel = wheel_mass
        self.m_pole = pole_mass
        self.g = gravity
        self.dt = dt
        
        # Moments of inertia
        self.I_wheel = 0.5 * self.m_wheel * self.R ** 2
        self.I_rod = (1.0 / 3.0) * self.m_pole * self.L ** 2
        
        # Pre-compute constants
        m_b = self.m_pole
        m_w = self.m_wheel
        l = self.L
        r = self.R
        I_b = self.I_rod
        I_w = self.I_wheel
        
        self.a11_const = I_b + l**2 * m_b
        self.a22_const = I_w + m_b * r**2 + m_w * r**2
        self.mlr = m_b * l * r
        self.glm = self.g * l * m_b
        
        # State
        self.states = np.zeros((n_envs, 4), dtype=np.float32)
        self.time = np.zeros(n_envs, dtype=np.float32)
        self.step_count = np.zeros(n_envs, dtype=np.int32)
        
        # JIT warmup (compile functions on first call)
        self._warmup()
    
    def _warmup(self):
        """Warm up JIT compilation with a dummy call."""
        dummy_states = np.zeros((2, 4), dtype=np.float32)
        dummy_actions = np.zeros(2, dtype=np.float32)
        _ = step_vectorized_numba(
            dummy_states, dummy_actions, self.dt,
            self.a11_const, self.a22_const, self.mlr, self.glm
        )
    
    def reset(self, env_ids=None, phi=None, phi_dot=None, theta=None, theta_dot=None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = np.arange(self.n_envs)
        
        env_ids = np.asarray(env_ids)
        n_reset = len(env_ids)
        
        if phi is None:
            phi = np.zeros(n_reset, dtype=np.float32)
        elif np.isscalar(phi):
            phi = np.full(n_reset, phi, dtype=np.float32)
            
        if phi_dot is None:
            phi_dot = np.zeros(n_reset, dtype=np.float32)
        elif np.isscalar(phi_dot):
            phi_dot = np.full(n_reset, phi_dot, dtype=np.float32)
            
        if theta is None:
            theta = np.zeros(n_reset, dtype=np.float32)
        elif np.isscalar(theta):
            theta = np.full(n_reset, theta, dtype=np.float32)
            
        if theta_dot is None:
            theta_dot = np.zeros(n_reset, dtype=np.float32)
        elif np.isscalar(theta_dot):
            theta_dot = np.full(n_reset, theta_dot, dtype=np.float32)
        
        self.states[env_ids, 0] = phi
        self.states[env_ids, 1] = phi_dot
        self.states[env_ids, 2] = theta
        self.states[env_ids, 3] = theta_dot
        
        self.time[env_ids] = 0.0
        self.step_count[env_ids] = 0
        
        return self.states[env_ids].copy()
    
    def step(self, actions):
        """
        Step all environments using Numba-accelerated parallel computation.
        
        Parameters:
        -----------
        actions : (n_envs,) array of torques
        
        Returns:
        --------
        states : (n_envs, 4) array of new states
        """
        actions = np.asarray(actions, dtype=np.float32)
        
        # Call JIT-compiled parallel kernel
        self.states = step_vectorized_numba(
            self.states, actions, self.dt,
            self.a11_const, self.a22_const, self.mlr, self.glm
        )
        
        self.time += self.dt
        self.step_count += 1
        
        return self.states.copy()
    
    def get_states(self):
        """Return current states."""
        return self.states.copy()


# ===========================================================================
# Performance comparison
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PERFORMANCE COMPARISON: Numba vs Vectorized vs Sequential")
    print("=" * 60)
    print()
    
    n_envs = 512
    n_steps = 2000
    total_steps = n_envs * n_steps
    
    print(f"Configuration: {n_envs} environments × {n_steps} steps")
    print(f"Total simulated steps: {total_steps:,}")
    print()
    
    # ========== Numba version (parallel) ==========
    print("Testing Numba-accelerated version...")
    numba_env = NumbaWheelPoleSystem(n_envs=n_envs, dt=0.02)
    numba_env.reset(theta=0.1)
    actions = np.zeros(n_envs, dtype=np.float32)
    
    start = time.time()
    for _ in range(n_steps):
        numba_env.step(actions)
    elapsed_numba = time.time() - start
    
    steps_per_sec_numba = total_steps / elapsed_numba
    
    print(f"  Time: {elapsed_numba:.4f}s")
    print(f"  → {steps_per_sec_numba:,.0f} env-steps/second")
    print(f"  → {steps_per_sec_numba / n_envs:,.0f} steps/sec per env")
    print()
    
    # ========== Pure NumPy vectorized ==========
    print("Testing pure NumPy vectorized version...")
    try:
        from vectorized_wheel_pole import VectorizedWheelPoleSystem
        
        vec_env = VectorizedWheelPoleSystem(n_envs=n_envs, dt=0.02)
        vec_env.reset(theta=0.1)
        
        start = time.time()
        for _ in range(n_steps):
            vec_env.step(actions)
        elapsed_vec = time.time() - start
        
        steps_per_sec_vec = total_steps / elapsed_vec
        
        print(f"  Time: {elapsed_vec:.4f}s")
        print(f"  → {steps_per_sec_vec:,.0f} env-steps/second")
        print(f"  → {steps_per_sec_vec / n_envs:,.0f} steps/sec per env")
        print()
        
        speedup_vs_vec = elapsed_vec / elapsed_numba
        print(f"Numba speedup vs NumPy vectorized: {speedup_vs_vec:.1f}x")
        print()
        
    except ImportError:
        print("  (vectorized_wheel_pole.py not found, skipping)")
        print()
    
    # ========== Sequential baseline ==========
    print("Testing sequential version (single env)...")
    try:
        from src.systems.wheel_pole_system import WheelPoleSystem
        
        single_env = WheelPoleSystem(rod_length=1.0, wheel_radius=0.2,
                                      wheel_mass=1.0, pole_mass=0.1, dt=0.02)
        single_env.set_initial_state(theta=0.1)
        
        start = time.time()
        for _ in range(n_steps):
            single_env.step(0.0)
        elapsed_single = time.time() - start
        
        steps_per_sec_single = n_steps / elapsed_single
        
        print(f"  Time: {elapsed_single:.4f}s for {n_steps} steps")
        print(f"  → {steps_per_sec_single:,.0f} steps/second")
        print()
        
        speedup_vs_single = (elapsed_single * n_envs) / elapsed_numba
        
        print(f"Numba speedup vs sequential: {speedup_vs_single:.1f}x")
        print()
        
    except ImportError:
        print("  (wheel_pole_system.py not found, skipping)")
        print()
    
    print("=" * 60)
    print("SUMMARY:")
    print(f"  Numba achieves {steps_per_sec_numba / 1e6:.1f} MILLION env-steps/second!")
    print(f"  Perfect for large-scale RL training with parallel envs")
    print("=" * 60)