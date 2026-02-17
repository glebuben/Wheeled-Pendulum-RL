import numpy as np
from scipy.linalg import solve_continuous_are


class WheelPoleController:
    """
    Nonlinear controller for the wheel-pole balancing system.

    Combines two control strategies:
      1. Energy-based swing-up (Lyapunov / gradient descent):
         drives the pendulum energy toward the upright target value.
      2. LQR stabilisation: takes over when BOTH |θ| < theta_threshold
         AND |θ̇| < vel_threshold.

    ── Energy definition ───────────────────────────────────────────────────

    From the Lagrangian the pendulum subsystem energy is:

        E_pend = (1/2)·(I_b + m_b·l²)·θ̇²
               + m_b·l·r·cosθ·φ̇·θ̇
               + m_b·g·l·(1 + cosθ)

    Reference levels  (θ̇ = 0, φ̇ = 0):
        E (hanging,  θ = ±π) = 0           ← zero reference
        E*(upright,  θ =  0) = 2·m_b·g·l   ← target

    ── Swing-up control law ────────────────────────────────────────────────

    Lyapunov candidate:  V = (E − E*)² / 2
    Time derivative:     V̇ = (E − E*) · Ė

    The torque τ affects Ė through both channels of the coupled EOM:

        Ė|_τ = [∂E/∂θ̇ · (∂θ̈/∂τ)  +  ∂E/∂φ̇ · (∂φ̈/∂τ)] · τ
             = σ · τ

    where the full energy-rate sensitivity to τ is:

        σ = (∂E/∂θ̇) · c_τ→θ̈  +  (∂E/∂φ̇) · c_τ→φ̈

        ∂E/∂θ̇ = (I_b + m_b·l²)·θ̇ + m_b·l·r·cosθ·φ̇   ← dominant channel
        ∂E/∂φ̇ = m_b·l·r·cosθ·θ̇
        c_τ→θ̈ = −M₁₂/det        (always < 0)
        c_τ→φ̈ = M₁₁/det         (always > 0)

    Choosing τ_pump = kE · (E − E*) · σ yields:
        V̇ = kE · σ² · (E − E*)² · ... → drives V toward 0.

    Note the POSITIVE sign: since σ < 0 for the standard parameters,
    a positive energy error (E > E*) yields τ < 0, and a negative error
    (E < E*) yields τ > 0 — both correct for energy regulation.

    A damping term −kd·φ̇ prevents unbounded wheel speed accumulation.

    ── LQR linearisation ───────────────────────────────────────────────────

    State: x = [phi, phi_dot, theta, theta_dot].
    Around upright (sinθ ≈ θ, cosθ ≈ 1, θ̇² ≈ 0):

        θ̈ = (M₂₂·m_b·g·l / det)·θ + c_τ→θ̈ · τ
        φ̈ = −(M₁₂·m_b·g·l / det)·θ + c_τ→φ̈ · τ
    """

    def __init__(
        self,
        rod_length: float,
        wheel_radius: float,
        wheel_mass: float,
        pole_mass: float,
        gravity: float = 9.81,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        kE: float = 5.0,
        kd: float = 1.0,
        tau_max: float = 1.0,
        theta_threshold: float = 1.0,
        vel_threshold: float = 1.0,
    ):
        """
        Parameters
        ----------
        rod_length : float
            Length of the rod l (m).
        wheel_radius : float
            Radius of the wheel r (m).
        wheel_mass : float
            Mass of the wheel m_w (kg).
        pole_mass : float
            Mass of the rod m_b (kg).
        gravity : float
            Gravitational acceleration g (m/s²).
        Q : (4, 4) array, optional
            LQR state-cost matrix.
            Default: diag(1, 1, 100, 10).
        R : (1, 1) array, optional
            LQR control-cost matrix. Default: [[0.1]].
        kE : float
            Gain on the energy-error·sensitivity product in swing-up.
            Default 5.0.
        kd : float
            Gain on the wheel-velocity damping term during swing-up.
            Prevents unbounded φ̇ accumulation. Default 1.0.
        tau_max : float
            Hard saturation limit on the output torque (N·m). Default 10.
        theta_threshold : float
            |θ_wrapped| (rad) below which LQR *may* take over.
            Default 0.30 rad ≈ 17°.
        vel_threshold : float
            |θ̇| (rad/s) below which LQR *may* take over.
            LQR activates only when BOTH conditions hold simultaneously.
            Default 1.0 rad/s.
        """
        # ── System parameters ──────────────────────────────────────────
        self.l   = rod_length
        self.r   = wheel_radius
        self.m_w = wheel_mass
        self.m_b = pole_mass
        self.g   = gravity

        # Moments of inertia
        self.I_b = (1.0 / 3.0) * self.m_b * self.l ** 2   # rod about its end
        self.I_w = 0.5 * self.m_w * self.r ** 2            # wheel (solid cylinder)

        # ── Controller parameters ──────────────────────────────────────
        self.kE              = kE
        self.kd              = kd
        self.tau_max         = tau_max
        self.theta_threshold = theta_threshold
        self.vel_threshold   = vel_threshold

        # ── LQR design ─────────────────────────────────────────────────
        self.A, self.B = self._linearise()

        if Q is None:
            Q = np.diag([1.0, 1.0, 100.0, 10.0])
        if R is None:
            R = np.array([[0.1]])

        self.Q = Q
        self.R = R
        self.K = self._compute_lqr_gain(self.A, self.B, Q, R)

        # ── Precomputed constants ──────────────────────────────────────
        M11 = self.I_b + self.m_b * self.l ** 2
        M12 = self.m_b * self.l * self.r
        M22 = self.I_w + (self.m_b + self.m_w) * self.r ** 2
        det = M11 * M22 - M12 ** 2

        self._M11     = M11            # I_b + m_b·l²
        self._c_ta2th = -M12 / det     # ∂θ̈/∂τ  (< 0)
        self._c_ta2ph =  M11 / det     # ∂φ̈/∂τ  (> 0)

        # E* = m_b·g·l·(1 + cos 0) = 2·m_b·g·l
        self.E_target = 2.0 * self.m_b * self.g * self.l

    # ──────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ──────────────────────────────────────────────────────────────────────

    def _linearise(self) -> tuple:
        """
        Build the linearised (A, B) matrices around the upright equilibrium.
        State: [phi, phi_dot, theta, theta_dot]
        """
        l, r, m_b, g = self.l, self.r, self.m_b, self.g

        M11   = self.I_b + l ** 2 * m_b
        M12_0 = l * m_b * r
        M22   = self.I_w + (m_b + self.m_w) * r ** 2
        det   = M11 * M22 - M12_0 ** 2

        c_th2th = ( M22 * g * l * m_b) / det
        c_ta2th = -M12_0 / det
        c_th2ph = -(M12_0 * g * l * m_b) / det
        c_ta2ph =  M11 / det

        A = np.array([
            [0, 1, 0,        0],
            [0, 0, c_th2ph,  0],
            [0, 0, 0,        1],
            [0, 0, c_th2th,  0],
        ])
        B = np.array([[0], [c_ta2ph], [0], [c_ta2th]])
        return A, B

    @staticmethod
    def _compute_lqr_gain(A, B, Q, R) -> np.ndarray:
        """Solve the continuous-time ARE and return K."""
        P = solve_continuous_are(A, B, Q, R)
        return np.linalg.inv(R) @ B.T @ P

    # ──────────────────────────────────────────────────────────────────────
    # Pendulum energy  (user's formula, exact)
    # ──────────────────────────────────────────────────────────────────────

    def pendulum_energy(self, theta: float, theta_dot: float,
                        phi_dot: float) -> float:
        """
        Pendulum subsystem energy (Lagrangian-derived, coupling included).

            E = (1/2)·(I_b + m_b·l²)·θ̇²
              + m_b·l·r·cosθ·φ̇·θ̇
              + m_b·g·l·(1 + cosθ)

        Parameters
        ----------
        theta     : float – pole angle (rad), wrapped to [−π, π]
        theta_dot : float – pole angular velocity (rad/s)
        phi_dot   : float – wheel angular velocity (rad/s)
        """
        m   = self.m_b
        l   = self.l
        r   = self.r
        g   = self.g
        I_b = self.I_b

        kinetic   = 0.5 * (I_b + m * l**2) * theta_dot**2 \
                    + m * l * r * np.cos(theta) * phi_dot * theta_dot
        potential = m * g * l * (1.0 + np.cos(theta))
        return kinetic + potential

    # ──────────────────────────────────────────────────────────────────────
    # Swing-up (Lyapunov energy-gradient law, corrected)
    # ──────────────────────────────────────────────────────────────────────

    def swing_up_lyapunov(self, theta: float, theta_dot: float,
                          phi_dot: float) -> float:
        """
        Lyapunov-based energy swing-up.

        Minimises V = (E − E*)² / 2 by following the gradient of Ė w.r.t. τ:

            σ = ∂E/∂θ̇ · c_τ→θ̈ + ∂E/∂φ̇ · c_τ→φ̈

            ∂E/∂θ̇ = (I_b + m_b·l²)·θ̇ + m_b·l·r·cosθ·φ̇
            ∂E/∂φ̇ = m_b·l·r·cosθ·θ̇

        Control law:
            τ = kE · (E − E*) · σ  −  kd · φ̇

        The positive sign in front of kE is correct for this system:
        σ < 0 for the default parameters, so:
          - when E < E* (too little energy): τ > 0 — injects energy
          - when E > E* (too much energy): τ < 0 — removes energy

        Parameters
        ----------
        theta     : float – pole angle (rad), wrapped to [−π, π]
        theta_dot : float – pole angular velocity (rad/s)
        phi_dot   : float – wheel angular velocity (rad/s)
        """
        m = self.m_b
        l = self.l
        r = self.r

        E = self.pendulum_energy(theta, theta_dot, phi_dot)

        # Partial derivatives of E w.r.t. generalised velocities
        dE_dtheta_dot = self._M11 * theta_dot + m * l * r * np.cos(theta) * phi_dot
        dE_dphi_dot   = m * l * r * np.cos(theta) * theta_dot

        # Full energy-rate sensitivity to torque (via both EOM channels)
        sigma = dE_dtheta_dot * self._c_ta2th + dE_dphi_dot * self._c_ta2ph

        tau = (
              self.kE * (E - self.E_target) * sigma   # energy-shaping term
            - self.kd * phi_dot                        # wheel damping term
        )
        return float(np.clip(tau, -self.tau_max, self.tau_max))

    # ──────────────────────────────────────────────────────────────────────
    # LQR stabilisation
    # ──────────────────────────────────────────────────────────────────────

    def _lqr(self, state: np.ndarray) -> float:
        """Linear state-feedback: τ = −K·x, saturated to ±tau_max."""
        tau = float((-self.K @ state)[0])
        return float(np.clip(tau, -self.tau_max, self.tau_max))

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def control(self, state: np.ndarray) -> float:
        """
        Compute the control torque for the current system state.

        Switching logic:
            if |θ_wrapped| < theta_threshold AND |θ̇| < vel_threshold:
                use LQR stabilisation
            else:
                use energy-based swing-up (Lyapunov gradient law)

        Both branches saturate the output to ±tau_max.

        Parameters
        ----------
        state : array-like, shape (4,)
            [phi, phi_dot, theta, theta_dot]

        Returns
        -------
        tau : float  – torque to apply to the wheel (N·m).
        """
        phi, phi_dot, theta, theta_dot = state

        # Wrap theta to [−π, π]
        theta_w = (theta + np.pi) % (2.0 * np.pi) - np.pi

        if abs(theta_w) < self.theta_threshold and abs(theta_dot) < self.vel_threshold:
            lqr_state = np.array([phi, phi_dot, theta_w, theta_dot])
            return self._lqr(lqr_state)
        else:
            return self.swing_up_lyapunov(theta_w, theta_dot, phi_dot)

    # ──────────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        print("=" * 55)
        print("WheelPoleController summary")
        print("=" * 55)
        print(f"  Rod length        l     = {self.l} m")
        print(f"  Wheel radius      r     = {self.r} m")
        print(f"  Wheel mass        m_w   = {self.m_w} kg")
        print(f"  Rod mass          m_b   = {self.m_b} kg")
        print(f"  Gravity           g     = {self.g} m/s²")
        print(f"  I_b (rod)               = {self.I_b:.5f} kg·m²")
        print(f"  I_w (wheel)             = {self.I_w:.5f} kg·m²")
        print(f"  E* (target energy)      = {self.E_target:.4f} J")
        print(f"  c_τ→θ̈  (∂θ̈/∂τ)       = {self._c_ta2th:.5f}")
        print(f"  c_τ→φ̈  (∂φ̈/∂τ)       = {self._c_ta2ph:.5f}")
        print(f"  Swing-up gain  kE       = {self.kE}")
        print(f"  Wheel damping  kd       = {self.kd}")
        print(f"  Torque saturation       = ±{self.tau_max} N·m")
        print(f"  Angle threshold         = {self.theta_threshold:.3f} rad "
              f"({np.degrees(self.theta_threshold):.1f}°)")
        print(f"  Velocity threshold      = {self.vel_threshold:.3f} rad/s")
        print(f"  LQR gain K              = {self.K}")
        print("=" * 55)


# ===========================================================================
# Smoke-test
# ===========================================================================
if __name__ == "__main__":
    from wheel_pole_system import WheelPoleSystem
    import matplotlib.pyplot as plt

    params = dict(rod_length=1.0, wheel_radius=0.2,
                  wheel_mass=1.0, pole_mass=0.1)

    system = WheelPoleSystem(**params, dt=0.02)
    ctrl   = WheelPoleController(**params)
    ctrl.print_summary()

    # Full swing-up from near-hanging with small velocity nudge
    system.set_initial_state(theta=np.pi - 0.05, theta_dot=0.3)

    states, torques, modes = [system.get_state()], [], []

    for _ in range(1500):
        state = system.get_state()
        phi, phi_dot, theta, theta_dot = state
        theta_w = (theta + np.pi) % (2.0 * np.pi) - np.pi
        in_lqr  = (abs(theta_w) < ctrl.theta_threshold and
                   abs(theta_dot) < ctrl.vel_threshold)
        modes.append(1 if in_lqr else 0)
        tau = ctrl.control(state)
        torques.append(tau)
        system.step(tau)
        states.append(system.get_state())

    states    = np.array(states)
    torques   = np.array(torques)
    modes     = np.array(modes)
    t         = np.arange(len(states)) * system.dt
    theta_deg = np.degrees((states[:, 2] + np.pi) % (2.0 * np.pi) - np.pi)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t, theta_deg, color="tab:blue")
    axes[0].axhline(0, color="k", lw=0.7, ls="--")
    axes[0].fill_between(t[:-1], -200, 200, where=modes.astype(bool),
                         alpha=0.15, color="green", label="LQR active")
    axes[0].set_ylim(-200, 200)
    axes[0].set_ylabel("θ (deg)")
    axes[0].set_title("Wheel-Pole: Lyapunov swing-up + LQR")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    axes[1].plot(t, states[:, 3], color="tab:orange")
    axes[1].axhline(0, color="k", lw=0.7, ls="--")
    axes[1].set_ylabel("θ̇ (rad/s)")
    axes[1].grid(True)

    axes[2].plot(t, states[:, 1], color="tab:red")
    axes[2].set_ylabel("φ̇ (rad/s)")
    axes[2].grid(True)

    axes[3].step(t[:-1], torques, where="post", color="tab:purple")
    axes[3].set_ylabel("τ (N·m)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig("controller_test.png", dpi=120)
    plt.show()
    print("Done – plot saved to controller_test.png")