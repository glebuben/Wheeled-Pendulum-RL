import pygame
import numpy as np
import sys
from wheel_pole_system import WheelPoleSystem
from controller import WheelPoleController


class WheelPoleSimulation:
    """
    Interactive simulation of the wheel-pole system using pygame.
    Allows adjustment of system parameters and initial conditions.
    Supports manual torque input and automatic nonlinear controller.
    """

    def __init__(self, width=1200, height=700):
        pygame.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Wheel-Pole Balancing System")

        self.clock = pygame.time.Clock()
        self.fps = 60

        # Colors
        self.WHITE      = (255, 255, 255)
        self.BLACK      = (0,   0,   0)
        self.RED        = (220, 50,  50)
        self.BLUE       = (50,  100, 220)
        self.GREEN      = (50,  200, 50)
        self.ORANGE     = (230, 140, 30)
        self.GRAY       = (200, 200, 200)
        self.DARK_GRAY  = (100, 100, 100)
        self.LIGHT_BLUE = (150, 200, 255)

        # Simulation viewport
        self.sim_x      = 50
        self.sim_y      = 50
        self.sim_width  = 700
        self.sim_height = 600

        # Control panel
        self.panel_x     = 770
        self.panel_y     = 50
        self.panel_width = 400

        # System parameters (defaults)
        self.params = {
            'rod_length':   1.0,
            'wheel_radius': 0.2,
            'wheel_mass':   1.0,
            'pole_mass':    0.1,
        }

        # Initial conditions
        self.initial_conditions = {
            'phi':   0.0,
            'theta': 0.1,
        }

        # Scale: pixels per metre
        self.scale = 200

        # Runtime state
        self.running        = False
        self.use_controller = False   # C key toggles this
        self.torque         = 0.0
        self.max_torque     = 1.0

        # Node dragging
        self.dragging_node = None   # 'wheel' | 'pole_end'
        self.node_radius   = 12

        # Trajectory: persistent surface so we never lose history
        self.traj_surface = pygame.Surface(
            (self.sim_width, self.sim_height), pygame.SRCALPHA
        )
        self.traj_surface.fill((0, 0, 0, 0))
        self.prev_pole_screen = None   # last drawn pole-end position on surface

        # Fonts
        self.font_large  = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small  = pygame.font.Font(None, 20)

        # Build system + controller
        self.system     = None
        self.controller = None
        self.create_system()

        # Sliders
        self.sliders       = self.create_sliders()
        self.active_slider = None

    # ------------------------------------------------------------------ #
    # System / controller creation                                         #
    # ------------------------------------------------------------------ #

    def create_system(self):
        """Instantiate a fresh WheelPoleSystem and matching controller."""
        self.system = WheelPoleSystem(
            rod_length=self.params['rod_length'],
            wheel_radius=self.params['wheel_radius'],
            wheel_mass=self.params['wheel_mass'],
            pole_mass=self.params['pole_mass'],
        )
        self.system.set_initial_state(
            phi=self.initial_conditions['phi'],
            theta=self.initial_conditions['theta'],
        )
        self.controller = WheelPoleController(
            rod_length=self.params['rod_length'],
            wheel_radius=self.params['wheel_radius'],
            wheel_mass=self.params['wheel_mass'],
            pole_mass=self.params['pole_mass'],
        )
        self._clear_trajectory()

    def _clear_trajectory(self):
        self.traj_surface.fill((0, 0, 0, 0))
        self.prev_pole_screen = None

    # ------------------------------------------------------------------ #
    # Coordinate helpers                                                   #
    # ------------------------------------------------------------------ #

    def _origin(self):
        """Screen-space origin: centre-bottom of the simulation viewport."""
        cx = self.sim_x + self.sim_width  // 2
        cy = self.sim_y + self.sim_height - 100
        return cx, cy

    def get_screen_positions(self):
        """Return (wheel_screen, pole_screen) in absolute screen coords."""
        wheel_world, pole_world = self.system.get_cartesian_positions()
        cx, cy = self._origin()
        wx = cx + int(wheel_world[0] * self.scale)
        wy = cy - int(wheel_world[1] * self.scale)
        px = cx + int(pole_world[0]  * self.scale)
        py = cy - int(pole_world[1]  * self.scale)
        return (wx, wy), (px, py)

    def set_state_from_screen_positions(self, wheel_screen, pole_screen):
        """Update state from dragged node, keeping the other angle fixed."""
        cx, cy = self._origin()

        wheel_x = (wheel_screen[0] - cx) / self.scale
        wheel_y = (cy - wheel_screen[1]) / self.scale

        pole_x  = (pole_screen[0]  - cx) / self.scale
        pole_y  = (cy - pole_screen[1])  / self.scale

        new_phi   = wheel_x / self.system.R
        dx, dy    = pole_x - wheel_x, pole_y - wheel_y
        new_theta = np.arctan2(dx, dy)

        if self.dragging_node == 'wheel':
            self.system.state[0] = new_phi
        elif self.dragging_node == 'pole_end':
            self.system.state[2] = new_theta

    # ------------------------------------------------------------------ #
    # Sliders                                                              #
    # ------------------------------------------------------------------ #

    def create_sliders(self):
        sliders  = []
        y_offset = self.panel_y + 50

        param_defs = [
            ('Rod Length (m)',   'rod_length',   0.5,    2.0,  self.params['rod_length']),
            ('Wheel Radius (m)', 'wheel_radius', 0.1,    0.5,  self.params['wheel_radius']),
            ('Wheel Mass (kg)',  'wheel_mass',   0.5,    3.0,  self.params['wheel_mass']),
            ('Pole Mass (kg)',   'pole_mass',    0.05,   0.5,  self.params['pole_mass']),
        ]
        for name, key, lo, hi, val in param_defs:
            sliders.append({'name': name, 'key': key, 'type': 'param',
                            'min': lo, 'max': hi, 'value': val,
                            'rect': pygame.Rect(self.panel_x, y_offset, 300, 20)})
            y_offset += 60

        y_offset += 20  # separator

        ic_defs = [
            ('Initial φ (rad)',  'phi',   -np.pi, np.pi, self.initial_conditions['phi']),
            ('Initial θ (rad)',  'theta', -np.pi, np.pi, self.initial_conditions['theta']),
        ]
        for name, key, lo, hi, val in ic_defs:
            sliders.append({'name': name, 'key': key, 'type': 'initial',
                            'min': lo, 'max': hi, 'value': val,
                            'rect': pygame.Rect(self.panel_x, y_offset, 300, 20)})
            y_offset += 60

        return sliders

    def draw_slider(self, slider):
        disabled = self.running
        lc = self.DARK_GRAY if disabled else self.BLACK
        hc = self.DARK_GRAY if disabled else self.BLUE
        tc = self.DARK_GRAY if disabled else self.GRAY

        self.screen.blit(
            self.font_small.render(slider['name'], True, lc),
            (slider['rect'].x, slider['rect'].y - 25))
        self.screen.blit(
            self.font_small.render(f"{slider['value']:.3f}", True, hc),
            (slider['rect'].x + 310, slider['rect'].y - 25))

        pygame.draw.rect(self.screen, tc, slider['rect'], border_radius=10)

        norm  = (slider['value'] - slider['min']) / (slider['max'] - slider['min'])
        hx    = slider['rect'].x + int(norm * slider['rect'].width)
        pygame.draw.rect(self.screen, hc,
                         pygame.Rect(hx - 8, slider['rect'].y - 5, 16, 30),
                         border_radius=8)

    def update_slider(self, slider, mouse_x):
        norm  = (mouse_x - slider['rect'].x) / slider['rect'].width
        value = slider['min'] + max(0.0, min(1.0, norm)) * (slider['max'] - slider['min'])
        slider['value'] = value
        if slider['type'] == 'param':
            self.params[slider['key']] = value
        else:
            self.initial_conditions[slider['key']] = value

    # ------------------------------------------------------------------ #
    # Drawing                                                              #
    # ------------------------------------------------------------------ #

    def _update_trajectory(self, pole_screen):
        """Append a segment to the persistent trajectory surface."""
        # Surface-local coordinates
        local = (pole_screen[0] - self.sim_x, pole_screen[1] - self.sim_y)
        if self.prev_pole_screen is not None:
            col = (*self.ORANGE, 210) if self.use_controller else (*self.RED, 210)
            pygame.draw.line(self.traj_surface, col,
                             self.prev_pole_screen, local, 2)
        self.prev_pole_screen = local

    def draw_system(self):
        (wheel_sx, wheel_sy), (pole_sx, pole_sy) = self.get_screen_positions()
        cx, cy = self._origin()

        # Ground
        ground_y = wheel_sy + int(self.system.R * self.scale)
        pygame.draw.line(self.screen, self.DARK_GRAY,
                         (self.sim_x, ground_y),
                         (self.sim_x + self.sim_width, ground_y), 3)

        # Full trajectory (blitted every frame — no data is ever discarded)
        self.screen.blit(self.traj_surface, (self.sim_x, self.sim_y))

        # Wheel circle + spokes
        r_px = int(self.system.R * self.scale)
        pygame.draw.circle(self.screen, self.BLUE, (wheel_sx, wheel_sy), r_px, 3)
        phi = self.system.state[0]
        for k in range(4):
            angle = phi + k * np.pi / 2
            sx2 = wheel_sx + int(r_px * np.sin(angle))
            sy2 = wheel_sy - int(r_px * np.cos(angle))
            pygame.draw.line(self.screen, self.BLUE,
                             (wheel_sx, wheel_sy), (sx2, sy2), 1)

        # Pole rod
        pygame.draw.line(self.screen, self.RED,
                         (wheel_sx, wheel_sy), (pole_sx, pole_sy), 5)

        # Bob
        bob_col = self.GREEN if self.dragging_node == 'pole_end' else self.RED
        pygame.draw.circle(self.screen, bob_col, (pole_sx, pole_sy), self.node_radius)

        # Hub
        hub_col = self.GREEN if self.dragging_node == 'wheel' else self.BLUE
        pygame.draw.circle(self.screen, hub_col, (wheel_sx, wheel_sy), self.node_radius)

        # Extend trajectory while running
        if self.running:
            self._update_trajectory((pole_sx, pole_sy))

    def draw_ui(self):
        # Viewport border
        pygame.draw.rect(self.screen, self.BLACK,
                         (self.sim_x, self.sim_y, self.sim_width, self.sim_height), 2)

        # Panel
        panel_rect = pygame.Rect(self.panel_x - 10, self.panel_y - 10,
                                 self.panel_width, self.height - 100)
        pygame.draw.rect(self.screen, self.LIGHT_BLUE, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.BLACK,      panel_rect, 2, border_radius=10)

        self.screen.blit(
            self.font_large.render("Controls", True, self.BLACK),
            (self.panel_x + 100, self.panel_y))

        for slider in self.sliders:
            self.draw_slider(slider)

        # State readout
        state = self.system.get_state()
        theta_w = (state[2] + np.pi) % (2 * np.pi) - np.pi
        ctrl_label  = "CONTROLLER" if self.use_controller else "MANUAL"
        ctrl_colour = self.ORANGE   if self.use_controller else self.DARK_GRAY

        y_pos = self.sim_y + 10
        info = [
            (f"Step : {self.system.get_current_step()}",       self.BLACK),
            (f"Time : {self.system.get_time():.2f} s",         self.BLACK),
            (f"phi  : {state[0]:.3f} rad",                     self.BLACK),
            (f"phi' : {state[1]:.3f} rad/s",                   self.BLACK),
            (f"theta: {np.degrees(theta_w):+.1f} deg",         self.BLACK),
            (f"tht' : {state[3]:.3f} rad/s",                   self.BLACK),
            (f"tau  : {self.torque:+.2f} N*m",                 self.BLACK),
            (f"Mode : {ctrl_label}",                            ctrl_colour),
            (f"      {'RUNNING' if self.running else 'PAUSED'}", self.BLACK),
        ]
        for i, (text, col) in enumerate(info):
            self.screen.blit(self.font_small.render(text, True, col),
                             (self.sim_x + 10, y_pos + i * 25))

        # Key legend
        y_pos = self.height - 270
        legend = [
            ("SPACE",      "Start / Stop"),
            ("R",          "Reset"),
            ("T",          "Clear trajectory"),
            ("C",          "Toggle controller"),
            ("← →",        "Manual torque"),
            ("Drag wheel", "Change phi"),
            ("Drag bob",   "Change theta"),
        ]
        for i, (key, desc) in enumerate(legend):
            self.screen.blit(self.font_small.render(key,  True, self.BLUE),
                             (self.panel_x, y_pos + i * 26))
            self.screen.blit(self.font_small.render(desc, True, self.BLACK),
                             (self.panel_x + 110, y_pos + i * 26))

    # ------------------------------------------------------------------ #
    # Event handling                                                       #
    # ------------------------------------------------------------------ #

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.running = not self.running

                elif event.key == pygame.K_r:
                    self.create_system()
                    self.running = False

                elif event.key == pygame.K_t:
                    self._clear_trajectory()

                elif event.key == pygame.K_c:
                    self.use_controller = not self.use_controller

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if not self.running:
                    (wx, wy), (px, py) = self.get_screen_positions()

                    if np.hypot(mx - wx, my - wy) <= self.node_radius:
                        self.dragging_node = 'wheel'
                        continue
                    if np.hypot(mx - px, my - py) <= self.node_radius:
                        self.dragging_node = 'pole_end'
                        continue

                    for slider in self.sliders:
                        if slider['rect'].collidepoint(mx, my):
                            self.active_slider = slider
                            self.update_slider(slider, mx)
                            self.create_system()
                            break

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.active_slider = None
                self.dragging_node = None

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if self.dragging_node and not self.running:
                    (wx, wy), (px, py) = self.get_screen_positions()
                    if self.dragging_node == 'wheel':
                        self.set_state_from_screen_positions((mx, my), (px, py))
                    else:
                        self.set_state_from_screen_positions((wx, wy), (mx, my))
                    self._clear_trajectory()
                elif self.active_slider and not self.running:
                    self.update_slider(self.active_slider, mx)
                    self.create_system()

        # Compute torque
        if self.use_controller:
            self.torque = float(self.controller.control(self.system.get_state()))
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.torque = -self.max_torque
            elif keys[pygame.K_RIGHT]:
                self.torque = self.max_torque
            else:
                self.torque = 0.0

        return True

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self):
        active = True
        while active:
            active = self.handle_events()

            if self.running:
                self.system.step(self.torque)

            self.screen.fill(self.WHITE)
            self.draw_system()
            self.draw_ui()

            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    WheelPoleSimulation().run()