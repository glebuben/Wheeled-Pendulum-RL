import pygame
import numpy as np
import torch
import sys
import os
from pathlib import Path
from wheel_pole_system import WheelPoleSystem
from policy_network import PolicyNetwork


class PolicyVisualizer:
    """
    Interactive visualization of trained REINFORCE policies.
    Loads checkpoint files and renders the agent's behavior.
    """
    
    def __init__(self, width=1200, height=700):
        pygame.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Wheel-Pole Policy Visualization")
        
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Colors
        self.WHITE      = (255, 255, 255)
        self.BLACK      = (0,   0,   0)
        self.RED        = (220, 50,  50)
        self.BLUE       = (50,  100, 220)
        self.GREEN      = (50,  200, 50)
        self.ORANGE     = (230, 140, 30)
        self.PURPLE     = (150, 50,  200)
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
        
        # System parameters (fixed for trained policy)
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
        self.show_trace     = True
        self.torque         = 0.0
        self.tau_max        = 10.0  # User-adjustable
        
        # Trajectory surface
        self.traj_surface  = pygame.Surface(
            (self.sim_width, self.sim_height), pygame.SRCALPHA
        )
        self.traj_surface.fill((0, 0, 0, 0))
        self.prev_pole_screen = None
        
        # Fonts
        self.font_large  = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small  = pygame.font.Font(None, 20)
        
        # Policy network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_bound = 10.0  # Default action bound, will be set from checkpoint or config
        self.policy_net = PolicyNetwork(
            observation_dim=4, 
            hidden_dim=128, 
            action_dim=1,
            action_bound=self.action_bound
        ).to(self.device)
        self.checkpoint_loaded = False
        self.checkpoint_path = None
        
        # Build system
        self.system = None
        self.create_system()
        
        # UI elements
        self.sliders       = self.create_sliders()
        self.buttons       = self.create_buttons()
        self.active_slider = None
        self.dragging_node = None
        self.node_radius   = 12

    def create_system(self):
        """Instantiate a fresh WheelPoleSystem."""
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
        self._clear_trajectory()

    def _clear_trajectory(self):
        self.traj_surface.fill((0, 0, 0, 0))
        self.prev_pole_screen = None

    def load_checkpoint(self, checkpoint_path):
        """Load a trained policy from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.policy_net.eval()
            self.checkpoint_loaded = True
            self.checkpoint_path = checkpoint_path
            print(f"Loaded checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            self.checkpoint_loaded = False
            return False

    def get_policy_action(self, state):
        """Get action from policy network (using mean, not sampling)."""
        if not self.checkpoint_loaded:
            return 0.0
        
        with torch.no_grad():
            obs = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
            mean, log_std, value = self.policy_net(obs)
            
            # Use mean action (deterministic policy for visualization)
            # Note: mean is already bounded by tanh * action_bound in the network
            action_value = float(mean.detach().cpu().numpy().squeeze())
            
            # Apply user-specified torque saturation (tau_max slider)
            # This allows testing the policy with different torque limits
            action_value = np.clip(action_value, -self.tau_max, self.tau_max)
            
            return action_value

    # ------------------------------------------------------------------ #
    # Coordinate helpers                                                 #
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
        """Update state from dragged node."""
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
    # Sliders                                                            #
    # ------------------------------------------------------------------ #

    def create_sliders(self):
        sliders  = []
        y_offset = self.panel_y + 50

        # Initial condition sliders
        sliders.append({
            'name':  'Initial φ (rad)',
            'key':   'phi',
            'type':  'initial',
            'min':   -np.pi,
            'max':   np.pi,
            'value': self.initial_conditions['phi'],
            'rect':  pygame.Rect(self.panel_x, y_offset, 300, 20)
        })
        y_offset += 60

        sliders.append({
            'name':  'Initial θ (rad)',
            'key':   'theta',
            'type':  'initial',
            'min':   -np.pi,
            'max':   np.pi,
            'value': self.initial_conditions['theta'],
            'rect':  pygame.Rect(self.panel_x, y_offset, 300, 20)
        })
        y_offset += 60

        # Torque limit slider
        sliders.append({
            'name':  'Max Torque (N·m)',
            'key':   'tau_max',
            'type':  'param',
            'min':   1.0,
            'max':   20.0,
            'value': self.tau_max,
            'rect':  pygame.Rect(self.panel_x, y_offset, 300, 20)
        })

        return sliders

    def create_buttons(self):
        """Create UI buttons."""
        buttons = []
        y_offset = self.panel_y + 240
        
        # Load checkpoint button
        buttons.append({
            'name': 'Load Checkpoint',
            'action': 'load_checkpoint',
            'rect': pygame.Rect(self.panel_x, y_offset, 200, 40),
            'color': self.BLUE,
        })
        y_offset += 50
        
        # Toggle trace button
        buttons.append({
            'name': f"Trace: {'ON' if self.show_trace else 'OFF'}",
            'action': 'toggle_trace',
            'rect': pygame.Rect(self.panel_x, y_offset, 200, 40),
            'color': self.GREEN if self.show_trace else self.GRAY,
        })
        
        return buttons

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
        
        if slider['type'] == 'initial':
            self.initial_conditions[slider['key']] = value
        elif slider['key'] == 'tau_max':
            self.tau_max = value

    def draw_button(self, button):
        """Draw a button."""
        color = button['color']
        pygame.draw.rect(self.screen, color, button['rect'], border_radius=10)
        pygame.draw.rect(self.screen, self.BLACK, button['rect'], 2, border_radius=10)
        
        text = self.font_small.render(button['name'], True, self.WHITE)
        text_rect = text.get_rect(center=button['rect'].center)
        self.screen.blit(text, text_rect)

    def handle_button_click(self, button):
        """Handle button click actions."""
        if button['action'] == 'load_checkpoint':
            # Open file dialog (simplified - user must specify path)
            checkpoint_path = self.prompt_checkpoint_path()
            if checkpoint_path:
                self.load_checkpoint(checkpoint_path)
        elif button['action'] == 'toggle_trace':
            self.show_trace = not self.show_trace
            button['name'] = f"Trace: {'ON' if self.show_trace else 'OFF'}"
            button['color'] = self.GREEN if self.show_trace else self.GRAY

    def prompt_checkpoint_path(self):
        """
        Simple checkpoint path prompt.
        In practice, you'd use a file dialog or command-line argument.
        """
        # For now, check for checkpoint in artifacts/checkpoints/
        artifacts_dir = Path('artifacts/checkpoints')
        if artifacts_dir.exists():
            checkpoints = sorted(artifacts_dir.glob('*.pt'))
            if checkpoints:
                # Use the most recent checkpoint
                return str(checkpoints[-1])
        return None

    # ------------------------------------------------------------------ #
    # Drawing                                                            #
    # ------------------------------------------------------------------ #

    def _update_trajectory(self, pole_screen):
        """Paint a new segment onto the persistent trajectory surface."""
        local = (pole_screen[0] - self.sim_x, pole_screen[1] - self.sim_y)

        if self.prev_pole_screen is not None and self.show_trace:
            color = (*self.PURPLE[:3], 200)
            pygame.draw.line(self.traj_surface, color,
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

        # Trajectory
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

        # Extend trajectory
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

        # Title
        title_text = "Policy Visualizer"
        if self.checkpoint_loaded:
            title_text += " ✓"
        title = self.font_large.render(title_text, True, self.BLACK)
        self.screen.blit(title, (self.panel_x + 60, self.panel_y))

        # Checkpoint info
        if self.checkpoint_loaded:
            cp_name = Path(self.checkpoint_path).name if self.checkpoint_path else "Loaded"
            cp_label = self.font_small.render(f"Model: {cp_name}", True, self.PURPLE)
            self.screen.blit(cp_label, (self.panel_x, self.panel_y + 35))

        # Sliders
        for slider in self.sliders:
            self.draw_slider(slider)

        # Buttons
        for button in self.buttons:
            self.draw_button(button)

        # State readout
        state = self.system.get_state()
        theta_w = (state[2] + np.pi) % (2 * np.pi) - np.pi

        y_pos = self.sim_y + 10
        status_color = self.GREEN if self.checkpoint_loaded else self.RED
        info = [
            (f"Policy: {'LOADED' if self.checkpoint_loaded else 'NOT LOADED'}", status_color),
            (f"Step : {self.system.get_current_step()}",       self.BLACK),
            (f"Time : {self.system.get_time():.2f} s",        self.BLACK),
            (f"phi  : {state[0]:.3f} rad",                    self.BLACK),
            (f"phi' : {state[1]:.3f} rad/s",                  self.BLACK),
            (f"theta: {np.degrees(theta_w):+.1f} deg",        self.BLACK),
            (f"tht' : {state[3]:.3f} rad/s",                  self.BLACK),
            (f"tau  : {self.torque:+.2f} N*m",                self.BLACK),
            (f"      {'RUNNING' if self.running else 'PAUSED'}", self.BLACK),
        ]
        for i, (text, col) in enumerate(info):
            self.screen.blit(self.font_small.render(text, True, col),
                             (self.sim_x + 10, y_pos + i * 25))

        # Instructions
        y_pos = self.height - 200
        legend = [
            ("SPACE",      "Start / Stop"),
            ("R",          "Reset"),
            ("T",          "Clear trace"),
            ("L",          "Load checkpoint"),
            ("Drag wheel", "Change phi"),
            ("Drag bob",   "Change theta"),
        ]
        for i, (key, desc) in enumerate(legend):
            self.screen.blit(self.font_small.render(key,  True, self.BLUE),
                             (self.panel_x, y_pos + i * 24))
            self.screen.blit(self.font_small.render(desc, True, self.BLACK),
                             (self.panel_x + 110, y_pos + i * 24))

    # ------------------------------------------------------------------ #
    # Event handling                                                     #
    # ------------------------------------------------------------------ #

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.checkpoint_loaded:
                        self.running = not self.running
                    else:
                        print("Load a checkpoint first!")

                elif event.key == pygame.K_r:
                    self.create_system()
                    self.running = False

                elif event.key == pygame.K_t:
                    self._clear_trajectory()

                elif event.key == pygame.K_l:
                    checkpoint_path = self.prompt_checkpoint_path()
                    if checkpoint_path:
                        self.load_checkpoint(checkpoint_path)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                
                # Check buttons
                for button in self.buttons:
                    if button['rect'].collidepoint(mx, my):
                        self.handle_button_click(button)
                        continue
                
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

        # Get policy action
        if self.checkpoint_loaded:
            state = self.system.get_state()
            self.torque = self.get_policy_action(state)
        else:
            self.torque = 0.0

        return True

    # ------------------------------------------------------------------ #
    # Main loop                                                          #
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
    # Check for checkpoint path as command-line argument
    checkpoint_path = None
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    visualizer = PolicyVisualizer()
    
    # Auto-load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        visualizer.load_checkpoint(checkpoint_path)
    else:
        # Try to auto-load most recent checkpoint
        auto_path = visualizer.prompt_checkpoint_path()
        if auto_path:
            visualizer.load_checkpoint(auto_path)
    
    visualizer.run()