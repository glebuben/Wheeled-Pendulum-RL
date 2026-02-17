import pygame
import numpy as np
import sys
from wheel_pole_system import WheelPoleSystem


class WheelPoleSimulation:
    """
    Interactive simulation of the wheel-pole system using pygame.
    Allows adjustment of system parameters and initial conditions.
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
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (220, 50, 50)
        self.BLUE = (50, 100, 220)
        self.GREEN = (50, 200, 50)
        self.GRAY = (200, 200, 200)
        self.DARK_GRAY = (100, 100, 100)
        self.LIGHT_BLUE = (150, 200, 255)
        
        # Simulation area
        self.sim_x = 50
        self.sim_y = 50
        self.sim_width = 700
        self.sim_height = 600
        
        # Control panel
        self.panel_x = 770
        self.panel_y = 50
        self.panel_width = 400
        
        # System parameters (defaults)
        self.params = {
            'rod_length': 1.0,
            'wheel_radius': 0.2,
            'wheel_mass': 1.0,
            'pole_mass': 0.1,
        }
        
        # Initial conditions
        self.initial_conditions = {
            'phi': 0.0,
            'theta': 0.1,  # Small initial angle
        }
        
        # Create system
        self.system = None
        self.create_system()
        
        # Control
        self.torque = 0.0
        # self.max_torque = 0.1
        self.max_torque = 0.5
        
        # Simulation state
        self.running = False
        self.show_trace = False
        self.trace_points = []
        self.max_trace_points = 500
        
        # Scale for visualization (pixels per meter)
        self.scale = 200
        
        # Node dragging
        self.dragging_node = None  # 'wheel' or 'pole_end'
        self.node_radius = 12  # Radius for clickable nodes
        
        # Font
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        
        # Sliders
        self.sliders = self.create_sliders()
        self.active_slider = None
        
    def create_system(self):
        """Create a new system with current parameters."""
        self.system = WheelPoleSystem(
            rod_length=self.params['rod_length'],
            wheel_radius=self.params['wheel_radius'],
            wheel_mass=self.params['wheel_mass'],
            pole_mass=self.params['pole_mass']
        )
        self.system.set_initial_state(
            phi=self.initial_conditions['phi'],
            theta=self.initial_conditions['theta']
        )
        self.trace_points = []
        
    def create_sliders(self):
        """Create UI sliders for parameters and initial conditions."""
        sliders = []
        y_offset = self.panel_y + 50
        
        # Parameter sliders
        sliders.append({
            'name': 'Rod Length (m)',
            'key': 'rod_length',
            'type': 'param',
            'min': 0.5,
            'max': 2.0,
            'value': self.params['rod_length'],
            'rect': pygame.Rect(self.panel_x, y_offset, 300, 20)
        })
        y_offset += 60
        
        sliders.append({
            'name': 'Wheel Radius (m)',
            'key': 'wheel_radius',
            'type': 'param',
            'min': 0.1,
            'max': 0.5,
            'value': self.params['wheel_radius'],
            'rect': pygame.Rect(self.panel_x, y_offset, 300, 20)
        })
        y_offset += 60
        
        sliders.append({
            'name': 'Wheel Mass (kg)',
            'key': 'wheel_mass',
            'type': 'param',
            'min': 0.5,
            'max': 3.0,
            'value': self.params['wheel_mass'],
            'rect': pygame.Rect(self.panel_x, y_offset, 300, 20)
        })
        y_offset += 60
        
        sliders.append({
            'name': 'Pole Mass (kg)',
            'key': 'pole_mass',
            'type': 'param',
            'min': 0.05,
            'max': 0.5,
            'value': self.params['pole_mass'],
            'rect': pygame.Rect(self.panel_x, y_offset, 300, 20)
        })
        y_offset += 80
        
        # Initial condition sliders
        sliders.append({
            'name': 'Initial φ (rad)',
            'key': 'phi',
            'type': 'initial',
            'min': -np.pi,
            'max': np.pi,
            'value': self.initial_conditions['phi'],
            'rect': pygame.Rect(self.panel_x, y_offset, 300, 20)
        })
        y_offset += 60
        
        sliders.append({
            'name': 'Initial θ (rad)',
            'key': 'theta',
            'type': 'initial',
            'min': -np.pi,
            'max': np.pi,
            'value': self.initial_conditions['theta'],
            'rect': pygame.Rect(self.panel_x, y_offset, 300, 20)
        })
        
        return sliders
    
    def draw_slider(self, slider):
        """Draw a slider on the screen."""
        # Determine if slider should be disabled
        disabled = self.running
        
        # Label
        color = self.DARK_GRAY if disabled else self.BLACK
        label = self.font_small.render(slider['name'], True, color)
        self.screen.blit(label, (slider['rect'].x, slider['rect'].y - 25))
        
        # Value
        value_text = f"{slider['value']:.3f}"
        value_color = self.DARK_GRAY if disabled else self.BLUE
        value_label = self.font_small.render(value_text, True, value_color)
        self.screen.blit(value_label, (slider['rect'].x + 310, slider['rect'].y - 25))
        
        # Slider track
        track_color = self.DARK_GRAY if disabled else self.GRAY
        pygame.draw.rect(self.screen, track_color, slider['rect'], border_radius=10)
        
        # Slider handle
        normalized = (slider['value'] - slider['min']) / (slider['max'] - slider['min'])
        handle_x = slider['rect'].x + int(normalized * slider['rect'].width)
        handle_rect = pygame.Rect(handle_x - 8, slider['rect'].y - 5, 16, 30)
        handle_color = self.DARK_GRAY if disabled else self.BLUE
        pygame.draw.rect(self.screen, handle_color, handle_rect, border_radius=8)
        
    def update_slider(self, slider, mouse_x):
        """Update slider value based on mouse position."""
        normalized = (mouse_x - slider['rect'].x) / slider['rect'].width
        normalized = max(0.0, min(1.0, normalized))
        value = slider['min'] + normalized * (slider['max'] - slider['min'])
        slider['value'] = value
        
        if slider['type'] == 'param':
            self.params[slider['key']] = value
        elif slider['type'] == 'initial':
            self.initial_conditions[slider['key']] = value
            
    def get_screen_positions(self):
        """Get screen positions of wheel center and pole end."""
        wheel_center, pole_end = self.system.get_cartesian_positions()
        
        # Convert to screen coordinates
        center_x = self.sim_x + self.sim_width // 2
        center_y = self.sim_y + self.sim_height - 100
        
        # Wheel center in screen coordinates
        wheel_screen_x = center_x + int(wheel_center[0] * self.scale)
        wheel_screen_y = center_y - int(wheel_center[1] * self.scale)
        
        # Pole end in screen coordinates
        pole_screen_x = center_x + int(pole_end[0] * self.scale)
        pole_screen_y = center_y - int(pole_end[1] * self.scale)
        
        return (wheel_screen_x, wheel_screen_y), (pole_screen_x, pole_screen_y)
    
    def set_state_from_screen_positions(self, wheel_screen_pos, pole_screen_pos):
        """Update system state based on screen positions of nodes."""
        center_x = self.sim_x + self.sim_width // 2
        center_y = self.sim_y + self.sim_height - 100
        
        # Get current state
        current_state = self.system.get_state()
        current_phi = current_state[0]
        current_theta = current_state[2]
        
        # Convert wheel screen position to world coordinates
        wheel_x = (wheel_screen_pos[0] - center_x) / self.scale
        wheel_y = (center_y - wheel_screen_pos[1]) / self.scale
        
        # Convert pole end screen position to world coordinates
        pole_end_x = (pole_screen_pos[0] - center_x) / self.scale
        pole_end_y = (center_y - pole_screen_pos[1]) / self.scale
        
        # Calculate what phi would be from wheel position
        new_phi = wheel_x / self.system.R
        
        # Calculate what theta would be from pole end position relative to wheel center
        dx = pole_end_x - wheel_x
        dy = pole_end_y - wheel_y
        new_theta = np.arctan2(dx, dy)
        
        # Determine which node is being dragged and update accordingly
        if self.dragging_node == 'wheel':
            # Moving wheel: update phi, keep theta constant
            self.system.state[0] = new_phi
            # theta stays the same: self.system.state[2] unchanged
        elif self.dragging_node == 'pole_end':
            # Moving pole end: keep phi constant, update theta
            # phi stays the same: self.system.state[0] unchanged
            self.system.state[2] = new_theta
        
        # Keep velocities unchanged: state[1] and state[3]
    
    def draw_system(self):
        """Draw the wheel-pole system."""
        # Get positions
        (wheel_screen_x, wheel_screen_y), (pole_screen_x, pole_screen_y) = self.get_screen_positions()
        
        center_x = self.sim_x + self.sim_width // 2
        center_y = self.sim_y + self.sim_height - 100
        
        # Draw ground
        ground_y = wheel_screen_y + int(self.system.R * self.scale)
        pygame.draw.line(self.screen, self.DARK_GRAY, 
                        (self.sim_x, ground_y), 
                        (self.sim_x + self.sim_width, ground_y), 3)
        
        # Draw trace
        if self.show_trace and len(self.trace_points) > 1:
            for i in range(len(self.trace_points) - 1):
                alpha = int(255 * (i / len(self.trace_points)))
                color = (*self.RED[:2], alpha)
                pygame.draw.line(self.screen, self.RED, 
                               self.trace_points[i], self.trace_points[i + 1], 2)
        
        # Draw wheel
        wheel_radius_pixels = int(self.system.R * self.scale)
        pygame.draw.circle(self.screen, self.BLUE, 
                          (wheel_screen_x, wheel_screen_y), 
                          wheel_radius_pixels, 3)
        
        # Draw pole
        pygame.draw.line(self.screen, self.RED, 
                        (wheel_screen_x, wheel_screen_y), 
                        (pole_screen_x, pole_screen_y), 5)
        
        # Draw pole end (bob)
        pole_color = self.GREEN if self.dragging_node == 'pole_end' else self.RED
        pygame.draw.circle(self.screen, pole_color, 
                          (pole_screen_x, pole_screen_y), self.node_radius)
        
        # Draw wheel center node
        wheel_color = self.GREEN if self.dragging_node == 'wheel' else self.BLUE
        pygame.draw.circle(self.screen, wheel_color, 
                          (wheel_screen_x, wheel_screen_y), self.node_radius)
        
        # Add to trace
        if self.running:
            self.trace_points.append((pole_screen_x, pole_screen_y))
            if len(self.trace_points) > self.max_trace_points:
                self.trace_points.pop(0)
                
    def draw_ui(self):
        """Draw the user interface."""
        # Simulation area border
        pygame.draw.rect(self.screen, self.BLACK, 
                        (self.sim_x, self.sim_y, self.sim_width, self.sim_height), 2)
        
        # Control panel
        panel_rect = pygame.Rect(self.panel_x - 10, self.panel_y - 10, 
                                 self.panel_width, self.height - 100)
        pygame.draw.rect(self.screen, self.LIGHT_BLUE, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.BLACK, panel_rect, 2, border_radius=10)
        
        # Title
        title = self.font_large.render("Controls", True, self.BLACK)
        self.screen.blit(title, (self.panel_x + 100, self.panel_y))
        
        # Draw sliders
        for slider in self.sliders:
            self.draw_slider(slider)
        
        # Instructions
        y_pos = self.height - 250
        instructions = [
            "Controls:",
            "LEFT/RIGHT arrows: Apply torque",
            "SPACE: Start/Stop simulation",
            "R: Reset system",
            "T: Toggle trace",
            "Drag wheel center: Change φ",
            "Drag pole end: Change θ",
            "Sliders frozen while running"
        ]
        
        for i, text in enumerate(instructions):
            label = self.font_small.render(text, True, self.BLACK)
            self.screen.blit(label, (self.panel_x, y_pos + i * 25))
        
        # State information
        state = self.system.get_state()
        y_pos = self.sim_y + 10
        info_texts = [
            f"Step: {self.system.get_current_step()}",
            f"Time: {self.system.get_time():.2f} s",
            f"φ (wheel): {state[0]:.3f} rad",
            f"φ_dot: {state[1]:.3f} rad/s",
            f"θ (pole): {state[2]:.3f} rad ({np.degrees(state[2]):.1f}°)",
            f"θ_dot: {state[3]:.3f} rad/s",
            f"Torque: {self.torque:.2f} N⋅m",
            f"Status: {'RUNNING' if self.running else 'PAUSED'}"
        ]
        
        for i, text in enumerate(info_texts):
            label = self.font_small.render(text, True, self.BLACK)
            self.screen.blit(label, (self.sim_x + 10, y_pos + i * 25))
            
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.running = not self.running
                    if self.running:
                        # Clear trace when starting
                        self.trace_points = []
                elif event.key == pygame.K_r:
                    self.create_system()
                    self.running = False
                elif event.key == pygame.K_t:
                    self.show_trace = not self.show_trace
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_x, mouse_y = event.pos
                    
                    # Check if clicking on nodes (only when not running)
                    if not self.running:
                        (wheel_x, wheel_y), (pole_x, pole_y) = self.get_screen_positions()
                        
                        # Check wheel center
                        dist_wheel = np.sqrt((mouse_x - wheel_x)**2 + (mouse_y - wheel_y)**2)
                        if dist_wheel <= self.node_radius:
                            self.dragging_node = 'wheel'
                            continue
                        
                        # Check pole end
                        dist_pole = np.sqrt((mouse_x - pole_x)**2 + (mouse_y - pole_y)**2)
                        if dist_pole <= self.node_radius:
                            self.dragging_node = 'pole_end'
                            continue
                        
                        # Check sliders (only if not running)
                        for slider in self.sliders:
                            if slider['rect'].collidepoint(mouse_x, mouse_y):
                                self.active_slider = slider
                                self.update_slider(slider, mouse_x)
                                self.create_system()
                                break
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.active_slider = None
                    self.dragging_node = None
                    
            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = event.pos
                
                # Handle node dragging
                if self.dragging_node is not None and not self.running:
                    (wheel_x, wheel_y), (pole_x, pole_y) = self.get_screen_positions()
                    
                    if self.dragging_node == 'wheel':
                        # Move wheel center, pole stays at same angle theta
                        # We pass the new wheel position and calculate where pole end should be
                        self.set_state_from_screen_positions((mouse_x, mouse_y), (pole_x, pole_y))
                    elif self.dragging_node == 'pole_end':
                        # Keep wheel center fixed, move pole end to change theta
                        self.set_state_from_screen_positions((wheel_x, wheel_y), (mouse_x, mouse_y))
                
                # Handle slider dragging (only if not running)
                elif self.active_slider is not None and not self.running:
                    self.update_slider(self.active_slider, mouse_x)
                    self.create_system()
        
        # Handle continuous key presses for torque control
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.torque = -self.max_torque
        elif keys[pygame.K_RIGHT]:
            self.torque = self.max_torque
        else:
            self.torque = 0.0
            
        return True
    
    def run(self):
        """Main simulation loop."""
        running = True
        
        while running:
            running = self.handle_events()
            
            # Update physics
            if self.running:
                self.system.step(self.torque)
            
            # Draw
            self.screen.fill(self.WHITE)
            self.draw_system()
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(self.fps)
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    simulation = WheelPoleSimulation()
    simulation.run()