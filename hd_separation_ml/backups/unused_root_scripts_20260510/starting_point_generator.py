"""
Starting Point Generator for HER Kinetic EIS Fitting
Generates diverse starting points for optimization algorithms
"""

import numpy as np
from scipy.stats import loguniform

class StartingPointGenerator:
    """Generates diverse starting points for HER kinetic parameters"""
    
    def __init__(self):
        # Parameter bounds for directly fitting the A, B, and C parameters
        # The ranges are widened further for parameters that often hit bounds
        self.param_bounds = {
            'A': (1e-12, 1e7),       # A parameter - admittance constant term
            'B': (-1e7, 1e7),        # B parameter - frequency-dependent admittance term
            'C': (1e-10, 1e8),       # C parameter - frequency constant
            'R_hf': (0.0, 100.0),    # High-frequency resistance (surface film / oxide layer, 0 means none)
            'C_hf': (1e-6, 1e-4),    # High-frequency capacitance (surface film / oxide layer)
            'W_sigma': (0.0, 1000.0), # Warburg coefficient (Ohm*s^(-1/2), 0 means no diffusion limitation)
        }
        
    def generate_diverse_points(self, n_points=50, estimated_C_dl=None, estimated_R_s=None):
        """
        Generate diverse starting points covering the parameter space
        Improvement: generate initial points that differ strongly in magnitude
        
        Args:
            n_points: Number of starting points to generate
            estimated_C_dl: Estimated C_dl value (fixed for all points)
            estimated_R_s: Estimated R_s value (fixed for all points)
        
        Returns:
            List of starting points
        """
        points = []
        
        # 1. Generate initial points with large magnitude differences
        points.extend(self._generate_magnitude_diverse_points(n_points))
        
        # 2. Add edge-case points
        edge_cases = self._edge_case_points()
        points.extend(edge_cases)
        
        # 3. If more points are still needed, generate random ones
        while len(points) < n_points:
            points.append(self._random_point())
        
        # Add C_dl, R_s to each point
        final_points = []
        for point in points[:n_points]:
            final_point = [
                point['A'],
                point['B'],
                point['C'],
                estimated_C_dl or 400e-6,  # Default 400 μF
                estimated_R_s or 1.0,     # Default 1 Ω
            ]
            final_points.append(final_point)
        
        return final_points
    
    def _generate_magnitude_diverse_points(self, n_points):
        """
        Generate initial points with large magnitude differences.
        Keep A and C positive, allow B to be positive or negative,
        and enforce clear scale differences between points.
        """
        points = []
        
        # Define combinations with different magnitudes
        magnitude_combinations = [
            # Extremely small-scale combinations
            {'A_mag': -6, 'B_mag': -3, 'C_mag': -4},  # A=1e-6, B=±1e-3, C=1e-4
            {'A_mag': -5, 'B_mag': -2, 'C_mag': -3},  # A=1e-5, B=±1e-2, C=1e-3
            {'A_mag': -4, 'B_mag': -1, 'C_mag': -2},  # A=1e-4, B=±1e-1, C=1e-2
            
            # Small-scale combinations
            {'A_mag': -3, 'B_mag': 0, 'C_mag': -1},   # A=1e-3, B=±1e0, C=1e-1
            {'A_mag': -2, 'B_mag': 1, 'C_mag': 0},   # A=1e-2, B=±1e1, C=1e0
            {'A_mag': -1, 'B_mag': 2, 'C_mag': 1},   # A=1e-1, B=±1e2, C=1e1
            
            # Medium-scale combinations
            {'A_mag': 0, 'B_mag': 0, 'C_mag': 2},    # A=1e0, B=±1e0, C=1e2
            {'A_mag': 1, 'B_mag': 1, 'C_mag': 3},    # A=1e1, B=±1e1, C=1e3
            {'A_mag': 2, 'B_mag': 2, 'C_mag': 1},    # A=1e2, B=±1e2, C=1e1
            
            # Large-scale combinations
            {'A_mag': 3, 'B_mag': 3, 'C_mag': 4},    # A=1e3, B=±1e3, C=1e4
            {'A_mag': 2, 'B_mag': 0, 'C_mag': 0},    # A=1e2, B=±1e0, C=1e0
            {'A_mag': 1, 'B_mag': -1, 'C_mag': 2},   # A=1e1, B=±1e-1, C=1e2
            
            # Extreme combinations
            {'A_mag': -6, 'B_mag': 3, 'C_mag': 4},   # A=1e-6, B=±1e3, C=1e4
            {'A_mag': 3, 'B_mag': -3, 'C_mag': -4},  # A=1e3, B=±1e-3, C=1e-4
            {'A_mag': -6, 'B_mag': -3, 'C_mag': 4},  # A=1e-6, B=±1e-3, C=1e4
            {'A_mag': 3, 'B_mag': 3, 'C_mag': -4},   # A=1e3, B=±1e3, C=1e-4
        ]
        
        # Generate multiple variants for each magnitude combination
        for i in range(n_points):
            if i < len(magnitude_combinations):
                combo = magnitude_combinations[i]
            else:
                # Select a magnitude combination in a repeating pattern
                combo = magnitude_combinations[i % len(magnitude_combinations)]
            
            # Generate A (always positive)
            A = 10 ** np.random.uniform(combo['A_mag'], combo['A_mag'] + 1)
            
            # Generate B (can be positive or negative)
            B_mag = 10 ** np.random.uniform(combo['B_mag'], combo['B_mag'] + 1)
            B = B_mag if np.random.random() > 0.5 else -B_mag
            
            # Generate C (always positive)
            C = 10 ** np.random.uniform(combo['C_mag'], combo['C_mag'] + 1)
            
            # Clip the values to the configured bounds
            A = np.clip(A, *self.param_bounds['A'])
            B = np.clip(B, *self.param_bounds['B'])
            C = np.clip(C, *self.param_bounds['C'])
            
            points.append({'A': A, 'B': B, 'C': C})
        
        return points
    
    def _latin_hypercube_sampling(self, n_points):
        """Generate points using Latin Hypercube Sampling (simplified)"""
        points = []
        for i in range(n_points):
            point = {}
            for param, (low, high) in self.param_bounds.items():
                # Simplified Latin hypercube sampling
                segment = (i + np.random.random()) / max(1, n_points)
                value = low * (high / low) ** segment
                point[param] = value
            points.append(point)
        return points
    
    def _log_uniform_sampling(self, n_points):
        """Generate points using log-uniform distribution"""
        points = []
        for _ in range(n_points):
            point = {}
            for param, (low, high) in self.param_bounds.items():
                value = 10 ** np.random.uniform(np.log10(low), np.log10(high))
                point[param] = value
            points.append(point)
        return points
    
    def _edge_case_points(self):
        """Generate edge case points for A, B, C parameters with magnitude diversity"""
        edge_cases = [
            # Extremely small-scale combinations
            {'A': 1e-6, 'B': 1e-4, 'C': 1e-3},    # Extremely small values
            {'A': 1e-6, 'B': -1e-4, 'C': 1e-3},   # Extremely small values, negative B
            {'A': 1e-5, 'B': 1e-3, 'C': 1e-2},    # Small values
            {'A': 1e-5, 'B': -1e-3, 'C': 1e-2},   # Small values, negative B
            
            # Small-scale combinations
            {'A': 1e-4, 'B': 1e-2, 'C': 1e-1},    # Small admittance values
            {'A': 1e-4, 'B': -1e-2, 'C': 1e-1},   # Small admittance values, negative B
            {'A': 1e-3, 'B': 1e-1, 'C': 1e0},     # Moderate admittance values
            {'A': 1e-3, 'B': -1e-1, 'C': 1e0},    # Moderate admittance values, negative B
            
            # Medium-scale combinations
            {'A': 1e-2, 'B': 1e0, 'C': 1e1},      # Moderate admittance values
            {'A': 1e-2, 'B': -1e0, 'C': 1e1},     # Moderate admittance values, negative B
            {'A': 1e-1, 'B': 1e1, 'C': 1e2},      # Large admittance values
            {'A': 1e-1, 'B': -1e1, 'C': 1e2},     # Large admittance values, negative B
            
            # Large-scale combinations
            {'A': 1e0, 'B': 1e2, 'C': 1e3},      # Large admittance values
            {'A': 1e0, 'B': -1e2, 'C': 1e3},     # Large admittance values, negative B
            {'A': 1e1, 'B': 1e1, 'C': 1e2},      # Very large admittance values
            {'A': 1e1, 'B': -1e1, 'C': 1e2},     # Very large admittance values, negative B
            
            # Extreme combinations
            {'A': 1e-6, 'B': 1e2, 'C': 1e3},     # Extremely small A, extremely large B
            {'A': 1e2, 'B': 1e-4, 'C': 1e-3},    # Extremely large A, extremely small B
            {'A': 1e-6, 'B': -1e2, 'C': 1e3},    # Extremely small A, large negative B
            {'A': 1e2, 'B': -1e-4, 'C': 1e-3},   # Extremely large A, small negative B
        ]
        return edge_cases
    
    def _random_point(self):
        """Generate a single random point"""
        point = {}
        for param, (low, high) in self.param_bounds.items():
            if param == 'B' and low < 0:
                # B can be negative, so use a linear distribution
                value = np.random.uniform(low, high)
            else:
                # Other parameters use a log-uniform distribution
                value = 10 ** np.random.uniform(np.log10(low), np.log10(high))
            point[param] = value
        return point
    
    def generate_adaptive_points(self, best_params, n_points=3, spread_factor=1.0):
        """
        Generate adaptive points around the best found parameters
        Optimized for speed with reduced number of points
        
        Args:
            best_params: Best parameters found so far
            n_points: Number of points to generate (optimized for 3)
            spread_factor: Reduced spread for faster convergence
        
        Returns:
            List of starting points
        """
        points = []
        A, B, C, C_dl, R_s = best_params[:5]  # Only use the first five parameters
        
        # Define specific spread factors for each parameter to balance exploration and speed
        spreads = {
            'A': spread_factor, 
            'B': spread_factor, 
            'C': spread_factor/2, 
        }
        
        # Generate points with controlled variation
        for i in range(n_points):
            # Use deterministic variations for the first few points
            if i == 0:
                # Best point
                A_new, B_new, C_new = A, B, C
            elif i == 1:
                # Point with increased values
                A_new = min(A * 10, self.param_bounds['A'][1])
                B_new = min(B * 10, self.param_bounds['B'][1])
                C_new = min(C * 1.5, self.param_bounds['C'][1])
            elif i == 2:
                # Point with decreased values
                A_new = max(A / 10, self.param_bounds['A'][0])
                B_new = max(B / 10, self.param_bounds['B'][0])
                C_new = max(C / 1.5, self.param_bounds['C'][0])
            else:
                # For additional points if needed, use controlled random variation
                A_new = A * (10 ** np.random.uniform(-spreads['A'], spreads['A']))
                # B can be negative, so vary it linearly
                B_new = B + np.random.uniform(-spreads['B'] * abs(B), spreads['B'] * abs(B))
                C_new = C * (10 ** np.random.uniform(-spreads['C'], spreads['C']))
                
                # Ensure bounds are respected
                A_new = np.clip(A_new, *self.param_bounds['A'])
                B_new = np.clip(B_new, *self.param_bounds['B'])
                C_new = np.clip(C_new, *self.param_bounds['C'])
            
            points.append([A_new, B_new, C_new, C_dl, R_s])
        
        return points
    
    def get_parameter_bounds(self, include_extended=False, rs_initial=None):
        """
        Return parameter bounds for optimization
        
        Args:
            include_extended: Whether to include the Warburg diffusion term
            rs_initial: Initial Rs value; if provided, use a ±30% range
        
        Returns:
            List of parameter bounds
        """
        # Rs bounds are left fully open and are not constrained by rs_initial
        # rs_initial is only used to generate starting points, not to limit optimization
        rs_bounds = (0.0001, 10.0)
        
        # Base parameter bounds
        base_bounds = [
            self.param_bounds['A'],
            self.param_bounds['B'],
            self.param_bounds['C'],
            (1e-12, 1e-1),    # C_dl bounds
            rs_bounds,        # R_s bounds (optimizable, with bounds)
        ]
        
        if include_extended:
            # Only add the Warburg diffusion term (without the high-frequency RC term)
            extended_bounds = [
                self.param_bounds['W_sigma'],  # Warburg coefficient
            ]
            return base_bounds + extended_bounds
        else:
            return base_bounds
