"""
Starting Point Generator for HER Kinetic EIS Fitting
Generates diverse starting points for optimization algorithms
"""

import numpy as np
from scipy.stats import loguniform

class StartingPointGenerator:
    """Generates diverse starting points for HER kinetic parameters"""
    
    def __init__(self):
        # Parameter bounds for A, B, C parameters - 直接拟合ABC参数
        # 进一步扩大容易到达边界的参数范围
        self.param_bounds = {
            'A': (1e-12, 1e7),       # A parameter - 导纳常数项
            'B': (-1e7, 1e7),        # B parameter - 导纳频率相关项
            'C': (1e-10, 1e8),       # C parameter - 频率常数
            'R_hf': (0.0, 100.0),    # 高频电阻 (表面膜/氧化层，0表示无表面膜)
            'C_hf': (1e-6, 1e-4),    # 高频电容 (表面膜/氧化层)
            'W_sigma': (0.0, 1000.0), # Warburg系数 (Ohm*s^(-1/2)，0表示无扩散限制)
        }
        
    def generate_diverse_points(self, n_points=50, estimated_C_dl=None, estimated_R_s=None):
        """
        Generate diverse starting points covering the parameter space
        改进：生成量级差异很大的初始点
        
        Args:
            n_points: Number of starting points to generate
            estimated_C_dl: Estimated C_dl value (fixed for all points)
            estimated_R_s: Estimated R_s value (fixed for all points)
        
        Returns:
            List of starting points
        """
        points = []
        
        # 1. 生成量级差异很大的初始点
        points.extend(self._generate_magnitude_diverse_points(n_points))
        
        # 2. 添加边界情况点
        edge_cases = self._edge_case_points()
        points.extend(edge_cases)
        
        # 3. 如果还需要更多点，生成随机点
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
        生成量级差异很大的初始点
        确保A和C为正，B可正可负，且不同点之间有量级差异
        """
        points = []
        
        # 定义不同的量级组合
        magnitude_combinations = [
            # 极小量级组合
            {'A_mag': -6, 'B_mag': -3, 'C_mag': -4},  # A=1e-6, B=±1e-3, C=1e-4
            {'A_mag': -5, 'B_mag': -2, 'C_mag': -3},  # A=1e-5, B=±1e-2, C=1e-3
            {'A_mag': -4, 'B_mag': -1, 'C_mag': -2},  # A=1e-4, B=±1e-1, C=1e-2
            
            # 小量级组合
            {'A_mag': -3, 'B_mag': 0, 'C_mag': -1},   # A=1e-3, B=±1e0, C=1e-1
            {'A_mag': -2, 'B_mag': 1, 'C_mag': 0},   # A=1e-2, B=±1e1, C=1e0
            {'A_mag': -1, 'B_mag': 2, 'C_mag': 1},   # A=1e-1, B=±1e2, C=1e1
            
            # 中量级组合
            {'A_mag': 0, 'B_mag': 0, 'C_mag': 2},    # A=1e0, B=±1e0, C=1e2
            {'A_mag': 1, 'B_mag': 1, 'C_mag': 3},    # A=1e1, B=±1e1, C=1e3
            {'A_mag': 2, 'B_mag': 2, 'C_mag': 1},    # A=1e2, B=±1e2, C=1e1
            
            # 大量级组合
            {'A_mag': 3, 'B_mag': 3, 'C_mag': 4},    # A=1e3, B=±1e3, C=1e4
            {'A_mag': 2, 'B_mag': 0, 'C_mag': 0},    # A=1e2, B=±1e0, C=1e0
            {'A_mag': 1, 'B_mag': -1, 'C_mag': 2},   # A=1e1, B=±1e-1, C=1e2
            
            # 极端组合
            {'A_mag': -6, 'B_mag': 3, 'C_mag': 4},   # A=1e-6, B=±1e3, C=1e4
            {'A_mag': 3, 'B_mag': -3, 'C_mag': -4},  # A=1e3, B=±1e-3, C=1e-4
            {'A_mag': -6, 'B_mag': -3, 'C_mag': 4},  # A=1e-6, B=±1e-3, C=1e4
            {'A_mag': 3, 'B_mag': 3, 'C_mag': -4},   # A=1e3, B=±1e3, C=1e-4
        ]
        
        # 为每个量级组合生成多个变体
        for i in range(n_points):
            if i < len(magnitude_combinations):
                combo = magnitude_combinations[i]
            else:
                # 随机选择量级组合
                combo = magnitude_combinations[i % len(magnitude_combinations)]
            
            # 生成A值（总是正）
            A = 10 ** np.random.uniform(combo['A_mag'], combo['A_mag'] + 1)
            
            # 生成B值（可正可负）
            B_mag = 10 ** np.random.uniform(combo['B_mag'], combo['B_mag'] + 1)
            B = B_mag if np.random.random() > 0.5 else -B_mag
            
            # 生成C值（总是正）
            C = 10 ** np.random.uniform(combo['C_mag'], combo['C_mag'] + 1)
            
            # 确保在边界内
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
            # 极小量级组合
            {'A': 1e-6, 'B': 1e-4, 'C': 1e-3},    # 极小值
            {'A': 1e-6, 'B': -1e-4, 'C': 1e-3},   # 极小值，B为负
            {'A': 1e-5, 'B': 1e-3, 'C': 1e-2},    # 小值
            {'A': 1e-5, 'B': -1e-3, 'C': 1e-2},   # 小值，B为负
            
            # 小量级组合
            {'A': 1e-4, 'B': 1e-2, 'C': 1e-1},    # 小导纳值
            {'A': 1e-4, 'B': -1e-2, 'C': 1e-1},   # 小导纳值，B为负
            {'A': 1e-3, 'B': 1e-1, 'C': 1e0},     # 中等导纳值
            {'A': 1e-3, 'B': -1e-1, 'C': 1e0},    # 中等导纳值，B为负
            
            # 中量级组合
            {'A': 1e-2, 'B': 1e0, 'C': 1e1},      # 中等导纳值
            {'A': 1e-2, 'B': -1e0, 'C': 1e1},     # 中等导纳值，B为负
            {'A': 1e-1, 'B': 1e1, 'C': 1e2},      # 大导纳值
            {'A': 1e-1, 'B': -1e1, 'C': 1e2},     # 大导纳值，B为负
            
            # 大量级组合
            {'A': 1e0, 'B': 1e2, 'C': 1e3},      # 大导纳值
            {'A': 1e0, 'B': -1e2, 'C': 1e3},     # 大导纳值，B为负
            {'A': 1e1, 'B': 1e1, 'C': 1e2},      # 极大导纳值
            {'A': 1e1, 'B': -1e1, 'C': 1e2},     # 极大导纳值，B为负
            
            # 极端组合
            {'A': 1e-6, 'B': 1e2, 'C': 1e3},     # A极小，B极大
            {'A': 1e2, 'B': 1e-4, 'C': 1e-3},    # A极大，B极小
            {'A': 1e-6, 'B': -1e2, 'C': 1e3},    # A极小，B极大负值
            {'A': 1e2, 'B': -1e-4, 'C': 1e-3},   # A极大，B极小负值
        ]
        return edge_cases
    
    def _random_point(self):
        """Generate a single random point"""
        point = {}
        for param, (low, high) in self.param_bounds.items():
            if param == 'B' and low < 0:
                # B参数可以为负，使用线性分布
                value = np.random.uniform(low, high)
            else:
                # 其他参数使用对数分布
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
        A, B, C, C_dl, R_s = best_params[:5]  # 只取前5个参数
        
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
                # B参数可以为负，使用线性变化
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
            include_extended: 是否包含Warburg扩散项
            rs_initial: Rs初始值，如果提供则使用±30%范围
        
        Returns:
            参数边界列表
        """
        # Rs边界 - 完全放开，不受rs_initial限制
        # Rs初始值仅用于生成初始参数点，不限制优化范围
        rs_bounds = (0.0001, 10.0)
        
        # 基础参数边界
        base_bounds = [
            self.param_bounds['A'],
            self.param_bounds['B'],
            self.param_bounds['C'],
            (1e-12, 1e-1),    # C_dl bounds
            rs_bounds,        # R_s bounds (可优化，带约束)
        ]
        
        if include_extended:
            # 只添加Warburg扩散项（去掉高频RC）
            extended_bounds = [
                self.param_bounds['W_sigma'],  # Warburg系数
            ]
            return base_bounds + extended_bounds
        else:
            return base_bounds
