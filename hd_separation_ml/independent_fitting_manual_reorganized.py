#!/usr/bin/env python3
"""
独立拟合脚本 - 重新组织版本
将A, B, C, C_dl, R_s对每个样本都视为独立变量进行拟合
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# 确保工作目录在脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"工作目录: {os.getcwd()}")
from scipy.optimize import differential_evolution, fsolve
from scipy.interpolate import interp1d, PchipInterpolator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib字体，使用系统中常见的字体
plt.rcParams["font.family"] = ["Arial", "Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

# 导入模块化组件
from improved_eis_data_loader import ImprovedEISDataLoader
from starting_point_generator import StartingPointGenerator

# 目录配置
DATA_DIR = "data_extraction/all_data/EIS"  # 更新为新的数据路径
CSV_PATH = 'independent_fitting_results.csv'
# HISTORICAL_RESULTS_PATH = 'historical_fitting_results.csv'
# RS_JSON_PATH = 'data_extraction/overpotentials.json'

class IndependentFitter:
    """独立拟合器，将所有参数视为独立变量"""
    
    # ============================================================================
    # 1. 初始化方法
    # ============================================================================
    
    def __init__(self, aggregated_json_dir='aggregated_json', debug_mass_transport=False):
        self.R = 8.314
        self.T = 298.15
        self.F = 96487
        self.debug_mass_transport = debug_mass_transport
        
        # 初始化组件
        self.data_loader = ImprovedEISDataLoader(DATA_DIR, json_dir=None)
        self.point_generator = StartingPointGenerator()
        
        # 历史结果存储（用于主动学习）
        self.historical_results = []
        
        # 目录路径
        self.aggregated_json_dir = aggregated_json_dir
        # 工作电极面积（假设为1 cm2）
        self.working_electrode_area = 1.0
        
        # 加载固定的Rs值
        self.rs_values = self.load_rs_values()
    
    
    # ============================================================================
    # 2. 数据加载和验证方法
    # ============================================================================
    
    def load_rs_values(self):
        """从ML-20250930(1)的all_substances.json加载固定的Rs值"""
        try:
            import json
            with open("data_extraction/all_substances.json" , 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 创建正向和反向映射
            rs_mapping = {}
            for entry in data:
                substance1 = entry['substance1']
                substance2 = entry['substance2']
                ir_value = entry['ir_value']
                
                # 同时存储正向和反向
                forward_name = f"{substance1}-{substance2}"
                reverse_name = f"{substance2}-{substance1}"
                rs_mapping[forward_name] = ir_value
                rs_mapping[reverse_name] = ir_value
            
            print(f"[OK] 已加载 {len(rs_mapping)//2} 个样本的固定Rs值")
            return rs_mapping
            
        except Exception as e:
            print(f"[WARNING] 无法加载Rs值: {e}")
            print(f"          将在拟合中优化Rs")
            return {}
    
    def get_fixed_rs(self, sample_name):
        """获取样本的固定Rs值，如果没有则返回None"""
        if sample_name in self.rs_values:
            return self.rs_values[sample_name]
        return None
    
    def validate_eis_data(self, freq, Z_exp):
        """验证EIS数据的有效性"""
        if freq is None:
            return False, "无有效数据"
        
        if len(freq) < 10:
            return False, "数据点太少"
        
        if np.min(freq) <= 0 or not np.all(np.isfinite(Z_exp)):
            return False, "无效的频率或阻抗数据"
        
        return True, "数据有效"
    
    # def load_historical_results(self):
    #     """加载历史拟合结果用于主动学习"""
    #     if os.path.exists(HISTORICAL_RESULTS_PATH):
    #         try:
    #             df = pd.read_csv(HISTORICAL_RESULTS_PATH)
    #             self.historical_results = df.to_dict('records')
    #             print(f"加载了 {len(self.historical_results)} 条历史结果")
    #         except Exception as e:
    #             print(f"无法加载历史结果: {e}")
    #             self.historical_results = []
    #     else:
    #         self.historical_results = []
    
    # ============================================================================
    # 3. 参数估计和初始化方法
    # ============================================================================
        
    def get_initial_parameters_from_generator(self, freq, Z_exp, sample_name=None):
        """使用starting_point_generator生成初始参数"""
        # 从频率最高的imag(Z)<=0的点估算Rs
        mask_neg_imag = Z_exp.imag <= 0
        if np.any(mask_neg_imag):
            freq_neg_imag = freq[mask_neg_imag]
            Z_neg_imag = Z_exp[mask_neg_imag]
            max_freq_idx = np.argmax(freq_neg_imag)
            R_s_est = Z_neg_imag[max_freq_idx].real
        else:
            R_s_est = max(np.real(Z_exp[0]), 0.1)  # 备用方法
        C_dl_est = 400e-6  # 400 uF/cm2 (按用户要求)
        
        # 使用starting_point_generator生成多样化的初始参数
        print(f"  使用starting_point_generator生成初始参数")
        diverse_points = self.point_generator.generate_diverse_points(
            n_points=50, 
            estimated_C_dl=C_dl_est, 
            estimated_R_s=R_s_est
        )
        
        # 返回所有生成的starting points，让调用者决定如何处理
        print(f"  生成了 {len(diverse_points)} 个初始参数点")
        return diverse_points
    
    # ============================================================================
    # 4. 核心拟合算法方法
    # ============================================================================
    
    def unified_objective_function(self, params, freq, Z_exp, debug=False):
        """目标函数：考虑EIS拟合误差，高频区域增加权重
        
        参数:
            params: 拟合参数 [A, B, C, C_dl, R_s] 或 [A, B, C, C_dl, R_s, W_sigma]
            freq: 频率数组 (已经过滤过虚部<=0的点)
            Z_exp: 实验阻抗数据 (已经过滤过虚部<=0的点)
            debug: 是否输出调试信息
        """
        try:
            # 直接使用传入的数据，不需要再次过滤
            Z_model = self.impedance_from_ABC(freq, params)
            
            # 计算加权误差
            # 只对1000Hz以上的高频区域增加权重
            weights = np.ones_like(freq)
            weights[freq > 1000] = 10.0  # 高频区域(>1kHz)权重×10（提高以更好拟合小半圆）
            
            # 计算加权残差
            residual_real = np.abs(Z_exp.real - Z_model.real)
            residual_imag = np.abs(Z_exp.imag - Z_model.imag)
            
            weighted_error = np.sum(weights * (residual_real**2 + residual_imag**2))
            
            # 计算标准R2用于显示（不加权）
            r_squared = self.calculate_r_squared(Z_exp, Z_model)
            
            if debug:
                print(f"    调试: R2={r_squared:.4f}, 加权误差={weighted_error:.4e}")
            
            # 返回原始R2值用于显示
            self._last_r_squared = r_squared
            
            return weighted_error
            
        except Exception as e:
            if debug:
                print(f"  目标函数计算错误: {str(e)}")
            return 1e10

    def calculate_r_squared(self, Z_exp, Z_fit):
        """计算拟合优度R2值 - 分别计算实部和虚部的R2"""
        # 分别计算实部和虚部的R2
        Z_exp_real = np.real(Z_exp)
        Z_exp_imag = np.imag(Z_exp)
        Z_fit_real = np.real(Z_fit)
        Z_fit_imag = np.imag(Z_fit)
        
        # 计算实部R2
        ss_res_real = np.sum((Z_exp_real - Z_fit_real)**2)
        ss_tot_real = np.sum((Z_exp_real - np.mean(Z_exp_real))**2)
        r2_real = 1 - (ss_res_real / ss_tot_real) if ss_tot_real > 0 else 0
        
        # 计算虚部R2
        ss_res_imag = np.sum((Z_exp_imag - Z_fit_imag)**2)
        ss_tot_imag = np.sum((Z_exp_imag - np.mean(Z_exp_imag))**2)
        r2_imag = 1 - (ss_res_imag / ss_tot_imag) if ss_tot_imag > 0 else 0
        
        # 返回平均值
        return (r2_real + r2_imag) / 2

    def fit_single_optimization(self, freq, Z_exp, initial_params, rs_initial=None):
        """
        单次优化拟合参数（包含Warburg扩散）
        
        Args:
            freq: 频率数组
            Z_exp: 实验阻抗数据
            initial_params: 初始参数 [A, B, C, C_dl, R_s] (基础模型)
            rs_initial: Rs初始值（从all_substances.json），用于设置Rs优化范围
        
        Returns:
            拟合参数数组 [A, B, C, C_dl, R_s, W_sigma] (6参数)
        """
        
        # 使用Warburg模型（6参数：去掉高频RC）
        bounds = self.point_generator.get_parameter_bounds(
            include_extended=True, 
            rs_initial=rs_initial
        )
        
        # 初始参数：基础5个 + Warburg
        if len(initial_params) == 5:
            # 扩展初始参数
            A, B, C, C_dl, R_s = initial_params
            # 初始猜测：中等的W_sigma
            initial_params_extended = [A, B, C, C_dl, R_s, 10.0]
        else:
            initial_params_extended = initial_params
        
        # 确保初始参数在边界内
        initial_params_extended = np.array(initial_params_extended)
        for i, (low, high) in enumerate(bounds):
            initial_params_extended[i] = np.clip(initial_params_extended[i], low, high)
        
        # 定义目标函数（6参数）
        def objective_extended(params_6):
            return self.unified_objective_function(params_6, freq, Z_exp)
        
        # 优化策略：先用L-BFGS-B快速优化，失败则用differential_evolution
        from scipy.optimize import minimize
        result = minimize(
            objective_extended,
            initial_params_extended,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            params = result.x
        else:
            # 如果L-BFGS-B失败，回退到differential_evolution
            print(f"    L-BFGS-B失败，回退到differential_evolution")
            result_de = differential_evolution(
                objective_extended,
                bounds, maxiter=100, popsize=15,
                tol=1e-4, polish=True, init='random'
            )
            params = result_de.x
        
        # 计算拟合质量
        Z_fit = self.impedance_from_ABC(freq, params)
        r_squared = self.calculate_r_squared(Z_exp, Z_fit)
        
        # 计算roughness因子（基于C_dl）
        A, B, C, C_dl, R_s = params[:5]  # 取前5个参数
        standard_c_dl = 25e-6  # 25 uF/cm2
        if C_dl > 1e-12:
            estimated_roughness = C_dl / standard_c_dl
            estimated_roughness = max(0.1, min(estimated_roughness, 1000.0))
        else:
            estimated_roughness = 25.0  # 默认值
        
        return {
            'params': params,
            'r_squared': r_squared,
            'iterations': 1,
            'roughness_factor': estimated_roughness
        }
    
    def fit_single_sample(self, freq, Z_exp, file_name, use_fixed_eta=True, use_unified_fitting=True):
        """处理单个样本的完整拟合流程
        
        参数:
            freq: 频率数组
            Z_exp: 实验阻抗数据
            file_name: 文件名
            use_fixed_eta: 是否使用固定过电位（保留参数以兼容性）
            use_unified_fitting: 是否使用统一目标函数拟合（保留参数以兼容性）
        """
        # 数据验证
        is_valid, message = self.validate_eis_data(freq, Z_exp)
        if not is_valid:
            print(f"  跳过: {message}")
            return None
        
        # 过滤掉高频(>100Hz)且虚部大于0的数据点
        # 这些点通常是仪器噪声或测量误差（电感效应）
        mask_valid = ~((freq > 100) & (Z_exp.imag > 0))
        freq_filtered = freq[mask_valid]
        Z_exp_filtered = Z_exp[mask_valid]
        
        n_removed = len(freq) - len(freq_filtered)
        if n_removed > 0:
            print(f"  过滤掉 {n_removed} 个高频(>100Hz)且虚部>0的点")
        print(f"  过滤后数据点数: {len(freq_filtered)} (原始: {len(freq)})")
        
        # 获取Rs初始值（如果有）作为初始猜测
        rs_initial = self.get_fixed_rs(file_name)
        if rs_initial is not None:
            print(f"  Rs初始值: {rs_initial:.2f} Ohm (优化范围: 0.0001-10 Ohm)")
        
        # 使用starting_point_generator生成初始参数
        initial_params_list = self.get_initial_parameters_from_generator(freq_filtered, Z_exp_filtered, file_name)
        
        # 尝试所有starting points，选择最佳结果
        best_fit_result = None
        best_r_squared = -float('inf')
        
        n_initial_tries = len(initial_params_list)
        print(f"  尝试 {n_initial_tries} 个不同的初始参数点...")
        
        for i, initial_params in enumerate(initial_params_list):
            # 使用单次优化拟合（传递Rs初始值作为约束）
            fit_result = self.fit_single_optimization(freq_filtered, Z_exp_filtered, initial_params, rs_initial=rs_initial)
            
            if fit_result is not None and fit_result['r_squared'] > best_r_squared:
                best_r_squared = fit_result['r_squared']
                best_fit_result = fit_result
        
        # 如果R2 < 0.99，再尝试100次随机初始化
        if best_r_squared < 0.99:
            print(f"  R2={best_r_squared:.4f} < 0.99, 再尝试100次随机初始化...")
            additional_params = self.point_generator.generate_diverse_points(
                n_points=100,
                estimated_C_dl=1e-3,
                estimated_R_s=rs_initial if rs_initial else 5.0
            )
            
            for i, initial_params in enumerate(additional_params):
                fit_result = self.fit_single_optimization(freq_filtered, Z_exp_filtered, initial_params, rs_initial=rs_initial)
                
                if fit_result is not None and fit_result['r_squared'] > best_r_squared:
                    best_r_squared = fit_result['r_squared']
                    best_fit_result = fit_result
            
            print(f"  额外尝试后最佳R2={best_r_squared:.4f}")
        
        fit_result = best_fit_result
        
        if fit_result is None:
            print("  跳过: 拟合失败")
            return None
        
        # 输出最佳拟合结果
        params = fit_result['params']
        if len(params) == 6:
            A, B, C, C_dl, R_s, W_sigma = params
            has_warburg = True
        else:
            A, B, C, C_dl, R_s = params
            W_sigma = 0
            has_warburg = False
        
        print(f"  拟合参数:")
        print(f"    A = {A:.6e}")
        print(f"    B = {B:.6e}")
        print(f"    C = {C:.6e}")
        print(f"    C_dl = {C_dl:.6e} F/cm2")
        print(f"    R_s = {R_s:.6e} Ohm")
        if has_warburg:
            if W_sigma > 1e-6:
                print(f"    W_sigma = {W_sigma:.6e} Ohm*s**(-1/2) (Warburg扩散)")
            else:
                print(f"    W_sigma = {W_sigma:.6e} Ohm*s**(-1/2) (无扩散限制)")
        
        # 检查参数是否到达边界
        bounds = self.point_generator.get_parameter_bounds(include_extended=has_warburg, rs_initial=rs_initial)
        param_names = ['A', 'B', 'C', 'C_dl', 'R_s']
        if has_warburg:
            param_names += ['W_sigma']
        
        boundary_warnings = []
        for i, (param_name, param_value) in enumerate(zip(param_names, params)):
            low, high = bounds[i]
            if abs(param_value - low) < 1e-10 or (low > 0 and abs(param_value / low - 1) < 0.01):
                boundary_warnings.append(f"    [WARNING] {param_name} 参数到达下界 {low:.2e}")
            elif abs(param_value - high) < 1e-10 or abs(param_value / high - 1) < 0.01:
                boundary_warnings.append(f"    [WARNING] {param_name} 参数到达上界 {high:.2e}")
        
        if boundary_warnings:
            for warning in boundary_warnings:
                print(warning)
        
        print(f"    估算roughness因子: {fit_result['roughness_factor']:.2f}")
        
        # 打印拟合质量
        r_squared = fit_result['r_squared']
        if r_squared >= 0.99:
            quality = "优秀"
        elif r_squared >= 0.95:
            quality = "良好"
        else:
            quality = "需要改进"
        
        print(f"\n拟合总结:")
        print(f"  最终R^2值: {r_squared:.4f}")
        print(f"  拟合质量: {quality}")
    
        # 绘制拟合图（使用原始数据）
        Z_fit = self.impedance_from_ABC(freq, fit_result['params'])
        self.plot_fit_result(freq, Z_exp, Z_fit, file_name, fit_result['r_squared'])
        
        # 构建结果字典（支持6参数Warburg模型）
        params = fit_result['params']
        result_dict = {
            'sample': file_name,
            'r_squared': fit_result['r_squared'],
            'A': params[0],
            'B': params[1],
            'C': params[2],
            'C_dl': params[3],
            'R_s': params[4],
            'iterations': fit_result['iterations'],
            'roughness_factor': fit_result['roughness_factor']
        }
        
        # 如果有Warburg参数，添加到结果中
        if len(params) == 6:
            result_dict['W_sigma'] = params[5]
        else:
            result_dict['W_sigma'] = 0.0
        
        return result_dict

    # ============================================================================
    # 5. 阻抗和模型计算方法
    # ============================================================================
    
    def impedance_from_ABC(self, freq, params):
        """
        计算阻抗，包含Warburg扩散项
        
        模型: Z = Rs + [Zf || Cdl] + Z_warburg
        
        参数顺序: A, B, C, C_dl, R_s, W_sigma (6参数)
        """
        if len(params) == 5:
            # 基础模型（向后兼容）
            A, B, C, C_dl, R_s = params
            W_sigma = 0
        elif len(params) == 6:
            # Warburg模型
            A, B, C, C_dl, R_s, W_sigma = params
        else:
            raise ValueError(f"参数数量错误: {len(params)}, 应为5或6")
        
        omega = 2 * np.pi * freq
        
        # 1. Faradaic导纳和阻抗
        Yf = A + B / (1j * omega + C)
        Zf = 1 / Yf
        
        # 2. 双电层电容阻抗
        Zdl = 1 / (1j * omega * C_dl)
        
        # 3. Zf和Cdl并联
        Z_parallel = (Zf * Zdl) / (Zf + Zdl)
        
        # 4. Warburg扩散阻抗 (无限扩散，OH-积累)
        # W_sigma = 0 表示无扩散限制
        if W_sigma > 1e-6:
            Z_warburg = W_sigma / np.sqrt(omega) * (1 - 1j)
        else:
            Z_warburg = 0
        
        # 总阻抗
        Z_total = R_s + Z_parallel + Z_warburg
        
        return Z_total
    

    
    # ============================================================================
    # 6. 辅助计算函数
    # ============================================================================

    def calculate_abc_from_kinetic_params(self, k1_0, k2_0, k3_0, m, eta, roughness):
        """从动力学参数计算A, B, C参数"""
        from scipy.optimize import fsolve
        
        # 物理常数
        RT_F = self.R * self.T / self.F
        gamma_max = 2e-9
        g = 2.0
        lambda_ = 0.5  # asymmetrical factor for Frumkin parameter
        beta1 = 0.5
        beta3 = 0.5
        
        # Surface concentrations ratios (surface concentration / bulk concentration)
        C_H2O_surface_ratio = 1
        C_OH_surface_ratio = 1
        C_H2_surface_ratio = 1
        
        # 调整后的反向速率常数
        k_1_0 = k1_0 / m
        k_2_0 = k2_0 * (m ** 2)
        k_3_0 = k3_0 * m
        
        # 计算与电位相关的速率常数
        k1 = k1_0 * np.exp(-beta1 * eta / RT_F)
        k_1 = k_1_0 * np.exp((1 - beta1) * eta / RT_F)
        k2 = k2_0
        k_2 = k_2_0
        k3 = k3_0 * np.exp(-beta3 * eta / RT_F)
        k_3 = k_3_0 * np.exp((1 - beta3) * eta / RT_F)
        
        # Solve for coverage at steady state
        def coverage_equation(theta):
            v1 = k1 * (1 - theta) * np.exp(-lambda_ * g * theta) * C_H2O_surface_ratio - k_1 * theta * np.exp((1-lambda_) * g * theta) * C_OH_surface_ratio
            v2 = k2 * theta**2 * np.exp(-2*lambda_ *g * theta) - k_2 * (1 - theta)**2 * np.exp(2*(1-lambda_) *g * theta) * C_H2_surface_ratio
            v3 = k3 * theta * np.exp(-lambda_ * g * theta) * C_H2O_surface_ratio - k_3 * (1 - theta) * np.exp((1-lambda_) * g * theta) * C_H2_surface_ratio * C_OH_surface_ratio
            return v1 - 2*v2 - v3
        
        try:
            theta = fsolve(coverage_equation, 0.5, xtol=1e-6)[0]
            theta = np.clip(theta, 0, 1)
        except:
            theta = 0.5
        
        # 计算偏导数
        dk1_dE = -k1 * beta1 / RT_F
        dk_1_dE = k_1 * (1 - beta1) / RT_F
        dk3_dE = -k3 * beta3 / RT_F
        dk_3_dE = k_3 * (1 - beta3) / RT_F
        
        dv1_dtheta = (-k1 * np.exp(-lambda_ * g * theta) * C_H2O_surface_ratio
                       - k1 * (1-theta) * np.exp(-lambda_ * g * theta) * C_H2O_surface_ratio * lambda_ * g
                       - k_1 * np.exp((1-lambda_) * g * theta) * C_OH_surface_ratio
                       - k_1 * theta * np.exp((1-lambda_) * g * theta) * C_OH_surface_ratio * (1-lambda_) * g
        )
                       
        dv2_dtheta = (2 * k2 * theta * np.exp(-2*lambda_ *g * theta) 
                        - 2 * k2 * theta**2 * np.exp(-2*lambda_ *g * theta) * lambda_ *g
                        + 2 * k_2 * C_H2_surface_ratio * (1 - theta) * np.exp(2*(1-lambda_) *g * theta) * C_H2_surface_ratio
                        - 2 * k_2 * (1 - theta)**2 * np.exp(2*(1-lambda_) *g * theta) * C_H2_surface_ratio * 2*(1-lambda_) *g
        )
                       
        dv3_dtheta = (k3 * np.exp(-lambda_ * g * theta) * C_H2O_surface_ratio
                       - k3 * theta * np.exp(-lambda_ * g * theta) * C_H2O_surface_ratio * lambda_ * g
                       + k_3 * np.exp((1-lambda_) * g * theta) * C_H2_surface_ratio * C_OH_surface_ratio
                       - k_3 * (1 - theta) * np.exp((1-lambda_) * g * theta) * C_H2_surface_ratio * C_OH_surface_ratio * (1-lambda_) * g
        )
        
        dv1_dE = (dk1_dE * (1 - theta) * np.exp(-lambda_ * g * theta) * C_H2O_surface_ratio 
                    - dk_1_dE * theta * np.exp((1-lambda_) * g * theta) * C_OH_surface_ratio
        )
        dv2_dE = 0
        dv3_dE = (dk3_dE * theta * np.exp(-lambda_ * g * theta) * C_H2O_surface_ratio 
                    - dk_3_dE * (1 - theta) * np.exp((1-lambda_) * g * theta) * C_H2_surface_ratio * C_OH_surface_ratio
        )
        
        # 计算A, B, C参数
        A = -self.F * (dv1_dE + dv3_dE) * gamma_max * roughness
        B = -self.F * (dv1_dtheta + dv3_dtheta) * (dv1_dE - 2*dv2_dE - dv3_dE) * gamma_max * roughness
        C = -(dv1_dtheta - 2*dv2_dtheta - dv3_dtheta)
        
        return A, B, C


    def get_fixed_overpotential_from_lsv(self, sample_name):
        """从LSV曲线读取固定过电位，在-10mA cm-2时的电位(vs Ag/AgCl)"""
        # 处理样本名称
        base_name = sample_name.replace('.json', '')
        
        # 查找对应的JSON文件
        json_files = os.listdir(self.aggregated_json_dir)
        matching_file = None
        
        for file in json_files:
            if file.startswith(base_name) and file.endswith('.json'):
                matching_file = file
                break
        
        if not matching_file:
            print(f"  警告: 未找到样本 {sample_name} 的LSV数据文件")
            return None
        
        try:
            # 加载JSON数据
            file_path = os.path.join(self.aggregated_json_dir, matching_file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'LSV' not in data:
                print(f"  警告: 文件 {matching_file} 中没有LSV数据")
                return None
            
            # 提取LSV数据
            lsv = data['LSV']
            E = np.array([pt['E(V)'] for pt in lsv])
            I = np.array([pt['i(A)'] for pt in lsv])
            
            # 转换为电流密度 (mA/cm2)
            current_density = I * 1000 / self.working_electrode_area
            target_current = -10.0  # mA/cm2 (HER阴极电流为负值)
            
            # 找到第一个比目标电流更负的值
            for i, current in enumerate(current_density):
                if current <= target_current:
                    E_at_target = E[i]
                    # 计算过电位：过电位 = E + 0.098 + 0.0592*14
                    overpotential = E_at_target + 0.098 + 0.0592*14
                    
                    print(f"  从LSV获取过电位: {overpotential:.4f} V (电流: {current:.2f} mA/cm2)")
                    return overpotential
            
            print(f"  警告: 无法在LSV数据中找到比-10mA/cm2更负的电流值，使用默认过电位-0.6V")
            return -0.6
            
        except Exception as e:
            print(f"  错误: 从LSV数据计算过电位时出错: {str(e)}")
            return None

    def calculate_local_ph_correction(self, current_density, diffusion_layer_thickness=30e-6):
        """计算传质限制导致的局部pH变化和过电位修正
        
        参数:
            current_density: 电流密度 (mA/cm2)
            diffusion_layer_thickness: 扩散层厚度 (m)，默认30微米
            
        返回:
            ph_correction: pH变化量
            eta_correction: 过电位修正量 (V)
        """
        # 物理常数
        F = self.F  # 法拉第常数 (C/mol)
        R = self.R  # 气体常数 (J/(mol·K))
        T = self.T  # 温度 (K)
        
        # 氢氧根扩散系数 (m2/s)
        D_OH = 5.3e-9  # 25°C下OH-的扩散系数
        
        # 体相pH (假设为14，即[OH-] = 1 M)
        bulk_ph = 14.0
        bulk_oh_concentration = 10**(bulk_ph - 14)  # mol/L = 1 M
        
        # 电流密度转换为A/m2
        current_density_A_m2 = current_density * 10  # mA/cm2 -> A/m2
        
        # 计算OH-的表面浓度变化
        # 根据Fick第一定律: J = -D * dC/dx
        # 在稳态下: J = D * (C_surface - C_bulk) / δ
        # 其中J = I/(n*F)，对于OH-，n=1
        # 注意：HER反应中OH-是产物，所以表面浓度应该高于体相浓度
        oh_flux = abs(current_density_A_m2) / F  # mol/(m2·s)
        
        # 计算表面OH-浓度（HER反应中OH-是产物，表面浓度升高）
        # 注意单位换算：oh_flux是mol/(m2·s)，diffusion_layer_thickness是m，D_OH是m2/s
        # 所以 oh_flux * diffusion_layer_thickness / D_OH 的单位是 mol/m3
        # 需要转换为 mol/L，即除以1000
        concentration_change = (oh_flux * diffusion_layer_thickness / D_OH) / 1000  # mol/L
        surface_oh_concentration = bulk_oh_concentration + concentration_change
        surface_oh_concentration = max(surface_oh_concentration, 1e-10)  # 避免负值
        
        # 计算局部pH
        surface_ph = 14 + np.log10(surface_oh_concentration)
        ph_correction = surface_ph - bulk_ph
        
        # 计算过电位修正
        # 在HER反应中，OH-是产物，表面OH-浓度增加会促进反应
        # 根据Nernst方程: E = E° - (RT/F) * ln([OH-])
        # 当OH-浓度增加时，过电位应该减小（变得更负）
        # 由于过电位为负值，要使其更负，传质修正应该是正值
        eta_correction = (R * T / F) * np.log(surface_oh_concentration / bulk_oh_concentration)
        
        return ph_correction, eta_correction

    # ============================================================================
    # 9. 历史结果管理
    # ============================================================================
    
    # def save_historical_result(self, sample_name, params, r_squared, features):
    #     """将拟合结果保存到历史数据库"""
    #     result = {
    #         'sample': sample_name,
    #         'A': params[0],
    #         'B': params[1],
    #         'C': params[2],
    #         'C_dl': params[3],
    #         'R_s': params[4],
    #         'r_squared': r_squared,
    #         'features': str(features)
    #     }
        
    #     self.historical_results.append(result)
        
    #     # 保存到CSV
    #     df = pd.DataFrame(self.historical_results)
    #     df.to_csv(HISTORICAL_RESULTS_PATH, index=False)
    
    def find_similar_samples(self, features, n_similar=5):
        """基于特征从历史结果中查找相似样本"""
        if not self.historical_results:
            return []
        
        # 简单的基于关键特征的相似度计算
        similarities = []
        
        for hist_result in self.historical_results:
            # 从历史结果中提取特征
            try:
                hist_features = {
                    'mean_Z_mag': hist_result.get('R_s', 1.0) * 10,  # 近似值
                    'log_freq_range': 4.0  # 典型范围
                }
                
                # 计算相似度得分
                score = 0
                for key in ['mean_Z_mag', 'log_freq_range']:
                    if key in features and key in hist_features:
                        score += abs(features[key] - hist_features[key])
                
                similarities.append((score, hist_result))
            except:
                continue
        
        # 按相似度排序并返回最佳匹配
        similarities.sort(key=lambda x: x[0])
        return [result for _, result in similarities[:n_similar]]
    
    # ============================================================================
    # 10. 可视化和绘图
    # ============================================================================
    
    def plot_fit_result(self, freq, Z_exp, Z_fit, sample_name, r_squared):
        """绘制EIS拟合结果"""
        try:
            # 确保绘图目录存在
            plot_dir = getattr(self, 'plot_dir', 'eis_fit_plots')
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            
            # 创建新图形
            plt.figure(figsize=(12, 10))
        
            # 绘制Nyquist图 (使用负虚部，传统方法)
            plt.subplot(2, 1, 1)
            plt.scatter(np.real(Z_exp), -np.imag(Z_exp), marker='o', color='blue', label='Experimental Data')
            plt.plot(np.real(Z_fit), -np.imag(Z_fit), 'r-', label=f'Fitting Result (R^2 = {r_squared:.4f})')
            plt.xlabel('Real Part Z\' (Ohm)')
            plt.ylabel('Imaginary Part -Z" (Ohm)')
            plt.title(f'Nyquist Plot of {sample_name}')
            plt.legend()
            plt.grid(True)
            
            # 绘制Bode图 (幅频特性)
            plt.subplot(2, 2, 3)
            plt.loglog(freq, np.abs(Z_exp), 'bo', label='Experimental Data')
            plt.loglog(freq, np.abs(Z_fit), 'r-', label=f'Fitting Result')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Impedance Magnitude |Z| (Ohm)')
            plt.title(f'Bode Magnitude Plot of {sample_name}')
            plt.legend()
            plt.grid(True)
            
            # 绘制Bode图 (相频特性)
            plt.subplot(2, 2, 4)
            plt.semilogx(freq, np.angle(Z_exp, deg=True), 'bo', label='Experimental Data')
            plt.semilogx(freq, np.angle(Z_fit, deg=True), 'r-', label=f'Fitting Result')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Phase Angle (deg)')
            plt.title(f'Bode Phase Plot of {sample_name}')
            plt.legend()
            plt.grid(True)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图像
            # 确保plot_dir目录存在
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir, exist_ok=True)
            
            # 替换非法字符和扩展名
            safe_file_name = sample_name.replace('.', '_').replace('\\', '_').replace('/', '_')
            plot_path = os.path.join(plot_dir, f'{safe_file_name}_fit.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  拟合图已保存到: {plot_path}")
        except Exception as e:
            print(f"  [WARNING] 绘图失败: {e}")
            # 确保关闭图形
            plt.close('all')
    
    def plot_iteration_results(self, r_squared_history, sample_name):
        """绘制迭代结果图"""
        try:
            import matplotlib.pyplot as plt
            import os
            
            plots_dir = 'eis_iteration_plots'
            os.makedirs(plots_dir, exist_ok=True)
            
            plot_filename = os.path.join(plots_dir, f'{sample_name}_iteration.png') if sample_name else os.path.join(plots_dir, 'iteration_plot.png')
            
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            
            # R2值变化
            ax1.plot(range(1, len(r_squared_history) + 1), r_squared_history, 'o-', color='blue', label='R^2值')
            ax1.axhline(y=0.99, color='r', linestyle='--', label='优秀拟合阈值(0.99)')
            ax1.set_xlabel('迭代次数')
            ax1.set_ylabel('R^2值')
            ax1.set_title('EIS拟合质量随迭代变化')
            ax1.grid(True)
            ax1.legend()
            ax1.set_ylim(max(0, min(r_squared_history) - 0.05), 1.0)
            
            plt.tight_layout()
            plt.savefig(plot_filename)
            plt.close()
            
            print(f"  迭代过程可视化图已保存到: {plot_filename}")
        except Exception as e:
            print(f"  生成迭代过程可视化图时出错: {str(e)}")
    
    # ============================================================================
    # 11. 分析和统计
    # ============================================================================

    def print_final_summary(self, r_squared):
        """打印最终总结"""
        print("\n拟合总结:")
        print(f"  最终R^2值: {r_squared:.4f}")
        if r_squared > 0.99:
            print("  拟合质量: 优秀")
        elif r_squared > 0.95:
            print("  拟合质量: 良好")
        else:
            print("  拟合质量: 需要改进")
    
    def print_fitting_summary(self, results, files):
        """打印拟合结果总结"""
        if not results:
            print("没有成功的拟合结果可保存。")
            return
        
        df = pd.DataFrame(results)
        successful_fits = len(results)
        
        print(f"\n{'='*80}")
        print(f"完成: {successful_fits}/{len(files)} 个成功拟合")
        print(f"平均R^2: {df['r_squared'].mean():.4f}")
        print(f"最佳R^2: {df['r_squared'].max():.4f}")
        print(f"最差R^2: {df['r_squared'].min():.4f}")
        print(f"平均迭代次数: {df['iterations'].mean():.1f}")
        print(f"结果保存到: {CSV_PATH}")
        
    
    # ============================================================================
    # 12. 主要接口方法
    # ============================================================================
    
    
    def run_fitting(self, use_fixed_eta=False, use_unified_fitting=True):
        """在所有样本上运行独立拟合（重构版本）
        
        参数:
            use_fixed_eta: 是否使用从LSV获取的固定过电位
            use_unified_fitting: 是否使用统一目标函数拟合（True）或迭代拟合（False）
        """
        fitting_method = "统一目标函数拟合" if use_unified_fitting else "迭代拟合"
        print(f"独立拟合策略 (重构版 - {fitting_method})")
        print("=" * 80)
        print("特点:")
        print("• 将A, B, C, C_dl, R_s对每个样本都视为独立变量")
        if use_unified_fitting:
            print("• 统一目标函数：同时优化EIS拟合和电流密度匹配")
            print("• EIS拟合权重: 70%, 电流密度权重: 30%")
            print("• 非迭代拟合，使用差分进化算法")
        else:
            print("• 迭代eta拟合（最多150次迭代）")
            print("• 基于电流比值的参数调整")
        print("• 主动学习机制，利用历史拟合结果优化初始参数")
        print("• 增强的初始参数估计方法")
        print("• 支持从standardized_json文件加载数据")
        print(f"• 使用固定过电位: {'是' if use_fixed_eta else '否'}")
        print("=" * 80)
        
        # 初始化数据加载器
        standardized_json_dir = 'standardized_json'
        if not os.path.exists(standardized_json_dir):
            print(f"错误: standardized_json目录 '{standardized_json_dir}' 不存在！")
            print("请确保您的JSON数据文件在正确的目录中")
            return
        
        self.data_loader = ImprovedEISDataLoader(data_dir=DATA_DIR, json_dir=standardized_json_dir)
        # 完全禁用历史数据加载，确保使用纯初始参数
        # self.load_historical_results()
        
        # 获取文件列表
        try:
            files = self.data_loader.get_all_json_files()
            if not files:
                print("错误: 没有找到JSON数据文件！")
                return
        except Exception as e:
            print(f"错误: 无法获取JSON数据文件列表！错误: {str(e)}")
            return
        
        # 处理所有样本
        results = []
        for i, fname in enumerate(files, 1):
            print(f"\n处理 {i}/{len(files)}: {fname}")
            
            try:
                # 加载数据
                file_name = os.path.basename(fname)
                data = self.data_loader.load_eis_data_from_json(fname)
                if data is None:
                    print(f"  跳过: 无法加载JSON数据")
                    continue
                
                freq = data['frequency']
                Z_exp = data['z_complex']
                
                # 处理单个样本
                result = self.fit_single_sample(freq, Z_exp, file_name, use_fixed_eta, use_unified_fitting)
                if result is not None:
                    results.append(result)
                    print(f"  [OK] R^2: {result['r_squared']:.4f}")
                    print(f"  [OK] 迭代次数: {result['iterations']}")
                    print(f"  [OK] roughness: {result['roughness_factor']:.2f}")
                
            except Exception as e:
                print(f"  跳过: 处理错误: {str(e)}")
                continue
        
        # 保存结果并打印总结
        if results:
            df = pd.DataFrame(results)
            df.to_csv(CSV_PATH, index=False)
            self.print_fitting_summary(results, files)
        else:
            print("没有成功的拟合结果可保存。")


if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='独立拟合EIS数据')
    parser.add_argument('--fixed-eta', action='store_true', 
                        help='使用从LSV曲线获取的固定过电位, 而不是拟合过电位')
    parser.add_argument('--roughness-factor', type=float, default=10.0, 
                        help='粗糙因子 (默认: 10.0)')
    parser.add_argument('--test-overpotential', action='store_true', 
                        help='测试get_fixed_overpotential_from_lsv方法')
    parser.add_argument('--use-iterative', action='store_true', 
                        help='使用迭代拟合方法而不是统一目标函数拟合')
    parser.add_argument('--eis-weight', type=float, default=0.999, 
                        help='EIS拟合权重 (默认: 0.999)')
    parser.add_argument('--current-weight', type=float, default=0.001, 
                        help='电流密度权重 (默认: 0.001)')
    parser.add_argument('--debug-mass-transport', action='store_true', 
                        help='启用传质修正调试输出')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 初始化拟合器
    fitter = IndependentFitter(debug_mass_transport=args.debug_mass_transport)
    
    # 如果启用了过电位测试
    if args.test_overpotential:
        print("\n===== 测试get_fixed_overpotential_from_lsv方法 =====")
        
        # 测试一个具体的样本名称（带.json扩展名）
        sample_name = "Co-Co.json"
        print(f"\n测试1: 样本名称 = {sample_name}")
        overpotential = fitter.get_fixed_overpotential_from_lsv(sample_name)
        print(f"  测试结果: 过电位 = {overpotential} V")
        
        # 测试另一个样本名称格式（没有.json扩展名）
        sample_name2 = "Co-Cr"
        print(f"\n测试2: 样本名称 = {sample_name2}")
        overpotential2 = fitter.get_fixed_overpotential_from_lsv(sample_name2)
        print(f"  测试结果: 过电位 = {overpotential2} V")
        
        print("\n===== 测试完成 =====")
        exit()  # 测试结束后退出程序
    
    # Fit all samples
    print("===== Fitting All Samples =====")
    
    # Clean plot directory
    import shutil
    import time
    plot_dir = 'eis_fit_plots'
    if os.path.exists(plot_dir):
        try:
            shutil.rmtree(plot_dir, ignore_errors=True)
            time.sleep(1)
            print(f"[OK] Cleared plot directory: {plot_dir}")
        except Exception as e:
            print(f"[WARNING] Failed to clear directory: {e}")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"[OK] Created plot directory: {plot_dir}")
    
    # Get all CSV files
    import os
    eis_dir = DATA_DIR
    all_files = [f for f in os.listdir(eis_dir) if f.lower().endswith(('.csv', '.txt'))]
    selected_samples = [os.path.splitext(f)[0] for f in all_files]

    
    print(f"Total samples: {len(selected_samples)}")
    print(f"Starting fitting process...\n")
    
    # Simple serial fitting (NO parallel code)
    results = []
    for i, sample_name in enumerate(selected_samples, 1):
        print(f"\n=== Fitting sample {i}/{len(selected_samples)}: {sample_name} ===")
        
        try:
            data = fitter.data_loader.load_sample_by_name(sample_name, from_json=False)
            
            if data is None:
                print(f"[SKIP] Cannot load data for: {sample_name}")
                continue
                
            freq = data['frequency']
            Z_exp = data['z_complex']
            
            print(f"  Data points: {len(freq)}")
            print(f"  Freq range: {freq.min():.1f} - {freq.max():.1f} Hz")
            
            # Run fitting
            result = fitter.fit_single_sample(freq, Z_exp, sample_name, use_fixed_eta=True, use_unified_fitting=True)
            
            if result:
                print(f"[OK] Fitting succeeded!")
                print(f"  R2: {result['r_squared']:.4f}")
                print(f"  Params: A={result['A']:.2e}, B={result['B']:.2e}, C={result['C']:.2e}")
                print(f"          Rs={result['R_s']:.2f} Ohm, C_dl={result['C_dl']:.2e} F")
                results.append(result)
            else:
                print("[FAIL] Fitting failed")
                
        except Exception as e:
            print(f"[ERROR] Exception for {sample_name}: {e}")
    
    # Summary
    print(f"\n===== Fitting Complete =====")
    print(f"Successful fits: {len(results)}/{len(selected_samples)}")
    
    if results:
        r_squared_values = [r['r_squared'] for r in results]
        print(f"R2 statistics:")
        print(f"  Mean R2: {np.mean(r_squared_values):.4f}")
        print(f"  Max R2: {np.max(r_squared_values):.4f}")
        print(f"  Min R2: {np.min(r_squared_values):.4f}")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(CSV_PATH, index=False)
        print(f"\n[OK] Results saved to: {CSV_PATH}")
        print(f"Columns: {list(df.columns)}")
    
    print("\n===== Done =====")

# ============================================================================
# 测试小半圆样品的拟合（仅在直接运行时执行）
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # 检查是否要运行小半圆测试
    if len(sys.argv) > 1 and sys.argv[1] == '--test-semicircle':
        print("\n" + "="*80)
        print("测试小半圆样品拟合（权重×20）")
        print("="*80)
        
        # 有高频小半圆未拟合的样品
        semicircle_samples = [
            'Y-Y', 'Y-Sc', 'Y-Mn', 'W-V', 'Sc-Y', 'Sc-Cd', 'Mn-Sc', 'Mn-Cr', 
            'La-V', 'Ga-Mn', 'Cr-Zn', 'Cr-Mn', 'Ce-Zn', 'Ce-La', 'Ce-Cr', 'Cd-La',
            'Co-Mo0', 'Co-Mo3', 'Co-Ni12', 'Eu-Cr', 'Ga-Cd', 'La-Ni', 'V-Ce', 'Zn-Mn'
        ]
        
        # 创建测试输出目录
        test_plot_dir = 'semicircle_samples_refit_plots'
        os.makedirs(test_plot_dir, exist_ok=True)
        
        # 创建拟合器并修改plot_dir
        fitter = IndependentFitter()
        fitter.plot_dir = test_plot_dir
        
        # 拟合指定样品
        test_results = []
        for i, sample in enumerate(semicircle_samples, 1):
            print(f"\n[{i}/{len(semicircle_samples)}] 拟合: {sample}")
            
            file_path = os.path.join(DATA_DIR, f'{sample}.csv')
            if not os.path.exists(file_path):
                print(f"  [SKIP] 文件不存在")
                continue
            
            try:
                # 读取数据
                import pandas as pd
                df = pd.read_csv(file_path)
                freq = df['freq(Hz)'].values
                Z_real = df['Z_re(Ohm)'].values
                Z_imag = df['Z_im(Ohm)'].values
                Z_exp = Z_real + 1j * Z_imag
                
                # 拟合
                result = fitter.fit_single_sample(freq, Z_exp, file_path)
                
                if result:
                    print(f"  [OK] R2={result['r_squared']:.4f}, Rs={result['R_s']:.4f}")
                    test_results.append(result)
                else:
                    print(f"  [FAIL]")
            except Exception as e:
                print(f"  [ERROR] {e}")
        
        print(f"\n完成: {len(test_results)}/{len(semicircle_samples)}")
        print(f"图片已保存到: {test_plot_dir}/")
    