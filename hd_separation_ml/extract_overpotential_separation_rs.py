#!/usr/bin/env python3
"""
提取过电位、分离系数和Rs值
1. 从Et_potential_means_1100_1200.xlsx提取电压并计算过电位
2. 从extracted_separation_coefficients.csv提取分离系数
3. 从all_substances.json提取Rs值
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 配置路径
DATA_DIR = './ML-20250930(1)/ML-20250926'
OUTPUT_FILE = 'sample_overpotentials_new.csv'

def extract_rs_from_json():
    """从all_substances.json提取Rs值"""
    json_file = os.path.join(DATA_DIR, 'all_substances.json')
    
    print("=" * 80)
    print("从all_substances.json提取Rs值")
    print("=" * 80)
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rs_mapping = {}
    for entry in data:
        substance1 = entry['substance1']
        substance2 = entry['substance2']
        ir_value = entry['ir_value']
        
        # 创建正向和反向样本名
        forward_name = f"{substance1}-{substance2}"
        reverse_name = f"{substance2}-{substance1}"
        
        rs_mapping[forward_name] = ir_value
        rs_mapping[reverse_name] = ir_value
    
    print(f"[OK] 已加载 {len(rs_mapping)//2} 个样本的Rs值")
    print(f"     总共生成 {len(rs_mapping)} 个映射（包含正反向）")
    
    return rs_mapping

def extract_potential_from_excel():
    """从Et_potential_means_1100_1200.xlsx提取电压"""
    excel_file = os.path.join(DATA_DIR, 'Et_potential_means_1100_1200.xlsx')
    
    print("\n" + "=" * 80)
    print("从Et_potential_means_1100_1200.xlsx提取电压")
    print("=" * 80)
    
    df = pd.read_excel(excel_file)
    print(f"[OK] Excel数据形状: {df.shape}")
    print(f"     列名: {df.columns.tolist()}")
    
    # 提取文件名和电压
    potential_mapping = {}
    for _, row in df.iterrows():
        filename = row['filename']
        potential = row['potential_mean_1100_1200']
        
        # 如果电压是有效的（非NaN）
        if not pd.isna(potential):
            potential_mapping[filename] = potential
    
    print(f"[OK] 已提取 {len(potential_mapping)} 个样本的电压")
    
    return potential_mapping

def extract_separation_coefficients():
    """从extracted_separation_coefficients.csv提取分离系数"""
    csv_file = os.path.join(DATA_DIR, 'extracted_separation_coefficients.csv')
    
    print("\n" + "=" * 80)
    print("从extracted_separation_coefficients.csv提取分离系数")
    print("=" * 80)
    
    df = pd.read_csv(csv_file)
    print(f"[OK] 分离系数数据形状: {df.shape}")
    print(f"     列名: {df.columns.tolist()}")
    
    # 提取分离系数
    separation_mapping = {}
    for _, row in df.iterrows():
        sample = row['sample'].replace('.json', '')  # 移除.json后缀
        coef = row['separation_coefficient']
        separation_mapping[sample] = coef
    
    print(f"[OK] 已提取 {len(separation_mapping)} 个样本的分离系数")
    
    return separation_mapping

def calculate_overpotential(E, Rs):
    """
    计算过电位
    公式: overpotential = E + 0.098 + 0.0592*14 + Rs*0.05
    
    参数:
        E: 实际电压 (V)
        Rs: 串联电阻 (Ohm)
    
    返回:
        overpotential (V)
    """
    # E是H/D分离时的实际电压
    # 0.098: 标准氢电极参考电位
    # 0.0592*14: Nernst校正 (pH=14)
    # Rs*0.05: IR降校正 (0.05A电流)
    
    overpotential = E + 0.098 + 0.0592 * 14 + Rs * 0.05
    return overpotential

def main():
    """主函数"""
    # 提取三个数据源
    rs_mapping = extract_rs_from_json()
    potential_mapping = extract_potential_from_excel()
    separation_mapping = extract_separation_coefficients()
    
    # 合并数据
    print("\n" + "=" * 80)
    print("合并数据并计算过电位")
    print("=" * 80)
    
    results = []
    matched_count = 0
    missing_rs = []
    missing_potential = []
    
    for sample_name in separation_mapping.keys():
        # 获取分离系数
        separation_coef = separation_mapping.get(sample_name)
        
        # 获取Rs值
        rs = rs_mapping.get(sample_name)
        if rs is None:
            missing_rs.append(sample_name)
            continue
        
        # 获取电压
        potential = potential_mapping.get(sample_name)
        if potential is None:
            missing_potential.append(sample_name)
            continue
        
        # 计算过电位
        overpotential = calculate_overpotential(potential, rs)
        
        results.append({
            'sample': sample_name,
            'E_actual(V)': potential,
            'Rs(Ohm)': rs,
            'overpotential(V)': overpotential,
            'separation_coefficient': separation_coef
        })
        matched_count += 1
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 保存到CSV
    df_results.to_csv(OUTPUT_FILE, index=False)
    
    print(f"[OK] 成功匹配 {matched_count}/{len(separation_mapping)} 个样本")
    print(f"     缺少Rs值的样本: {len(missing_rs)} 个")
    if missing_rs:
        print(f"     示例: {missing_rs[:5]}")
    print(f"     缺少电压值的样本: {len(missing_potential)} 个")
    if missing_potential:
        print(f"     示例: {missing_potential[:5]}")
    
    print(f"\n[OK] 结果已保存到: {OUTPUT_FILE}")
    print(f"     列名: {df_results.columns.tolist()}")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("数据统计")
    print("=" * 80)
    print(f"过电位范围: {df_results['overpotential(V)'].min():.4f} ~ {df_results['overpotential(V)'].max():.4f} V")
    print(f"过电位均值: {df_results['overpotential(V)'].mean():.4f} V")
    print(f"过电位中位数: {df_results['overpotential(V)'].median():.4f} V")
    
    print(f"\nRs范围: {df_results['Rs(Ohm)'].min():.2f} ~ {df_results['Rs(Ohm)'].max():.2f} Ohm")
    print(f"Rs均值: {df_results['Rs(Ohm)'].mean():.2f} Ohm")
    
    print(f"\n分离系数范围: {df_results['separation_coefficient'].min():.1f} ~ {df_results['separation_coefficient'].max():.1f}")
    print(f"分离系数均值: {df_results['separation_coefficient'].mean():.2f}")
    print(f"分离系数中位数: {df_results['separation_coefficient'].median():.2f}")
    
    # 绘制分布图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 过电位分布
    axes[0, 0].hist(df_results['overpotential(V)'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Overpotential (V)', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Overpotential Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rs分布
    axes[0, 1].hist(df_results['Rs(Ohm)'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Rs (Ohm)', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_title('Rs Distribution', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 分离系数分布
    axes[1, 0].hist(df_results['separation_coefficient'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Separation Coefficient', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Separation Coefficient Distribution', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 过电位 vs 分离系数
    axes[1, 1].scatter(df_results['overpotential(V)'], df_results['separation_coefficient'], alpha=0.5, s=20)
    axes[1, 1].set_xlabel('Overpotential (V)', fontsize=12)
    axes[1, 1].set_ylabel('Separation Coefficient', fontsize=12)
    axes[1, 1].set_title('Overpotential vs Separation Coefficient', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overpotential_separation_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] 分布图已保存: overpotential_separation_distribution.png")
    
    # 显示前几行数据
    print("\n" + "=" * 80)
    print("数据预览（前10行）")
    print("=" * 80)
    print(df_results.head(10).to_string())

if __name__ == "__main__":
    main()
