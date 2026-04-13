
import os
import numpy as np
import pandas as pd
import json

class ImprovedEISDataLoader:
    def __init__(self, data_dir=None, json_dir=None):
        self.data_dir = data_dir
        self.json_dir = json_dir
        self.formats = {
            '5_column': {'freq_col': 0, 'z_prime_col': 1, 'z_double_prime_col': 2},
            '9_column': {'freq_col': 0, 'z_prime_col': 4, 'z_double_prime_col': 5}
        }
    
    def detect_file_format(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith('#'):
                        continue
                    if 'Hz' in stripped or 'Ohm' in stripped:
                        continue
                    try:
                        cols = stripped.split()
                        if len(cols) >= 8:
                            return '9_column'
                        else:
                            return '5_column'
                    except:
                        continue
            return '5_column'
        except:
            return '5_column'
    
    def load_eis_data_from_txt(self, file_path):
        try:
            # Process data with simplified column handling
            frequency = []
            z_prime = []
            z_double_prime = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith('#'):
                        continue
                    if 'Hz' in stripped or 'Ohm' in stripped:
                        continue
                    try:
                        cols = stripped.split()
                        if len(cols) >= 3:
                            # Assume first three columns are frequency, Z', Z''
                            freq = float(cols[0])
                            z_re = float(cols[1])
                            z_im = float(cols[2])
                            
                            frequency.append(freq)
                            z_prime.append(z_re)
                            z_double_prime.append(z_im)
                    except:
                        continue
            
            if len(frequency) < 3:
                print(f"Insufficient valid data points in {file_path}")
                return None
            
            # Convert to numpy arrays
            frequency = np.array(frequency)
            z_prime = np.array(z_prime)
            z_double_prime = np.array(z_double_prime)
            
            # Data validation
            valid = (frequency > 0) & np.isfinite(frequency) & np.isfinite(z_prime) & np.isfinite(z_double_prime)
            frequency = frequency[valid]
            z_prime = z_prime[valid]
            z_double_prime = z_double_prime[valid]
            
            # Create complex impedance
            z_complex = z_prime + 1j * z_double_prime
            
            return {
                'frequency': frequency,
                'z_prime': z_prime,
                'z_double_prime': z_double_prime,
                'z_complex': z_complex
            }
        except Exception as e:
            print(f"Error loading EIS data from {file_path}: {e}")
            return None
            
    def load_eis_data_from_json(self, file_path):
        """从JSON文件加载EIS数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否包含EIS数据
            if 'EIS' not in data:
                return None
            
            eis_data = data['EIS']
            if not eis_data:
                return None
            
            # 检查数据格式并提取频率和阻抗数据
            if isinstance(eis_data, list):
                # 旧格式：数组格式
                freq = np.array([pt['Frequency(Hz)'] for pt in eis_data])
                z_prime = np.array([pt["Z'(Ohm)"] for pt in eis_data])
                z_double_prime = np.array([pt["Z''(Ohm)"] for pt in eis_data])
            elif isinstance(eis_data, dict):
                # 新格式：对象格式
                freq = np.array(eis_data['frequency'])
                z_prime = np.array(eis_data['z_real'])
                z_double_prime = np.array(eis_data['z_imag'])
            else:
                print(f"未知的EIS数据格式: {type(eis_data)}")
                return None
            
            # 数据验证
            valid = (freq > 0) & np.isfinite(freq) & np.isfinite(z_prime) & np.isfinite(z_double_prime)
            freq = freq[valid]
            z_prime = z_prime[valid]
            z_double_prime = z_double_prime[valid]
            
            # 计算复数阻抗（不反转虚部符号，因为绘图时会使用传统的负虚部显示）
            z_complex = z_prime + 1j * z_double_prime
            
            return {
                'frequency': freq,
                'z_prime': z_prime,
                'z_double_prime': z_double_prime,
                'z_complex': z_complex
            }
        except Exception as e:
            print(f"加载JSON文件出错: {e}")
            return None

    def get_all_txt_files(self, directory=None):
        directory = directory or self.data_dir
        if not directory or not os.path.exists(directory):
            return []
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.txt')]
        
    def get_all_json_files(self, directory=None):
        """获取指定目录下的所有JSON文件"""
        directory = directory or self.json_dir
        if not directory or not os.path.exists(directory):
            return []
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.json')]
    
    def load_all_samples(self, max_samples=None, directory=None, from_json=False):
        """加载所有样本数据，可以选择从TXT或JSON文件加载"""
        if from_json:
            files = self.get_all_json_files(directory)
            load_func = self.load_eis_data_from_json
        else:
            files = self.get_all_txt_files(directory)
            load_func = self.load_eis_data_from_txt
            
        if max_samples:
            files = files[:max_samples]
        
        samples = []
        for file_path in files:
            data = load_func(file_path)
            if data:
                samples.append({
                    'file_name': os.path.basename(file_path),
                    'data': data
                })
        
        return samples
        
    def load_eis_data_from_csv(self, file_path):
        """从CSV文件加载EIS数据（新格式）"""
        try:
            df = pd.read_csv(file_path)
            
            # 检查列名是否正确
            expected_cols = ['freq(Hz)', 'Z_re(Ohm)', 'Z_im(Ohm)']
            if not all(col in df.columns for col in expected_cols):
                print(f"CSV文件缺少必要列: {file_path}")
                return None
            
            # 提取数据
            frequency = df['freq(Hz)'].values
            z_prime = df['Z_re(Ohm)'].values
            z_double_prime = df['Z_im(Ohm)'].values
            
            # 数据验证
            valid = (frequency > 0) & np.isfinite(frequency) & np.isfinite(z_prime) & np.isfinite(z_double_prime)
            frequency = frequency[valid]
            z_prime = z_prime[valid]
            z_double_prime = z_double_prime[valid]
            
            if len(frequency) < 3:
                print(f"CSV文件数据点不足: {file_path}")
                return None
            
            # 创建复数阻抗
            # 新CSV数据中Z_im已经是正确的物理符号（容性为正）
            z_complex = z_prime + 1j * z_double_prime
            
            return {
                'frequency': frequency,
                'z_prime': z_prime,
                'z_double_prime': z_double_prime,
                'z_complex': z_complex
            }
        except Exception as e:
            print(f"加载CSV文件出错 {file_path}: {e}")
            return None
    
    def load_sample_by_name(self, sample_name, from_json=False):
        """通过样本名称加载单个样本数据，优先使用CSV格式"""
        # 优先尝试CSV文件（新格式）
        csv_file = os.path.join(self.data_dir, f"{sample_name}.csv")
        if os.path.exists(csv_file):
            return self.load_eis_data_from_csv(csv_file)
        
        if from_json and self.json_dir:
            # 从JSON文件加载
            json_file = os.path.join(self.json_dir, f"{sample_name}.json")
            if os.path.exists(json_file):
                return self.load_eis_data_from_json(json_file)
            else:
                print(f"JSON文件不存在: {json_file}")
                return None
        else:
            # 从TXT文件加载
            txt_file = os.path.join(self.data_dir, f"{sample_name}.txt")
            if os.path.exists(txt_file):
                return self.load_eis_data_from_txt(txt_file)
            else:
                print(f"TXT文件不存在: {txt_file}")
                return None
