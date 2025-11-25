#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DHS数据准备脚本
该脚本用于解压缩和组织DHS数据文件，并进行基本验证
"""

import os
import zipfile
import pandas as pd
import sys
from pathlib import Path

def check_dependencies():
    """检查必要的依赖包是否已安装"""
    required_packages = ['pandas', 'pyreadstat']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"[ERROR] {package} 未安装")
    
    if missing_packages:
        print("\n请安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def get_country_code_from_filename(filename):
    """从文件名中提取国家代码和年份"""
    # DHS文件命名格式: MWHR61DT.ZIP (MW=Malawi, HR=Household Recode, 61=年份代码)
    # 我们将使用前两个字母作为国家代码，后面的数字作为年份
    basename = os.path.basename(filename)
    
    # 提取国家代码（前两个字母）
    country_code = basename[:2].lower()
    
    # 提取年份代码（第3-4个字符）
    year_code = basename[2:4]
    
    # DHS年份代码映射（简化版本）
    year_mapping = {
        '61': '2011',
        '62': '2012', 
        '63': '2013',
        '64': '2014',
        '65': '2015',
        '66': '2016',
        '67': '2017',
        '68': '2018',
        '69': '2019',
        '6A': '2010',
        '6B': '2011',
        '6C': '2012'
    }
    
    year = year_mapping.get(year_code, '20' + year_code)
    
    # 国家代码映射
    country_mapping = {
        'mw': 'malawi',
        'ng': 'nigeria', 
        'rw': 'rwanda',
        'tz': 'tanzania',
        'ug': 'uganda'
    }
    
    country_name = country_mapping.get(country_code, country_code)
    
    return f"{country_name}_{year}"

def unzip_dhs_files():
    """解压缩DHS文件到相应的文件夹"""
    data_dir = Path("data/dhs")
    
    if not data_dir.exists():
        print(f"错误: 目录 {data_dir} 不存在")
        return False
    
    # 创建extracted目录
    extracted_dir = data_dir / "extracted"
    extracted_dir.mkdir(exist_ok=True)
    
    zip_files = list(data_dir.glob("*.ZIP"))
    survey_count = 0
    gps_count = 0
    
    print(f"找到 {len(zip_files)} 个ZIP文件")
    
    for zip_file in zip_files:
        print(f"\n处理: {zip_file.name}")
        
        # 确定文件类型（HR=调查数据，GE=GPS数据）
        if 'HR' in zip_file.name:
            survey_count += 1
            file_type = "survey"
        elif 'GE' in zip_file.name:
            gps_count += 1
            file_type = "gps"
        else:
            file_type = "unknown"
        
        # 创建目标文件夹
        folder_name = get_country_code_from_filename(zip_file.name)
        target_dir = extracted_dir / folder_name
        target_dir.mkdir(exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"  [OK] 解压到: {target_dir}")
        except Exception as e:
            print(f"  [ERROR] 解压失败: {e}")
            return False
    
    print(f"\n解压完成!")
    print(f"- 调查数据文件: {survey_count}")
    print(f"- GPS数据文件: {gps_count}")
    
    return True

def verify_extracted_data():
    """验证解压后的数据"""
    extracted_dir = Path("data/dhs/extracted")
    
    if not extracted_dir.exists():
        print("错误: extracted目录不存在")
        return False
    
    # 查找所有.dta文件
    dta_files = list(extracted_dir.rglob("*.dta"))
    
    print(f"\n找到 {len(dta_files)} 个.dta文件:")
    for dta_file in dta_files:
        print(f"  - {dta_file}")
    
    if not dta_files:
        print("警告: 没有找到.dta文件")
        return False
    
    # 尝试读取第一个.dta文件
    first_dta = dta_files[0]
    print(f"\n验证文件: {first_dta}")
    
    try:
        # 读取前5行数据
        df = pd.read_stata(first_dta)
        # 只显示前5行
        df_sample = df.head(5)
        print("[OK] 成功读取数据")
        print(f"  数据形状: {df.shape}")
        print(f"  前5行数据预览:")
        print(df_sample)
        print(f"  列名: {list(df.columns)}")
        
        # 检查是否包含财富指数相关字段
        wealth_columns = [col for col in df.columns if 'hv271' in col.lower() or 'wealth' in col.lower()]
        if wealth_columns:
            print(f"[OK] 找到财富指数相关列: {wealth_columns}")
        else:
            print("[WARNING] 未找到明显的财富指数列 (hv271)")
            print("  所有列名:", list(df.columns))
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {e}")
        return False

def main():
    """主函数"""
    print("=== DHS数据准备脚本 ===\n")
    
    # 1. 检查依赖
    print("1. 检查依赖包...")
    if not check_dependencies():
        sys.exit(1)
    
    # 2. 解压文件
    print("\n2. 解压DHS文件...")
    if not unzip_dhs_files():
        sys.exit(1)
    
    # 3. 验证数据
    print("\n3. 验证提取的数据...")
    if not verify_extracted_data():
        sys.exit(1)
    
    # 4. 总结
    print("\n=== 处理完成 ===")
    print("[OK] 所有DHS文件已成功解压和组织")
    print("[OK] 数据验证通过")
    print("[OK] 脚本执行完成")

if __name__ == "__main__":
    main()