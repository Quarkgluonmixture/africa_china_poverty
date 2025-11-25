import pandas as pd
import os
import glob
import shapefile

# 国家代码到文件夹名称的映射
country_mapping = {
    'MW': 'malawi',
    'NG': 'nigeria',
    'RW': 'rwanda',
    'TZ': 'tanzania',
    'UG': 'uganda'
}

# 存储所有国家数据的列表
all_countries_data = []

# 遍历5个国家
for country_code, country_name in country_mapping.items():
    print(f"处理 {country_name} ({country_code}) 的数据...")
    
    # 1. 加载调查数据 (.dta)
    survey_pattern = f"data/dhs/extracted/{country_name}_20HR/*.dta"
    survey_files = glob.glob(survey_pattern)
    
    if not survey_files:
        print(f"警告: 未找到 {country_name} 的调查数据文件")
        continue
        
    survey_file = survey_files[0]  # 使用第一个找到的文件
    print(f"读取调查数据: {survey_file}")
    
    # 读取调查数据，只需要的列
    survey_data = pd.read_stata(survey_file, columns=['hv001', 'hv271'])
    
    # 清理数据：转换hv271为float并除以100,000.0（DHS缩放因子）
    survey_data['hv271'] = pd.to_numeric(survey_data['hv271'], errors='coerce') / 100000.0
    
    # 聚合：按hv001分组并计算hv271的平均值
    wealth_by_cluster = survey_data.groupby('hv001')['hv271'].mean().reset_index()
    wealth_by_cluster.columns = ['cluster_id', 'wealth_index']
    
    # 2. 加载GPS数据 (.shp)
    gps_pattern = f"data/dhs/extracted/{country_name}_20GE/*.shp"
    gps_files = glob.glob(gps_pattern)
    
    if not gps_files:
        print(f"警告: 未找到 {country_name} 的GPS数据文件")
        continue
        
    gps_file = gps_files[0]  # 使用第一个找到的文件
    print(f"读取GPS数据: {gps_file}")
    
    # 使用shapefile库读取shapefile
    sf = shapefile.Reader(gps_file)
    
    # 获取字段名和记录
    fields = [field[0] for field in sf.fields[1:]]  # 跳过第一个字段（删除标志）
    records = sf.records()
    
    # 转换为DataFrame
    gps_data = pd.DataFrame(records, columns=fields)
    
    # 选择需要的列
    if 'DHSCLUST' in gps_data.columns and 'LATNUM' in gps_data.columns and 'LONGNUM' in gps_data.columns:
        gps_data = gps_data[['DHSCLUST', 'LATNUM', 'LONGNUM']]
        
        # 过滤：移除LATNUM == 0.0和LONGNUM == 0.0的行（DHS使用0,0表示缺失坐标）
        gps_data = gps_data[(gps_data['LATNUM'] != 0.0) & (gps_data['LONGNUM'] != 0.0)]
    else:
        print(f"警告: {country_name} 的GPS数据中缺少必要的列")
        print(f"可用列: {gps_data.columns.tolist()}")
        continue
    
    # 3. 合并数据
    # 重命名GPS数据的列以便合并
    gps_data = gps_data.rename(columns={'DHSCLUST': 'cluster_id'})
    
    # 合并财富数据和GPS数据
    merged_data = pd.merge(wealth_by_cluster, gps_data, on='cluster_id', how='inner')
    
    # 4. 添加国家列
    merged_data['country'] = country_code
    
    # 5. 创建全局唯一的cluster_id
    merged_data['unique_id'] = merged_data['country'] + '_' + merged_data['cluster_id'].astype(int).astype(str)
    
    # 添加到所有国家数据列表
    all_countries_data.append(merged_data)
    
    print(f"{country_name} 处理完成，找到 {len(merged_data)} 个聚类")

# 6. 合并所有国家数据
if all_countries_data:
    final_data = pd.concat(all_countries_data, ignore_index=True)
    
    # 重新排列列顺序，将unique_id放在第一列
    cols = ['unique_id'] + [col for col in final_data.columns if col != 'unique_id']
    final_data = final_data[cols]
    
    # 7. 保存结果
    output_path = "data/clusters.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_data.to_csv(output_path, index=False)
    
    # 8. 打印结果
    print("\n处理完成！")
    print("前5行数据:")
    print(final_data.head())
    print(f"\n总共找到的聚类数量: {len(final_data)}")
    print(f"唯一ID数量: {final_data['unique_id'].nunique()}")
    print(f"数据已保存到: {output_path}")
else:
    print("错误: 没有找到任何国家的数据")