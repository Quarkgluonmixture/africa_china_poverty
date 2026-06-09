import os
import pandas as pd
import ee
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 初始化 GEE
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# 读取 CSV
clusters_path = 'data/clusters.csv'
df = pd.read_csv(clusters_path)
print(f"Total Clusters: {len(df)}")

output_dir = 'data/images'
os.makedirs(output_dir, exist_ok=True)

def get_satellite_image(lat, lon, cluster_id):
    try:
        point = ee.Geometry.Point([lon, lat])
        roi = point.buffer(2000)  # 2km

        # --- 关键修改：扩大时间，使用中位数合成 ---
        # Sentinel-2 Harmonized (修正后的数据集)
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate('2015-07-01', '2017-12-31') \
            .filterBounds(roi) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) # 放宽到 30%

        # 使用中位数合成 (Median Composite) 去云
        # 只要这 2.5 年里有哪怕几张好图，中位数就能还原出地表
        image = collection.median().clip(roi)

        # 获取 URL
        url = image.getThumbURL({
            'dimensions': '224x224',
            'format': 'jpg',
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000
        })
        return url
    except Exception as e:
        # 如果还是失败，尝试放宽到 Landsat 8 (备用方案)
        # 但为了保持一致性，我们先只打印错误
        # print(f"Cluster {cluster_id} Error: {e}")
        return None

def download_single_image(row_data, output_dir):
    """下载单个图像（用于多线程）"""
    index, row = row_data
    
    if 'unique_id' in row:
        unique_id = row['unique_id']
    else:
        unique_id = f"{row['country']}_{int(row['cluster_id'])}"
    
    path = os.path.join(output_dir, f"{unique_id}.jpg")
    
    if os.path.exists(path):
        return 'skip'

    url = get_satellite_image(row['LATNUM'], row['LONGNUM'], unique_id)
    
    if url:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(r.content)
                return 'success'
        except:
            pass
    return 'fail'

# 主循环（多线程版本）
success = 0
fail = 0
skip = 0
max_workers = 6  # 并发数，可根据网络情况调整

print(f"Starting parallel download with {max_workers} workers...")
tasks = [(i, row) for i, row in df.iterrows()]

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(download_single_image, task, output_dir): task
               for task in tasks}
    
    for future in tqdm(as_completed(futures), total=len(tasks)):
        result = future.result()
        if result == 'success':
            success += 1
        elif result == 'fail':
            fail += 1
        else:
            skip += 1

print(f"Done. Success: {success}, Fail: {fail}, Skipped: {skip}")