import ee
import pandas as pd
import requests
import os

# Initialize Earth Engine
try:
    ee.Initialize()
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"EE Initialization failed: {e}")
    print("Please ensure you have authenticated with Earth Engine.")

def download_image(row, output_dir):
    try:
        # Parse coordinates
        lat = row['lat']
        lon = row['lon']
        name = row['name']
        label = row['label']
        
        # 1. 定义区域 (正方形 BBox，而不是圆形 Buffer)
        # 我们用 buffer 生成圆，然后取 bounds 变成正方形
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(2000).bounds() 
        
        # 2. 获取数据 (2023年, 30%云量)
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate('2023-01-01', '2023-12-31') \
            .filterBounds(region) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            
        # 3. 去云合成 (Median)
        image = collection.median()
        
        # 4. 关键修正：强制裁剪 + 视觉参数优化
        # 显式指定 scale=10 (10米分辨率) 和 crs='EPSG:4326' (经纬度投影)
        # 这样出来的图就是正的，不是斜的
        url = image.getThumbURL({
            'region': region,
            'dimensions': 500,
            'format': 'jpg',
            'min': 0,
            'max': 3000,
            'bands': ['B4', 'B3', 'B2'], # 确保是 RGB
            'crs': 'EPSG:4326'           # 强制投影
        })
        
        # Filename
        filename = f"{label}_{name}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            print(f"Skipping {name}")
            return

        # Download
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded: {name}")
        else:
            print(f"❌ Failed: {name}")
            
    except Exception as e:
        print(f"❌ Error {name}: {e}")

def main():
    # Define paths
    csv_file = 'china_coordinates.csv'
    output_dir = 'china_dataset_final'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Read CSV using pandas
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} locations from {csv_file}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Loop through each row and download
    for index, row in df.iterrows():
        download_image(row, output_dir)

if __name__ == "__main__":
    main()