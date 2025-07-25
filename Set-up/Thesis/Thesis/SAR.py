import numpy as np
import rasterio
import os
from scipy.ndimage import uniform_filter


'''
import ee
ee.Authenticate()
ee.Initialize(project='sar-thesis')
tasks = ee.data.getTaskList()

# Cancel all tasks that are not completed/failed/cancelled
for task in tasks:
    if task['state'] in ['READY', 'RUNNING']:
        print(f"Cancelling task {task['description']} (ID: {task['id']}) - Status: {task['state']}")
        ee.data.cancelTask(task['id'])

print("All READY and RUNNING tasks have been cancelled.")

# Set your AOI and date range
aoi = ee.Geometry.Rectangle([-6.520675667, 37.832876641, -6.188309907, 37.999978411])
start, end = '2019-01-01', '2024-12-31'

# Load Sentinel-1A collection with VV+VH
col = (ee.ImageCollection("COPERNICUS/S1_GRD")
       .filterDate(start, end)
       .filterBounds(aoi)
       .filter(ee.Filter.eq('instrumentMode', 'IW'))
       .filter(ee.Filter.eq('platform_number', 'A'))
       .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
       .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
       .select(['VV', 'VH']))

# Convert collection to list
images = col.toList(col.size())
n = images.size().getInfo()
print(f"Found {n} images to export")

for idx in range(n):
    img = ee.Image(images.get(idx))
    date = ee.Date(img.get('system:time_start')).format('YYYYMMdd').getInfo()
    desc = f"S1A_VV_VH_{date}"

    task = ee.batch.Export.image.toDrive(
        image=img.clip(aoi),
        description=desc,
        fileNamePrefix=desc,
        folder='S1_exports_Puertollano',
        region=aoi,
        crs='EPSG:32630',
        dimensions='415x273',
        maxPixels=1e9
    )
    task.start()
    print(f"Submitted export {idx + 1}/{n}: {desc}")
'''

# === Input and output folders ===
input_folder = r"E:/Studies/Thesis/SAR/SAR Villoslada raw"
output_folder = r"E:/Studies/Thesis/SAR/SAR Villoslada"
os.makedirs(output_folder, exist_ok=True)

# === Loop through all .tif files ===
for fname in os.listdir(input_folder):
    if not fname.lower().endswith(".tif"):
        continue

    input_path = os.path.join(input_folder, fname)
    output_path = os.path.join(output_folder, fname)

    with rasterio.open(input_path) as src:
        data = src.read(1).astype("float32")  # Read first band only
        print(data[:2])
        # Mask invalid values (NaNs, Infs)
        valid_mask = np.isfinite(data)
        if not np.any(valid_mask):
            print(f"⚠️ Skipping {fname}: no valid data found.")
            continue

        # Normalize using valid data
        min_val = np.nanmin(data[valid_mask])
        max_val = np.nanmax(data[valid_mask])
        normalized = (data - min_val) / (max_val - min_val + 1e-6)

        # Replace any NaNs/infs (from original invalid or division) with 0
        normalized[~np.isfinite(normalized)] = 0.0
        print(normalized[:2])

        # Update profile
        profile = src.profile.copy()
        profile.update({
            "dtype": "float32",
            "count": 1
        })

        # Save normalized image
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(normalized, 1)

    print(f"✅ Normalized and saved: {fname}")