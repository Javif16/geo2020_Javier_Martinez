'''
This file extracts and prepares NDVI data from Sentinel-2 for the main pre-processing stage in complementary.py, where
the data will be combined with thermal patches for preparation.
'''

import numpy as np
import rasterio
import os
from scipy.ndimage import uniform_filter

import ee

ee.Authenticate()
ee.Initialize(project='sar-thesis')
tasks = ee.data.getTaskList()

for task in tasks:
    if task['state'] in ['READY', 'RUNNING']:
        print(f"Cancelling task {task['description']} (ID: {task['id']}) - Status: {task['state']}")
        ee.data.cancelTask(task['id'])

print("All READY and RUNNING tasks have been cancelled.")

# Rectangle in order: W-S-E-N
aoi = ee.Geometry.Rectangle([-2.852681, 42.002377, -2.518803, 42.168736])
start, end = '2019-01-01', '2024-12-31'

# Sentinel-2 collection for NDVI calculation
col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")  # Using Surface Reflectance data
       .filterDate(start, end)
       .filterBounds(aoi)
       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))  # Filter out very cloudy images
       .select(['B4', 'B8', 'QA60']))  # Red, NIR, and Quality Assessment bands


# Function to calculate NDVI and mask clouds
def calculate_ndvi(image):
    # Calculate NDVI
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # Cloud masking using QA60 band
    qa = image.select('QA60')
    cloudBitMask = 1 << 10  # Bit 10: Cirrus clouds
    cirrusBitMask = 1 << 11  # Bit 11: Clouds
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))

    return ndvi.updateMask(mask).copyProperties(image, ['system:time_start'])


# Apply NDVI calculation to collection
ndvi_col = col.map(calculate_ndvi)

images = ndvi_col.toList(ndvi_col.size())
n = images.size().getInfo()
print(f"Found {n} NDVI images to export")

for idx in range(n):
    img = ee.Image(images.get(idx))
    date = ee.Date(img.get('system:time_start')).format('YYYYMMdd').getInfo()
    desc = f"S2_NDVI_{date}"

    task = ee.batch.Export.image.toDrive(
        image=img.clip(aoi),
        description=desc,
        fileNamePrefix=desc,
        folder='S2_NDVI_Villoslada',
        region=aoi,
        crs='EPSG:32630',
        dimensions='397x267',
        maxPixels=1e9
    )
    task.start()
    print(f"Submitted export {idx + 1}/{n}: {desc}")
'''

# Input and output folders for post-processing
input_folder = r"C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/NDVI/S2 NDVI Santa Olalla raw"
output_folder = r"C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/NDVI/S2 NDVI Santa Olalla"
os.makedirs(output_folder, exist_ok=True)

# Process .tif files following similar methodology as SAR processing
for fname in os.listdir(input_folder):
    if not fname.lower().endswith(".tif"):
        continue

    input_path = os.path.join(input_folder, fname)
    output_path = os.path.join(output_folder, fname)

    with rasterio.open(input_path) as src:
        data = src.read(1).astype("float32")
        print(f"Original NDVI data for {fname}:", data[:2])

        # Mask invalid values (NaNs, Infs) and values outside typical NDVI range
        valid_mask = np.isfinite(data) & (data >= -1.0) & (data <= 1.0)
        if not np.any(valid_mask):
            print(f"Skipping {fname}: no valid NDVI data found.")
            continue

        processed = data.copy()
        processed[~valid_mask] = 0.0
        print(f"Processed NDVI data for {fname}:", processed[:2])

        profile = src.profile.copy()
        profile.update({"dtype": "float32", "count": 1})

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(processed, 1)

    print(f"Saved: {fname}")

print("NDVI extraction and processing completed!")
'''