'''
This file extracts and prepares RGB files from Landsat for the main pre-processing stage in complementary.py, where the
data will be combined with thermal patches for preparation.
'''

import os
import rasterio
import numpy as np
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject

# Bit masks for QA_PIXEL
CLOUD_BIT = 1 << 3
CLOUD_CONF_MASK = 0b11 << 8
CLOUD_CONF_HIGH = 0b11 << 8

# Input and output folders
thermal_path = r'E:/Studies/Thesis/ECO_example_Villoslada.tif'
optical_folder = r'E:/Studies/Thesis/RGB/Villoslada RGB raw'
output_folder = r'E:/Studies/Thesis/RGB/Villoslada RGB'
os.makedirs(output_folder, exist_ok=True)


# Thermal extent and shape
with rasterio.open(thermal_path) as thermal_src:
    thermal_bounds = thermal_src.bounds
    thermal_crs = thermal_src.crs
    thermal_shape = (thermal_src.height, thermal_src.width)

image_groups = {}
for fname in os.listdir(optical_folder):
    if "_SR_B2" in fname or "_SR_B3" in fname or "_SR_B4" in fname:
        base_id = fname.split("_SR_B")[0]
        image_groups.setdefault(base_id, []).append(fname)

# B2, B3, B4
for base_id, band_files in image_groups.items():
    if len(band_files) < 3:
        print(f"Skipping {base_id}: missing one or more RGB bands")
        continue

    qa_name = base_id + "_QA_PIXEL.TIF"
    qa_path = os.path.join(optical_folder, qa_name)
    if not os.path.exists(qa_path):
        print(f"Skipping {base_id}: QA_PIXEL not found")
        continue

    with rasterio.open(qa_path) as qa_src:
        if qa_src.crs != thermal_crs:
            print(f"Skipping {base_id}: CRS mismatch in QA_PIXEL")
            continue
        try:
            qa_window = from_bounds(*thermal_bounds, transform=qa_src.transform)
        except ValueError:
            print(f"Failed windowing QA_PIXEL for {base_id}, skipping")
            continue

        qa_crop = qa_src.read(1, window=qa_window, boundless=True)
        qa_resized = np.empty(thermal_shape, dtype=qa_crop.dtype)
        reproject(
            source=qa_crop,
            destination=qa_resized,
            src_transform=qa_src.window_transform(qa_window),
            src_crs=qa_src.crs,
            dst_transform=rasterio.transform.from_bounds(*thermal_bounds, *thermal_shape[::-1]),
            dst_crs=qa_src.crs,
            resampling=Resampling.nearest
        )

        # QA checks
        total_pixels = qa_resized.size
        cloud_pixels = np.sum(((qa_resized & CLOUD_BIT) != 0) & ((qa_resized & CLOUD_CONF_MASK) == CLOUD_CONF_HIGH))
        cloud_pct = (cloud_pixels / total_pixels) * 100

        if cloud_pct > 10:
            print(f"Skipping {base_id}, Cloud={cloud_pct:.2f}%")
            continue

    # Load, crop, resize, normalize bands
    bands_sorted = sorted(band_files)
    normalized_stack = []

    base_profile = None
    for fname in bands_sorted:
        input_path = os.path.join(optical_folder, fname)

        with rasterio.open(input_path) as src:
            if src.crs != thermal_crs:
                print(f"Skipping {fname}: CRS mismatch")
                continue

            if base_profile is None:
                base_profile = src.profile.copy()

            try:
                window = from_bounds(*thermal_bounds, transform=src.transform)
            except ValueError:
                print(f"Failed windowing for {fname}, skipping")
                continue

            cropped = src.read(1, window=window, boundless=True)
            resized = np.empty(thermal_shape, dtype=cropped.dtype)
            reproject(
                source=cropped,
                destination=resized,
                src_transform=src.window_transform(window),
                src_crs=src.crs,
                dst_transform=rasterio.transform.from_bounds(*thermal_bounds, *thermal_shape[::-1]),
                dst_crs=src.crs,
                resampling=Resampling.bilinear
            )

            normalized = resized.astype('float32')
            band_min, band_max = np.nanmin(normalized), np.nanmax(normalized)
            normalized = (normalized - band_min) / (band_max - band_min + 1e-6)
            normalized_stack.append(normalized)

    if len(normalized_stack) != 3:
        print(f"Skipping {base_id}: Could not process all 3 bands")
        continue

    stacked = np.stack(normalized_stack, axis=0)
    out_profile = src.profile.copy()
    out_profile.update({
        'dtype': 'float32',
        'count': 3,
        'height': thermal_shape[0],
        'width': thermal_shape[1],
        'transform': rasterio.transform.from_bounds(*thermal_bounds, *thermal_shape[::-1])
    })

    band_suffixes = ["B2", "B3", "B4"]
    for i, band_array in enumerate(normalized_stack):
        out_name = f"{base_id}_{band_suffixes[i]}.tif"
        band_profile = base_profile.copy()
        band_profile.update({
            'dtype': 'float32',
            'count': 1,
            'height': thermal_shape[0],
            'width': thermal_shape[1],
            'transform': rasterio.transform.from_bounds(*thermal_bounds, *thermal_shape[::-1])
        })
        with rasterio.open(os.path.join(output_folder, out_name), 'w', **band_profile) as dst:
            dst.write(band_array, 1)
        print(f"Saved: {out_name}")

print("All valid RGB images processed and saved.")
