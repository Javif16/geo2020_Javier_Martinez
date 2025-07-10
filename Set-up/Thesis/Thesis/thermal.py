'''
File that will be used to obtain all the thermal images and extract the desired attributes,
such as emissivity or surface temperature.

It will be able to extract files in formats like GeoTIFF or HDF.

Normalization of images will also take place here, in order for the deep learning algorithms
to always take the same type of data. From 0 to 1, easier for algorithms to process.

Then, the necessary images or attributes will be able to be passed to other files, in order
for the deep learning algorithms established in other files to work with the output produced
in this program.
'''

import numpy as np
import cv2
import os
import re
import math
import glob
import h5py
import rasterio
from collections import defaultdict
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ---------- FILES & DATASETS ------------------------------------------------------------------------------------------
folder_path = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Images/Geology/Villoslada de Cameros"
file_path = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Images/Geology/Villoslada de Cameros/ECOSTRESS_L2_LSTE_25405_006_20221229T022351_0601_01.h5"
datasets = ['SDS/LST', 'SDS/Emis1', 'SDS/Emis2', 'SDS/Emis3', 'SDS/Emis4', 'SDS/Emis5', 'SDS/QC']

'''
def explore_h5_group(name, obj):
    # group or dataset
    if isinstance(obj, h5py.Group):
        print(f"📂 Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"📄 Dataset: {name} | Shape: {obj.shape} | Data Type: {obj.dtype}")


with h5py.File(file_path, 'r') as f:
    print(f"📖 Exploring HDF5 File: {file_path}")
    f.visititems(explore_h5_group)
'''
geo_map_path = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Images/Geology/Villoslada de Cameros/Geo_map_vill.tif"

with rasterio.open(geo_map_path) as src:
    class_band = src.read(2)  # Read the class labels band
    unique_values = np.unique(class_band)
    print("Unique class values:", unique_values)
    print("Number of unique classes:", len(unique_values))
    print("Min class value:", np.min(unique_values))
    print("Max class value:", np.max(unique_values))


def decode_qc_value(qc_value):
    """Decodes a 16-bit QC value according to the ECOSTRESS LST QC table."""
    bits = format(qc_value, '016b')  # Convert to 16-bit binary string

    qc_info = {
        "LST Accuracy": bits[0:2],  # Bits 15 & 14
        "Emissivity Accuracy": bits[2:4],  # Bits 13 & 12
        "MMD": bits[4:6],  # Bits 11 & 10
        "Atmospheric Opacity": bits[6:8],  # Bits 9 & 8
        "Iterations": bits[8:10],  # Bits 7 & 6
        "Cloud/Ocean Flag": bits[10:12],  # Bits 5 & 4
        "Data Quality Flag": bits[12:14],  # Bits 3 & 2
        "Mandatory QA": bits[14:16],  # Bits 1 & 0
    }

    return qc_info

'''
# Example usage:
qc_value = 40975
decoded_qc = decode_qc_value(qc_value)
print(decoded_qc)

qc_values = [
    15, 16399, 32770, 32783, 32898, 32962, 32966, 33218, 33222, 33474,
    33478, 33986, 33990, 34242, 34246, 34498, 34502, 35010, 35266, 35270,
    35522, 35526, 36290, 36294, 36546, 36550, 36866, 36879, 36994, 37058,
    37062, 37314, 37318, 37568, 37570, 37574, 38082, 38086, 38338, 38342,
    38592, 38594, 38597, 38598, 39106, 39362, 39366, 39616, 39617, 39618,
    39621, 39622, 40130, 40386, 40390, 40640, 40642, 40645, 40646, 40975
]

decoded_qc_values = {val: decode_qc_value(val) for val in qc_values}

# Print results
for qc, decoded in decoded_qc_values.items():
    print(f"\nQC Value: {qc}")
    print(decoded)
'''


# ------------------- QC ------------------------------------------------
# checking pixel quality
def check_pixel_quality(qc_value):
    # bits extracted according to ECOSTRESS manual guide
    qc_value = int(qc_value)

    # Bits 1 & 2 - General Pixel Quality
    bit1 = (qc_value >> 0) & 1  # Bit 1
    bit2 = (qc_value >> 1) & 1  # Bit 2

    if bit1 == 0 and bit2 == 0:
        pixel_quality = 1.0  # Best Quality
    elif bit1 == 1 and bit2 == 0:
        pixel_quality = 0.75  # Nominal Quality
    elif bit1 == 0 and bit2 == 1:
        pixel_quality = 0.25  # Cloud Detected
    else:
        pixel_quality = 0.0  # Missing/Bad Data

    # Bits 10 & 11 - MMD (Mineralogical Mixing Diagnostic)
    bit10 = (qc_value >> 10) & 1
    bit11 = (qc_value >> 11) & 1
    mmd_value = (bit11 << 1) | bit10  # Combine bits

    mmd_map = {
        0b00: "> 0.15 (Most silicate rocks)",
        0b01: "0.1 - 0.15 (Rocks, sand, some soils)",
        0b10: "0.03 - 0.1 (Mostly soils, mixed pixel)",
        0b11: "< 0.03 (Vegetation, snow, water, ice)"
    }
    mmd_quality = mmd_map.get(mmd_value, "Unknown")

    # Bits 12 & 13 - Emissivity Accuracy
    bit12 = (qc_value >> 12) & 1
    bit13 = (qc_value >> 13) & 1
    emissivity_value = (bit13 << 1) | bit12  # Combine bits

    emissivity_map = {
        0b00: "> 0.02 (Poor performance)",
        0b01: "0.015 - 0.02 (Marginal performance)",
        0b10: "0.01 - 0.015 (Good performance)",
        0b11: "< 0.01 (Excellent performance)"
    }
    emissivity_quality = emissivity_map.get(emissivity_value, "Unknown")

    # Bits 14 & 15 - LST (Land Surface Temperature) Accuracy
    bit14 = (qc_value >> 14) & 1
    bit15 = (qc_value >> 15) & 1
    lst_value = (bit15 << 1) | bit14  # Combine bits

    lst_map = {
        0b00: "> 2 K (Poor performance)",
        0b01: "1.5 - 2 K (Marginal performance)",
        0b10: "1 - 1.5 K (Good performance)",
        0b11: "< 1 K (Excellent performance)"
    }
    lst_quality = lst_map.get(lst_value, "Unknown")

    # Return all extracted information
    return {
        "Pixel Quality": pixel_quality,
        "MMD": mmd_quality,
        "Emissivity Accuracy": emissivity_quality,
        "LST Accuracy": lst_quality
    }


def extract_weights(qc_data):
    def get_pixel_weight(qc_value):
        return check_pixel_quality(qc_value)["Pixel Quality"]

    vectorized_check = np.vectorize(get_pixel_weight)
    return vectorized_check(qc_data)


# ---------- AOI -------------------------------------------------------------------------------------------------------
# area of interest (same area for all images)
SW_lat, SW_lon = 42.002377, -2.852681
NE_lat, NE_lon = 42.168736, -2.518803


def get_aoi(data, transform, sw_lat, sw_lon, ne_lat, ne_lon, north, south, east, west):
    """Extracts the Area of Interest (AOI) ensuring it is within the image bounds."""

    # coordinates inside bounds
    sw_lat = max(south, min(sw_lat, north))
    ne_lat = max(south, min(ne_lat, north))
    sw_lon = max(west, min(sw_lon, east))
    ne_lon = max(west, min(ne_lon, east))

    # get transform
    top_left_x = transform.c.item()  # West (X origin)
    top_left_y = transform.f.item()  # North (Y origin)
    pixel_width = transform.a.item()
    pixel_height = abs(transform.e.item())  # Should be positive

    # lat/lon to row/col indices
    row_start = int((top_left_y - ne_lat) / pixel_height)
    row_end = int((top_left_y - sw_lat) / pixel_height)
    col_start = int((sw_lon - top_left_x) / pixel_width)
    col_end = int((ne_lon - top_left_x) / pixel_width)

    # indices inside bounds
    row_start = max(0, min(row_start, data.shape[0] - 1))
    row_end = max(0, min(row_end, data.shape[0]))
    col_start = max(0, min(col_start, data.shape[1] - 1))
    col_end = max(0, min(col_end, data.shape[1]))

    if row_end <= row_start or col_end <= col_start:
        return None, None  # AOI outside or too small

    # get AOI
    aoi_data = data[row_start:row_end, col_start:col_end]
    print(f"AOI data shape: {aoi_data.shape}")

    aoi_transform = rasterio.transform.from_origin(
        top_left_x + col_start * pixel_width,
        top_left_y - row_start * pixel_height,
        pixel_width, pixel_height
    )

    return aoi_data, aoi_transform


def tiff_aoi(file, output, datasets, sw_lat, sw_lon, ne_lat, ne_lon):
    """Converts AOI from HDF5 to GeoTIFF ensuring correct spatial alignment."""

    with h5py.File(file, 'r') as f:
        # bbox
        north = f['StandardMetadata/NorthBoundingCoordinate'][()]
        south = f['StandardMetadata/SouthBoundingCoordinate'][()]
        east = f['StandardMetadata/EastBoundingCoordinate'][()]
        west = f['StandardMetadata/WestBoundingCoordinate'][()]

        rows, cols = f[datasets[0]].shape
        pixel_width = (east - west) / cols
        pixel_height = (north - south) / rows
        transform = from_origin(west, north, pixel_width, -pixel_height)  # Note: negative height

        weight_map = None

        for dataset in datasets:
            dset = f[dataset]
            data = dset[:].astype(np.float32)

            # scale, offset and fill values
            scale_factor = dset.attrs.get("scale_factor", [1.0])[0]
            add_offset = dset.attrs.get("add_offset", [0.0])[0]
            fill_value = dset.attrs.get("_FillValue", [None])[0]

            data = data * scale_factor + add_offset
            if fill_value is not None:
                data[data == fill_value] = 0.0

            # AOI inside image
            aoi_data, aoi_transform = get_aoi(data, transform, sw_lat, sw_lon, ne_lat, ne_lon, north, south, east, west)

            if aoi_data is None:
                print(f"Skipping {dataset} because AOI is outside image bounds or empty!")
                continue
            print(aoi_transform)

            # Kelvin to Celsius
            if dataset.endswith("LST"):
                data -= 273.15

            # quality pixels
            if dataset == "SDS/QC":
                weight_map = extract_weights(aoi_data)
                np.save(os.path.join(output, f"{os.path.basename(file).replace('.h5', '')}_weights.npy"), weight_map)
                continue

            safe_dataset = dataset.replace("/", "_")
            output_path = os.path.join(output, f"{os.path.basename(file).replace('.h5', '')}_{safe_dataset}.tif")
            with rasterio.open(
                    output_path, 'w', driver='GTiff',
                    height=aoi_data.shape[0], width=aoi_data.shape[1],
                    count=1, dtype=aoi_data.dtype,
                    crs='EPSG:4326', transform=aoi_transform,
                    nodata=np.nan
            ) as dst:
                dst.write(aoi_data, 1)

            print(f"Saved {dataset} to {output_path}")


# -------------- NORMALIZATION -----------------------------------------------------------------------------------------
# normalize values to range [0, 1]
def normalize(data):
    data = np.nan_to_num(data, nan=0.0)

    min_val = np.min(data)
    max_val = np.max(data)

    # CHANGE: Handle the case where max_val equals min_val
    if max_val == min_val:
        return np.zeros_like(data)

    return (data - min_val) / (max_val - min_val)


# -------------- PATCHING ----------------------------------------------------------------------------------------------
def patching(images, patch_size=64, stride=48, label=None):     # stride=128 means 50% overlapping
    """
    Extracts overlapping patches from the h5 images.
    :param image: 2D NumPy array (grayscale) or 3D NumPy array (multi-channel)
    :param patch_size: Size of patches
    :param stride: same as patch size to ensure no overlapping
    :return: List of patches
    """
    patches = []
    positions = []
    if len(images.shape) == 2:  # Grayscale (H, W)
        h, w = images.shape
        c = 1  # Set channels to 1 for consistency
        images = images[..., np.newaxis]  # Convert to (H, W, 1)
    elif len(images.shape) == 3:  # Multi-channel (H, W, C)
        h, w, c = images.shape

    for i in range(0, h - patch_size + 1, stride):  # height
        for j in range(0, w - patch_size + 1, stride):  # width
            patch = images[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
            positions.append((i, j))

    if len(patches) == 0:
        print(
            f"⚠️ No patches extracted from image of shape {images.shape} with patch size {patch_size} and stride {stride}.")

    return np.array(patches), positions


# -------------- PRE-PROCESSING ----------------------------------------------------------------------------------------
# CROPPING to same SHAPE

# UPSCALING
def upscale_thermal_image(image_path, target_shape=(742, 1176)):
    """Upscale a thermal image to match the geological map size."""
    with rasterio.open(image_path) as src:
        image = src.read(1).astype(np.float32)

    upscaled_image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
    return upscaled_image  # Shape: (744, 1171)


def upscale_weight_map(weight_path, target_shape=(742, 1176)):
    """Upscale a weight map (.npy) to match the geological map size."""
    weights = np.load(weight_path).astype(np.float32)  # Load .npy file

    upscaled_weights = cv2.resize(weights, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return upscaled_weights  # Shape: (744, 1171)


'''
# DOWNSAMPLING
from scipy.stats import mode


def downsample_majority_vote(label_image, target_shape):
    H, W = label_image.shape
    target_H, target_W = target_shape

    scale_H = H // target_H
    scale_W = W // target_W

    downsampled = np.zeros((target_H, target_W), dtype=np.uint8)

    for i in range(target_H):
        for j in range(target_W):
            block = label_image[
                i * scale_H : (i + 1) * scale_H,
                j * scale_W : (j + 1) * scale_W
            ]
            downsampled[i, j] = mode(block, axis=None, keepdims=False).mode

    return downsampled


def downsample_weights(weight_map, target_shape):
    H, W = weight_map.shape
    target_H, target_W = target_shape

    scale_H = H // target_H
    scale_W = W // target_W

    downsampled = np.zeros((target_H, target_W), dtype=np.float32)

    for i in range(target_H):
        for j in range(target_W):
            block = weight_map[
                i * scale_H : (i + 1) * scale_H,
                j * scale_W : (j + 1) * scale_W
            ]
            downsampled[i, j] = np.mean(block)

    return downsampled
'''


# NaNs
def check_and_fix_nans(data_array, replace_value=0.0):
    """Check for NaN values in the array and replace them with the specified value."""
    nan_count = np.isnan(data_array).sum()
    if nan_count > 0:
        print(f"⚠️ Found {nan_count} NaN values, replacing with {replace_value}")
        return np.nan_to_num(data_array, nan=replace_value)
    return data_array


def geo_tiffs_preprocess(folder_path, num_classes=14):
    tiff_files = glob.glob(os.path.join(folder_path, "*.tif"))
    weight_files = glob.glob(os.path.join(folder_path, "*_weights.npy"))
    geo_map_path = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Images/Geology/Villoslada de Cameros/Geo_map_Vill.tif"

    print(f"Found {len(tiff_files)} TIFF files and {len(weight_files)} weight files in {folder_path}.")
    if len(tiff_files) == 0:
        raise FileNotFoundError("No TIFF files found! Check your file paths and AOI extraction.")

    # Date grouping for thermal images
    grouped_files = {}
    weight_files_dict = {}
    for file in os.listdir(folder_path):
        if file.endswith(".tif"):
            for dataset in ["SDS_LST", "SDS_Emis1", "SDS_Emis2", "SDS_Emis3", "SDS_Emis4", "SDS_Emis5", "SDS_QC"]:
                if file.endswith(f"{dataset}.tif"):
                    date_key = file.split("_")[5][:8]  # Extract YYYYMMDD from filename
                    time_key = file.split("_")[5][9:]
                    if date_key not in grouped_files:
                        grouped_files[date_key] = {}
                    grouped_files[date_key][dataset] = os.path.join(folder_path, file)
                    grouped_files[date_key]['time'] = time_key
        elif file.endswith("weights.npy"):
            date_key = file.split("_")[5][:8]  # Extract YYYYMMDD from filename
            weight_files_dict[date_key] = os.path.join(folder_path, file)

    all_thermals = []
    all_labels = []
    all_weights = []
    all_date_indices = []  # NEW: Track which date each patch comes from

    # Open the geology map once here
    with rasterio.open(geo_map_path) as src:
        print(src.read(2)[:10])
        class_band = src.read(2)  # Class labels (e.g., 0, 1, 2, 3, etc.)
        weight_band = src.read(1)  # Class weights

        geo_class_image = class_band  # shape (H, W)
        print(geo_class_image.shape)
        geo_weight_image = weight_band  # shape (H, W)

    for date_idx, date_key in enumerate(sorted(grouped_files.keys())):
        print(date_idx)
        dataset_files = grouped_files[date_key]
        time_key = dataset_files.get("time", 'UNKOWN')
        required_datasets = ["SDS_LST", "SDS_Emis1", "SDS_Emis2", "SDS_Emis3", "SDS_Emis4", "SDS_Emis5"]

        # Check if all required datasets are present in the grouped dataset files
        if not all(ds in dataset_files for ds in required_datasets):
            print(f"⚠️ Missing datasets for {date_key}, skipping...")
            continue

        # Check if the weight file exists in weight_dict for the same date_key
        if date_key not in weight_files_dict:
            print(f"⚠️ Missing weight file for {date_key}, skipping...")
            continue

        # If both datasets and weight files exist for the date_key, proceed with processing
        formatted_date = f"{date_key[6:8]}/{date_key[4:6]}/{date_key[0:4]}"  # DD/MM/YYYY
        formatted_time = f"{time_key[0:2]}:{time_key[2:4]}:{time_key[4:6]}"  # HH:MM:SS
        print(f"Processing data: {formatted_date} - {formatted_time}")

        stacked_images = []
        for ds in required_datasets:
            # with rasterio.open(dataset_files[ds]) as src:  # this line and the next ones go out if upscaling
              #  image = src.read(1).astype(np.float32)
               # normalized_image = normalize(image)
                #stacked_images.append(normalized_image)
            upscaled_image = upscale_thermal_image(dataset_files[ds])  # Upscale .tif file
            normalized_image = normalize(upscaled_image)  # Normalize
            stacked_images.append(normalized_image)

        stacked_images = np.stack(stacked_images, axis=-1)  # (744, 1171, 6)

        # thermal_shape = stacked_images.shape[:2]
        # down_labels = downsample_majority_vote(geo_class_image, thermal_shape)
        # down_weights = downsample_weights(geo_weight_image, thermal_shape)
        upscaled_weights = upscale_weight_map(weight_files_dict[date_key])  # Upscale .npy weights
        if stacked_images.shape[:2] != geo_class_image.shape:
        # if stacked_images.shape[:2] != down_labels.shape:
            print(f"⚠️ Image shape mismatch for {date_key}, skipping...")
            continue

        combined_weights = upscaled_weights * geo_weight_image  # shape (H, W)
        # combined_weights = down_weights

        all_thermals.append(stacked_images)  # shape (H, W, 6)
        all_labels.append(geo_class_image)  # shape (H, W)
        all_weights.append(combined_weights)  # shape (H, W)
        all_date_indices.append(date_idx)

    # Combine all patches into single arrays
    all_thermals = np.array(all_thermals)
    all_labels = np.array(all_labels)
    all_weights = np.array(all_weights)
    all_date_indices = np.array(all_date_indices)

    print(
        f"Final arrays shape: thermal={all_thermals.shape}, labels={all_labels.shape}, weights={all_weights.shape}, dates={all_date_indices.shape}")

    # Fix NaN values
    all_thermal_images = check_and_fix_nans(all_thermals)
    all_label_images = check_and_fix_nans(all_labels)
    all_weight_images = check_and_fix_nans(all_weights)

    return all_thermal_images, all_label_images, all_weight_images, all_date_indices


# -------------- WORKFLOW ----------------------------------------------------------------------------------------------
output_path = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Images/Examples/Villoslada_tif"

os.makedirs(output_path, exist_ok=True)
# for h5_file in glob.glob(os.path.join(folder_path, "*.h5")):
   # tiff_aoi(h5_file, output_path, datasets, SW_lat, SW_lon, NE_lat, NE_lon)

tiff_thermal, labels, weight_maps, date_indices = geo_tiffs_preprocess(output_path, num_classes=14)
print(f"Number of thermal images extracted: {len(tiff_thermal)}")
print(f"Number of dates: {len(np.unique(date_indices))}")
if len(tiff_thermal) == 0:
    raise ValueError("No thermal images were extracted. Check the TIFF processing pipeline!")

# -------------- CNN ----------------------------------------------------------------------------------------------
cnn_patches = tiff_thermal.astype(np.float32)
cnn_labels = labels.astype(np.float32)
cnn_weights = weight_maps.astype(np.float32)
cnn_dates = date_indices

# First split: train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp, w_train, w_temp, date_train, date_temp = train_test_split(
    cnn_patches, cnn_labels, cnn_weights, cnn_dates, test_size=0.3, random_state=42, shuffle=True
)

# Second split: temp -> val (15%) and test (15%)
X_val, X_test, y_val, y_test, w_val, w_test, date_val, date_test = train_test_split(
    X_temp, y_temp, w_temp, date_temp, test_size=0.5, random_state=42, shuffle=True
)

# y_train = y_train.reshape(-1)
# y_val = y_val.reshape(-1)
# y_test = y_test.reshape(-1)

print("Final thermal train cnn shape:", X_train.shape)
print("Final labels train cnn shape:", y_train.shape)
print("Final weights train cnn shape:", w_train.shape)
print("Final dates train cnn shape:", date_train.shape)

print("Final thermal val cnn shape:", X_val.shape)
print("Final labels val cnn shape:", y_val.shape)
print("Final weights val cnn shape:", w_val.shape)
print("Final dates val cnn shape:", date_val.shape)

print("Final thermal test cnn shape:", X_test.shape)
print("Final labels test cnn shape:", y_test.shape)
print("Final weights test cnn shape:", w_test.shape)
print("Final dates test cnn shape:", date_test.shape)

# Save all data including positions
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
np.save("y_test.npy", y_test)
np.save("weights_train.npy", w_train)
np.save("weights_val.npy", w_val)
np.save("weights_test.npy", w_test)
np.save("date_indices_train.npy", date_train)
np.save("date_indices_val.npy", date_val)
np.save("date_indices_test.npy", date_test)
print("✅ Processing complete! Data saved for CNN training.")


# -------------- ConvLSTM ----------------------------------------------------------------------------------------------
def generate_temporal_sequences(data, labels, weights, dates, seq_length=5, step=3, max_sequences=None):
    """
    Generate temporal sequences from full images (not patches).

    Args:
        data: (N, H, W, C) - Full thermal images
        labels: (N, H, W) - Ground truth labels per image
        weights: (N, H, W) - Weight maps per image
        dates: (N,) - Dates corresponding to each full image
        seq_length: Number of time steps in each sequence
        step: Temporal stride between sequences
        max_sequences: Max number of sequences to generate (optional)

    Returns:
        Tuple of sequences:
        - thermal_sequences: (N_seq, seq_length, H, W, C)
        - label_sequences: (N_seq, seq_length, H, W)
        - weight_sequences: (N_seq, seq_length, H, W)
        - date_sequences: (N_seq, seq_length)
    """
    thermal_sequences = []
    label_sequences = []
    weight_sequences = []
    date_sequences = []

    total_sequences = len(data) - seq_length + 1
    for i in range(0, total_sequences, step):
        if max_sequences is not None and len(thermal_sequences) >= max_sequences:
            break

        thermal_seq = data[i:i + seq_length]
        label_seq = labels[i:i + seq_length]
        weight_seq = weights[i:i + seq_length]
        date_seq = dates[i:i + seq_length]

        thermal_sequences.append(thermal_seq)
        label_sequences.append(label_seq)
        weight_sequences.append(weight_seq)
        date_sequences.append(date_seq)

    return (np.array(thermal_sequences), np.array(label_sequences),
            np.array(weight_sequences), np.array(date_sequences))


print("Creating ConvLSTM temporal sequences...")
seq_length = 5
step = 2
clstm_patches, clstm_labels, clstm_weights, clstm_dates = generate_temporal_sequences(
    tiff_thermal, labels, weight_maps, date_indices,
    seq_length=seq_length, step=step
)

print("clstm_patches:", clstm_patches.shape)  # (N_seq, seq_length, H, W, C)
print("clstm_labels:", clstm_labels.shape)    # (N_seq, seq_length, H, W)
print("clstm_weights:", clstm_weights.shape)  # (N_seq, seq_length, H, W)
print("clstm_dates:", clstm_dates.shape)      # (N_seq, seq_length)

X_train_lstm, X_temp_lstm, y_train_lstm, y_temp_lstm, w_train_lstm, w_temp_lstm, date_train_lstm, date_temp_lstm = train_test_split(
    clstm_patches, clstm_labels, clstm_weights, clstm_dates, test_size=0.3, random_state=42, shuffle=False
)

X_val_lstm, X_test_lstm, y_val_lstm, y_test_lstm, w_val_lstm, w_test_lstm, date_val_lstm, date_test_lstm = train_test_split(
    X_temp_lstm, y_temp_lstm, w_temp_lstm, date_temp_lstm, test_size=0.5, random_state=42, shuffle=False
)

# y_train_lstm = y_train_lstm.reshape(-1)
# y_val_lstm = y_val_lstm.reshape(-1)
# y_test_lstm = y_test_lstm.reshape(-1)
print("Final thermal train lstm shape:", X_train_lstm.shape)
print("Final labels train lstm shape:", y_train_lstm.shape)
print("Final weights train lstm shape:", w_train_lstm.shape)
print("Final dates train lstm shape:", date_train_lstm.shape)

# ConvLSTM - Save all data including positions
np.save("X_train_lstm.npy", X_train_lstm)
np.save("X_val_lstm.npy", X_val_lstm)
np.save("X_test_lstm.npy", X_test_lstm)
np.save("y_train_lstm.npy", y_train_lstm)
np.save("y_val_lstm.npy", y_val_lstm)
np.save("y_test_lstm.npy", y_test_lstm)
np.save("weights_train_lstm.npy", w_train_lstm)
np.save("weights_val_lstm.npy", w_val_lstm)
np.save("weights_test_lstm.npy", w_test_lstm)
np.save("dates_train_lstm.npy", date_train_lstm)
np.save("dates_val_lstm.npy", date_val_lstm)
np.save("dates_test_lstm.npy", date_test_lstm)

print("✅ Processing complete! Data saved for ConvLSTM training.")
