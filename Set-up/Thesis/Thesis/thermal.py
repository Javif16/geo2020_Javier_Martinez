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
folder_path = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Images/Geology/Santa Olalla del Cala"
file_path = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Images/Geology/Santa Olalla del Cala/ECOSTRESS_L2_LSTE_25405_006_20221229T022351_0601_01.h5"
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
geo_map_path = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Images/Geology/Santa Olalla del Cala/Geo_map.tif"

with rasterio.open(geo_map_path) as src:
    class_band = src.read(1)  # Read the class labels band
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
SW_lat, SW_lon = 37.832876641, -6.520675667
NE_lat, NE_lon = 37.999978411, -6.188309907


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
def patching(images, patch_size=64, stride=64, label=None):     # stride=128 means 50% overlapping
    """
    Extracts non-overlapping patches from the h5 images.
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
def upscale_thermal_image(image_path, target_shape=(744, 1171)):
    """Upscale a thermal image to match the geological map size."""
    with rasterio.open(image_path) as src:
        image = src.read(1).astype(np.float32)

    upscaled_image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
    return upscaled_image  # Shape: (744, 1171)


def upscale_weight_map(weight_path, target_shape=(744, 1171)):
    """Upscale a weight map (.npy) to match the geological map size."""
    weights = np.load(weight_path).astype(np.float32)  # Load .npy file

    upscaled_weights = cv2.resize(weights, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return upscaled_weights  # Shape: (744, 1171)


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
    geo_map_path = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Images/Geology/Santa Olalla del Cala/Geo_map.tif"

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
                    if date_key not in grouped_files:
                        grouped_files[date_key] = {}
                    grouped_files[date_key][dataset] = os.path.join(folder_path, file)
        elif file.endswith("weights.npy"):
            date_key = file.split("_")[5][:8]  # Extract YYYYMMDD from filename
            weight_files_dict[date_key] = os.path.join(folder_path, file)

    all_thermal_patches = []
    all_label_patches = []
    all_weight_patches = []
    all_positions = []
    all_date_indices = []  # NEW: Track which date each patch comes from

    # Open the geology map once here
    with rasterio.open(geo_map_path) as src:
        class_band = src.read(1)  # Class labels (e.g., 0, 1, 2, 3, etc.)
        weight_band = src.read(2)  # Class weights

        # Extract patches from geology map (class labels)
        geo_class_patches, class_positions = patching(class_band)  # (num_patches, patch_size, patch_size)
        geo_weight_patches, weights_positions = patching(weight_band)  # (num_patches, patch_size, patch_size)
        assert class_positions == weights_positions, "Label missmatch"

    for date_idx, date_key in enumerate(sorted(grouped_files.keys())):
        dataset_files = grouped_files[date_key]
        print(date_idx)
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
        print(f"Processing data for {date_key}...")

        stacked_images = []
        for ds in required_datasets:
            upscaled_image = upscale_thermal_image(dataset_files[ds])  # Upscale .tif file
            normalized_image = normalize(upscaled_image)  # Normalize
            stacked_images.append(normalized_image)

        stacked_images = np.stack(stacked_images, axis=-1)  # (744, 1171, 6)
        upscaled_weights = upscale_weight_map(weight_files_dict[date_key])  # Upscale .npy weights

        # Extract patches
        thermal_patches, thermal_positions = patching(stacked_images)  # (num_patches, patch_size, patch_size, 6)
        thermal_weight_patches, weight_thermal_positions = patching(upscaled_weights)  # (num_patches, patch_size, patch_size)
        assert thermal_positions == class_positions, "Thermal missmatch"

        # Ensure the number of patches is consistent between the geology map and thermal images
        if thermal_patches.shape[0] != geo_class_patches.shape[0] or thermal_patches.shape[0] != \
                geo_weight_patches.shape[0]:
            print(f"⚠️ Mismatch in number of patches for {date_key}, skipping...")
            continue

        # Combine the thermal image weights and geology class weights by multiplying them
        for i in range(thermal_patches.shape[0]):
            all_thermal_patches.append(thermal_patches[i])
            all_label_patches.append(geo_class_patches[i])  # Same for all dates
            combined_weights = thermal_weight_patches[i] * geo_weight_patches[i]
            all_weight_patches.append(combined_weights)
            all_date_indices.append(date_idx)  # NEW: Track date index

            # patch positions
            if len(all_positions) < len(all_thermal_patches):
                all_positions.append(class_positions[i])

    # Combine all patches into single arrays
    all_thermal_patches = np.array(all_thermal_patches)
    all_label_patches = np.array(all_label_patches)
    all_weight_patches = np.array(all_weight_patches)
    all_positions = np.array(all_positions)
    all_date_indices = np.array(all_date_indices)  # NEWwx

    print(
        f"Final arrays shape: thermal={all_thermal_patches.shape}, labels={all_label_patches.shape}, weights={all_weight_patches.shape}, positions={all_positions.shape}, dates={all_date_indices.shape}")

    # Fix NaN values
    all_thermal_patches = check_and_fix_nans(all_thermal_patches)
    all_label_patches = check_and_fix_nans(all_label_patches)
    all_weight_patches = check_and_fix_nans(all_weight_patches)

    # Convert labels to one-hot encoding
    one_hot_labels = []
    for patch in all_label_patches:
        patch_2d = np.squeeze(patch)
        # Convert each patch to one-hot encoding
        one_hot_patch = np.zeros((patch_2d.shape[0], patch_2d.shape[1], num_classes), dtype=np.float32)

        # Loop through each possible class
        for class_idx in range(num_classes):
            # Create a boolean mask where patch == class_idx
            mask = (patch_2d == class_idx)
            # Set those positions to 1 in the appropriate channel
            one_hot_patch[mask, class_idx] = 1.0

        one_hot_labels.append(one_hot_patch)

    one_hot_labels = np.array(one_hot_labels)

    print(f"One-hot labels shape: {one_hot_labels.shape}")

    return all_thermal_patches, one_hot_labels, all_weight_patches, all_positions, all_date_indices


# -------------- SPLITS ------------------------------------------------------------------------------------------------

def splitting(patches, labels, weights, positions, date_indices):
    if patches.ndim == 5:  # ConvLSTM case: (N_seq, seq_length, H, W, C)
        sequence_start_dates = date_indices[:, 0]  # shape: (N_seq,) - use first date of each sequence
        # unique dates based on start time
        unique_dates = np.unique(sequence_start_dates)
        n_dates = len(unique_dates)

        # split dates
        train_dates = unique_dates[:int(n_dates * 0.7)]
        val_dates = unique_dates[int(n_dates * 0.7):int(n_dates * 0.9)]
        test_dates = unique_dates[int(n_dates * 0.9):]

        # Create masks based on sequence start dates
        train_mask = np.isin(sequence_start_dates, train_dates)
        val_mask = np.isin(sequence_start_dates, val_dates)
        test_mask = np.isin(sequence_start_dates, test_dates)

    else:  # CNN case: (N, H, W, C)
        unique_dates = np.unique(date_indices)
        n_dates = len(unique_dates)

        # split dates
        train_dates = unique_dates[:int(n_dates * 0.7)]
        val_dates = unique_dates[int(n_dates * 0.7):int(n_dates * 0.9)]
        test_dates = unique_dates[int(n_dates * 0.9):]

        # Create masks
        train_mask = np.isin(date_indices, train_dates)
        val_mask = np.isin(date_indices, val_dates)
        test_mask = np.isin(date_indices, test_dates)

    return (patches[train_mask], patches[val_mask], patches[test_mask],
            labels[train_mask], labels[val_mask], labels[test_mask],
            weights[train_mask], weights[val_mask], weights[test_mask],
            positions[train_mask], positions[val_mask], positions[test_mask],
            date_indices[train_mask], date_indices[val_mask], date_indices[test_mask])


# -------------- WORKFLOW ----------------------------------------------------------------------------------------------
output_path = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Images/Examples/Santa_Olalla_tif"

os.makedirs(output_path, exist_ok=True)
# for h5_file in glob.glob(os.path.join(folder_path, "*.h5")):
    # tiff_aoi(h5_file, output_path, datasets, SW_lat, SW_lon, NE_lat, NE_lon)

tiff_patches, labels, weight_maps, patch_positions, date_indices = geo_tiffs_preprocess(output_path, num_classes=14)
print(f"Number of patches extracted: {len(tiff_patches)}")
print(f"Number of positions extracted: {len(patch_positions)}")
print(f"Number of dates: {len(np.unique(date_indices))}")
if len(tiff_patches) == 0:
    raise ValueError("No patches were extracted. Check the TIFF processing pipeline!")

'''
# CNN
cnn_patches = tiff_patches.astype(np.float32)
cnn_labels = labels.astype(np.float32)
cnn_weights = weight_maps.astype(np.float32)
cnn_positions = patch_positions
cnn_dates = date_indices

print("Shapes before splitting:")
print("cnn_patches:", cnn_patches.shape)
print("cnn_labels:", cnn_labels.shape)
print("cnn_weights:", cnn_weights.shape)
print("cnn_positions:", cnn_positions.shape)
print("cnn_dates:", cnn_dates.shape)

(X_train, X_val, X_test, y_train, y_val, y_test,
 w_train, w_val, w_test, pos_train, pos_val, pos_test,
 date_train, date_val, date_test) = splitting(
    cnn_patches, cnn_labels, cnn_weights, cnn_positions, cnn_dates)

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
np.save("positions_train.npy", pos_train)
np.save("positions_val.npy", pos_val)
np.save("positions_test.npy", pos_test)
np.save("date_indices_train.npy", date_train)
np.save("date_indices_val.npy", date_val)
np.save("date_indices_test.npy", date_test)

print("✅ Processing complete! Data saved for CNN training.")
'''


# ConvLSTM
def generate_temporal_sequences(data, labels, weights, positions, dates, seq_length=5, step=3, max_sequences=None):
    # step=2 gives memory error, this is max
    """
        Generate temporal sequences by grouping patches from the SAME spatial position across different dates.

        Args:
            thermal_data: (22572, 64, 64, 6) - All patches
            labels: (22572, 64, 64, 14) - All patch labels
            weights: (22572, 64, 64, 1) - All patch weights
            positions: (22572, 2) - Spatial positions of patches
            date_indices: (22572,) - Date index for each patch
            seq_length: Length of temporal sequences
            step: Step between sequence start dates
            max_sequences: Maximum number of sequences to generate

        Returns:
            Temporal sequences where each sequence contains the same spatial patch across different dates
        """

    print("Organizing patches by spatial position and date...")

    # Group patches by spatial position
    position_groups = defaultdict(list)

    for i in range(len(data)):
        # Use position as key (convert to tuple for hashing)
        pos_key = tuple(positions[i])
        position_groups[pos_key].append({
            'index': i,
            'date': dates[i],
            'thermal': data[i],
            'label': labels[i],
            'weight': weights[i],
            'position': positions[i]
        })

    print(f"Found {len(position_groups)} unique spatial positions")

    # Sort each position group by date
    for pos_key in position_groups:
        position_groups[pos_key].sort(key=lambda x: x['date'])

    # Generate temporal sequences
    thermal_sequences = []
    label_sequences = []
    weight_sequences = []
    position_sequences = []
    date_sequences = []

    sequence_count = 0

    for pos_key, patches_at_position in position_groups.items():
        # Check if we have enough dates for this position
        if len(patches_at_position) < seq_length:
            continue

        # Generate sequences for this spatial position
        for start_idx in range(0, len(patches_at_position) - seq_length + 1, step):
            if max_sequences is not None and sequence_count >= max_sequences:
                break

            # Extract sequence of patches from the same position across different dates
            sequence_patches = patches_at_position[start_idx:start_idx + seq_length]

            # Build sequences
            thermal_seq = np.array([p['thermal'] for p in sequence_patches])
            label_seq = np.array([p['label'] for p in sequence_patches])
            weight_seq = np.array([p['weight'] for p in sequence_patches])
            position_seq = np.array([p['position'] for p in sequence_patches])
            date_seq = np.array([p['date'] for p in sequence_patches])

            thermal_sequences.append(thermal_seq)
            label_sequences.append(label_seq)
            weight_sequences.append(weight_seq)
            position_sequences.append(position_seq)
            date_sequences.append(date_seq)

            sequence_count += 1

            if sequence_count % 100 == 0:
                print(f"Generated {sequence_count} temporal sequences...")

        if max_sequences is not None and sequence_count >= max_sequences:
            break

    print(f"Generated {sequence_count} temporal sequences from {len(position_groups)} spatial positions")

    return (np.array(thermal_sequences), np.array(label_sequences),
            np.array(weight_sequences), np.array(position_sequences),
            np.array(date_sequences))


# Print shapes and collect sequences if needed
print("Creating ConvLSTM temporal sequences...")

max_sequences = 5867
seq_length = 5
step = 4
clstm_patches, clstm_labels, clstm_weights, clstm_positions, clstm_dates = generate_temporal_sequences(
    tiff_patches, labels, weight_maps, patch_positions, date_indices,
    seq_length=seq_length, step=step, max_sequences=max_sequences
)
print("ConvLSTM shapes after sequence generation:")
print("clstm_patches:", clstm_patches.shape)      # (num_sequences, seq_length, H, W, C)
print("clstm_labels:", clstm_labels.shape)        # (num_sequences, seq_length, H, W, num_classes)
print("clstm_weights:", clstm_weights.shape)      # (num_sequences, seq_length, H, W)
print("clstm_positions:", clstm_positions.shape)  # (num_sequences, seq_length, 2) - keeps all positions in the sequence, not only the first one
print("clstm_dates:", clstm_dates.shape)          # (num_sequences, seq_length)

(X_train_lstm, X_val_lstm, X_test_lstm,
 y_train_lstm, y_val_lstm, y_test_lstm,
 w_train_lstm, w_val_lstm, w_test_lstm,
 pos_train_lstm, pos_val_lstm, pos_test_lstm,
 date_train_lstm, date_val_lstm, date_test_lstm) = splitting(
    clstm_patches, clstm_labels, clstm_weights, clstm_positions, clstm_dates)

print("Final thermal train lstm shape:", X_train_lstm.shape)
print("Final labels train lstm shape:", y_train_lstm.shape)
print("Final weights train lstm shape:", w_train_lstm.shape)
print("Final positions train lstm shape:", pos_train_lstm.shape)
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
np.save("positions_train_lstm.npy", pos_train_lstm)
np.save("positions_val_lstm.npy", pos_val_lstm)
np.save("positions_test_lstm.npy", pos_test_lstm)
np.save("dates_train_lstm.npy", date_train_lstm)
np.save("dates_val_lstm.npy", date_val_lstm)
np.save("dates_test_lstm.npy", date_test_lstm)

print("✅ Processing complete! Data saved for ConvLSTM training.")
