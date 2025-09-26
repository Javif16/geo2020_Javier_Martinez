'''
This file carries out all the main pre-processing phase of the thermal images and the geological maps.
It also combines all the different datasets to generate multi-modal patches ready for CNN and ConvLSTM models.
'''

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
import tifffile
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
import re
from datetime import datetime


def extract_date_from_filename(filename):
    """
    Format: *_doyYYYYDDDHHMMSS_* where YYYY=year, DDD=day of year, HHMMSS=time
    """
    match = re.search(r'doy(\d{4})(\d{3})(\d{6})', filename)
    if match:
        year = match.group(1)
        day_of_year = match.group(2)
        time = match.group(3)
        return f"{year}{day_of_year}{time}"
    return None


def group_files_by_date(directory):
    """
    ECOSTRESS files by date and type.
    """
    grouped_files = {}

    all_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    for f in all_files:
        print(f"  {f}")
    if len(all_files) > 10:
        print(f"  ... and {len(all_files) - 10} more files")

    for filename in os.listdir(directory):
        if filename.endswith('.tif'):
            date = extract_date_from_filename(filename)
            if date:
                if date not in grouped_files:
                    grouped_files[date] = {}

                if '_LST_' in filename:
                    grouped_files[date]['lst'] = os.path.join(directory, filename)
                elif 'EmisWB' in filename:
                    grouped_files[date]['emiswb'] = os.path.join(directory, filename)
                elif 'cloud' in filename:
                    grouped_files[date]['cloud'] = os.path.join(directory, filename)
                elif 'QC' in filename:
                    grouped_files[date]['qc'] = os.path.join(directory, filename)
                elif 'water' in filename:
                    grouped_files[date]['water'] = os.path.join(directory, filename)

    print(f"Grouped {len(grouped_files)} dates:")
    for date, files in list(grouped_files.items())[:3]:  # Show first 3 dates
        print(f"  {date}: {list(files.keys())}")

    return grouped_files


def resample_to_match(source_file, target_file, output_path=None):
    """
    Resample to match shape.
    """
    with rasterio.open(target_file) as target:
        target_profile = target.profile.copy()
        target_transform = target.transform
        target_crs = target.crs
        target_shape = (target.height, target.width)

    with rasterio.open(source_file) as source:
        resampled_data = np.zeros(target_shape, dtype=source.dtypes[0])

        # Re-projection
        reproject(
            source=rasterio.band(source, 1),
            destination=resampled_data,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

    if output_path:
        target_profile.update(dtype=resampled_data.dtype)
        with rasterio.open(output_path, 'w', **target_profile) as dst:
            dst.write(resampled_data, 1)
        return output_path
    else:
        return resampled_data


def filter_dates_by_quality(grouped_files, max_cloud_percentage=10, max_bad_percentage=10):
    """
    Quality assessment for cloud coverage and QA quality.
    """
    # Good quality bits - based on documentation
    good_quality = [
        0, 16384, 16448, 16512, 16576, 16832, 17088, 17344, 17600, 17856,
        18112, 18368, 18624, 18880, 19136, 19392, 19904, 20160, 32768, 32832,
        32896, 32960, 33216, 33472, 33728, 33984, 34240, 34496, 34752, 35008,
        35264, 35520, 35776, 36288, 36544, 36800, 36928, 36992, 37056, 37312,
        37568, 37824, 38080, 38336, 38592, 38848, 39104, 39360, 39616, 39872,
        40128, 40384, 40640, 40896, 41408, 41664, 41920, 42432, 42688, 42944,
        43456, 43712, 43968, 44480, 44736, 44992, 49856, 50112, 50880, 51136,
        51904, 52160, 52928, 53184, 53952, 54208, 54976, 55232, 56000, 56256,
        57280, 58048, 58304, 59072, 59328, 60096, 60352, 61120, 61376, 63168,
        63424, 64192, 64448, 65216, 65472,
        16513, 16577, 16833, 17089, 17345, 17601, 17857, 18113, 18625, 18881,
        19137, 32897, 32961, 33217, 33473, 33985, 34241, 34497, 34753, 35265,
        35521, 35777, 36929, 36993, 37057, 37313, 37569, 37825, 38081, 38337,
        38593, 38849, 39105, 39361, 39617, 39873, 41409, 41665, 41921, 42689,
        42945, 43713, 43969, 49857, 50113, 50881, 51137, 51905, 52161, 53953,
        54209, 54977, 55233, 56001, 56257, 58049, 58305, 59073, 59329, 60097,
        60353, 64449
    ]

    filtered_dates = []
    date_masks = {}

    for date, files in grouped_files.items():
        required_files = ['lst', 'emiswb', 'cloud', 'qc']
        if not all(file_type in files for file_type in required_files):
            print(f"Date {date}: Missing required files, skipping")
            continue

        print(f"Processing date {date}...")
        try:
            # LST is reference for grid alignment - since lowest resolution
            lst_file = files['lst']
            # Match the cloud and qa data
            cloud_data = resample_to_match(files['cloud'], lst_file)
            qc_data = resample_to_match(files['qc'], lst_file)

            # cloud percentage
            total_pixels = cloud_data.size
            cloud_pixels = np.sum(cloud_data == 1)  # 1 means cloudy
            cloud_percentage = (cloud_pixels / total_pixels) * 100
            print(f"  Cloud coverage: {cloud_percentage:.2f}%")

            # bad quality percentage
            quality_mask = np.isin(qc_data, good_quality).astype(int)
            bad_pixels = np.sum(quality_mask == 0)
            bad_percentage = (bad_pixels / total_pixels) * 100
            print(f"  Bad quality: {bad_percentage:.2f}%")

            if cloud_percentage <= max_cloud_percentage and bad_percentage <= max_bad_percentage:
                filtered_dates.append(date)
                combined_mask = (cloud_data == 0) & (quality_mask == 1)
                date_masks[date] = combined_mask.astype(int)
                print(f"  -> ACCEPTED")
            else:
                reasons = []
                if cloud_percentage > max_cloud_percentage:
                    reasons.append(f"cloud coverage {cloud_percentage:.1f}% > {max_cloud_percentage}%")
                if bad_percentage > max_bad_percentage:
                    reasons.append(f"bad quality {bad_percentage:.1f}% > {max_bad_percentage}%")
                print(f"  -> REJECTED ({', '.join(reasons)})")
        except Exception as e:
            print(f"  -> ERROR processing date {date}: {e}")
            continue

    return filtered_dates, date_masks


def load_combined_thermal_data_filtered(grouped_files, filtered_dates):
    """
    Combine the LST and EmisWB data for filtered dates.
    """
    combined_data = []

    for date in filtered_dates:
        files = grouped_files[date]
        with rasterio.open(files['lst']) as lst_src:
            lst_data = lst_src.read(1)

        # EmisWB resampled to LST - 2 channels for thermal datasets
        emiswb_data = resample_to_match(files['emiswb'], files['lst'])
        thermal_combined = np.stack([lst_data, emiswb_data], axis=0)
        combined_data.append(thermal_combined)

    return combined_data


def create_thermal_dataset(thermal_data_list, quality_masks, mask_file, window_size=64, stride=64,
                           directory="./data_thermal"):
    """
    Manage dataset through windowing thermal images and applying quality mask.
    """
    os.makedirs(f"{directory}/thermal", exist_ok=True)
    os.makedirs(f"{directory}/mask", exist_ok=True)

    image_counter = 0
    total_images = len(thermal_data_list)

    for idx, (thermal_array, quality_mask) in enumerate(zip(thermal_data_list, quality_masks)):
        print(f"Processing thermal image {idx + 1} of {total_images}")
        n_bands, height, width = thermal_array.shape
        x_ind = 0
        while x_ind < (height - window_size):
            y_ind = 0
            while y_ind < (width - window_size):
                save_thermal_window(x_ind, y_ind, window_size, image_counter, thermal_array, quality_mask, mask_file, n_bands, directory)
                image_counter += 1
                y_ind += stride
            x_ind += stride
    print(f"Created {image_counter} thermal windows")


def save_thermal_window(x_ind, y_ind, window_size, image_id, thermal_array, quality_mask,
                        mask_file, n_bands, directory):
    """
    Windowed thermal image and corresponding mask.
    """
    window = Window(y_ind, x_ind, window_size, window_size)
    thermal_window = thermal_array[:, x_ind:x_ind + window_size, y_ind:y_ind + window_size]
    quality_window = quality_mask[x_ind:x_ind + window_size, y_ind:y_ind + window_size]

    # Quality application on thermal data
    for band in range(n_bands):
        thermal_window[band] = thermal_window[band] * quality_window
    # Validity of window - has to be more than 25%
    valid_pixel_ratio = np.sum(quality_window) / (window_size * window_size)
    if valid_pixel_ratio < 0.25:
        return

    try:
        # Geology mask window
        with rasterio.open(mask_file) as mask_raster:
            mask_window_data = mask_raster.read(window=window)
        if np.any(mask_window_data == -999):
            return

        thermal_profile = {
            'driver': 'GTiff',
            'dtype': thermal_window.dtype,
            'count': n_bands,
            'height': window_size,
            'width': window_size
        }

        thermal_path = f"{directory}/thermal/{image_id}.tif"
        with rasterio.open(thermal_path, 'w', **thermal_profile) as thermal_out:
            thermal_out.write(thermal_window)

        mask_profile = {
            'driver': 'GTiff',
            'dtype': mask_window_data.dtype,
            'count': mask_window_data.shape[0],
            'height': window_size,
            'width': window_size
        }

        mask_path = f"{directory}/mask/{image_id}.tif"
        with rasterio.open(mask_path, 'w', **mask_profile) as mask_out:
            mask_out.write(mask_window_data)

    except Exception as e:
        print(f"Warning: Could not save window {image_id}: {e}")


def load_thermal_data(thermal_path, mask_path):
    """Sorted lists of thermal and mask TIFF filenames."""
    thermal_files = []
    mask_files = []

    for file in os.listdir(thermal_path):
        if file.endswith('.tif'):
            thermal_files.append(file)
    for file in os.listdir(mask_path):
        if file.endswith('.tif'):
            mask_files.append(file)

    # Correspondence by sorting
    thermal_files.sort()
    mask_files.sort()

    return thermal_files, mask_files


def preprocess_thermal_data(thermal_files, mask_files, target_shape_thermal, target_shape_mask,
                            thermal_path, mask_path):
    """Preprocess everything for training in models."""
    # Dimensions
    m = len(thermal_files)
    t_h, t_w, t_c = target_shape_thermal
    m_h, m_w, m_c = target_shape_mask

    X = np.zeros((m, t_h, t_w, t_c), dtype=np.float32)
    y = np.zeros((m, m_h, m_w, m_c), dtype=np.int32)

    # Thermal images and corresponding masks
    for idx, thermal_file in enumerate(thermal_files):
        thermal_path_full = os.path.join(thermal_path, thermal_file)
        thermal_image = tifffile.imread(thermal_path_full)

        # Reshaping
        if len(thermal_image.shape) == 3:
            thermal_image = np.transpose(thermal_image, (1, 2, 0))
        thermal_image = np.reshape(thermal_image, (t_h, t_w, t_c))
        X[idx] = thermal_image

        # Corresponding mask
        mask_file = mask_files[idx]
        mask_path_full = os.path.join(mask_path, mask_file)
        mask_image = Image.open(mask_path_full)
        mask_image = mask_image.resize((m_h, m_w))
        mask_image = np.reshape(mask_image, (m_h, m_w, m_c))
        y[idx] = mask_image

    return X, y


def split_and_normalize_thermal_data(X, y, test_size=0.15, validation_size=0.15, random_state=42):
    """Splitting and normalizing thermal data."""
    # Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Test and Validating sets
    val_size_relative = validation_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=val_size_relative, random_state=random_state
    )

    count_label = np.count_nonzero(y_test < 25)
    count_no_data = np.count_nonzero(y_test == 25)
    total_pixels = np.count_nonzero(y_test)
    percent_labelled = count_label / total_pixels if total_pixels > 0 else 0

    print(f'Test sample randomization: {random_state}')
    print(f'Test sample size: {count_label}')
    print(f'Test percent_labelled: {percent_labelled:.4f}')
    print(f'Dataset shapes:')
    print(f'  X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'  X_valid: {X_valid.shape}, y_valid: {y_valid.shape}')
    print(f'  X_test: {X_test.shape}, y_test: {y_test.shape}')

    # No-data to 0 in masks
    for i in range(y_train.shape[0]):
        y_train[i, :, :] = tf.where(y_train[i, :, :] == 25, 0, y_train[i, :, :])
    for i in range(y_valid.shape[0]):
        y_valid[i, :, :] = tf.where(y_valid[i, :, :] == 25, 0, y_valid[i, :, :])
    for i in range(y_test.shape[0]):
        y_test[i, :, :] = tf.where(y_test[i, :, :] == 25, 0, y_test[i, :, :])

    # Normalizing
    print("Normalizing thermal data...")
    for band in range(X_train.shape[3]):
        print(f"Processing band {band + 1}/{X_train.shape[3]}")
        train_band_data = X_train[:, :, :, band].flatten()
        # Remove zero values AND NaN values
        valid_pixels = train_band_data[(train_band_data != 0) & (~np.isnan(train_band_data))]

        if len(valid_pixels) > 0:
            # Min-max normalization
            min_val = np.min(valid_pixels)
            max_val = np.max(valid_pixels)
            print(f"  Band {band + 1}: min={min_val:.2f}, max={max_val:.2f}, valid_pixels={len(valid_pixels)}")
            if max_val != min_val:
                # mask for valid pixels
                train_mask = (X_train[:, :, :, band] != 0) & (~np.isnan(X_train[:, :, :, band]))
                valid_mask = (X_valid[:, :, :, band] != 0) & (~np.isnan(X_valid[:, :, :, band]))
                test_mask = (X_test[:, :, :, band] != 0) & (~np.isnan(X_test[:, :, :, band]))

                # Normalizing and invalid pixels to 0
                X_train[:, :, :, band] = np.where(train_mask, (X_train[:, :, :, band] - min_val) / (max_val - min_val), 0)
                X_valid[:, :, :, band] = np.where(valid_mask, (X_valid[:, :, :, band] - min_val) / (max_val - min_val), 0)
                X_test[:, :, :, band] = np.where(test_mask, (X_test[:, :, :, band] - min_val) / (max_val - min_val), 0)
            else:
                print(f"  Band {band + 1} has constant values")
                X_train[:, :, :, band] = np.where(~np.isnan(X_train[:, :, :, band]), X_train[:, :, :, band], 0)
                X_valid[:, :, :, band] = np.where(~np.isnan(X_valid[:, :, :, band]), X_valid[:, :, :, band], 0)
                X_test[:, :, :, band] = np.where(~np.isnan(X_test[:, :, :, band]), X_test[:, :, :, band], 0)

            print("\nPost-normalization statistics:")
            for band in range(X_train.shape[3]):
                band_data = X_train[:, :, :, band]
                valid_data = band_data[(band_data != 0) & (~np.isnan(band_data))]
                if len(valid_data) > 0:
                    print(f"  Band {band + 1} normalized: min={np.min(valid_data):.4f}, max={np.max(valid_data):.4f}")
        else:
            print(f"  Band {band + 1} has no valid pixels, setting all to 0")
            X_train[:, :, :, band] = 0
            X_valid[:, :, :, band] = 0
            X_test[:, :, :, band] = 0
    print("Normalization complete!")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def create_time_sequences_from_patches(X_train, X_valid, X_test, y_train, y_valid, y_test,
                                       sequence_length=5, overlap=2, min_valid_frames=3):
    """
    Create time sequences from existing CNN patches for ConvLSTM training.

    Args:
        X_train, X_valid, X_test: CNN patch arrays (samples, height, width, channels)
        y_train, y_valid, y_test: Corresponding mask arrays
        sequence_length: Number of time steps in each sequence
        overlap: Number of overlapping frames between consecutive sequences
        min_valid_frames: Minimum number of non-zero frames required in a sequence

    Returns:
        ConvLSTM-compatible arrays: (samples, timesteps, height, width, channels)
    """

    def create_sequences_from_split(X_split, y_split, split_name):
        """Helper function to create sequences from a data split"""
        print(f"Creating {split_name} sequences...")

        n_samples, height, width, channels = X_split.shape
        stride = sequence_length - overlap

        # Calculate number of possible sequences
        n_sequences = max(0, (n_samples - sequence_length) // stride + 1)

        if n_sequences == 0:
            print(f"Warning: Not enough samples in {split_name} to create sequences")
            return np.array([]).reshape(0, sequence_length, height, width, channels), \
                np.array([]).reshape(0, height, width, 1)

        # Initialize sequence arrays
        X_sequences = []
        y_sequences = []

        valid_sequence_count = 0

        for i in range(0, n_samples - sequence_length + 1, stride):
            # Extract sequence
            X_seq = X_split[i:i + sequence_length]  # (seq_len, h, w, c)
            y_seq = y_split[i + sequence_length - 1]  # Use last frame's mask (h, w, 1)

            # Quality check: count non-zero frames in sequence
            valid_frames = 0
            for frame_idx in range(sequence_length):
                # Check if frame has meaningful data (not all zeros)
                frame_sum = np.sum(X_seq[frame_idx])
                if frame_sum > 0:
                    valid_frames += 1

            # Only keep sequences with enough valid frames
            if valid_frames >= min_valid_frames:
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)
                valid_sequence_count += 1

        if len(X_sequences) == 0:
            print(f"Warning: No valid sequences found in {split_name}")
            return np.array([]).reshape(0, sequence_length, height, width, channels), \
                np.array([]).reshape(0, height, width, 1)

        # Convert to numpy arrays
        X_sequences = np.array(X_sequences)  # (n_seq, seq_len, h, w, c)
        y_sequences = np.array(y_sequences)  # (n_seq, h, w, 1)

        print(f"  {split_name}: {n_samples} patches → {valid_sequence_count} sequences")
        print(f"  Shape: {X_sequences.shape}")

        return X_sequences, y_sequences

    # Create sequences for each split
    X_train_seq, y_train_seq = create_sequences_from_split(X_train, y_train, "Training")
    X_valid_seq, y_valid_seq = create_sequences_from_split(X_valid, y_valid, "Validation")
    X_test_seq, y_test_seq = create_sequences_from_split(X_test, y_test, "Test")

    print(f"\nConvLSTM sequence creation complete!")
    print(f"Sequence length: {sequence_length}, Overlap: {overlap}")
    print(f"Final shapes:")
    print(f"  X_train_seq: {X_train_seq.shape}")
    print(f"  X_valid_seq: {X_valid_seq.shape}")
    print(f"  X_test_seq: {X_test_seq.shape}")

    return X_train_seq, X_valid_seq, X_test_seq, y_train_seq, y_valid_seq, y_test_seq


def process_ecostress_thermal_data(data_dir, mask_file, target_shape_thermal=(64, 64, 2), target_shape_mask=(64, 64, 1),
                                   max_cloud_percentage=10, max_bad_percentage=10, sequence_length=5, overlap=2):
    """
    Complete pipeline for processing ECOSTRESS thermal data.
    """

    print("Step 1: Grouping files by date...")
    grouped_files = group_files_by_date(data_dir)
    print(f"Found {len(grouped_files)} unique dates")

    print("\nStep 2: Filtering dates by quality...")
    filtered_dates, date_masks = filter_dates_by_quality(grouped_files, max_cloud_percentage, max_bad_percentage)
    print(f"Selected {len(filtered_dates)} dates after quality filtering")

    if len(filtered_dates) == 0:
        print("ERROR: No dates passed quality filtering!")
        return None

    print("\nStep 3: Loading thermal data for filtered dates...")
    thermal_data_combined = load_combined_thermal_data_filtered(grouped_files, filtered_dates)
    quality_masks = [date_masks[date] for date in filtered_dates]

    print(f"Loaded {len(thermal_data_combined)} thermal images")

    print("\nStep 4: Creating windowed dataset...")
    create_thermal_dataset(thermal_data_combined, quality_masks, mask_file)

    print("\nStep 5: Loading windowed data...")
    thermal_list, mask_list = load_thermal_data("./data_thermal/thermal", "./data_thermal/mask")

    print("\nStep 6: Preprocessing data...")
    X, y = preprocess_thermal_data(thermal_list, mask_list, target_shape_thermal, target_shape_mask, "./data_thermal/thermal", "./data_thermal/mask")
    print(f"Debug: NaN count in X: {np.sum(np.isnan(X))}")
    print(f"Debug: Zero count in X: {np.sum(X == 0)}")
    print(f"Debug: Total pixels in X: {X.size}")

    print("\nStep 7: Splitting and normalizing data...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_and_normalize_thermal_data(X, y)
    print(f"\nFinal dataset shapes:")
    print(f"Thermal data (X): {X_train.shape}")  # (samples, 64, 64, 2)
    print(f"Mask data (y): {y_train.shape}")  # (samples, 64, 64, 1)
    print(f"\nCNN compatibility: {'✓' if len(X_train.shape) == 4 else '✗'}")

    print("\nStep 8: Saving processed dataset...")
    save_dir = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Outputs/Santa"
    os.makedirs(save_dir, exist_ok=True)
    # CNN
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_valid.npy'), X_valid)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_valid.npy'), y_valid)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    print("Dataset saved for CNN")

    print("\nStep 9: Saving ConvLSTM processed dataset...")
    X_train_seq, X_valid_seq, X_test_seq, y_train_seq, y_valid_seq, y_test_seq = create_time_sequences_from_patches(
        X_train, X_valid, X_test, y_train, y_valid, y_test, sequence_length=sequence_length, overlap=overlap)

    # Save ConvLSTM data
    np.save(os.path.join(save_dir, 'X_train_seq.npy'), X_train_seq)
    np.save(os.path.join(save_dir, 'X_valid_seq.npy'), X_valid_seq)
    np.save(os.path.join(save_dir, 'X_test_seq.npy'), X_test_seq)
    np.save(os.path.join(save_dir, 'y_train_seq.npy'), y_train_seq)
    np.save(os.path.join(save_dir, 'y_valid_seq.npy'), y_valid_seq)
    np.save(os.path.join(save_dir, 'y_test_seq.npy'), y_test_seq)

    return {'cnn': (X_train, X_valid, X_test, y_train, y_valid, y_test),
            'convlstm': (X_train_seq, X_valid_seq, X_test_seq, y_train_seq, y_valid_seq, y_test_seq)}


results = process_ecostress_thermal_data(
        data_dir="C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/THERMAL/Santa_full/",
        mask_file="C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Masks/Santa Olalla masks/Geo_map_resized_Santa.tif",
        sequence_length=5, overlap=2)

X_train, X_valid, X_test, y_train, y_valid, y_test = results['cnn']
X_train_seq, X_valid_seq, X_test_seq, y_train_seq, y_valid_seq, y_test_seq = results['convlstm']