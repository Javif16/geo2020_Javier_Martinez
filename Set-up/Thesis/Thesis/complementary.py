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
from PIL import Image as PILImage
import tensorflow as tf
from sklearn.model_selection import train_test_split
import re
from datetime import datetime, timedelta
import json


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


def extract_date_from_ndvi_filename(filename):
    """
    Extract date from NDVI filename to match thermal dates.
    """
    match = re.search(r'S2_NDVI_(\d{8})', filename)
    if match:
        date_str = match.group(1)
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        return date_obj.strftime('%Y%j')  # %j gives day of year
    return None


def extract_date_from_optical_filename(filename):
    """Date from optical filename format: LC08_L2SP_200033_20210801_..."""
    match = re.search(r'_(\d{8})_', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d').strftime('%Y%j')
    return None


def extract_date_from_sar_filename(filename):
    """Date from SAR filename format: S1A_VV_VH_20190110"""
    match = re.search(r'_(\d{8})$', filename.replace('.tif', ''))
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d').strftime('%Y%j')
    return None


def group_ndvi_files_by_date(ndvi_dir):
    """
    Group NDVI files by date.
    Returns: dict mapping date strings (YYYYDDD format) to file paths
    """
    grouped_files = {}

    print(f"\nScanning NDVI directory: {ndvi_dir}")
    for filename in os.listdir(ndvi_dir):
        if filename.endswith('.tif'):
            date = extract_date_from_ndvi_filename(filename)
            if date:
                grouped_files[date] = os.path.join(ndvi_dir, filename)
                print(f"  Found NDVI for date {date}: {filename}")
            else:
                print(f"  Warning: Could not extract date from {filename}")

    print(f"Total NDVI dates found: {len(grouped_files)}")
    return grouped_files


def group_optical_files_by_date(optical_dir):
    """Optical files (R, G, B) by date."""
    grouped_files = {}
    for filename in os.listdir(optical_dir):
        if filename.endswith('.tif'):
            date = extract_date_from_optical_filename(filename)
            if date:
                if date not in grouped_files:
                    grouped_files[date] = {}
                if '_B2' in filename or 'Blue' in filename:
                    grouped_files[date]['blue'] = os.path.join(optical_dir, filename)
                elif '_B3' in filename or 'Green' in filename:
                    grouped_files[date]['green'] = os.path.join(optical_dir, filename)
                elif '_B4' in filename or 'Red' in filename:
                    grouped_files[date]['red'] = os.path.join(optical_dir, filename)
    return grouped_files


def group_sar_files_by_date(sar_dir):
    """SAR files by date."""
    grouped_files = {}
    for filename in os.listdir(sar_dir):
        if filename.endswith('.tif'):
            date = extract_date_from_sar_filename(filename)
            if date:
                grouped_files[date] = os.path.join(sar_dir, filename)
    return grouped_files


def resample_to_match_array(source_array, target_shape):
    """Resample a 2D array to match target shape."""
    if source_array.shape == target_shape:
        return source_array
    pil_image = PILImage.fromarray(source_array)
    resized = pil_image.resize((target_shape[1], target_shape[0]), PILImage.LANCZOS)
    return np.array(resized)


def parse_thermal_datetime(date_str):
    """Parse thermal date string to datetime object"""
    match = re.match(r"(\d{4})(\d{3})(\d{6})", date_str)
    if match:
        year = int(match.group(1))
        doy = int(match.group(2))
        time_str = match.group(3)
        base_date = datetime(year, 1, 1)
        date_obj = base_date + timedelta(days=doy - 1)
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        date_obj = date_obj.replace(hour=hour, minute=minute, second=second)
        return date_obj
    return None


def match_ndvi_to_thermal_dates(thermal_dates, ndvi_grouped, max_days_diff=16):
    """
    Match NDVI dates to thermal dates within a temporal window.

    Args:
        thermal_dates: list of thermal date strings (YYYYDDDHHMMSS format)
        ndvi_grouped: dict from group_ndvi_files_by_date
        max_days_diff: maximum days difference to consider a match

    Returns:
        dict mapping thermal_date -> ndvi_file_path
    """

    def parse_date(date_str):
        """Parse YYYYDDD format to datetime"""
        year = int(date_str[:4])
        doy = int(date_str[4:7])
        return datetime(year, 1, 1) + timedelta(days=doy - 1)

    matched_ndvi = {}

    for thermal_date in thermal_dates:
        # Convert thermal date to YYYYDDD format
        thermal_date_short = thermal_date[:7]  # Extract YYYYDDD from YYYYDDDHHMMSS
        thermal_dt = parse_date(thermal_date_short)

        # Find closest NDVI date
        best_match = None
        min_diff = float('inf')

        for ndvi_date, ndvi_path in ndvi_grouped.items():
            ndvi_dt = parse_date(ndvi_date)
            days_diff = abs((thermal_dt - ndvi_dt).days)

            if days_diff < min_diff and days_diff <= max_days_diff:
                min_diff = days_diff
                best_match = ndvi_path

        if best_match:
            matched_ndvi[thermal_date] = best_match
            print(f"  Thermal {thermal_date} -> NDVI {os.path.basename(best_match)} ({min_diff} days)")
        else:
            print(f"  Warning: No NDVI match found for thermal date {thermal_date}")

    print(f"\nMatched {len(matched_ndvi)} thermal dates with NDVI")
    return matched_ndvi


def is_day_time(dt):
    """Datetime is day time (6:00-17:59)"""
    return 6 <= dt.hour <= 17


def is_summer(dt):
    """Datetime is summer (March 22 - September 21)"""
    march_22 = datetime(dt.year, 3, 22)
    sept_21 = datetime(dt.year, 9, 21)
    return march_22 <= dt <= sept_21


def filter_dates_by_quality(grouped_files, max_cloud_percentage=12, max_bad_percentage=15):
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


def create_multimodal_full_images(thermal_data_list, quality_masks, filtered_dates, optical_grouped=None, sar_grouped=None):
    """Combine thermal with optical/SAR data at full resolution."""
    def thermal_date_to_standard(date_str):
        """Convert thermal date format to YYYYDDD for matching"""
        match = re.match(r"(\d{4})(\d{3})(\d{6})", date_str)
        if match:
            return f"{match.group(1)}{match.group(2)}"
        return None

    results = {
        'thermal_only': ([], []),
        'thermal_optical': ([], []),
        'thermal_sar': ([], []),
        'thermal_optical_sar': ([], [])
    }

    for idx, (thermal_array, quality_mask, date_str) in enumerate(
            zip(thermal_data_list, quality_masks, filtered_dates)):

        results['thermal_only'][0].append(thermal_array)
        results['thermal_only'][1].append(quality_mask)

        thermal_date = thermal_date_to_standard(date_str)
        if not thermal_date:
            continue

        _, thermal_height, thermal_width = thermal_array.shape
        optical_data = None
        sar_data = None

        # Resize optical data
        if optical_grouped and thermal_date in optical_grouped:
            optical_files = optical_grouped[thermal_date]
            if all(channel in optical_files for channel in ['red', 'green', 'blue']):
                try:
                    red = tifffile.imread(optical_files['red'])
                    green = tifffile.imread(optical_files['green'])
                    blue = tifffile.imread(optical_files['blue'])

                    if len(red.shape) > 2:
                        red = red[:, :, 0] if len(red.shape) == 3 else red
                    if len(green.shape) > 2:
                        green = green[:, :, 0] if len(green.shape) == 3 else green
                    if len(blue.shape) > 2:
                        blue = blue[:, :, 0] if len(blue.shape) == 3 else blue

                    red_resized = resample_to_match_array(red, (thermal_height, thermal_width))
                    green_resized = resample_to_match_array(green, (thermal_height, thermal_width))
                    blue_resized = resample_to_match_array(blue, (thermal_height, thermal_width))

                    optical_data = np.stack([red_resized, green_resized, blue_resized], axis=0)
                    print(f"  ✓ Optical data loaded: {optical_data.shape}")
                except Exception as e:
                    print(f"  ✗ Error loading optical: {e}")

        # Resize SAR data
        if sar_grouped and thermal_date in sar_grouped:
            try:
                sar_image = tifffile.imread(sar_grouped[thermal_date])
                if len(sar_image.shape) == 3 and sar_image.shape[0] == 2:
                    vv_resized = resample_to_match_array(sar_image[0], (thermal_height, thermal_width))
                    vh_resized = resample_to_match_array(sar_image[1], (thermal_height, thermal_width))
                    sar_data = np.stack([vv_resized, vh_resized], axis=0)
                elif len(sar_image.shape) == 3 and sar_image.shape[2] == 2:
                    sar_transposed = np.transpose(sar_image, (2, 0, 1))
                    vv_resized = resample_to_match_array(sar_transposed[0], (thermal_height, thermal_width))
                    vh_resized = resample_to_match_array(sar_transposed[1], (thermal_height, thermal_width))
                    sar_data = np.stack([vv_resized, vh_resized], axis=0)
                elif len(sar_image.shape) == 2:
                    sar_resized = resample_to_match_array(sar_image, (thermal_height, thermal_width))
                    sar_data = np.stack([sar_resized, sar_resized], axis=0)
                print(f"  ✓ SAR data loaded: {sar_data.shape}")
            except Exception as e:
                print(f"  ✗ Error loading SAR: {e}")

        # Combinations
        if optical_data is not None:
            thermal_optical = np.concatenate([thermal_array, optical_data], axis=0)
            results['thermal_optical'][0].append(thermal_optical)
            results['thermal_optical'][1].append(quality_mask)

        if sar_data is not None:
            thermal_sar = np.concatenate([thermal_array, sar_data], axis=0)
            results['thermal_sar'][0].append(thermal_sar)
            results['thermal_sar'][1].append(quality_mask)

        if optical_data is not None and sar_data is not None:
            thermal_optical_sar = np.concatenate([thermal_array, optical_data, sar_data], axis=0)
            results['thermal_optical_sar'][0].append(thermal_optical_sar)
            results['thermal_optical_sar'][1].append(quality_mask)

    return results


def filter_thermal_data_by_time_season(thermal_data_list, quality_masks, filtered_dates):
    """Filter thermal data into day/night and summer/winter categories"""
    categories = {
        'thermal_day': ([], []),
        'thermal_night': ([], []),
        'thermal_summer': ([], []),
        'thermal_winter': ([], [])
    }

    for thermal_array, quality_mask, date_str in zip(thermal_data_list, quality_masks, filtered_dates):
        dt = parse_thermal_datetime(date_str)
        if dt is None:
            continue

        if is_day_time(dt):
            categories['thermal_day'][0].append(thermal_array)
            categories['thermal_day'][1].append(quality_mask)
        else:
            categories['thermal_night'][0].append(thermal_array)
            categories['thermal_night'][1].append(quality_mask)

        if is_summer(dt):
            categories['thermal_summer'][0].append(thermal_array)
            categories['thermal_summer'][1].append(quality_mask)
        else:
            categories['thermal_winter'][0].append(thermal_array)
            categories['thermal_winter'][1].append(quality_mask)

    return categories


def create_thermal_dataset(thermal_data_list, quality_masks, mask_file, filtered_dates, matched_ndvi, window_size=128, stride=64, directory="./data_thermal"):
    """
    Dataset with full coverage and position tracking.
    """
    os.makedirs(f"{directory}/thermal", exist_ok=True)
    os.makedirs(f"{directory}/mask", exist_ok=True)
    os.makedirs(f"{directory}/ndvi", exist_ok=True)

    image_counter = 0
    total_images = len(thermal_data_list)
    positions_dict = {}
    ndvi_metadata = {}

    for date_idx, (thermal_array, quality_mask, date_str) in enumerate(
            zip(thermal_data_list, quality_masks, filtered_dates)):

        print(f"Processing thermal image {date_idx + 1} of {total_images} (date: {date_str})")
        n_bands, height, width = thermal_array.shape

        # Load and resample NDVI if available for this date
        ndvi_array = None
        if date_str in matched_ndvi:
            try:
                ndvi_raw = tifffile.imread(matched_ndvi[date_str])
                # Handle multi-band NDVI (take first band if needed)
                if len(ndvi_raw.shape) == 3:
                    ndvi_raw = ndvi_raw[0] if ndvi_raw.shape[0] < ndvi_raw.shape[1] else ndvi_raw[:, :, 0]

                # Resample NDVI to match thermal grid
                ndvi_array = resample_to_match_array(ndvi_raw, (height, width))
                print(f"  ✓ NDVI loaded and resampled: {ndvi_array.shape}")
            except Exception as e:
                print(f"  ✗ Error loading NDVI for {date_str}: {e}")
        else:
            print(f"  ○ No NDVI available for {date_str}")

        # Sliding window
        x_ind = 0
        while x_ind <= (height - window_size):
            y_ind = 0
            while y_ind <= (width - window_size):
                success = save_thermal_window(
                    x_ind, y_ind, window_size, image_counter, thermal_array,
                    quality_mask, mask_file, n_bands, directory, date_idx,
                    positions_dict, ndvi_array, ndvi_metadata)
                if success:
                    image_counter += 1
                y_ind += stride
            x_ind += stride

    # Save positions metadata
    positions_file = os.path.join(directory, 'patch_positions.json')
    with open(positions_file, 'w') as f:
        json.dump(positions_dict, f, indent=2)
    ndvi_meta_file = os.path.join(directory, 'ndvi_metadata.json')
    with open(ndvi_meta_file, 'w') as f:
        json.dump(ndvi_metadata, f, indent=2)

    print(f"Created {image_counter} thermal windows")
    print(f"Patches with NDVI: {len(ndvi_metadata)}")
    print(f"Saved position metadata to {positions_file}")
    print(f"Saved NDVI metadata to {ndvi_meta_file}")

    return positions_dict, ndvi_metadata


def save_thermal_window(x_ind, y_ind, window_size, image_id, thermal_array, quality_mask, mask_file, n_bands, directory,
                        date_idx, positions_dict, ndvi_array, ndvi_metadata):
    """
    Windowed thermal image and corresponding mask with position tracking.
    """
    window = Window(y_ind, x_ind, window_size, window_size)
    thermal_window = thermal_array[:, x_ind:x_ind + window_size, y_ind:y_ind + window_size]
    quality_window = quality_mask[x_ind:x_ind + window_size, y_ind:y_ind + window_size]

    for band in range(n_bands):
        thermal_window[band] = thermal_window[band] * quality_window

    # Validity of window - more than 25%
    valid_pixel_ratio = np.sum(quality_window) / (window_size * window_size)
    if valid_pixel_ratio < 0.25:
        return False

    try:
        # Geology mask window
        with rasterio.open(mask_file) as mask_raster:
            mask_window_data = mask_raster.read(window=window)
        if np.any(mask_window_data == -999):
            return False

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

        if ndvi_array is not None:
            ndvi_window = ndvi_array[x_ind:x_ind + window_size, y_ind:y_ind + window_size]

            # Apply same quality mask to NDVI
            ndvi_window = ndvi_window * quality_window

            ndvi_profile = {
                'driver': 'GTiff',
                'dtype': ndvi_window.dtype,
                'count': 1,
                'height': window_size,
                'width': window_size
            }
            ndvi_path = f"{directory}/ndvi/{image_id}.tif"
            with rasterio.open(ndvi_path, 'w', **ndvi_profile) as ndvi_out:
                ndvi_out.write(ndvi_window, 1)

            # Record that this patch has NDVI
            ndvi_metadata[str(image_id)] = {
                'has_ndvi': True,
                'ndvi_path': ndvi_path,
                'valid_ndvi_ratio': float(np.sum(ndvi_window != 0) / (window_size * window_size))
            }
        else:
            ndvi_metadata[str(image_id)] = {'has_ndvi': False}

        # Position metadata
        positions_dict[str(image_id)] = {
            'x': int(x_ind),
            'y': int(y_ind),
            'date_idx': int(date_idx)}

        return True

    except Exception as e:
        print(f"Warning: Could not save window {image_id}: {e}")
        return False


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


def load_ndvi_data_for_split(patch_ids, ndvi_dir, ndvi_metadata):
    """
    Load NDVI data for a specific data split (train/valid/test).

    Args:
        patch_ids: list of patch IDs (e.g., ['0', '1', '2', ...])
        ndvi_dir: directory containing NDVI patches
        ndvi_metadata: metadata dict from ndvi_metadata.json

    Returns:
        ndvi_array: (n_patches, 64, 64) array with NDVI values
        has_ndvi_mask: (n_patches,) boolean array indicating which patches have NDVI
    """
    n_patches = len(patch_ids)
    ndvi_array = np.zeros((n_patches, 128, 128), dtype=np.float32)
    has_ndvi_mask = np.zeros(n_patches, dtype=bool)

    for idx, patch_id in enumerate(patch_ids):
        patch_id_str = str(patch_id)
        if patch_id_str in ndvi_metadata and ndvi_metadata[patch_id_str]['has_ndvi']:
            try:
                ndvi_path = os.path.join(ndvi_dir, f"{patch_id}.tif")
                ndvi_patch = tifffile.imread(ndvi_path)
                if len(ndvi_patch.shape) == 3:
                    ndvi_patch = ndvi_patch[0]  # Take first band
                ndvi_array[idx] = ndvi_patch
                has_ndvi_mask[idx] = True
            except Exception as e:
                print(f"Warning: Could not load NDVI for patch {patch_id}: {e}")

    print(f"Loaded NDVI for {np.sum(has_ndvi_mask)}/{n_patches} patches")
    return ndvi_array, has_ndvi_mask


def preprocess_thermal_data(thermal_files, mask_files, target_shape_thermal,
                                           target_shape_mask, thermal_path, mask_path,
                                           positions_file):
    """
    Preprocess data and load position metadata.
    """
    with open(positions_file, 'r') as f:
        positions_dict = json.load(f)
    positions_array = []

    m = len(thermal_files)
    t_h, t_w, t_c = target_shape_thermal
    m_h, m_w, m_c = target_shape_mask

    X = np.zeros((m, t_h, t_w, t_c), dtype=np.float32)
    y = np.zeros((m, m_h, m_w, m_c), dtype=np.int32)

    for idx, thermal_file in enumerate(thermal_files):
        # Thermal and mask
        thermal_path_full = os.path.join(thermal_path, thermal_file)
        thermal_image = tifffile.imread(thermal_path_full)

        if len(thermal_image.shape) == 3:
            thermal_image = np.transpose(thermal_image, (1, 2, 0))
        thermal_image = np.reshape(thermal_image, (t_h, t_w, t_c))
        X[idx] = thermal_image

        mask_file = mask_files[idx]
        mask_path_full = os.path.join(mask_path, mask_file)
        mask_image = Image.open(mask_path_full)
        mask_image = mask_image.resize((m_h, m_w))
        mask_image = np.reshape(mask_image, (m_h, m_w, m_c))
        y[idx] = mask_image

        # Position for patch
        patch_id = thermal_file.replace('.tif', '')
        if patch_id in positions_dict:
            pos = positions_dict[patch_id]
            positions_array.append([pos['x'], pos['y'], pos['date_idx']])
        else:
            print(f"Warning: No position found for {patch_id}")
            positions_array.append([0, 0, 0])

    positions_array = np.array(positions_array)
    return X, y, positions_array


def split_thermal_data(X, y, positions, test_size=0.15, validation_size=0.15):
    """Splitting and normalizing thermal data."""
    # Training
    X_train, X_test, y_train, y_test, pos_train, pos_test = train_test_split(X, y, positions, test_size=test_size, shuffle=False)
    # Testing
    val_size_relative = validation_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid, pos_train, pos_valid = train_test_split(X_train, y_train, pos_train, test_size=val_size_relative, shuffle=False)

    count_label = np.count_nonzero(y_test < 25)
    total_pixels = np.count_nonzero(y_test)
    percent_labelled = count_label / total_pixels if total_pixels > 0 else 0

    print(f'Test sample size: {count_label}')
    print(f'Test percent_labelled: {percent_labelled:.4f}')
    print(f'Dataset shapes:')
    print(f'  X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'  X_valid: {X_valid.shape}, y_valid: {y_valid.shape}')
    print(f'  X_test: {X_test.shape}, y_test: {y_test.shape}')
    print(f'  Positions shapes: train={pos_train.shape}, valid={pos_valid.shape}, test={pos_test.shape}')

    # No-data to 0 in masks
    for i in range(y_train.shape[0]):
        y_train[i, :, :] = tf.where(y_train[i, :, :] == 25, 0, y_train[i, :, :])
    for i in range(y_valid.shape[0]):
        y_valid[i, :, :] = tf.where(y_valid[i, :, :] == 25, 0, y_valid[i, :, :])
    for i in range(y_test.shape[0]):
        y_test[i, :, :] = tf.where(y_test[i, :, :] == 25, 0, y_test[i, :, :])

    # Normalizing thermal data
    print("Normalizing thermal data...")
    for band in range(X_train.shape[3]):
        train_band_data = X_train[:, :, :, band].flatten()
        valid_pixels = train_band_data[(train_band_data != 0) & (~np.isnan(train_band_data))]

        if len(valid_pixels) > 0:
            min_val = np.min(valid_pixels)
            max_val = np.max(valid_pixels)
            if max_val != min_val:
                train_mask = (X_train[:, :, :, band] != 0) & (~np.isnan(X_train[:, :, :, band]))
                valid_mask = (X_valid[:, :, :, band] != 0) & (~np.isnan(X_valid[:, :, :, band]))
                test_mask = (X_test[:, :, :, band] != 0) & (~np.isnan(X_test[:, :, :, band]))

                X_train[:, :, :, band] = np.where(train_mask, (X_train[:, :, :, band] - min_val) / (max_val - min_val),
                                                  0)
                X_valid[:, :, :, band] = np.where(valid_mask, (X_valid[:, :, :, band] - min_val) / (max_val - min_val),
                                                  0)
                X_test[:, :, :, band] = np.where(test_mask, (X_test[:, :, :, band] - min_val) / (max_val - min_val), 0)

    print("Normalization complete!")

    return X_train, X_valid, X_test, y_train, y_valid, y_test, pos_train, pos_valid, pos_test


def create_convlstm_positions(pos_train, pos_valid, pos_test, sequence_length=5, overlap=2):
    """
    Create position metadata for ConvLSTM sequences.
    Each sequence inherits the position of its last frame.
    """

    def sequences_from_positions(positions, split_name):
        n_samples = len(positions)
        stride = sequence_length - overlap
        n_sequences = max(0, (n_samples - sequence_length) // stride + 1)

        seq_positions = []
        for i in range(0, n_samples - sequence_length + 1, stride):
            # Use the last frame's position for the sequence
            last_frame_pos = positions[i + sequence_length - 1]
            seq_positions.append(last_frame_pos)

        return np.array(seq_positions)

    pos_train_seq = sequences_from_positions(pos_train, "Training")
    pos_valid_seq = sequences_from_positions(pos_valid, "Validation")
    pos_test_seq = sequences_from_positions(pos_test, "Test")

    print(f"\nConvLSTM position shapes:")
    print(f"  pos_train_seq: {pos_train_seq.shape}")
    print(f"  pos_valid_seq: {pos_valid_seq.shape}")
    print(f"  pos_test_seq: {pos_test_seq.shape}")

    return pos_train_seq, pos_valid_seq, pos_test_seq


def create_time_sequences_from_patches(X_train, X_valid, X_test, y_train, y_valid, y_test,
                                       sequence_length=5, overlap=2, min_valid_frames=3):
    """
    Time sequences from existing CNN patches for ConvLSTM training.
    """
    def create_sequences_from_split(X_split, y_split, split_name):
        """Sequences from a data split"""
        print(f"Creating {split_name} sequences...")

        n_samples, height, width, channels = X_split.shape
        stride = sequence_length - overlap
        n_sequences = max(0, (n_samples - sequence_length) // stride + 1)

        if n_sequences == 0:
            print(f"Warning: Not enough samples in {split_name} to create sequences")
            return np.array([]).reshape(0, sequence_length, height, width, channels), \
                np.array([]).reshape(0, height, width, 1)

        # Sequences
        X_sequences = []
        y_sequences = []

        valid_sequence_count = 0

        for i in range(0, n_samples - sequence_length + 1, stride):
            X_seq = X_split[i:i + sequence_length]  # (seq_len, h, w, c)
            y_seq = y_split[i + sequence_length - 1]  # Use last frame's mask (h, w, 1)

            # QC: count non-zero frames in sequence
            valid_frames = 0
            for frame_idx in range(sequence_length):
                frame_sum = np.sum(X_seq[frame_idx])
                if frame_sum > 0:
                    valid_frames += 1

            if valid_frames >= min_valid_frames:
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)
                valid_sequence_count += 1

        if len(X_sequences) == 0:
            print(f"Warning: No valid sequences found in {split_name}")
            return np.array([]).reshape(0, sequence_length, height, width, channels), \
                np.array([]).reshape(0, height, width, 1)

        X_sequences = np.array(X_sequences)  # (n_seq, seq_len, h, w, c)
        y_sequences = np.array(y_sequences)  # (n_seq, h, w, 1)

        print(f"  {split_name}: {n_samples} patches → {valid_sequence_count} sequences")
        print(f"  Shape: {X_sequences.shape}")

        return X_sequences, y_sequences

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


def create_ndvi_sequences_from_patches(ndvi_patches, has_ndvi_mask, sequence_length=5,
                                       overlap=2, min_valid_frames=3):
    """
    Create NDVI sequences that correspond to ConvLSTM thermal sequences.
    """
    n_patches, height, width = ndvi_patches.shape
    stride = sequence_length - overlap
    n_sequences = max(0, (n_patches - sequence_length) // stride + 1)

    if n_sequences == 0:
        return (np.array([]).reshape(0, sequence_length, height, width),
                np.array([]).reshape(0),
                np.array([]).reshape(0))

    ndvi_sequences = np.zeros((n_sequences, sequence_length, height, width), dtype=np.float32)
    has_ndvi_seq_mask = np.zeros(n_sequences, dtype=bool)
    ndvi_frame_counts = np.zeros(n_sequences, dtype=int)

    seq_idx = 0
    for i in range(0, n_patches - sequence_length + 1, stride):
        # Extract sequence
        ndvi_seq = ndvi_patches[i:i + sequence_length]  # (seq_len, h, w)
        has_ndvi_seq = has_ndvi_mask[i:i + sequence_length]  # (seq_len,)

        # Count how many frames have valid NDVI
        valid_frame_count = np.sum(has_ndvi_seq)

        ndvi_sequences[seq_idx] = ndvi_seq
        ndvi_frame_counts[seq_idx] = valid_frame_count

        # Mark sequence as having NDVI if it meets minimum threshold
        if valid_frame_count >= min_valid_frames:
            has_ndvi_seq_mask[seq_idx] = True

        seq_idx += 1

    print(f"\nNDVI Sequence Statistics:")
    print(f"  Total sequences: {n_sequences}")
    print(f"  Sequences with ≥{min_valid_frames} NDVI frames: {np.sum(has_ndvi_seq_mask)}")
    print(f"  Average NDVI frames per sequence: {np.mean(ndvi_frame_counts):.1f}")

    return ndvi_sequences, has_ndvi_seq_mask, ndvi_frame_counts


def aggregate_sequence_ndvi(ndvi_sequence, has_ndvi_frames):
    """
    Aggregate NDVI from a sequence into a single value.
    """
    valid_indices = np.where(has_ndvi_frames)[0]

    if len(valid_indices) == 0:
        return np.zeros((128, 128), dtype=np.float32)

    # Mean of all valid frames
    valid_frames = ndvi_sequence[valid_indices]
    aggregated = np.mean(valid_frames, axis=0)

    return aggregated


def process_ecostress_thermal_data(data_dir, mask_file, ndvi_dir=None, optical_dir=None, sar_dir=None,
                                   target_shape_thermal=(64, 64, 2), target_shape_mask=(128, 128, 1),
                                   max_cloud_percentage=12, max_bad_percentage=15, sequence_length=5, overlap=2,
                                   max_ndvi_days_diff=7):
    """
    Complete pipeline for processing ECOSTRESS thermal data for diurnal, seasonal and multi-modal patches.
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

    matched_ndvi = {}
    if ndvi_dir and os.path.exists(ndvi_dir):
        print("\nStep 3.5: Processing remaining data...")
        ndvi_grouped = group_ndvi_files_by_date(ndvi_dir)
        matched_ndvi = match_ndvi_to_thermal_dates(
            filtered_dates, ndvi_grouped, max_days_diff=max_ndvi_days_diff
        )
    else:
        print("\nStep 3.5: No NDVI directory provided, skipping NDVI processing")
    optical_grouped = None
    sar_grouped = None

    if optical_dir and os.path.exists(optical_dir):
        optical_grouped = group_optical_files_by_date(optical_dir)
        print(f"Found RGB data for {len(optical_grouped)} dates")

    if sar_dir and os.path.exists(sar_dir):
        sar_grouped = group_sar_files_by_date(sar_dir)
        print(f"Found SAR data for {len(sar_grouped)} dates")

    print("\nStep 4: Creating multimodal combinations...")
    multimodal_datasets = create_multimodal_full_images(thermal_data_combined, quality_masks, filtered_dates, optical_grouped, sar_grouped)
    time_season_categories = filter_thermal_data_by_time_season(thermal_data_combined, quality_masks, filtered_dates)
    multimodal_datasets.update(time_season_categories)

    print("\nStep 5: Processing each combination...")
    all_results = {}

    for combo_name, (combined_data_list, combined_masks) in multimodal_datasets.items():
        if len(combined_data_list) == 0:
            print(f"Skipping {combo_name} - no data")
            continue

        print(f"\n=== Processing {combo_name} ===")
        directory = "./data_thermal_thermal_only"
        # Create windowed dataset
        positions_dict, ndvi_metadata = create_thermal_dataset(
            thermal_data_combined, quality_masks, mask_file,
            filtered_dates, matched_ndvi,
            window_size=128, stride=64, directory=f"./data_thermal_{combo_name}")

        # Load windowed data
        thermal_list, mask_list = load_thermal_data(
            f"./data_thermal_{combo_name}/thermal",
            f"./data_thermal_{combo_name}/mask"
        )

        # Determine channel count
        channels = combined_data_list[0].shape[0]
        target_shape = (128, 128, channels)

        # Preprocess
        X, y, positions = preprocess_thermal_data(
            thermal_list, mask_list, target_shape, target_shape_mask,
            f"./data_thermal_{combo_name}/thermal",
            f"./data_thermal_{combo_name}/mask",
            f"./data_thermal_{combo_name}/patch_positions.json"
        )

        # Split
        X_train, X_valid, X_test, y_train, y_valid, y_test, pos_train, pos_valid, pos_test = \
            split_thermal_data(X, y, positions)

        # Extract patch IDs for each split
        n_train = len(X_train)
        n_valid = len(X_valid)
        train_ids = list(range(n_train))
        valid_ids = list(range(n_train, n_train + n_valid))
        test_ids = list(range(n_train + n_valid, len(X)))

        # Load NDVI for each split
        print("\nLoading NDVI data for splits...")
        ndvi_train, has_ndvi_train = load_ndvi_data_for_split(
            train_ids, f"{directory}/ndvi", ndvi_metadata
        )
        ndvi_valid, has_ndvi_valid = load_ndvi_data_for_split(
            valid_ids, f"{directory}/ndvi", ndvi_metadata
        )
        ndvi_test, has_ndvi_test = load_ndvi_data_for_split(
            test_ids, f"{directory}/ndvi", ndvi_metadata
        )

        print("\nCreating NDVI sequences for ConvLSTM...")
        ndvi_train_seq, has_ndvi_train_seq, ndvi_train_counts = create_ndvi_sequences_from_patches(
            ndvi_train, has_ndvi_train, sequence_length, overlap
        )
        ndvi_valid_seq, has_ndvi_valid_seq, ndvi_valid_counts = create_ndvi_sequences_from_patches(
            ndvi_valid, has_ndvi_valid, sequence_length, overlap
        )
        ndvi_test_seq, has_ndvi_test_seq, ndvi_test_counts = create_ndvi_sequences_from_patches(
            ndvi_test, has_ndvi_test, sequence_length, overlap
        )

        # ADD THIS: Aggregate NDVI sequences
        print("\nAggregating NDVI sequences...")
        stride = sequence_length - overlap
        ndvi_train_agg = np.array([
            aggregate_sequence_ndvi(ndvi_train_seq[i],
                                    has_ndvi_train[i * stride:i * stride + sequence_length])
            for i in range(len(ndvi_train_seq))
        ])
        ndvi_valid_agg = np.array([
            aggregate_sequence_ndvi(ndvi_valid_seq[i],
                                    has_ndvi_valid[i * stride:i * stride + sequence_length])
            for i in range(len(ndvi_valid_seq))
        ])
        ndvi_test_agg = np.array([
            aggregate_sequence_ndvi(ndvi_test_seq[i],
                                    has_ndvi_test[i * stride:i * stride + sequence_length])
            for i in range(len(ndvi_test_seq))
        ])

        # ConvLSTM sequences
        X_train_seq, X_valid_seq, X_test_seq, y_train_seq, y_valid_seq, y_test_seq = \
            create_time_sequences_from_patches(X_train, X_valid, X_test, y_train, y_valid, y_test, sequence_length, overlap)

        pos_train_seq, pos_valid_seq, pos_test_seq = create_convlstm_positions(pos_train, pos_valid, pos_test, sequence_length, overlap)

        print("\nStep 6: Saving processed dataset...")
        save_dir = f"C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Outputs/Villoslada/{combo_name}"
        os.makedirs(save_dir, exist_ok=True)
        # CNN
        np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(save_dir, 'X_valid.npy'), X_valid)
        np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(save_dir, 'y_valid.npy'), y_valid)
        np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
        np.save(os.path.join(save_dir, 'pos_train.npy'), pos_train)
        np.save(os.path.join(save_dir, 'pos_valid.npy'), pos_valid)
        np.save(os.path.join(save_dir, 'pos_test.npy'), pos_test)
        print(f"Dataset saved for CNN in {save_dir}")

        # NDVI data (separate files)
        np.save(os.path.join(save_dir, 'ndvi_train.npy'), ndvi_train)
        np.save(os.path.join(save_dir, 'ndvi_valid.npy'), ndvi_valid)
        np.save(os.path.join(save_dir, 'ndvi_test.npy'), ndvi_test)
        np.save(os.path.join(save_dir, 'has_ndvi_train.npy'), has_ndvi_train)
        np.save(os.path.join(save_dir, 'has_ndvi_valid.npy'), has_ndvi_valid)
        np.save(os.path.join(save_dir, 'has_ndvi_test.npy'), has_ndvi_test)

        # ConvLSTM
        np.save(os.path.join(save_dir, 'X_train_seq.npy'), X_train_seq)
        np.save(os.path.join(save_dir, 'X_valid_seq.npy'), X_valid_seq)
        np.save(os.path.join(save_dir, 'X_test_seq.npy'), X_test_seq)
        np.save(os.path.join(save_dir, 'y_train_seq.npy'), y_train_seq)
        np.save(os.path.join(save_dir, 'y_valid_seq.npy'), y_valid_seq)
        np.save(os.path.join(save_dir, 'y_test_seq.npy'), y_test_seq)
        np.save(os.path.join(save_dir, 'pos_train_seq.npy'), pos_train_seq)
        np.save(os.path.join(save_dir, 'pos_valid_seq.npy'), pos_valid_seq)
        np.save(os.path.join(save_dir, 'pos_test_seq.npy'), pos_test_seq)
        print(f"Dataset saved for ConvLSTM in {save_dir}")

        # NDVI sequences
        np.save(os.path.join(save_dir, 'ndvi_train_seq.npy'), ndvi_train_seq)
        np.save(os.path.join(save_dir, 'ndvi_valid_seq.npy'), ndvi_valid_seq)
        np.save(os.path.join(save_dir, 'ndvi_test_seq.npy'), ndvi_test_seq)
        np.save(os.path.join(save_dir, 'has_ndvi_train_seq.npy'), has_ndvi_train_seq)
        np.save(os.path.join(save_dir, 'has_ndvi_valid_seq.npy'), has_ndvi_valid_seq)
        np.save(os.path.join(save_dir, 'has_ndvi_test_seq.npy'), has_ndvi_test_seq)
        np.save(os.path.join(save_dir, 'ndvi_train_counts.npy'), ndvi_train_counts)
        np.save(os.path.join(save_dir, 'ndvi_valid_counts.npy'), ndvi_valid_counts)
        np.save(os.path.join(save_dir, 'ndvi_test_counts.npy'), ndvi_test_counts)

        # Aggregated NDVI
        np.save(os.path.join(save_dir, 'ndvi_train_agg.npy'), ndvi_train_agg)
        np.save(os.path.join(save_dir, 'ndvi_valid_agg.npy'), ndvi_valid_agg)
        np.save(os.path.join(save_dir, 'ndvi_test_agg.npy'), ndvi_test_agg)

        print(f"  - NDVI coverage: {np.sum(has_ndvi_test)}/{len(has_ndvi_test)} test patches")
        print(f"  - ConvLSTM sequences: {len(X_train_seq)} train, {len(X_valid_seq)} valid, {len(X_test_seq)} test")
        print(f"  - NDVI sequence coverage: {np.sum(has_ndvi_test_seq)}/{len(has_ndvi_test_seq)} test sequences")

        all_results[combo_name] = {
            'cnn': (X_train, X_valid, X_test, y_train, y_valid, y_test, pos_train, pos_valid, pos_test),
            'ndvi': (ndvi_train, ndvi_valid, ndvi_test, has_ndvi_train, has_ndvi_valid, has_ndvi_test),
            'convlstm': (X_train_seq, X_valid_seq, X_test_seq, y_train_seq, y_valid_seq, y_test_seq,
                         pos_train_seq, pos_valid_seq, pos_test_seq),
            'ndvi_seq': (ndvi_train_seq, ndvi_valid_seq, ndvi_test_seq, has_ndvi_train_seq, has_ndvi_valid_seq,
                         has_ndvi_test_seq, ndvi_train_counts, ndvi_valid_counts, ndvi_test_counts),
            'ndvi_agg': (ndvi_train_agg, ndvi_valid_agg, ndvi_test_agg)}

    return all_results


results = process_ecostress_thermal_data(
    data_dir="C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/THERMAL/Villoslada_full/",
    mask_file="C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Masks/Villoslada masks/Geo_map_resized_Vill.tif",
    ndvi_dir="C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/NDVI/S2 NDVI Villoslada/",
    optical_dir=None, #"C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/RGB/Villoslada RGB/",
    sar_dir=None, #"C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/SAR/SAR Villoslada/",
    sequence_length=5, overlap=2, max_ndvi_days_diff=15)
