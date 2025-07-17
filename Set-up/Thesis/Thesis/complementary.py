import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
import numpy as np
import os
import glob
from pathlib import Path
import re
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, generic_filter


class ECOSTRESSPreprocessor:
    def __init__(self, base_directory, output_directory, geology_map_path=None, window_size=64, stride=64,
                 temporal_stacking=True, max_temporal_gap_days=30, min_temporal_images=3,
                 max_temporal_images=10, cloud_threshold=0.1, missing_data_threshold=0.05,
                 min_coverage_threshold=0.95, require_all_layers=False):
        """
        Initialize the ECOSTRESS data preprocessor for both CNN and ConvLSTM

        Args:
            base_directory: Path to directory containing ECOSTRESS data organized by year
            output_directory: Path where processed patches will be saved
            window_size: Size of square patches (default 64x64)
            stride: Step size for moving window (default 64)
            temporal_stacking: If True, creates temporal stacks for ConvLSTM
            max_temporal_gap_days: Maximum days between images for temporal stacking
            min_temporal_images: Minimum number of temporal images for ConvLSTM
            max_temporal_images: Maximum number of temporal images for ConvLSTM
            cloud_threshold: Maximum fraction of cloudy pixels allowed (0-1)
            missing_data_threshold: Maximum fraction of missing data allowed (0-1)
            min_coverage_threshold: Minimum coverage of study area required (0-1)
            require_all_layers: If False, allows processing with partial layer sets
        """
        self.base_directory = Path(base_directory)
        self.output_directory = Path(output_directory)
        self.window_size = window_size
        self.stride = stride
        self.counter = 0
        self.temporal_stacking = temporal_stacking
        self.max_temporal_gap_days = max_temporal_gap_days
        self.min_temporal_images = min_temporal_images
        self.max_temporal_images = max_temporal_images
        self.cloud_threshold = cloud_threshold
        self.missing_data_threshold = missing_data_threshold
        self.min_coverage_threshold = min_coverage_threshold
        self.require_all_layers = require_all_layers

        self.all_cnn_patches = []
        self.all_cnn_metadata = []
        self.all_convlstm_stacks = []
        self.all_convlstm_metadata = []

        # labels
        self.geology_map_path = geology_map_path
        self.geology_data = None
        self.geology_transform = None
        self.geology_crs = None

        self.all_cnn_labels = []
        self.all_convlstm_labels = []

        # output dir
        self.create_output_directories()

        # thermal layer checks
        self.thermal_layers = ['LST', 'EmisWB']
        self.quality_layers = ['LST_err', 'QC', 'water', 'cloud_final']

        # QC
        self.qc_flags = {
            'good_quality': [
                # Primary good quality pixels
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

                # Secondary good quality pixels (with minor LSTE degradation)
                16513, 16577, 16833, 17089, 17345, 17601, 17857, 18113, 18625, 18881,
                19137, 32897, 32961, 33217, 33473, 33985, 34241, 34497, 34753, 35265,
                35521, 35777, 36929, 36993, 37057, 37313, 37569, 37825, 38081, 38337,
                38593, 38849, 39105, 39361, 39617, 39873, 41409, 41665, 41921, 42689,
                42945, 43713, 43969, 49857, 50113, 50881, 51137, 51905, 52161, 53953,
                54209, 54977, 55233, 56001, 56257, 58049, 58305, 59073, 59329, 60097,
                60353, 64449
            ],
            'cloud_free': [0],  # 0 = not cloudy
        }

        self.processing_stats = {
            'total_files_found': 0,
            'files_excluded_incomplete_coverage': 0,
            'files_excluded_clouds': 0,
            'files_excluded_missing_data': 0,
            'files_excluded_artifacts': 0,
            'files_excluded_incomplete_layers': 0,
            'patches_created_cnn': 0,
            'patches_created_convlstm': 0,
            'temporal_stacks_created': 0
        }

        self.processed_dates = []

        if self.geology_map_path:
            self.load_geology_map()

    def create_output_directories(self):
        """Create necessary output directories for both CNN and ConvLSTM"""
        directories = [
            'cnn_patches/thermal',
            'cnn_patches/metadata',
            'convlstm_patches/thermal_stacks',
            'convlstm_patches/metadata',
            'convlstm_patches/temporal_info',
            'quality_reports',
            'excluded_files'
        ]

        for directory in directories:
            (self.output_directory / directory).mkdir(parents=True, exist_ok=True)

    def find_ecostress_files(self, years):
        """Find and organize ECOSTRESS files by date and layer"""
        file_groups = {}

        for year in years:
            year_dir = self.base_directory / str(year)
            if not year_dir.exists():
                print(f"Warning: Year directory {year_dir} not found")
                continue

            tiff_files = list(year_dir.glob('**/*.tif'))
            self.processing_stats['total_files_found'] += len(tiff_files)

            for tiff_file in tiff_files:
                date_match = self.extract_date_from_filename(tiff_file.name)
                if date_match:
                    date_key = date_match

                    if date_key not in file_groups:
                        file_groups[date_key] = {}

                    layer_type = self.identify_layer_type(tiff_file.name)
                    if layer_type:
                        file_groups[date_key][layer_type] = str(tiff_file)

        # Filter complete or partial sets based on requirements
        complete_sets = {}

        if self.require_all_layers:
            # Original strict requirement
            required_layers = set(self.thermal_layers + self.quality_layers)
        else:
            # More flexible - only require at least one thermal layer
            required_layers = set(self.thermal_layers)

        for date_key, layers in file_groups.items():
            available_layers = set(layers.keys())

            if self.require_all_layers:
                # Must have ALL required layers
                if available_layers >= required_layers:
                    complete_sets[date_key] = layers
                else:
                    missing = required_layers - available_layers
            else:
                # Must have at least one thermal layer
                thermal_available = available_layers & set(self.thermal_layers)
                if thermal_available:
                    complete_sets[date_key] = layers
                    if len(thermal_available) < len(self.thermal_layers):
                        missing_thermal = set(self.thermal_layers) - thermal_available

        return complete_sets

    def validate_image_quality(self, file_paths: Dict[str, str]) -> Tuple[bool, Dict[str, float]]:
        """
        Validate image quality based on the exclusion criteria
        Now more flexible with missing layers

        Returns:
            Tuple of (is_valid, quality_metrics)
        """
        quality_metrics = {}

        try:
            # Load available layers
            layers = {}
            for layer_name, file_path in file_paths.items():
                try:
                    with rasterio.open(file_path) as src:
                        layers[layer_name] = src.read(1)
                except Exception as e:
                    print(f"Warning: Could not load layer {layer_name}: {e}")
                    continue

            # Check if we have at least one thermal layer
            available_thermal = [layer for layer in self.thermal_layers if layer in layers]
            if not available_thermal:
                return False, {'error': 'No thermal layers available'}

            # Use the first available thermal layer for basic validation
            primary_thermal = available_thermal[0]
            height, width = layers[primary_thermal].shape
            total_pixels = height * width

            # Check 1: Coverage (no missing data in available thermal layers)
            thermal_valid = np.ones((height, width), dtype=bool)
            for layer_name in available_thermal:
                layer_data = layers[layer_name]
                valid_layer = ~np.isnan(layer_data) & (layer_data != -9999)
                thermal_valid &= valid_layer

            coverage_fraction = np.sum(thermal_valid) / total_pixels
            quality_metrics['coverage'] = coverage_fraction

            if coverage_fraction < self.min_coverage_threshold:
                return False, quality_metrics

            # Check 2: Cloud coverage (if available)
            if 'cloud_final' in layers:
                cloud_pixels = np.sum(layers['cloud_final'] > 0)
                cloud_fraction = cloud_pixels / total_pixels
                quality_metrics['cloud_fraction'] = cloud_fraction

                if cloud_fraction > self.cloud_threshold:
                    return False, quality_metrics
            else:
                quality_metrics['cloud_fraction'] = 0  # Assume no clouds if no cloud layer

            # Check 3: Quality control flags (if available)
            if 'QC' in layers:
                qc_good = np.isin(layers['QC'], self.qc_flags['good_quality'])
                qc_fraction = np.sum(qc_good) / total_pixels
                quality_metrics['qc_good_fraction'] = qc_fraction

                if qc_fraction < self.min_coverage_threshold:
                    return False, quality_metrics
            else:
                quality_metrics['qc_good_fraction'] = 1.0  # Assume good quality if no QC layer

            # Check 4: Missing data in available thermal layers
            missing_thermal = np.sum(~thermal_valid)
            missing_fraction = missing_thermal / total_pixels
            quality_metrics['missing_fraction'] = missing_fraction

            if missing_fraction > self.missing_data_threshold:
                return False, quality_metrics

            # Check 5: Artifacts (simple check for extreme values)
            primary_data = layers[primary_thermal]
            lst_mean = np.nanmean(primary_data)
            lst_std = np.nanstd(primary_data)
            extreme_values = np.sum(np.abs(primary_data - lst_mean) > 5 * lst_std)
            artifact_fraction = extreme_values / total_pixels
            quality_metrics['artifact_fraction'] = artifact_fraction

            if artifact_fraction > 0.01:  # 1% threshold for artifacts
                return False, quality_metrics

            return True, quality_metrics

        except Exception as e:
            print(f"Error validating image quality: {e}")
            return False, {'error': str(e)}

    def load_geology_map(self):
        try:
            with rasterio.open(self.geology_map_path) as src:
                self.geology_data = src.read(1)
                self.geology_transform = src.transform
                self.geology_crs = src.crs
                print(f"Geology map shape: {self.geology_data.shape}")
                print(f"Class range: {np.min(self.geology_data)} to {np.max(self.geology_data)}")
                print("Classes:", np.unique(self.geology_data))
        except Exception as e:
            print(f"Error loading map: {e}")
            self.geology_data = None

    def align_to_common_shape(self, file_paths: Dict[str, str]) -> Tuple[
        Dict[str, np.ndarray], rasterio.Affine, str, rasterio.windows.Window]:
        """
        Crop all layers to the largest common shape and return transform info
        """
        # Get shapes and reference info from first thermal layer
        layer_shapes = {}
        reference_transform = None
        thermal_crs = None

        aligned_layers = {}
        for layer_name, file_path in file_paths.items():
            with rasterio.open(file_path) as src:
                layer_shapes[layer_name] = (src.height, src.width)
                if layer_name in self.thermal_layers:
                    crs = src.crs
                    if crs is None:
                        print(f"⚠️ CRS missing for {layer_name}, assigning EPSG:32630 as default")
                        crs = rasterio.crs.CRS.from_epsg(32630)

                    if reference_transform is None:
                        reference_transform = src.transform
                        thermal_crs = crs
                        thermal_bounds = src.bounds

        if reference_transform is None:
            print("⚠️ No thermal layers found, using first layer as reference")
            first_layer = list(file_paths.keys())[0]
            with rasterio.open(file_paths[first_layer]) as src:
                reference_transform = src.transform
                thermal_crs = src.crs or rasterio.crs.CRS.from_epsg(32630)

        # Find common dimensions
        min_height = min(shape[0] for shape in layer_shapes.values())
        min_width = min(shape[1] for shape in layer_shapes.values())
        crop_window = rasterio.windows.Window(0, 0, min_width, min_height)

        # Load and crop all layers
        for layer_name, file_path in file_paths.items():
            with rasterio.open(file_path) as src:
                data = src.read(1)
                aligned_layers[layer_name] = data[0:min_height, 0:min_width]

        return aligned_layers, reference_transform, thermal_crs, crop_window, (min_height, min_width)

    def align_geology_to_thermal(self, thermal_shape, thermal_transform, thermal_crs):
        # Force CRS assignment if missing
        if thermal_crs is None or str(thermal_crs) == 'EPSG:4326':
            print("⚠️ Thermal CRS is missing. Assigning default EPSG:32630.")
            thermal_crs = rasterio.crs.CRS.from_epsg(32630)

        try:
            aligned_geology = np.zeros(thermal_shape, dtype=self.geology_data.dtype)
            reproject(
                source=self.geology_data,
                destination=aligned_geology,
                src_transform=self.geology_transform,
                src_crs=self.geology_crs,
                dst_transform=thermal_transform,
                dst_crs=thermal_crs,
                resampling=Resampling.nearest
            )

            unique_classes = np.unique(aligned_geology)
            print(f"Geo map realigned shape: {aligned_geology.shape}, unique classes: {unique_classes}")
            return aligned_geology
        except Exception as e:
            print(f"Error aligning geo map: {e}")
            return None

    def normalize_lst_data(self, lst_data):
        """Normalize LST data to 0-1 range"""
        # Remove invalid values for normalization calculation
        valid_mask = ~np.isnan(lst_data) & (lst_data != -9999)

        if not np.any(valid_mask):
            return lst_data  # Return as is if no valid data

        # Calculate min/max from valid data only
        valid_data = lst_data[valid_mask]
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)

        # Normalize
        normalized = np.copy(lst_data)
        normalized[valid_mask] = (lst_data[valid_mask] - min_val) / (max_val - min_val)

        return normalized

    def handle_nan_values(self, patch_data, mask_patch):
        """
        Handle NaN values in patches using multiple strategies

        Args:
            patch_data: numpy array of shape (n_layers, height, width)
            mask_patch: boolean mask indicating valid pixels

        Returns:
            patch_data with NaN values handled
        """
        processed_patch = np.copy(patch_data)

        for layer_idx in range(patch_data.shape[0]):
            layer = processed_patch[layer_idx]

            # Find NaN pixels
            nan_mask = np.isnan(layer)

            if np.any(nan_mask):
                # Strategy 1: Use spatial interpolation for isolated NaN pixels
                layer = self.spatial_interpolation(layer, nan_mask, mask_patch)

                # Strategy 2: Fill remaining NaN with layer mean of valid pixels
                if np.any(np.isnan(layer)):
                    valid_pixels = layer[~np.isnan(layer) & mask_patch]
                    if len(valid_pixels) > 0:
                        layer_mean = np.mean(valid_pixels)
                        layer[np.isnan(layer)] = layer_mean
                    else:
                        # If no valid pixels, fill with global mean (you might want to skip this patch instead)
                        layer[np.isnan(layer)] = 0  # or some default value

                processed_patch[layer_idx] = layer

        return processed_patch

    def spatial_interpolation(self, layer, nan_mask, mask_patch):
        """
        Perform spatial interpolation to fill NaN values
        Uses nearest neighbor interpolation for small gaps
        """
        from scipy.ndimage import distance_transform_edt
        from scipy.ndimage import generic_filter

        # Only interpolate within the valid mask
        interpolation_mask = nan_mask & mask_patch

        if not np.any(interpolation_mask):
            return layer

        # Create a copy to work with
        interpolated_layer = np.copy(layer)

        # Use distance transform to find nearest valid values
        valid_mask = ~np.isnan(layer) & mask_patch

        if np.any(valid_mask):
            # Distance transform gives distance to nearest valid pixel
            distances, indices = distance_transform_edt(
                ~valid_mask,
                return_indices=True
            )

            # For NaN pixels that are close to valid pixels (distance < 2), interpolate
            close_nan_mask = interpolation_mask & (distances < 2)

            if np.any(close_nan_mask):
                # Use nearest neighbor interpolation
                interpolated_layer[close_nan_mask] = layer[
                    indices[0][close_nan_mask],
                    indices[1][close_nan_mask]
                ]

        return interpolated_layer

    def create_patches_cnn(self, file_paths: Dict[str, str], date_key: str):
        """Create individual patches for CNN training - collect for single file"""
        patches_created = 0

        try:
            # Load and align all layers to common shape
            aligned_layers, reference_transform, reference_crs, crop_window, (height, width) = self.align_to_common_shape(file_paths)
            aligned_geology = None
            if self.geology_data is not None:
                aligned_geology = self.align_geology_to_thermal(
                    (height, width),
                    reference_transform,
                    reference_crs
                )
            # Get reference dimensions
            first_thermal = next(iter([layer for layer in self.thermal_layers if layer in aligned_layers]))
            height, width = aligned_layers[first_thermal].shape

            # Separate thermal and quality layers
            thermal_data = {k: v for k, v in aligned_layers.items() if k in self.thermal_layers}
            quality_data = {k: v for k, v in aligned_layers.items() if k in self.quality_layers}
            available_thermal = list(thermal_data.keys())

            # Create quality mask with aligned data
            quality_mask = self.create_quality_mask(thermal_data, quality_data)

            # Extract patches
            for i in range(0, height - self.window_size + 1, self.stride):
                for j in range(0, width - self.window_size + 1, self.stride):
                    window = Window(j, i, self.window_size, self.window_size)

                    # Extract patch from quality mask
                    mask_patch = quality_mask[i:i + self.window_size, j:j + self.window_size]

                    # Skip if patch doesn't have enough valid pixels
                    if np.sum(mask_patch) < (self.window_size * self.window_size * 0.8):
                        continue

                    # Extract thermal patches
                    patch_data = np.full((len(self.thermal_layers), self.window_size, self.window_size), np.nan)

                    for idx, layer_name in enumerate(self.thermal_layers):
                        if layer_name in thermal_data:
                            layer_patch = thermal_data[layer_name][i:i + self.window_size, j:j + self.window_size]
                            layer_patch = np.where(mask_patch, layer_patch, np.nan)
                            if layer_name == 'LST':
                                layer_patch = self.normalize_lst_data(layer_patch)
                            patch_data[idx] = layer_patch

                    label_patch = None
                    if aligned_geology is not None:
                        label_patch = aligned_geology[i:i + self.window_size, j:j + self.window_size]
                        label_patch = np.where(mask_patch, label_patch, np.nan)

                    # Handle NaN values in patch
                    patch_data = self.handle_nan_values(patch_data, mask_patch)

                    # Skip patch if it still contains NaN after processing
                    if np.any(np.isnan(patch_data)):
                        continue

                    # Convert to CNN format and add to global collection
                    patch_cnn_format = np.transpose(patch_data, (1, 2, 0))  # (height, width, channels)
                    self.all_cnn_patches.append(patch_cnn_format)

                    # add labels
                    if label_patch is not None:
                        label_patch = np.expand_dims(label_patch, axis=-1)
                        self.all_cnn_labels.append(label_patch)

                    # Calculate actual geospatial window coordinates
                    actual_col = crop_window.col_off + j
                    actual_row = crop_window.row_off + i
                    patch_transform = reference_transform * rasterio.Affine.translation(actual_col, actual_row)

                    # Add metadata to global collection
                    metadata = {
                        'date': date_key,
                        'patch_id': self.counter,
                        'window': [actual_row, actual_col, self.window_size, self.window_size],
                        'patch_window_in_cropped': [i, j, self.window_size, self.window_size],
                        'crop_window': [crop_window.row_off, crop_window.col_off, crop_window.height,
                                        crop_window.width],
                        'transform': list(reference_transform),
                        'patch_transform': list(patch_transform),
                        'crs': str(reference_crs),
                        'layers': self.thermal_layers,
                        'available_layers': available_thermal,
                        'valid_pixels': int(np.sum(mask_patch))
                    }

                    self.all_cnn_metadata.append(metadata)
                    self.counter += 1
                    patches_created += 1

            self.processing_stats['patches_created_cnn'] += patches_created
            return patches_created

        except Exception as e:
            print(f"Error creating CNN patches for {date_key}: {e}")
            return 0

    def create_patches_convlstm(self, temporal_group: List[Tuple[str, Dict[str, str]]]):
        """Create temporal stacks for ConvLSTM training - collect for single file"""
        if len(temporal_group) < self.min_temporal_images:
            return 0

        # Sort by date and limit temporal images
        temporal_group.sort(key=lambda x: x[0])
        if len(temporal_group) > self.max_temporal_images:
            temporal_group = temporal_group[:self.max_temporal_images]

        patches_created = 0
        stack_id = f"stack_{len(temporal_group)}_{temporal_group[0][0]}_{temporal_group[-1][0]}"

        try:
            # Load and align all temporal data
            temporal_aligned_data = []
            temporal_quality_data = []
            reference_transform = None
            reference_crs = None
            reference_crop_window = None

            for date_key, file_paths in temporal_group:
                aligned_layers, transform, crs, crop_window, (height, width) = self.align_to_common_shape(file_paths)
                aligned_geology = None
                if self.geology_data is not None:
                    aligned_geology = self.align_geology_to_thermal(
                        (height, width),
                        reference_transform,
                        reference_crs
                    )
                if reference_transform is None:
                    reference_transform = transform
                    reference_crs = crs
                    reference_crop_window = crop_window

                thermal_data = {k: v for k, v in aligned_layers.items() if k in self.thermal_layers}
                quality_data = {k: v for k, v in aligned_layers.items() if k in self.quality_layers}

                temporal_aligned_data.append(thermal_data)
                temporal_quality_data.append(quality_data)

            # Get dimensions from first temporal step
            if temporal_aligned_data:
                first_thermal = next(
                    iter([layer for layer in self.thermal_layers if layer in temporal_aligned_data[0]]))
                height, width = temporal_aligned_data[0][first_thermal].shape
            else:
                return 0

            # Extract patches
            for i in range(0, height - self.window_size + 1, self.stride):
                for j in range(0, width - self.window_size + 1, self.stride):
                    window = Window(j, i, self.window_size, self.window_size)

                    # Check if patch is valid across all time steps
                    valid_across_time = True
                    temporal_masks = []

                    for t_idx, (thermal_data, quality_data) in enumerate(
                            zip(temporal_aligned_data, temporal_quality_data)):
                        if not thermal_data:
                            valid_across_time = False
                            break

                        quality_mask = self.create_quality_mask(thermal_data, quality_data)
                        mask_patch = quality_mask[i:i + self.window_size, j:j + self.window_size]
                        temporal_masks.append(mask_patch)

                        if np.sum(mask_patch) < (self.window_size * self.window_size * 0.8):
                            valid_across_time = False
                            break

                    if not valid_across_time:
                        continue

                    # Extract temporal stack
                    temporal_stack = np.full((len(temporal_group), len(self.thermal_layers),
                                              self.window_size, self.window_size), np.nan)

                    for t_idx, (thermal_data, quality_data) in enumerate(
                            zip(temporal_aligned_data, temporal_quality_data)):
                        mask_patch = temporal_masks[t_idx]

                        for layer_idx, layer_name in enumerate(self.thermal_layers):
                            if layer_name in thermal_data:
                                layer_patch = thermal_data[layer_name][i:i + self.window_size, j:j + self.window_size]
                                layer_patch = np.where(mask_patch, layer_patch, np.nan)
                                if layer_name == 'LST':
                                    layer_patch = self.normalize_lst_data(layer_patch)
                                temporal_stack[t_idx, layer_idx] = layer_patch

                    label_patch = None
                    if aligned_geology is not None:
                        label_patch = aligned_geology[i:i + self.window_size, j:j + self.window_size]
                        label_patch = np.where(temporal_masks[0], label_patch, np.nan)

                    # Handle NaN values in temporal stack
                    for t_idx in range(len(temporal_group)):
                        temporal_stack[t_idx] = self.handle_nan_values(temporal_stack[t_idx], temporal_masks[t_idx])

                    # Skip stack if it still contains NaN after processing
                    if np.any(np.isnan(temporal_stack)):
                        continue

                    # Convert to ConvLSTM format and add to global collection
                    stack_convlstm_format = np.transpose(temporal_stack,
                                                         (0, 2, 3, 1))  # (timesteps, height, width, channels)
                    self.all_convlstm_stacks.append(stack_convlstm_format)

                    if label_patch is not None:
                        label_patch = np.expand_dims(label_patch, axis=-1)
                        self.all_convlstm_labels.append(label_patch)

                    # Calculate actual geospatial window coordinates
                    actual_col = reference_crop_window.col_off + j
                    actual_row = reference_crop_window.row_off + i
                    patch_transform = reference_transform * rasterio.Affine.translation(actual_col, actual_row)

                    # Add metadata to global collection
                    metadata = {
                        'stack_id': stack_id,
                        'patch_id': self.counter,
                        'temporal_length': len(temporal_group),
                        'dates': [date_key for date_key, _ in temporal_group],
                        'window': [actual_row, actual_col, self.window_size, self.window_size],
                        'patch_window_in_cropped': [i, j, self.window_size, self.window_size],
                        'crop_window': [reference_crop_window.row_off, reference_crop_window.col_off,
                                        reference_crop_window.height, reference_crop_window.width],
                        'transform': list(reference_transform),
                        'patch_transform': list(patch_transform),
                        'crs': str(reference_crs),
                        'layers': self.thermal_layers,
                        'shape': list(temporal_stack.shape)
                    }

                    self.all_convlstm_metadata.append(metadata)
                    self.counter += 1
                    patches_created += 1

            self.processing_stats['patches_created_convlstm'] += patches_created
            self.processing_stats['temporal_stacks_created'] += 1
            return patches_created

        except Exception as e:
            print(f"Error creating ConvLSTM patches for {stack_id}: {e}")
            return 0

    def save_all_patches(self):
        """Save all collected patches to single files"""

        # Save CNN patches
        if self.all_cnn_patches:
            cnn_array = np.stack(self.all_cnn_patches, axis=0)  # (n_patches, height, width, channels)

            cnn_patches_path = self.output_directory / 'cnn_patches' / 'thermal' / 'all_cnn_patches.npy'
            cnn_metadata_path = self.output_directory / 'cnn_patches' / 'metadata' / 'all_cnn_metadata.json'

            np.save(cnn_patches_path, cnn_array)
            with open(cnn_metadata_path, 'w') as f:
                json.dump(self.all_cnn_metadata, f, indent=2)

            print(f"CNN patches saved - Shape: {cnn_array.shape} - (n_patches, height, width, channels)")
            print(f"Total CNN patches: {len(self.all_cnn_patches)}")

            if self.all_cnn_labels:
                cnn_labels_array = np.stack(self.all_cnn_labels, axis=0)
                cnn_labels_path = self.output_directory / 'cnn_patches' / 'thermal' / 'all_cnn_labels.npy'
                np.save(cnn_labels_path, cnn_labels_array)
                print(f"CNN labels saved - Shape: {cnn_labels_array.shape} - (n_patches, height, width)")

        # Save ConvLSTM stacks
        if self.all_convlstm_stacks:
            shapes = [stack.shape for stack in self.all_convlstm_stacks]
            unique_shapes = list(set(shapes))
            if len(unique_shapes) == 1:
                convlstm_array = np.stack(self.all_convlstm_stacks,
                                      axis=0)  # (n_stacks, timesteps, height, width, channels)
            else:
                print("Normalizing ConvLSTM stack shapes...")
                min_timesteps = min(shape[0] for shape in unique_shapes)
                min_height = min(shape[1] for shape in unique_shapes)
                min_width = min(shape[2] for shape in unique_shapes)
                print(f"Target dimensions: {min_timesteps} timesteps, {min_height}x{min_width} spatial")

                normalized_stacks = []
                for i, stack in enumerate(self.all_convlstm_stacks):
                    timesteps, height, width, channels = stack.shape
                    if timesteps > min_timesteps:
                        stack = stack[:min_timesteps, :, :, :]
                    h_start = (height - min_height) // 2
                    w_start = (width - min_width) // 2
                    cropped_stack = stack[:,
                                    h_start:h_start + min_height,
                                    w_start:w_start + min_width,
                                    :]
                    normalized_stacks.append(cropped_stack)

                    if i < len(self.all_convlstm_metadata):
                        self.all_convlstm_metadata[i]['original_shape'] = list(stack.shape)
                        self.all_convlstm_metadata[i]['cropped_shape'] = list(cropped_stack.shape)
                        self.all_convlstm_metadata[i]['temporal_crop'] = min_timesteps if timesteps > min_timesteps else 0
                        self.all_convlstm_metadata[i]['crop_offset'] = [h_start, w_start]

                convlstm_array = np.stack(normalized_stacks, axis=0)
                print(f"Normalized {len(normalized_stacks)} stacks to common shape")

            convlstm_stacks_path = self.output_directory / 'convlstm_patches' / 'thermal_stacks' / 'all_convlstm_stacks.npy'
            convlstm_metadata_path = self.output_directory / 'convlstm_patches' / 'metadata' / 'all_convlstm_metadata.json'

            np.save(convlstm_stacks_path, convlstm_array)
            with open(convlstm_metadata_path, 'w') as f:
                json.dump(self.all_convlstm_metadata, f, indent=2)

            print(f"ConvLSTM stacks saved - Shape: {convlstm_array.shape} - (n_stacks, timesteps, height, width, channels)")
            print(f"Total ConvLSTM stacks: {len(self.all_convlstm_stacks)}")

            if self.all_convlstm_labels:
                convlstm_labels_array = np.stack(self.all_convlstm_labels, axis=0)
                convlstm_labels_path = self.output_directory / 'convlstm_patches' / 'thermal_stacks' / 'all_convlstm_labels.npy'
                np.save(convlstm_labels_path, convlstm_labels_array)
                print(f"ConvLSTM labels saved - Shape: {convlstm_labels_array.shape} - (n_stacks, height, width)")

    def create_quality_mask(self, thermal_data: Dict, quality_data: Dict) -> np.ndarray:
        """Create a quality mask combining all quality criteria"""
        # Get dimensions from first available thermal layer
        first_thermal = list(thermal_data.values())[0]
        height, width = first_thermal.shape
        mask = np.ones((height, width), dtype=bool)

        # Mask invalid thermal data
        for layer_name in thermal_data:
            layer_data = thermal_data[layer_name]
            valid_thermal = ~np.isnan(layer_data) & (layer_data != -9999)
            mask &= valid_thermal

        # Apply quality control masks (if available)
        if 'QC' in quality_data:
            qc_mask = np.isin(quality_data['QC'], self.qc_flags['good_quality'])
            mask &= qc_mask

        if 'cloud_final' in quality_data:
            cloud_mask = quality_data['cloud_final'] == 0  # Assuming 0 = no cloud
            mask &= cloud_mask

        if 'water' in quality_data:
            # Assuming you want to exclude water pixels (adjust as needed)
            water_mask = quality_data['water'] == 0  # Assuming 0 = no water
            mask &= water_mask

        return mask

    def group_temporal_sequences(self, complete_sets: Dict[str, Dict[str, str]]):
        """Group files into temporal sequences for ConvLSTM"""
        # Convert date keys to datetime objects
        date_files = []
        for date_key, file_paths in complete_sets.items():
            try:
                date_obj = datetime.strptime(date_key.split('T')[0], '%Y%m%d')
                date_files.append((date_obj, date_key, file_paths))
            except:
                continue

        # Sort by date
        date_files.sort()

        # Group into temporal sequences
        temporal_groups = []
        current_group = []

        for i, (date_obj, date_key, file_paths) in enumerate(date_files):
            if not current_group:
                current_group.append((date_key, file_paths))
            else:
                # Check if within temporal gap
                last_date = datetime.strptime(current_group[-1][0].split('T')[0], '%Y%m%d')
                gap_days = (date_obj - last_date).days

                if gap_days <= self.max_temporal_gap_days:
                    current_group.append((date_key, file_paths))
                else:
                    # Save current group if it meets minimum requirements
                    if len(current_group) >= self.min_temporal_images:
                        temporal_groups.append(current_group)
                    current_group = [(date_key, file_paths)]

        # Add final group
        if len(current_group) >= self.min_temporal_images:
            temporal_groups.append(current_group)

        return temporal_groups

    def format_date_for_display(self, date_key: str) -> str:
        """Convert date key to DD/MM/YYYY - Time: HH:MM:SS format"""
        try:
            # Extract date and time parts
            if 'T' in date_key:
                date_part, time_part = date_key.split('T')
            else:
                date_part = date_key
                time_part = "000000"

            # Parse date (YYYYMMDD)
            date_obj = datetime.strptime(date_part, '%Y%m%d')

            # Parse time (HHMMSS)
            if len(time_part) >= 6:
                hours = time_part[:2]
                minutes = time_part[2:4]
                seconds = time_part[4:6]
            else:
                hours = minutes = seconds = "00"

            # Format as requested
            formatted = f"{date_obj.strftime('%d/%m/%Y')} - Time: {hours}:{minutes}:{seconds}"
            return formatted
        except:
            return f"Invalid date format: {date_key}"

    def print_files_by_date(self, file_groups: Dict[str, Dict[str, str]]):
        """Print available files for each date in a formatted way"""
        print(f"\n" + "=" * 80)
        print("FILES FOUND BY DATE")
        print("=" * 80)

        # Sort dates for chronological display
        sorted_dates = sorted(file_groups.keys())

        for date_key in sorted_dates:
            formatted_date = self.format_date_for_display(date_key)
            available_files = list(file_groups[date_key].keys())
            file_count = len(available_files)

            print(f"Date: {formatted_date}")
            print(f"  Files found ({file_count}): {', '.join(sorted(available_files))}")
            print()

    def filter_suitable_dates(self, file_groups: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """Filter dates to only include those with both thermal layers and preferably 5+ files"""

        filtered_sets = {}

        for date_key, available_files in file_groups.items():
            file_set = set(available_files.keys())
            file_count = len(available_files)

            has_lst = 'LST' in file_set
            has_emiswb = 'EmisWB' in file_set

            # Only include dates with both thermal layers
            if has_lst and has_emiswb:
                # Prefer dates with 5+ files, but include all dates with both thermal layers
                if file_count >= 5:
                    filtered_sets[date_key] = available_files
                else:
                    # Still include but note it's minimal
                    filtered_sets[date_key] = available_files
            else:
                print(f"✗ Excluding {date_key} - missing thermal layers")

        return filtered_sets

    def process_data(self, years: List[int]):
        """Main processing function"""
        print(f"Starting ECOSTRESS preprocessing for years: {years}")
        print(f"Require all layers: {self.require_all_layers}")

        # Find all files
        complete_sets = self.find_ecostress_files(years)
        print(f"Found {len(complete_sets)} complete file sets")

        self.print_files_by_date(complete_sets)

        print(f"\n" + "=" * 80)
        print("FILTERING FOR TRAINING SUITABILITY")
        print("=" * 80)
        suitable_sets = self.filter_suitable_dates(complete_sets)
        print(f"\nFiltered to {len(suitable_sets)} suitable dates for training")

        # Validate quality and filter
        valid_sets = {}
        quality_report = {}

        for date_key, file_paths in suitable_sets.items():
            is_valid, quality_metrics = self.validate_image_quality(file_paths)
            quality_report[date_key] = quality_metrics

            if is_valid:
                valid_sets[date_key] = file_paths
                print(f"✓ {date_key}: Valid (coverage: {quality_metrics.get('coverage', 0):.2f})")
            else:
                print(f"✗ {date_key}: Invalid - {quality_metrics}")
                # Log exclusion reason
                if quality_metrics.get('coverage', 1) < self.min_coverage_threshold:
                    self.processing_stats['files_excluded_incomplete_coverage'] += 1
                elif quality_metrics.get('cloud_fraction', 0) > self.cloud_threshold:
                    self.processing_stats['files_excluded_clouds'] += 1
                elif quality_metrics.get('missing_fraction', 0) > self.missing_data_threshold:
                    self.processing_stats['files_excluded_missing_data'] += 1
                elif quality_metrics.get('artifact_fraction', 0) > 0.01:
                    self.processing_stats['files_excluded_artifacts'] += 1

        # Save quality report
        quality_report_path = self.output_directory / 'quality_reports' / 'quality_report.json'
        with open(quality_report_path, 'w') as f:
            json.dump(quality_report, f, indent=2)

        print(f"Valid sets after quality filtering: {len(valid_sets)}")

        # Track dates that actually produced patches
        dates_with_patches = set()

        # Process CNN patches (individual images)
        if valid_sets:
            print("\nCreating CNN patches...")
            for date_key, file_paths in valid_sets.items():
                patches_created = self.create_patches_cnn(file_paths, date_key)
                if patches_created > 0:
                    dates_with_patches.add(date_key)

            # Process ConvLSTM patches (temporal sequences)
            if self.temporal_stacking:
                print("\nCreating ConvLSTM temporal stacks...")
                temporal_groups = self.group_temporal_sequences(valid_sets)
                print(f"Created {len(temporal_groups)} temporal groups")

                for group in temporal_groups:
                    patches_created = self.create_patches_convlstm(group)
                    if patches_created > 0:
                        # Add all dates in the group that produced patches
                        for date_key, _ in group:
                            dates_with_patches.add(date_key)

        # Save processing statistics
        stats_path = self.output_directory / 'processing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.processing_stats, f, indent=2)

        # Print only dates that actually produced patches
        print(f"\n" + "=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Total dates that produced patches: {len(dates_with_patches)}")

        if dates_with_patches:
            print("\nDates that produced patches:")
            for date_key in sorted(dates_with_patches):
                formatted_date = self.format_date_for_display(date_key)
                print(f"Date: {formatted_date}")
        else:
            print("\nNo dates produced patches.")

        print(f"\n" + "=" * 80)
        print("SAVING PATCHES")
        print("=" * 80)
        self.save_all_patches()

        print(f"\nProcessing complete!")
        print(f"Statistics: {self.processing_stats}")

    def extract_date_from_filename(self, filename):
        """Extract date from ECOSTRESS filename"""
        match = re.search(r'doy(\d{4})(\d{3})(\d{6})', filename)
        if match:
            year, doy, time = match.groups()
            date = datetime.strptime(f"{year}{doy}", "%Y%j").strftime("%Y%m%d")
            return f"{date}T{time}"
        return None

    def identify_layer_type(self, filename):
        """Identify the layer type based on filename"""
        if 'LST_err' in filename:
            return 'LST_err'
        elif 'EmisWB' in filename:
            return 'EmisWB'
        elif 'QC' in filename:
            return 'QC'
        elif 'water' in filename:
            return 'water'
        elif 'Cloud_final' in filename:
            return 'cloud_final'
        elif 'LST' in filename:
            return 'LST'

        return None


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor with more flexible settings
    preprocessor = ECOSTRESSPreprocessor(
        base_directory="C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Villoslada_full",
        output_directory="C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Outputs/Villoslada",
        geology_map_path="C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Images/Geology/Villoslada de Cameros/Geo_map_Vill.tif",
        window_size=64,
        stride=32,  # no overlap
        temporal_stacking=True,
        max_temporal_gap_days=10,
        min_temporal_images=3,
        max_temporal_images=10,
        cloud_threshold=0.10,  # 10% cloud coverage max
        missing_data_threshold=0.10,  # 10% missing data max
        min_coverage_threshold=0.75,  # 75% coverage required
        require_all_layers=False
    )

    # Process data
    years_to_process = [2019, 2020, 2021, 2022, 2023, 2024]
    preprocessor.process_data(years_to_process)