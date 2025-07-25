import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os
import glob
from datetime import datetime, timedelta
import re

import pandas as pd

def interpolate_nans(values):
    """
    Interpolate NaNs in a list or array using linear interpolation.
    """
    series = pd.Series(values)
    return series.interpolate(method='linear', limit_direction='both').to_numpy()


class ThermalSignatureAnalyzer:
    def __init__(self, geo_map_path, thermal_images_dir):
        """
        Initialize the thermal signature analyzer

        Parameters:
        geo_map_path (str): Path to the geo map with class labels
        thermal_images_dir (str): Directory containing ECOSTRESS thermal images
        """
        self.geo_map_path = geo_map_path
        self.thermal_images_dir = thermal_images_dir

        # Class definitions
        self.class_names = {
            0: 'NaN values',
            1: 'Sand (Arenas)',
            2: 'Clay (Arcillas)',
            3: 'Chalk (Tiza)',
            4: 'Silt (Limos)',
            5: 'Peat (Turba)',
            6: 'Loam (Marga)',
            7: 'Detritic',
            8: 'Carbonate',
            9: 'Volcanic',
            10: 'Plutonic',
            11: 'Foliated',
            12: 'Non-foliated',
            13: 'Water'
        }

        # Colors for plotting (similar to reference image)
        self.colors = {
            1: '#1f77b4',  # blue
            2: '#ff7f0e',  # orange
            3: '#2ca02c',  # green
            4: '#d62728',  # red
            5: '#9467bd',  # purple
            6: '#8c564b',  # brown
            7: '#e377c2',  # pink
            8: '#7f7f7f',  # gray
            9: '#bcbd22',  # olive
            10: '#17becf',  # cyan
            11: '#ffbb78',  # light orange
            12: '#98df8a',  # light green
            13: '#aec7e8'  # light blue
        }

        self.geo_map = None
        self.thermal_data = {}

    def load_geo_map(self):
        """Load the geo map with class labels"""
        try:
            with rasterio.open(self.geo_map_path) as dataset:
                self.geo_map = dataset.read(1)
                print(f"Unique classes found: {np.unique(self.geo_map)}")

        except Exception as e:
            print(f"Error loading geo map: {e}")
            return False
        return True

    def extract_date_from_filename(self, filename):
        """
        Extract date from ECOSTRESS filename
        Example: ECO_L2T_LSTE.002_LST_doy2023171114725_aid0001_30N
        """
        try:
            # Extract the DOY (Day of Year) part
            match = re.search(r'doy(\d{7})', filename)
            if match:
                doy_str = match.group(1)
                year = int(doy_str[:4])
                doy = int(doy_str[4:7])

                # Convert DOY to date
                date = datetime(year, 1, 1) + timedelta(days=doy - 1)
                return date
        except Exception as e:
            print(f"Error extracting date from {filename}: {e}")

        return None

    def load_thermal_images(self):
        """Load all thermal images from the directory"""
        # Look specifically for ECOSTRESS LST files (exclude cloud files)
        pattern = os.path.join(self.thermal_images_dir, "*ECO_L2T_LSTE*LST_*.tif")
        thermal_files = glob.glob(pattern)

        if not thermal_files:
            # Alternative pattern for LST files
            pattern = os.path.join(self.thermal_images_dir, "*LST_doy*.tif")
            thermal_files = glob.glob(pattern)

        # Filter out cloud files explicitly
        thermal_files = [f for f in thermal_files if 'err' not in os.path.basename(f).lower()]

        print(f"Found {len(thermal_files)} thermal image files")

        for file_path in thermal_files:
            filename = os.path.basename(file_path)
            date = self.extract_date_from_filename(filename)

            if date is None:
                # Use filename as identifier if date extraction fails
                date = filename

            try:
                with rasterio.open(file_path) as dataset:
                    thermal_array = dataset.read()

                    # Handle different dimensions
                    if len(thermal_array.shape) == 3:
                        # If 3D, take first band
                        thermal_array = thermal_array[0]

                    print(f"Loaded thermal image: {filename}, shape: {thermal_array.shape}")

                    self.thermal_data[date] = thermal_array

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def align_dimensions(self, thermal_image):
        """
        Align thermal image dimensions with geo map
        Handle cases where thermal images have slightly different dimensions
        """
        geo_h, geo_w = self.geo_map.shape
        therm_h, therm_w = thermal_image.shape

        # Crop or pad thermal image to match geo map
        if therm_h > geo_h:
            thermal_image = thermal_image[:geo_h, :]
        elif therm_h < geo_h:
            pad_h = geo_h - therm_h
            thermal_image = np.pad(thermal_image, ((0, pad_h), (0, 0)), mode='constant', constant_values=np.nan)

        if therm_w > geo_w:
            thermal_image = thermal_image[:, :geo_w]
        elif therm_w < geo_w:
            pad_w = geo_w - therm_w
            thermal_image = np.pad(thermal_image, ((0, 0), (0, pad_w)), mode='constant', constant_values=np.nan)

        return thermal_image

    def calculate_class_statistics(self, use_median=True):
        """
        Calculate temporal thermal statistics for each class

        Parameters:
        use_median (bool): Use median instead of mean for more robust statistics
        """
        if self.geo_map is None:
            print("Geo map not loaded!")
            return None

        dates = sorted(self.thermal_data.keys())
        classes = [c for c in range(1, 14) if c != 0]  # Exclude NaN values (class 0)

        # Initialize results dictionary
        results = {cls: [] for cls in classes}
        valid_dates = []

        print("Calculating class statistics...")

        for date in dates:
            thermal_image = self.thermal_data[date]

            # Align dimensions
            thermal_aligned = self.align_dimensions(thermal_image)

            # Check if alignment was successful
            if thermal_aligned.shape != self.geo_map.shape:
                print(f"Skipping {date}: dimension mismatch after alignment")
                continue

            valid_dates.append(date)

            # Calculate statistics for each class
            for class_id in classes:
                class_mask = (self.geo_map == class_id)
                class_thermal_values = thermal_aligned[class_mask]

                # Remove NaN and invalid values
                valid_values = class_thermal_values[~np.isnan(class_thermal_values)]
                valid_values = valid_values[valid_values > 0]  # Remove invalid thermal values

                if len(valid_values) > 0:
                    if use_median:
                        stat_value = np.median(valid_values)
                    else:
                        stat_value = np.mean(valid_values)

                    results[class_id].append(stat_value)
                    print(f"Class {class_id}, Date {date}: {len(valid_values)} pixels, temp = {stat_value:.2f}K")
                else:
                    results[class_id].append(np.nan)
                    print(f"Class {class_id}, Date {date}: No valid pixels found")

        print(f"Processed {len(valid_dates)} dates with {len(results)} classes")
        return results, valid_dates

    def plot_yearly_signatures(self, results, dates, save_dir=None):
        """
        Plot thermal signatures separately for each year
        """
        # Group indices of dates by year
        from collections import defaultdict
        yearly_indices = defaultdict(list)
        for idx, date in enumerate(dates):
            year = date.year if hasattr(date, 'year') else None
            if year:
                yearly_indices[year].append(idx)

        for year, indices in yearly_indices.items():
            if not indices:
                continue

            # Extract the subset of dates and results for this year
            year_dates = [dates[i] for i in indices]

            # Subset results for each class
            year_results = {}
            for class_id, values in results.items():
                year_results[class_id] = [values[i] for i in indices]

            # Plot using your existing plot function, but tweak to accept subset
            print(f"Plotting year {year} with {len(indices)} dates")
            self.plot_temporal_signatures(year_results, year_dates,
                                          save_path=(os.path.join(save_dir,
                                                                  f'temporal_thermal_signatures_{year}.png') if save_dir else None))

    def plot_temporal_signatures(self, results, dates, save_path=None):
        """
        Create the temporal thermal signature plot
        """
        plt.figure(figsize=(12, 8))

        # Convert dates to indices for x-axis
        x_indices = range(len(dates))

        # Debug: Print what we're plotting
        print(f"Plotting {len(dates)} time points")

        # Plot each class
        plotted_classes = 0
        for class_id in sorted(results.keys()):
            if class_id == 0:  # Skip NaN values
                continue

            y_values = results[class_id]
            y_values = interpolate_nans(y_values)

            # Skip classes with no valid data
            valid_count = sum(1 for v in y_values if not np.isnan(v))
            if valid_count == 0:
                print(f"Skipping Class {class_id}: No valid data points")
                continue

            color = self.colors.get(class_id, 'black')
            label = f"C_{class_id} - {self.class_names[class_id].split('(')[0].strip()}"

            plt.plot(x_indices, y_values,
                     color=color,
                     linewidth=1.5,
                     label=label)

            plotted_classes += 1
            print(f"Plotted Class {class_id}: {valid_count}/{len(y_values)} valid points, "
                  f"temp range: {np.nanmin(y_values):.1f}-{np.nanmax(y_values):.1f}K")

        print(f"Total classes plotted: {plotted_classes}")

        plt.xlabel('Date Index of Thermal Imagery')
        plt.ylabel('Temperature (K)')
        plt.title('Temporal Thermal Signatures by Soil Class')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Set y-axis limits similar to reference image
        plt.ylim(260, 330)

        # Format x-axis to show date indices
        # Create date labels for display (actual dates)
        date_labels = []
        for d in dates:
            if hasattr(d, 'strftime'):
                date_labels.append(d.strftime('%Y-%m-%d'))
            else:
                date_labels.append(str(d)[:10])

        # Show every nth label to avoid overcrowding
        if len(x_indices) > 0:
            step = max(1, len(x_indices) // 10)
            xtick_positions = x_indices[::step]
            xtick_labels = date_labels[::step]

            plt.xticks(xtick_positions, xtick_labels, rotation=45)

            # Add secondary x-axis with indices
            ax = plt.gca()
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(xtick_positions)
            ax2.set_xticklabels([str(i) for i in xtick_positions])
            ax2.set_xlabel('Image Index')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()

    def run_analysis(self, thermal_images_dir=None, use_median=True, save_plot=None):
        """
        Run the complete thermal signature analysis
        """
        if thermal_images_dir:
            self.thermal_images_dir = thermal_images_dir

        print("Starting Thermal Signature Analysis...")

        # Load geo map
        if not self.load_geo_map():
            return

        # Load thermal images
        self.load_thermal_images()

        if not self.thermal_data:
            print("No thermal data loaded!")
            return

        # Calculate statistics
        results, dates = self.calculate_class_statistics(use_median=use_median)

        if results and dates:
            analyzer.plot_yearly_signatures(results, dates)
        if results is None:
            print("Failed to calculate statistics!")
            return

        # Create plot
        self.plot_temporal_signatures(results, dates, save_path=save_plot)

        return results, dates


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    geo_map_path = r"C:\Users\txiki\OneDrive\Documents\Studies\MSc_Geomatics\2Y\Thesis\Masks\Geo_map_resized_Santa.tif"
    thermal_images_dir = r"C:\Users\txiki\OneDrive\Documents\Studies\MSc_Geomatics\2Y\Thesis\Santa_full"

    analyzer = ThermalSignatureAnalyzer(geo_map_path, thermal_images_dir)

    # Run analysis
    results, dates = analyzer.run_analysis(
        thermal_images_dir=thermal_images_dir,
        use_median=True,
        save_plot="temporal_thermal_signatures.png"
    )

    # Print summary statistics
    if results:
        print("\nSummary Statistics:")
        for class_id, values in results.items():
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                print(f"Class {class_id} ({analyzer.class_names[class_id]}): "
                      f"Mean = {np.mean(valid_values):.2f}K, "
                      f"Std = {np.std(valid_values):.2f}K, "
                      f"Range = {np.min(valid_values):.2f}-{np.max(valid_values):.2f}K")