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
import rasterio
import matplotlib.pyplot as plt


def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def process_tif(input_file, output_file):
    if "LST" in input_file:
        file = "Land Surface Temperature"
    elif "EMISS" in input_file:
        file = "Emissivity"
    else:
        raise ValueError("File name must contain 'LST' or 'B10'.")

    with rasterio.open(input_file) as src:
        data = src.read(1)
        profile = src.profile  # metadata

        # ignore invalid data (e.g., nodata values)
        data = np.ma.masked_invalid(data)

        # min/max values
        min_val = data.min()
        max_val = data.max()

        print(f"Processing {file} file")
        print(f"Minimum {file} value: {min_val}")
        print(f"Maximum {file} value: {max_val}")

        # normalizing
        normalized_data = normalize(data, min_val, max_val)

        profile.update(dtype=rasterio.float32, nodata=None)

        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(normalized_data.filled(0).astype(rasterio.float32), 1)

    print(f"Normalized TIF file saved as: {output_file}")

    plt.figure(figsize=(10, 6))
    plt.title(f"Normalized {file} Image")
    plt.imshow(normalized_data, cmap='viridis')
    plt.colorbar(label=f"Normalized {file}")
    plt.xlabel("Pixel Index")
    plt.ylabel("Pixel Index")
    plt.show()

    return normalized_data


# Emissivity
emissivity_file = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Code/Set-up/Thesis/Thesis/Images/Thermal/ECO_EMISS_1812.tif"
output_emissivity = "Normalized_emissivity.tif"
process_tif(emissivity_file, output_emissivity)

# Surface Temperature
lst_file = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Images/Thermal/ECO_LST_1812.tif"
output_temperature = "Normalized_LST.tif"
process_tif(lst_file, output_temperature)
