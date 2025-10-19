"""
Utility functions shared between CNN and ConvLSTM models.
Contains functions for map reconstruction and size determination.
"""

import numpy as np


def get_map_size_from_path(data_path):
    """
    Gets map dimensions based on the area in the data path.
    """
    if 'Puertollano' in data_path:
        return (273, 415)  # height, width
    elif 'Santa Olalla' in data_path:
        return (278, 425)
    elif 'Villoslada' in data_path:
        return (267, 397)
    else:
        raise ValueError(f"Unknown area in path: {data_path}")


def reconstruct_map_from_patches(patches, positions, map_size, patch_size=128, handle_overlap=True):
    """
    Reconstruct full map from patches using stored positions.
    """
    map_h, map_w = map_size
    has_channels = len(patches.shape) == 4
    if handle_overlap:
        # For overlapping patches: accumulate and average
        if has_channels:
            reconstructed = np.zeros((map_h, map_w, patches.shape[-1]), dtype=np.float32)
            count_map = np.zeros((map_h, map_w, patches.shape[-1]), dtype=np.float32)
        else:
            reconstructed = np.zeros((map_h, map_w), dtype=np.float32)
            count_map = np.zeros((map_h, map_w), dtype=np.float32)

        # Accumulate patches
        for i, (patch, pos) in enumerate(zip(patches, positions)):
            x_start = int(pos[0])
            y_start = int(pos[1])

            x_end = min(x_start + patch_size, map_h)
            y_end = min(y_start + patch_size, map_w)

            patch_h = x_end - x_start
            patch_w = y_end - y_start

            if has_channels:
                reconstructed[x_start:x_end, y_start:y_end, :] += patch[:patch_h, :patch_w, :]
                count_map[x_start:x_end, y_start:y_end, :] += 1
            else:
                reconstructed[x_start:x_end, y_start:y_end] += patch[:patch_h, :patch_w]
                count_map[x_start:x_end, y_start:y_end] += 1

        # Average where patches overlap
        valid_mask = count_map > 0
        reconstructed[valid_mask] = reconstructed[valid_mask] / count_map[valid_mask]
        if not has_channels and patches.dtype in [np.int32, np.int64]:
            reconstructed = np.round(reconstructed).astype(patches.dtype)

    else:
        # For non-overlapping patches: direct placement
        if has_channels:
            reconstructed = np.zeros((map_h, map_w, patches.shape[-1]), dtype=patches.dtype)
        else:
            reconstructed = np.zeros((map_h, map_w), dtype=patches.dtype)

        for i, (patch, pos) in enumerate(zip(patches, positions)):
            x_start = int(pos[0])
            y_start = int(pos[1])

            x_end = min(x_start + patch_size, map_h)
            y_end = min(y_start + patch_size, map_w)

            patch_h = x_end - x_start
            patch_w = y_end - y_start

            if has_channels:
                reconstructed[x_start:x_end, y_start:y_end, :] = patch[:patch_h, :patch_w, :]
            else:
                reconstructed[x_start:x_end, y_start:y_end] = patch[:patch_h, :patch_w]

    return reconstructed