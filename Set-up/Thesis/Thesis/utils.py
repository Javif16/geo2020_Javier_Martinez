"""
Utility functions shared between CNN and ConvLSTM models.
Contains functions for map reconstruction and size determination.
"""

import numpy as np


def get_map_size_from_path(data_path):
    """
    Get map dimensions based on the area in the data path.

    Args:
        data_path: Path string containing area name

    Returns:
        Tuple of (height, width) for the map

    Raises:
        ValueError: If area name is not recognized
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

    Args:
        patches: Array of patches (can be 3D for single-channel or 4D for multi-channel)
        positions: Array of positions for each patch (N, 3) where columns are [x, y, date]
        map_size: Tuple of (height, width) for the full map
        patch_size: Size of each patch in pixels (default 64, use 128 for new patches)
        handle_overlap: If True, average overlapping predictions (for overlapping patches)
                       If False, last patch overwrites (for non-overlapping patches)

    Returns:
        Reconstructed map as numpy array
    """
    map_h, map_w = map_size

    # Determine if patches have channel dimension
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

            # Calculate actual size to copy (handle edge patches)
            x_end = min(x_start + patch_size, map_h)
            y_end = min(y_start + patch_size, map_w)

            patch_h = x_end - x_start
            patch_w = y_end - y_start

            # Add patch values and increment count
            if has_channels:
                reconstructed[x_start:x_end, y_start:y_end, :] += patch[:patch_h, :patch_w, :]
                count_map[x_start:x_end, y_start:y_end, :] += 1
            else:
                reconstructed[x_start:x_end, y_start:y_end] += patch[:patch_h, :patch_w]
                count_map[x_start:x_end, y_start:y_end] += 1

        # Average where patches overlap (avoid division by zero)
        valid_mask = count_map > 0
        reconstructed[valid_mask] = reconstructed[valid_mask] / count_map[valid_mask]

        # For classification labels (single channel, integer values), round to nearest class
        if not has_channels and patches.dtype in [np.int32, np.int64]:
            reconstructed = np.round(reconstructed).astype(patches.dtype)

    else:
        # For non-overlapping patches: direct placement (original behavior)
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