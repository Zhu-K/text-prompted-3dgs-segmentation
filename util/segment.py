""" 
Helper functions for the final segmentation of gaussians 
"""

import numpy as np 
import torch 
from plyfile import PlyElement, PlyData


def create_segmented_ply(pcd, segmented_pcd_array: torch.Tensor, segmented_gaussian_mask , fname: str):  
    """
    fname: name of file 
    """ 

    dtype = [(property.name, property.val_dtype) for property in pcd.properties]

    # Create a structured array with the same shape as your pcd_array
    structured_array = np.empty(segmented_pcd_array.shape[0], dtype=dtype)

    # Assign the coordinates and opacities to the structured array
    structured_array['x'] = segmented_pcd_array[:, 0]
    structured_array['y'] = segmented_pcd_array[:, 1]
    structured_array['z'] = segmented_pcd_array[:, 2]
    
    for property in pcd.properties: 
        structured_array[property.name] = pcd[property.name][segmented_gaussian_mask]

    # Create a PlyElement from the structured array
    vertex_element = PlyElement.describe(structured_array, 'vertex')

    # Create a PlyData object and write to a file
    ply_data = PlyData([vertex_element])
    ply_data.write(f'./outputs/{fname}')