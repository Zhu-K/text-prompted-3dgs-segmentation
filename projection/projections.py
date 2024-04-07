import torch 
import open3d as o3d  
import numpy as np 
   
def world_to_camera(points: torch.Tensor, extrinsic: torch.Tensor) -> torch.Tensor: 
    rot = extrinsic[:3, :3] 
    pos = extrinsic[:3, 3:4] 
    return (rot @ points.transpose(0,1)) + pos 

def camera_to_image(points: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor: 

    # Discarding points that have positive Z value, because these points lie behind the camera 
    front_of_cam_mask = points[2, :] >= 0
    points = points[:, front_of_cam_mask]
    
    pixel_coords = intrinsic @ points  
    pixel_coords[:2, :] = pixel_coords[:2, :] / pixel_coords[2:3, :] 
    pixel_coords , point_depths = pixel_coords[:2, :].type(torch.int) , pixel_coords[2:3, :] 

    return pixel_coords, point_depths 

def filter_pixel_points(pixel_coords: torch.Tensor, width: int, height: int): 
    valid_mask = (pixel_coords[0, :] >= 0) & (pixel_coords[0, :] < width) & \
                (pixel_coords[1, :] >= 0) & (pixel_coords[1, :] < height)
    
    print(valid_mask)

    filtered_points = pixel_coords[:, valid_mask]
    filtered_indices = torch.nonzero(valid_mask).squeeze()  

    return filtered_points, filtered_indices  


