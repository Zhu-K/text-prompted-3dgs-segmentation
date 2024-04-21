import torch 
import open3d as o3d  
import numpy as np 
   
def world_to_camera(points: torch.Tensor, extrinsic: torch.Tensor) -> torch.Tensor: 
    rot = extrinsic[:3, :3] 
    pos = extrinsic[:3, 3:4] 
    return (rot @ points.transpose(0,1)) + pos 

def camera_to_image(points: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor: 

    indices = torch.arange(points.shape[1])

    # Discarding points that have positive Z value, because these points lie behind the camera 
    front_of_cam_mask = points[2, :] >= 0
    points = points[:, front_of_cam_mask]
    
    pixel_coords = intrinsic @ points  
    pixel_coords[:2, :] = pixel_coords[:2, :] / pixel_coords[2:3, :] 
    pixel_coords , point_depths = pixel_coords[:2, :].type(torch.int) , pixel_coords[2:3, :] 

    return pixel_coords, point_depths.squeeze(), indices[front_of_cam_mask]

def filter_pixel_points(pixel_coords: torch.Tensor, depths: torch.Tensor, indices: torch.Tensor ,width: int, height: int): 
    valid_mask = (pixel_coords[0, :] >= 0) & (pixel_coords[0, :] < width) & \
                (pixel_coords[1, :] >= 0) & (pixel_coords[1, :] < height)
    
    filtered_points = pixel_coords[:, valid_mask] 
    filtered_depths = depths[valid_mask]
    filtered_indices =  indices[valid_mask]

    return filtered_points, filtered_depths, filtered_indices


def filter_projection_by_distance(projections: torch.Tensor, depths: torch.Tensor, indices: torch.Tensor, user_points: torch.Tensor, eps: float, opacities: torch.Tensor = None, opacity_threshold: float = 0.5) -> torch.Tensor:
    '''
    projections: (2,n) pixel coordinates of the projections of all visible 3D points in the camera view 
    depths: (1,n) depths for each projection 
    user_points: (2,m) set of user clicked points  
    eps: distance within which to allow 3D point for consideration 
    '''

    # Calculate pairwise distances between user_points and projections
    # The result will be a tensor of shape (m, n), where distances[i, j] is the distance
    # between user_points[:, i] and projections[:, j] 

    distances = torch.cdist(user_points.t().type(torch.float64), projections.t().type(torch.float64)) 

    if opacities is not None:
        opacities = torch.sigmoid(opacities[indices]).unsqueeze(0)
        mask = (distances < eps) & (opacities > opacity_threshold)
    else:
        mask = distances < eps

    # Filter projections for each user_point based on the mask
    filtered_projections = [projections[:, mask[i]] for i in range(mask.size(0))]
    filtered_depths = [depths[mask[i]] for i in range(mask.size(0))]  
    filtered_indices = [indices[mask[i]] for i in range(mask.size(0))]

    return filtered_projections, filtered_depths, filtered_indices
    

def get_3D_indices(projections: torch.Tensor, depths: torch.Tensor, indices: torch.Tensor, user_points: torch.Tensor, eps: float, opacities: torch.Tensor = None) -> torch.Tensor:

    '''
    projections: (2,n) pixel coordinates of the projections of all visible 3D points in the camera view 
    depths: (1,n) depths for each projection 
    user_points: (2,m) set of user clicked points  
    eps: distance within which to allow 3D point for consideration 
    ''' 

    gaussian_indices = [] 

    total_filt_projections, filt_depths, filt_indices = filter_projection_by_distance(
        projections, depths, indices, user_points, eps, opacities)


    # For each user point, select projection which is least far away (depth-wise)
    for i in range(user_points.shape[1]):  
        print(f'Point {i+1}: ')
        print(f'\t Depth: {filt_depths[i][torch.argmin(filt_depths[i])]}')
        print(f'\t (x,y) => {total_filt_projections[i][:, torch.argmin(filt_depths[i])]}')
        mapped_gaussian_index = filt_indices[i][torch.argmin(filt_depths[i])]
        gaussian_indices.append(mapped_gaussian_index)
    
    return torch.tensor(gaussian_indices)  
    

