import torch 
from projection.projections import world_to_camera, camera_to_image
from projection.utils import get_extrinsic_matrix, get_intrinsic_matrix, get_cameras

def test_3D_mapping(pred_3D_points: torch.Tensor, expected_xyzs: torch.Tensor, 
                    ext: torch.Tensor, int: torch.Tensor):  

    expected_xyzs = expected_xyzs.t()
    
    camera_coords = world_to_camera(pred_3D_points, ext)   
    pixel_coords, points_depths, indices = camera_to_image(camera_coords, int) 
    
    pred_xyz = torch.vstack((pixel_coords, points_depths)) 

    print(pred_xyz) 
    print(expected_xyzs == pred_xyz) 