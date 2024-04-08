import open3d as o3d
import torch
import json  
import numpy as np 

def get_cameras(json_file):   
    cameras = None 
    with open(json_file) as f: 
        cameras = json.load(f) 
    
    return cameras 

def get_extrinsic_matrix(camera: dict): 
    translation =  -torch.tensor(camera['position'], dtype=torch.float64).unsqueeze(dim=1) 
    rotation = torch.tensor(camera['rotation'], dtype=torch.float64).T

    translation = rotation @ translation

    extrinsic = torch.zeros((4,4), dtype=torch.float64) 
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3:4] = translation
    extrinsic[3:4, 3:4] = 1   

    return extrinsic 


def get_intrinsic_matrix(camera: dict): 
    width = camera['width'] 
    height = camera['height']  

    f_x = camera['fx'] 
    f_y = camera['fy'] 
    c_x = width / 2 
    c_y = height / 2  

    intrinsic = torch.tensor([
        [f_x, 0 , c_x], 
        [0 , f_y, c_y], 
        [0, 0, 1]
        ], dtype=torch.float64)   

    return intrinsic  

def get_example_camera(cameras): 
    test_camera = cameras[0] 

    img_name = test_camera['img_name'] 
    width = test_camera['width'] 
    height = test_camera['height']  
    translation =  torch.tensor(test_camera['position'], dtype=torch.float64).unsqueeze(dim=1) 
    rotation = torch.tensor(test_camera['rotation'], dtype=torch.float64) 
    f_x = test_camera['fx'] 
    f_y = test_camera['fy'] 
    c_x = width // 2 
    c_y = height // 2 

    print(f'Image dimension: ({width},{height})') 
    print(f'Translation vector: \n\t {translation}') 
    print(f'Rotation Matrix: \n\t {rotation}') 

def save_ply(pcd_array:torch.Tensor): 
    pcd = o3d.geometry.PointCloud()  
    pcd.points = o3d.utility.Vector3dVector(pcd_array.numpy()) 
 
    o3d.io.write_point_cloud('saved.ply', pcd)   