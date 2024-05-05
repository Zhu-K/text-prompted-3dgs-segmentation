"""
Main file that generates the ply file for the segmented gaussians (and colors it white for masking purposes)  

"""

import sys
sys.dont_write_bytecode = True

import torch 
import numpy as np 
from matplotlib import pyplot as plt
from plyfile import PlyData, PlyElement
from projection.projections import camera_to_image, world_to_camera, filter_pixel_points, get_3D_indices
from util.view import Viewset 
import sam 
from sam.sam import SAM
from sam.lang_sam import LangSAM
from sam.util import get_best_mask, shrink_mask, sample_points_from_mask
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm 
import pickle 
import numpy as np 
from util.voting import make_filter_matrix, majority_voting
from util.segment import create_segmented_ply 

def predict_mask_from_view(prompt_gaussians, view):
    points, labels = view.project(prompt_gaussians).to_sam_input(view.scale)
    if points.shape[0] == 0:
        # handle cases when none of the prompt gaussians are in view
        return None

    mask_view, scores_view, _ = sam.predict(view.image, points, labels, multimask=True)

    # get mask with the best score
    max_score = 0
    max_ind = 0
    for i, score in enumerate(scores_view):
        if score > max_score:
            max_score = score
            max_ind = i

    return mask_view[max_ind], points 

def color_ply_white(fname:str): 
    # Modify segmented file to be pure white so that mask creation can be done 
    ply = PlyData.read(f'./outputs/{fname}')
    vertices = np.array(ply['vertex'].data)

    # Modify the color properties to white
    # Assuming the values are normalized between 0 and 1
    vertices['f_dc_0'] = np.ones(len(vertices), dtype=np.float32)
    vertices['f_dc_1'] = np.ones(len(vertices), dtype=np.float32)
    vertices['f_dc_2'] = np.ones(len(vertices), dtype=np.float32)

    # set all other SH params to 0 to clear view-dependent colouring
    for i in range(45):
        vertices[f'f_rest_{i}'] = np.zeros(len(vertices), dtype=np.float32)

    # Create a new PlyElement for the modified vertices
    updated_vertices = PlyElement.describe(vertices, 'vertex')

    # Save the modified PLY data back to a new file
    ply = PlyData([updated_vertices], text=ply.text)
    ply.write(OUTPUT_FILE)

if __name__ == '__main__': 
    parser = ArgumentParser(description="Mask generation parameters") 
    parser.add_argument("--gaussian_file", "--g", required=True, type=str, default=None, help="Path to segmented gaussian ply file") 
    parser.add_argument("--output", "--o", type=str, default=None, help="Path to output pointcloud gaussian ply file")     
    parser.add_argument("--camera_file", "--c", required=True, type=str, default=None, help="Path to cameras.json file") 
    parser.add_argument("--images", "--i", required=True, type=str, default=None, help="Path to images folder")     
    parser.add_argument("--prompt", type=str, default="truck", help="Text prompt for object selection")     
    parser.add_argument("--name", required=True, type=str, help="Name for segmented ply file")   
    parser.add_argument("--white", action="store_true" , help="flag to color segmented ply white (required for metric generation)")     
    args = parser.parse_args()

    # sampling radius for mapping 2D prompts to 3D gaussians
    EPS = 10
    MASK_PADDING = 10 # do not sample initial points from within padding
    GAUSSIAN_FILE = args.gaussian_file 
    OUTPUT_FILE = GAUSSIAN_FILE if args.output == None else args.output 
    CAMERAS_JSON = args.camera_file 
    IMAGE_PATH = args.images  

    init_view_count = 4 # randomly sample some views to generate initial prompt points from
    prompts_per_view = 2
    text_prompt = args.prompt 

    # create masks from a random subset of views if rate < 1
    view_sampling_rate = 0.5 
    
    
    # Initialization 
    pcd = PlyData.read(GAUSSIAN_FILE)['vertex']
    pcd_array = np.array([pcd['x'], pcd['y'], pcd['z']], dtype=np.double).T
    opacities = pcd['opacity']

    pcd_array = torch.from_numpy(pcd_array)
    opacities = torch.from_numpy(opacities)

    # load sam and lang_sam
    torch.cuda.empty_cache()
    sam = SAM("./sam/sam_vit_h_4b8939.pth")
    langsam = LangSAM(sam.predictor)  

    # Load views, select initial views and 2D prompts 
    views = Viewset(camera_json=CAMERAS_JSON, dataset_path=IMAGE_PATH)
    init_views = views.sample(init_view_count, k_means=True) 

    init_prompts = [] # tuples of (view, prompt_coords)

    for view in init_views:
        masks, boxes, phrases, logits = langsam.predict(Image.fromarray(view.image), text_prompt)
        # plt.plot(init_prompts[0,:], init_prompts[1,:], 'b*')
        mask, logit = get_best_mask(masks, logits)
        # maybe filter out mask with low logits here
        mask_shrunk = shrink_mask(np.uint8(mask.numpy()), MASK_PADDING)
        
        prompts = sample_points_from_mask(mask_shrunk, prompts_per_view).T
        init_prompts.append((view, prompts))

    # Map 2D input points to 3D gaussians 
    prompt_gaussians = torch.empty((0, 3))

    for view, prompts in init_prompts:

        projection = view.project(pcd_array)

        # scale user points in case image mismatches sizes defined in cameras.json
        user_points = view.scale_from_image(prompts)

        pcd_indices = get_3D_indices(projection.coords, projection.depths, 
                                    projection.indices, user_points, EPS, opacities)

        gaussians = pcd_array[pcd_indices, :] 
        prompt_gaussians = torch.vstack((prompt_gaussians, gaussians)) 


    # Create masks from multiple views based on view_sampling_rate 
    torch.cuda.empty_cache()
    views_subset = views.sample(round(view_sampling_rate * views.count))

    # results is a list of result sublist each containing [view, mask, prompt_point_pos]
    results = []
    missing_view_indices = []
    for view in tqdm(views_subset):
        result = predict_mask_from_view(prompt_gaussians, view)
        if result is not None:
            results.append([view, *result])
        else:
            missing_view_indices.append(view.index)

    print("Views containing no visible prompt points:", missing_view_indices) 

    # save serialized results 
    with open(f"./results_{args.name}.pkl", "wb") as f:
        pickle.dump(results, f)  

    # load results
    with open(f"./results_{args.name}.pkl", 'rb') as f:
        results = pickle.load(f) 

    # Majority Voting 
    num_views = len(results) 
    num_points = pcd_array.shape[0]
    L = make_filter_matrix(results, pcd_array, num_views, num_points)  
    segmented_gaussian_mask = majority_voting(L) 
    segmented_pcd_array = pcd_array[segmented_gaussian_mask, :]  
    
    # Create segmented gaussian file 
    create_segmented_ply(pcd, segmented_pcd_array, segmented_gaussian_mask, f'{args.name}.ply')  

    if args.white: 
        color_ply_white(f'{args.name}.ply') 

    print(f'Segmented gaussian file created at {OUTPUT_FILE}') 
    print('Render to see results')
