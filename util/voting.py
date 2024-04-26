''' 
All helper functions for finding majority voting of gaussian points 
''' 

import torch  
from util.view import View

# using 0.6 since based on SA3D paper it gave best results
THRESHOLD = 0.6

# Filter the points based on whether the land within the mask or not
def make_filter_matrix(masked_results: list[list[View, torch.Tensor, torch.Tensor]] , pcd: torch.Tensor , num_views: int, num_points: int):
    """
    Input: 
    - masked_results: List of len = num_views, where each row is a list [view, mask, prompt_point_pos]

    For the resultant filter matrix L:
    - Rows -> Number of 3D points
    - Columns -> Number of image views
    """

    L = torch.zeros(num_points, num_views)

    for i, (view, mask, _) in enumerate(masked_results): 

        # Project pcd on view 
        view_proj = view.project(pcd) 

        # Filter points which are within the mask 

        coords = view_proj.coords 
        indices = view_proj.indices    
        coords = view.scale_to_image(coords)  
        coords = coords.type(torch.int) 

        mask_values = mask[coords[1], coords[0]]   # Value in the mask array at each 3D point coordinate
        filtered_indices = indices[mask_values] 

        # Update L matrix 
        L[filtered_indices, i] = 1  

    return L

def majority_voting(L):

    num_points, num_views = L.shape

    # Calculate the sum of each row (how many views agree the point is within the mask)
    votes_sum = L.sum(dim=1)

    # Majority is determined if more than half of the views agree
    majority_threshold = num_views * THRESHOLD

    # Perform majority voting
    majority_vote = (votes_sum > majority_threshold)

    return majority_vote

