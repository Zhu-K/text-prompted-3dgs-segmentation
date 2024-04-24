import numpy as np
import torch
import cv2


def get_best_mask(masks, logits):
    max_logit = 0
    max_index = 0
    for i in range(len(masks)):
        if logits[i].item() > max_logit:
            max_logit = logits[i].item()
            max_index = i

    return masks[max_index], max_logit


def shrink_mask(mask, n):
    # shrink mask by n pixels
    kernel = np.ones((2*n+1, 2*n+1), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    return eroded_mask


def sample_points_from_mask(mask, k):
    # randomly sample k points from the mask
    y_indices, x_indices = np.where(mask == 1)
    points = np.array(list(zip(x_indices, y_indices)))

    if k > len(points):
        raise ValueError("Not enough points to sample from")

    # Randomly select k points
    sampled_points = points[np.random.choice(len(points), k, replace=False)]

    return torch.Tensor(sampled_points)
