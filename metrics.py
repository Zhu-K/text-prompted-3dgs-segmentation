""" 
Responsible for printing metrics and generating folder with following structure. gt has all the ground truth 2D masks for each view. renders has the masks generated from projecting the segmented gaussian points. 

<output-location>
|---gt
    |---<image 0>
    |---<image 1>
    |---...
|---renders
    |---<image 0>
    |---<image 1>
    |---... 

"""  

import os  
import cv2
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt  
from argparse import ArgumentParser  
from PIL import Image

if __name__ == '__main__': 

    parser = ArgumentParser(description="Metrics generation parameters") 
    parser.add_argument("--dataset", required=True, type=str, default=None, help="Path to SPIN-NERF Scene folder")  
    parser.add_argument("--output", type=str, default='../outputRenders', help="Output folder")  
    parser.add_argument("--gaussian_output", type=str, required=True, default=None, help="Gaussian splatting output folder path")  
    args = parser.parse_args() 

    SPINNERF_DATA = args.dataset  
    DEST_DIR = args.output 
    RENDERED_PATH = f'{args.gaussian_output}/allRenders/ours_30000/renders'  

    new_size = None

    # Create the destination directory if it doesn't exist
    if not os.path.exists(DEST_DIR):
        os.makedirs(os.path.join(DEST_DIR, 'gt'))  
        os.makedirs(os.path.join(DEST_DIR, 'renders'))

    for filename in os.listdir(SPINNERF_DATA):
        # Check if the file name contains 'pseudo'
        if 'pseudo' in filename:
            # Construct the full file path
            src_file = os.path.join(SPINNERF_DATA, filename)
            
            # Extract the file number before the first underscore in the file name
            new_filename = filename.split('_pseudo')[0] + '.png'   
            if '_' in new_filename: 
                new_filename = new_filename.split('_')[1]  
            
            # Construct the destination ground file path
            dst_file = os.path.join(DEST_DIR, 'gt' , new_filename) 

            # Construct the destination render file path 
            rend_file = os.path.join(RENDERED_PATH, new_filename) 

            # Create binary mask out of the gt and renders 
            gt_img = cv2.imread(src_file) 

            if new_size == None: 
                new_size = gt_img.shape[:2][::-1] 

            rend_img = cv2.imread(rend_file)   
            rend_img =  cv2.resize(rend_img, new_size, interpolation = cv2.INTER_LINEAR)

            gray_rend = cv2.cvtColor(rend_img, cv2.COLOR_BGR2GRAY)  
            _, binary_rend = cv2.threshold(gray_rend, int(0.6 * 255), 255, cv2.THRESH_BINARY) 

            gray_gt = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)  
            _, binary_gt = cv2.threshold(gray_gt, int(0.0 * 255), 255, cv2.THRESH_BINARY) 

            dst_file2 = os.path.join(DEST_DIR, 'renders' , new_filename)

            # Copy and rename the file
            cv2.imwrite(dst_file, binary_gt) 
            cv2.imwrite(dst_file2, binary_rend)    
    
    iou = 0.0 
    accuracy = 0.0
    count = 0

    for filename in os.listdir(os.path.join(DEST_DIR, 'gt')):  

        count += 1 

        gt_file = os.path.join(DEST_DIR, 'gt', filename) 
        rend_file = os.path.join(DEST_DIR, 'renders', filename)

        # Load images
        gt_image = np.asarray(Image.open(gt_file).convert('L'))  # Convert to grayscale
        pred_image = np.asarray(Image.open(rend_file).convert('L'))  # Convert to grayscale   

        # Threshold images to get binary masks, assuming non-black (non-zero) pixels are foreground
        gt_mask = np.array(gt_image) > 0
        pred_mask = np.array(pred_image) > 0

        # Calculate intersection and union
        intersection = np.logical_and(gt_mask, pred_mask)
        union = np.logical_or(gt_mask, pred_mask)

        # Calculate IoU
        iou += intersection.sum() / union.sum()

        # Calculate pixel accuracy
        accuracy += np.mean(gt_mask == pred_mask)  


    iou /= count 
    accuracy /= count 

    print(f'IoU: {iou:.4f}')
    print(f'Accuracy: {accuracy:.4f}')