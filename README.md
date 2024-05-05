# Prompt-GS: Segment Anything in 3D Gaussians using Multi-View Text Prompting 

This repository contains the unofficial implementation of [Segment Anything in 3D Gaussians](https://arxiv.org/abs/2401.17857)* , with the added functionality of multi-view text prompting to get the "initial user points" for object segmentation. 

For detailed qualitative and quantitative comparisons with the original paper, please refer to our [report](https://drive.google.com/file/d/1i6wGrMlp3WK64zs21UmKRM7YhhdSrIWT/view?usp=sharing) 

> \* _The Gaussian Decomposition functionality has yet to be implemented. If you are interested, please feel free to make a pull request!_  

# What did we change? 

The small change we added which lead to better overall results was using lang-sam to sample initial points not only from one view, but from different views which are as diverse as possible (ensured using K-Means Clustering on views based on position and orientation of cameras). 
This allows better set of points to be selected initially and subsequently does better segmentation. For full detailed reasoning and results, please refer to report above. 

# Notebook Setup 

You can run/view the commands we run in `main_pipeline_langsam.ipynb` for this implementation. The `main_pipeline.ipynb` has the implementation without lang-sam, where the user clicked points are used on a single view for initialization. We discard the use of this in the `main_segment.py` script since lang-sam does a much better job!

# Running Steps to Reproduce Results  

Our implementation is based off the Spin-nerf dataset (for reasons outlined in original paper). So, first please download the [SPIn-NeRF Dataset](https://drive.google.com/drive/folders/1N7D4-6IutYD40v9lfXGSVbWrd47UdJEC) 

### main_segment.py

Run the `main_segment.py` file with the following arguments: 

```
python main_segment.py

--gaussian_file <path to trained 3D gaussian ply file>
--camera_file <path to cameras.json>
--images <path to scene images folder>
--output <path to output folder>
--name <name of segmented ply file (without extension)>
--white <enter if you want to get metrics later> 
```

At present, you need to manually add the `cameras.json` and `cfg_args` file to the `--output` folder. We will add the automation for setting that up soon.  

The `--white` command creates a copy of the segmented gaussian ply file and colors it pure white so that its projection can be interpreted as a mask and directly compute IoU and accuracy metrics with the ground truth masks.  
The unaltered segmented gaussian file can be found at `./outputs/`

### gaussian-splatting 

After running the previous command, you should have a ply file ready for rendering. To do this, we make use of the original gaussian splatting codebase. However, for the sake of easy use, we made the following change to `render.py` in the original gaussian splatting codebase: 

```
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "allRenders", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "allRenders", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

```

Run the `render.py` file with the following arguments: 

```
python render.py 
-m <path to trained guassian output folder (replace original ply with segmented ply if --white not passed; else --output path)>  
-s <path to colmap output folder having "images" and "sparse" folders.  
--data_device cuda 
```

> \* _Please make sure that if `--white` was not passed previously, you create a new folder with `cameras.json`, `cfg_args` and `point_cloud>iteration_30000>point_cloud.ply` where the segmented ply file in ./outputs is renamed to point_cloud.ply_ 

### metrics.py 

Run this to print the metrics. 

```
python metrics.py
--dataset <path to spin-nerf dataset scene of your choice> (truck used by us)
--gaussian_output <path to trained guassian output folder (replace original ply with segmented ply if --white not passed; else --output path)>
``` 
