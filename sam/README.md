### Installation

1. install SAM with instructions from https://github.com/facebookresearch/segment-anything (only the parts under INSTALLATION)

2. Download pretrained checkpoint to this dir https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

3. Run `sam_test.py` to try prompt selection and mask generation

### Sample Usage
```python
from sam import SAM

MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"


image = ...         # load image
input_points = ...  # shape (N, 2), N sets of x y prompt point coordinates
input_labels = ...  # shape (N), 1s to mark inclusive points, 0s indicate exclusive


sam = SAM(CHECKPOINT_PATH, MODEL_TYPE)
masks, scores, logits = sam.predict(
    image=image,
    input_points=input_points,
    input_labels=input_labels,
)
```