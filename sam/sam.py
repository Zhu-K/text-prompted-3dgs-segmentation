import torch
from segment_anything import SamPredictor, sam_model_registry


class SAM:
    def __init__(self, checkpoint="./sam_vit_h_4b8939.pth", model="vit_h") -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")   

        print("Loading pth...", end="")
        sam = sam_model_registry[model](checkpoint=checkpoint)
        sam.to(self.device)
        print("DONE")

        self.predictor = SamPredictor(sam)

    def predict(self, image, input_points, input_labels, multimask=True):
        self.predictor.set_image(image)
        return self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=multimask
        )
