
import cv2
from gui import SelectorGUI
from sam import SAM

MODEL_TYPE = "vit_h"
IMAGE_PATH = "./truck.jpg"
CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

# load image
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam = SAM(CHECKPOINT_PATH, MODEL_TYPE)
gui = SelectorGUI(image)

# loop until esc is pressed
while True:
    input_points, input_labels = gui.get_prompt()

    if input_points.shape[0] == 0:
        exit()

    # inference
    masks, scores, logits = sam.predict(
        image=image,
        input_points=input_points,
        input_labels=input_labels,
        # multimask_output=False,
    )

    print("mask shape: ", masks.shape)
    print("score shape: ", scores.shape)
    print("logits shape: ", logits.shape)

    print("Scores: ", scores)

    # display results, show scores for each mask
    for i, mask in enumerate(masks):
        label = "BEST SCORE!" if scores[i] == scores.max() else ""
        gui.draw_mask(mask, scores[i], label)
