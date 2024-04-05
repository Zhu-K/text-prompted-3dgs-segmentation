from matplotlib import pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import torch

MODEL_TYPE = "vit_h"
IMAGE_PATH = "./truck.jpg"
CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

# each click adds a point, close first drawn window to generate mask
def onclick(event):
    global input_point
    global input_label
    input_point = np.append(
        input_point, np.array([[event.xdata, event.ydata]]), axis=0)
    input_label = np.append(input_label, 1)
    plt.plot(event.xdata, event.ydata, 'x')
    print(f'input points: {input_point}')
    plt.draw()


image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device)

predictor = SamPredictor(sam)
predictor.set_image(image)

input_point = np.empty((0, 2), float)
input_label = np.array([], int)

# display original image for selection
# click to choose input position
fig, ax = plt.subplots(figsize=(10,10))
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.imshow(image)
plt.show()
fig.canvas.mpl_disconnect(cid)

# inference
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    # multimask_output=False,
)


print("mask shape: ", masks.shape)
print("score shape: ", scores.shape)
print("logits shape: ", logits.shape)

print("Scores: ", scores)

# display results, show scores for each mask
for i, mask in enumerate(masks):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f"Score: {scores[i]}")
    plt.plot(input_point[:, 0], input_point[:, 1], 'x')
    plt.imshow(mask[:,:,np.newaxis] * image)
    plt.axis('off')


    plt.subplot(1, 2, 2)
    # mark mask with best score
    plt.title("BEST SCORE!" if scores[i] == scores.max() else "")
    plt.plot(input_point[:, 0], input_point[:, 1], 'x')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
