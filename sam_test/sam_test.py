from matplotlib import pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import torch

def onclick(event):
    global input_point
    input_point = np.array([[event.xdata, event.ydata]])
    print(f'Updated input_point to: {input_point}')
    plt.close()

image = cv2.imread('./truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model_type = "vit_h"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[model_type](checkpoint="./sam_vit_h_4b8939.pth")
sam.to(device)

predictor = SamPredictor(sam)
predictor.set_image(image)

input_point = np.array([[500, 375]])
input_label = np.array([1])

# click to choose input position
fig, ax = plt.subplots(figsize=(10,10))
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.imshow(image)
plt.show()
fig.canvas.mpl_disconnect(cid)

masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    # multimask_output=False,
)

print(masks.shape)

for i, mask in enumerate(masks):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(input_point[0, 0], input_point[0, 1], 'x')
    plt.imshow(mask[:,:,np.newaxis] * image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.plot(input_point[0, 0], input_point[0, 1], 'x')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
