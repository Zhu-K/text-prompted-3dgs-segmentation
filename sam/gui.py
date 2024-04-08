from matplotlib import pyplot as plt
import numpy as np


class SelectorGUI:
    def __init__(self, image) -> None:
        self.image = image

    def get_prompt(self):
        self.input_point = np.empty((0, 2), float)
        self.input_label = np.array([], int)

        # display original image for selection
        # click to choose input position
        fig, ax = plt.subplots(figsize=(10, 10))
        cid_click = fig.canvas.mpl_connect('button_press_event', self._onclick)
        cid_key = fig.canvas.mpl_connect('key_press_event', self._onkey)
        plt.imshow(self.image)
        plt.title("Click to select points, ESC to quit")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        fig.canvas.mpl_disconnect(cid_click)

        return self.input_point, self.input_label

    def draw_mask(self, mask, score, label):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title(f"Score: {score}")
        plt.plot(self.input_point[:, 0], self.input_point[:, 1], 'x')
        plt.imshow(mask[:, :, np.newaxis] * self.image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        # mark mask with best score
        plt.title(label)
        plt.plot(self.input_point[:, 0], self.input_point[:, 1], 'x')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # each click adds a point, close first drawn window to generate mask

    def _onclick(self, event):
        self.input_point = np.append(
            self.input_point, np.array([[event.xdata, event.ydata]]), axis=0)
        self.input_label = np.append(self.input_label, 1)
        plt.plot(event.xdata, event.ydata, 'x')
        print(f'input points: {self.input_point}')
        plt.draw()

    # exit program only when escape is pressed

    def _onkey(self, event):
        if event.key == 'escape':
            exit()
