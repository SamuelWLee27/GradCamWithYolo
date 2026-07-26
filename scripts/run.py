from src.image.preprocess import ImagePreprocessor
from src.yolo.model import YoloModel
from src.yolo.wrapper import YoloWrapper
import torch
import cv2
import numpy as np
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo = YoloModel("yolov8n.pt", device)
detection_model = yolo.val()

pre = ImagePreprocessor()
pre.load_image("../data/dog.jpeg")
input_tensor = pre.get_preprocessed_image().to(device)

target_layers = [detection_model.model[-2]]

model = YoloWrapper(detection_model)

with EigenCAM(model, target_layers) as cam:
    grayscale_cam = cam(input_tensor, eigen_smooth=True)[0]

rgb = np.array(pre.image, dtype=np.float32) / 255.0

cam_image = show_cam_on_image(rgb, grayscale_cam, use_rgb=True,
                              colormap=cv2.COLORMAP_VIRIDIS)   # uint8 RGB
res = yolo.model.predict("../data/dog.jpeg", verbose=False)[0]
print(res.boxes.cls, res.boxes.conf)
cv2.imwrite("../result/cam.jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
print("cropped shape:", np.array(pre.image).shape)   # expect (640, 640, 3)