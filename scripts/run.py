from src.image.preprocess import ImagePreprocessor
from src.yolo.model import YoloModel
from src.yolo.wrapper import YoloWrapper
from src.grad_cam.cam import eigen_cam, grad_cam
import torch
import cv2
import numpy as np

model_name = "yolov8n.pt"
image_path = "data/dog.jpeg"
output_eigen_path = "result/eigen_cam.jpg"
output_grad_path = "result/grad_cam.jpg"
DOG = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo = YoloModel(model_name, device)
detection_model = yolo.val()

pre = ImagePreprocessor()
pre.load_image(image_path)
input_tensor = pre.get_preprocessed_image().to(device)

target_layers = [detection_model.model[-2]]

model = YoloWrapper(detection_model)

rgb = np.array(pre.image, dtype=np.float32) / 255.0

eigen_cam_image = eigen_cam(model, input_tensor, target_layers, rgb)
grad_cam_image = grad_cam(model, input_tensor, target_layers, rgb, DOG)

cv2.imwrite(output_eigen_path, cv2.cvtColor(eigen_cam_image, cv2.COLOR_RGB2BGR))
cv2.imwrite(output_grad_path, cv2.cvtColor(grad_cam_image, cv2.COLOR_RGB2BGR))
print(f"Saved to {output_eigen_path} and {output_grad_path}")