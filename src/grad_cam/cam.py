from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.yolo.wrapper import YoloClassScoreTarget
import torch

def eigen_cam(model, input_tensor, target_layers, rgb):
    cam = EigenCAM(model, target_layers)
    grayscale_cam = cam(input_tensor, eigen_smooth=True)[0]

    return show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

def grad_cam(model, input_tensor, target_layers, rgb, target):
    for p in model.parameters():
        p.requires_grad_(True)

    with torch.enable_grad():
        cam = GradCAM(model, target_layers)
        grayscale_cam = cam(input_tensor, targets=[YoloClassScoreTarget(target)],
                            eigen_smooth=True)[0]

    return show_cam_on_image(rgb, grayscale_cam, use_rgb=True)