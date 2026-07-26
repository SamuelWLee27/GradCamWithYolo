from ultralytics import YOLO
import torch

class YoloModel:
    def __init__(self, model_name: str, device):
        self.model = YOLO(model_name)
        self.device = device

    def val(self):
        assert isinstance(self.model.model, torch.nn.Module)
        return self.model.model.to(self.device).eval()
