import torch.nn as nn

class YoloWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out

class YoloClassScoreTarget:
    def __init__(self, class_id):
        self.class_id = class_id
    def __call__(self, output):
        return output[4 + self.class_id, :].max()