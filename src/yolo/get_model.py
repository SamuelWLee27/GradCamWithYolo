from ultralytics import YOLO

class YoloModel:
    def __init__(self, model_name: str):
        self.model = YOLO(model_name)

    def train(self, data=None, epochs: int=5):
        self.model.train(data=data, epochs=epochs)

    def val(self, data=None):
        return self.model.val(data=data)

    def detect(self, image):
        return self.model(image)

    def export(self, path: str, export_format: str="onnx"):
        self.model.export(path=path, format=export_format)