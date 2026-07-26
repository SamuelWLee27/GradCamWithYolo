import cv2
import torchvision.transforms as transforms

class ImagePreprocessor:
    def __init__(self):
        self.image = None
        self.image_transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_image(self, path):
        self.image = cv2.imread(path, cv2.IMREAD_COLOR)

    def get_preprocessed_image(self):
        img = self.image_transform(self.image)
        img = img.unsqueeze(0)
        return img
