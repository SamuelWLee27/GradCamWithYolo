from PIL import Image
import torchvision.transforms as transforms


class ImagePreprocessor:
    def __init__(self):
        self.image = None
        self.transform = transforms.Compose([
            transforms.Resize(640),
            transforms.CenterCrop(640),
        ])

        self.tensor = None

    def load_image(self, path):
        self.image = Image.open(path)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        self.image = self.transform(self.image.convert("RGB"))
        self.tensor = transforms.ToTensor()(self.image).float()

    def get_preprocessed_image(self):
        img = self.tensor
        return img.unsqueeze(0)
