import torch
import torchvision.models as models
import torchvision.transforms as T

class FeatureExtractor:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
        ])

    def extract(self, image):
        with torch.no_grad():
            image = self.transform(image).unsqueeze(0)
            features = self.model(image)
            return features.squeeze().numpy()
