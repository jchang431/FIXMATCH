import torch.nn as nn
from torchvision.models import resnet18

class FixMatchModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(512, cfg.dataset.num_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)
