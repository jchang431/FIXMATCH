import torch.nn as nn
from torchvision.models import resnet18


class SupervisedModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        num_classes = 10
        if hasattr(cfg, "data") and hasattr(cfg.data, "num_classes"):
            num_classes = cfg.data.num_classes

        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
