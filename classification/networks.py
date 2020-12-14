import torch
import torch.nn as nn
import torchvision
from unet import UNet
from model import Net
CLASSES=100



class ResNetContrast(torchvision.models.ResNet):
    def __init__(self, block, layers, num_classes=100):
        super(ResNetContrast, self).__init__(block, layers, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        y = torch.flatten(x, 1)
        z = self.fc(y)
        if(self.training):
            y2 = nn.functional.normalize(y, dim=1, p=2)
            return z,y2
        return z


class ResNetContrastWFC(torchvision.models.ResNet):
    def __init__(self, block, layers, num_classes=100):
        super(ResNetContrastWFC, self).__init__(block, layers, num_classes)
        self.featfc = nn.Linear(512, 512)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        y = torch.flatten(x, 1)
        yfeat = self.featfc(y)
        z = self.fc(yfeat)
        if(self.training):
            y2 = nn.functional.normalize(yfeat, dim=1, p=2)
            return z,y2
        return z

