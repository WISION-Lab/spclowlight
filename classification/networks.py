import torch
import torch.nn as nn
import torchvision
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
            return z,x
        return z


class ResNetContrast2(torchvision.models.ResNet):
    def __init__(self, block, layers, num_classes=100):
        super(ResNetContrast2, self).__init__(block, layers, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        a = self.avgpool(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        b = self.avgpool(x)
        x = self.layer2(x)
        c = self.avgpool(x)
        x = self.layer3(x)
        d = self.avgpool(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        e = x
        y = torch.flatten(x, 1)
        z = self.fc(y)
        if(self.training):
            return z,(a,b,c,d,e)
        return z


class ResNetPerceptual(torchvision.models.ResNet):
    def __init__(self, block, layers, num_classes=100, teacher_model=False):
        super(ResNetPerceptual, self).__init__(block, layers, num_classes)
        self.teacher_model = teacher_model
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        a = x
        x = self.maxpool(x)
        x = self.layer1(x)
        b = x
        x = self.layer2(x)
        c = x
        x = self.layer3(x)
        d = x
        x = self.layer4(x)
        e = x
        x = self.avgpool(x)

        y = torch.flatten(x, 1)
        z = self.fc(y)
        if(self.training or self.teacher_model):
            return z,(a,b,c,d,e)
        return z

