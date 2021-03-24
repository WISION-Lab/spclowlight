import torch
import torchvision
CLASSES=1000

class ResNetEncoder(torchvision.models.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNetEncoder, self).__init__(block, layers, num_classes)
         
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        a = x
        x = self.maxpool(x)

        x = self.layer1(x)
        b=x
        x = self.layer2(x)
        c=x
        x = self.layer3(x)
        d=x
        x = self.layer4(x)
        e=x
        x = self.avgpool(x)

        y = torch.flatten(x, 1)
        z = self.fc(y)
        return z,a,b,c,d,e,x 


