"""AlexNet architecture pytorch model."""

from torch import nn
import sys
sys.path.append('/home/tiany/label-refinery')
from models import blocks


class ExpressionNet(nn.Module):
    """This is the original AlexNet architecture, and not the version introduced
    in the "one weird trick" paper."""
    #LR_REGIME = [1, 27, 0.05, 28, 95, 0.005, 96, 105, 0.0005]
    #LR_REGIME = [1, 140, 0.1, 141, 170, 0.01, 171, 200, 0.001]
    LR_REGIME = [1, 5000, 0.05, 5001, 18000, 0.005, 18001, 20000, 0.0005]
    def __init__(self):
        super().__init__()
        self.conv1 = blocks.Conv2dBnRelu(1, 64, 3, 1, 1,
                                         pooling=nn.MaxPool2d(2))
        self.conv2 = blocks.Conv2dBnRelu(64, 96, 3, 1, 1,
                                         pooling=nn.MaxPool2d(2))
        self.conv3 = blocks.Conv2dBnRelu(96, 128, 3, 1, 1)
        self.conv4 = blocks.Conv2dBnRelu(128, 128, 3, 1, 1,
                                         pooling=nn.MaxPool2d(2))
        self.conv5 = blocks.Conv2dBnRelu(128, 256, 3, 1, 1)
        self.conv6 = blocks.Conv2dBnRelu(256, 256, 3, 1, 1)

        self.fc7 = blocks.LinearBnRelu(256 * 11 * 11, 2000)
        self.fc8 = nn.Linear(2000, 7, bias=False)

    def convolutions(self, x):
        return nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4,
                             self.conv5, self.conv6)(x)

    def fully_connecteds(self, x):
        return nn.Sequential(self.fc7, self.fc8)(x)

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connecteds(x)
        return x
