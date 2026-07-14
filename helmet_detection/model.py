#一个简单模型，适用yolo格式
import torch.onnx
from torch import nn
from torchvision.models import vgg16
from torchvision.transforms import transforms
from collections import OrderedDict

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extract =vgg16().features
        self.seq = nn.Sequential(
            nn.Flatten()  # 展平
            ,nn.Linear(25088,4096)
            ,nn.ReLU()
            ,nn.Linear(4096,1024)
            ,nn.ReLU()
            ,nn.Linear(1024,8)  #pred = [x, y, w, h, c1, c2, c3, c4]
    )
    def forward(self, x):
        x =self.feature_extract(x)
        return self.seq(x)

