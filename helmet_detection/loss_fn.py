# 一个简单的loss_fn，对应yolo格式
from torch import nn
class Loss_fn(nn.Module):
    def __init__(self):
        super().__init__()
        self.location_loss = nn.MSELoss(reduction='mean')
        self.class_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self,pred,target):
        pred_location = pred[:,0:4]
        pred_class = pred[:,4:8]
        target_location = target[:,0:4]
        target_class = target[:,4:8]
        location_loss_value = self.location_loss(pred_location,target_location)
        # CrossEntropyLoss 需要类别索引 (N,)，不能直接用 one-hot (N, 4)
        target_class_idx = target_class.argmax(dim=1)
        class_loss_value = self.class_loss(pred_class,target_class_idx)
        return location_loss_value + class_loss_value

