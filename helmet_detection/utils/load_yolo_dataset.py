#加载自定义yolo格式数据集
import os
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class YOLOdataset(Dataset):
    def __init__(self, img_folder, label_folder, transform, label_transform):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.img_name = os.listdir(img_folder)

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_name = self.img_name[idx]
        img_path = os.path.join(self.img_folder,img_name)
        img = Image.open(img_path).convert("RGB")
        label_name = img_name.split('.')[0]+'.txt'
        label_path = os.path.join(self.label_folder, label_name)
        with open(label_path,'r',encoding='utf-8') as f:
            label = f.read()
            label_info = label.strip().split('\n')

        # 只取第【0】物体，转为模型需要的格式 [x, y, w, h, c1, c2, c3, c4]
        info_list = label_info[0].strip().split(' ')
        class_id = int(info_list[0])
        x_center = float(info_list[1])
        y_center = float(info_list[2])
        width = float(info_list[3])
        height = float(info_list[4])

        # class_id 转 one-hot (4个类别: no helmet, motor, number, with helmet)
        one_hot = [0.0, 0.0, 0.0, 0.0]
        one_hot[class_id] = 1.0

        target = [x_center, y_center, width, height] + one_hot  # 共8个值
        target = torch.tensor(target)
        if self.transform:
            img = self.transform(img)
        return img, target

#测试
if __name__ == '__main__':

    train_set = YOLOdataset(r'../dataset/HelmetDataset-YOLO-Train/images'
                            ,r'../dataset/HelmetDataset-YOLO-Train/labels'
                            ,transforms.Compose([transforms.ToTensor()
                                                 ,transforms.Resize((224,224))]),None)
    print(train_set[0])

