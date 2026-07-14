#加载自定义voc格式数据集
import os
import torch
import xmltodict
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class VOCdataset(Dataset):
    def __init__(self, img_folder, label_folder, transform, label_transform):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.img_name = os.listdir(img_folder)
        self.class_list = ['no helmet', 'motor', 'number', 'with helmet']  #把文字类别改为数字

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_name = self.img_name[idx]
        img_path = os.path.join(self.img_folder,img_name)
        img = Image.open(img_path).convert("RGB")
        label_name = img_name.split('.')[0]+'.xml'
        label_path = os.path.join(self.label_folder, label_name)
        with open(label_path,'r',encoding='utf-8') as f:
            label = f.read()
            label_dict = xmltodict.parse(label)
            objects = label_dict['annotation']['object']
            target = []
            for object in objects:
                obj_name = object['name']
                obj_id = self.class_list.index(obj_name)
                obj_xmax = float(object['bndbox']['xmax'])
                obj_xmin = float(object['bndbox']['xmin'])
                obj_ymax = float(object['bndbox']['ymax'])
                obj_ymin = float(object['bndbox']['ymin'])
                target.extend([obj_id, obj_xmax, obj_ymax, obj_xmin, obj_ymin])


        target = torch.tensor(target)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

#测试
if __name__ == '__main__':
    train_set = VOCdataset(r'../dataset/HelmetDataset-VOC/train/images'
                            ,r'../dataset/HelmetDataset-VOC/train/labels'
                            ,transforms.Compose([transforms.ToTensor()]),None)
    print(len(train_set))
    print(train_set[0])
