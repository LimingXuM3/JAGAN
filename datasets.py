import json
import torch
from torch.utils.data import Dataset
import numpy as np
from lib.utils import pil_loader

class ImageCaptionDataset(Dataset):

    def __init__(self,
                 dataset,
                 model,
                 split_type,
                 use_img_feats,
                 transform,
                 img_src_path,
                 processed_data_path,
                 cnn_architecture):

        super(ImageCaptionDataset, self).__init__() #调用父类方法
        self.split_type = split_type
        self.use_img_feats = use_img_feats
        self.transform = transform
        self.img_src_path = img_src_path
        self.processed_data_path = processed_data_path
        self.dataset = dataset
        self.cnn_architecture = cnn_architecture
        self.model = model
        
        with open(processed_data_path + '/' + dataset + '/' + split_type + '/img_names.json') as f:
            self.img_names = json.load(f)
        
        with open(processed_data_path + '/' + dataset + '/' + split_type + '/captions.json') as f:
            self.caps = json.load(f)

        with open(processed_data_path + '/' + dataset + '/' + split_type + '/captions_len.json') as f:
            self.cap_lens = json.load(f)

        if split_type == 'val':
            with open(processed_data_path + '/' + dataset + '/' + split_type + '/caps_per_img.json') as f:
                self.caps_per_img = json.load(f)

    def __getitem__(self, index):
        img_name = self.img_names[index]

        if self.use_img_feats:
            img_feats = np.load(self.processed_data_path + '/' +
                                self.dataset + '/' + self.split_type +
                                '/image_features/' + self.cnn_architecture + '/' +
                                img_name.split('.')[0] + '.npy')
            #torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
            img_feats = torch.from_numpy(img_feats)
            cap = torch.LongTensor(self.caps[index])
            cap_len = torch.LongTensor([self.cap_lens[index]])

            if self.split_type == 'train':
                if self.model == 'discriminator':#错误匹配
                    mismatched_index = np.random.randint(len(self.img_names))#产生离散均匀分布的整数
                    mismatched_img_name = self.img_names[mismatched_index]
                    mismatched_img_feats = np.load(self.processed_data_path + '/' +
                                                   self.dataset + '/' + self.split_type +
                                                   '/image_features/' + self.cnn_architecture + '/' +
                                                   mismatched_img_name.split('.')[0] + '.npy')
                    mismatched_img_feats = torch.from_numpy(mismatched_img_feats)#改成张量
                    return img_feats, mismatched_img_feats, cap, cap_len
                else:
                    return img_feats, cap, cap_len
            else:
                all_caps = self.caps_per_img[img_name]
                return img_feats, cap, cap_len, torch.LongTensor(all_caps)
        else:
            img = pil_loader(self.img_src_path + '/' + self.dataset + '/' + img_name)
            img = self.transform(img)  # 通过找中心和缩放等实现标准化
            
            img = torch.FloatTensor(img)
            cap = torch.LongTensor(self.caps[index])
            cap_len = torch.LongTensor([self.cap_lens[index]])

            if self.split_type == 'train':
                if self.model == 'discriminator':
                    mismatched_index = np.random.randint(len(self.img_names))
                    mismatched_img_name = self.img_names[mismatched_index]
                    mismatched_img = pil_loader(self.img_src_path + '/' + self.dataset + '/' + mismatched_img_name)
                    mismatched_img = self.transform(mismatched_img)
                    mismatched_img = torch.FloatTensor(mismatched_img)
                    return img, mismatched_img, cap, cap_len
                else:
                    return img, cap, cap_len
            else:
                all_caps = self.caps_per_img[img_name]
                return img, cap, cap_len, torch.LongTensor(all_caps)
    
    def __len__(self): #字幕长度
        return len(self.caps)
    

class ImageDataset(Dataset):
    def __init__(self, split_type, dataset, transform, img_src_path, processed_data_path):
        super(ImageDataset, self).__init__()#对继承自父类的属性进行初始化,而且是用父类的初始化方法来初始化继承的属性.也就是说,子类继承了父类的所有属性和方法,
        self.split_type = split_type
        self.transform = transform
        self.processed_data_path = processed_data_path
        self.img_src_path = img_src_path
        self.dataset = dataset

        with open(processed_data_path + '/' + dataset + '/' + split_type + '/img_names.json') as f:
            self.img_names = json.load(f)

        self.img_names = sorted(set(self.img_names))#因为一张图片可能对应不同的说明性文字，所以图片数据集中存在重复的问题。set函数去重

        
    def __getitem__(self, index):
        img_name = self.img_names[index]
        
        img = pil_loader(self.img_src_path + '/' + self.dataset + '/' + img_name)
        img = self.transform(img)
        img = torch.FloatTensor(img)
        
        return img, img_name
    
    def __len__(self):
        return len(self.img_names)



