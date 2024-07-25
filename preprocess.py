import json
import pickle
from collections import Counter
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import argparse
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from lib.utils import encode_captions, encode_captions1
from datasets import ImageDataset
from models import Encoder
#ToTensor将 PIL Image 或者 ndarray 转换为 tensor，并且归一化至[0-1]
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.is_available())
#对句子词汇图像特征进行预处理，便于后续数据集再处理
def process_data(storage,
                 dataset,
                 karpathy_split_path,
                 min_freq, 
                 dest_path,
                 max_caps_per_img#每张图片最大caption数
):
    with open(storage + '/' + karpathy_split_path + '/dataset_{}.json'.format(dataset), 'r') as f:
        karpathy_split = json.load(f)
    
    caps_train = []#训练集报告
    img_names_train = []#训练集图片名

    caps_per_img_train = {}  # 每张图的caption数
    caps_per_img_val = {}  # 每张图的caption数
    caps_per_img_train_new={}
    caps_per_img_val_new={}

    caps_val = []#验证集报告
    img_names_val = []#验证集图片名
    
    word_freq = Counter()#计数器 词频

    max_len = 0
    for img in karpathy_split['images']:
        captions_count = 0
        img_name = img['filename']

        for cap in img['sentences']:
            word_freq.update(cap['tokens'])#token是已经做了分词的caption  统计词频
            if captions_count < max_caps_per_img:
                captions_count += 1
            else:
                break

            #img_name=img_name.split('.')[0]
            if img['split'] == 'train':
                caps_train.append(cap['tokens'])
                img_names_train.append(img_name)

                if img_name not in caps_per_img_train:
                    caps_per_img_train[img_name] = [cap['tokens']]
                else:
                    caps_per_img_train[img_name].append(cap['tokens'])

                if img_name not in caps_per_img_train_new:
                    caps_per_img_train_new[img_name] = [cap['tokens']]
                else:
                    caps_per_img_train_new[img_name].append(cap['tokens'])

            elif img['split'] == 'val':
                caps_val.append(cap['tokens'])
                img_names_val.append(img_name)

                if img_name not in caps_per_img_val:
                    caps_per_img_val[img_name] = [cap['tokens']]
                else:
                    caps_per_img_val[img_name].append(cap['tokens'])

                if img_name not in caps_per_img_val_new:
                    caps_per_img_val_new[img_name] = [cap['tokens']]
                else:
                    caps_per_img_val_new[img_name].append(cap['tokens'])



            max_len = max(max_len, len(cap['tokens']))
        #print(max_len)
        #print(captions_count)
        #print(img_names_train)

    # 下载字典及定义反编码函数
    words = [word for word in word_freq.keys() if word_freq[word] > min_freq]#词频#根据词频来筛掉单词
    # 构建一个word_map出来
    #word_index={}


    word_index = {word: idx + 1 for idx, word in enumerate(words)}#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    '''
    word_index['<pad>'] = 0  # 填充
    word_index['<start>'] = 0
    word_index['<end>'] = 0
    '''
    word_index['<unk>'] = 0  # 开头、结尾、未知、填充

    index_word = {v: k for k, v in word_index.items()}#本质上就是新建一个字典，与原字典相比k和v互换
    
    with open(os.path.join(storage + '/' + dest_path, dataset, 'word_index.json'), 'w') as f:
        json.dump(word_index, f)#写入json

    with open(os.path.join(storage + '/' + dest_path, dataset, 'index_word.json'), 'w') as f:
        json.dump(index_word, f)
    
    enc_caps_train, cap_lens_train = encode_captions(caps_train, word_index, max_len)
    
    enc_caps_val, cap_lens_val = encode_captions(caps_val, word_index, max_len)

    #caps_per_img_train_new=caps_per_img_train
    #caps_per_img_val_new=caps_per_img_val


    for img_name in caps_per_img_train:
        caps_per_img_train[img_name], _ = encode_captions(caps_per_img_train[img_name], word_index, max_len)

    for img_name in caps_per_img_val:
        caps_per_img_val[img_name], _ = encode_captions(caps_per_img_val[img_name], word_index, max_len)

    for img_name in caps_per_img_train_new:
        caps_per_img_train_new[img_name], _ = encode_captions1(caps_per_img_train_new[img_name], word_index, max_len)

    for img_name in caps_per_img_val_new:
        caps_per_img_val_new[img_name], _ = encode_captions1(caps_per_img_val_new[img_name], word_index, max_len)

    with open(os.path.join(storage + '/' + dest_path, dataset, 'train', 'captions.json'), 'w') as f:
        json.dump(enc_caps_train, f)#将一个Python数据结构转换为JSON
        
    with open(os.path.join(storage + '/' + dest_path, dataset, 'train', 'captions_len.json'), 'w') as f:
        json.dump(cap_lens_train, f)
        
    with open(os.path.join(storage + '/' + dest_path, dataset, 'val', 'captions.json'), 'w') as f:
        json.dump(enc_caps_val, f)

    with open(os.path.join(storage + '/' + dest_path, dataset, 'val', 'captions_len.json'), 'w') as f:
        json.dump(cap_lens_val, f)
    
    with open(os.path.join(storage + '/' + dest_path, dataset, 'train', 'img_names.json'), 'w') as f:
        json.dump(img_names_train, f)

    with open(os.path.join(storage + '/' + dest_path, dataset, 'val', 'img_names.json'), 'w') as f:
        json.dump(img_names_val, f) 

    with open(os.path.join(storage + '/' + dest_path, dataset, 'train', 'caps_per_img.json'), 'w') as f:
        json.dump(caps_per_img_train, f)

    with open(os.path.join(storage + '/' + dest_path, dataset, 'val', 'caps_per_img.json'), 'w') as f:
        json.dump(caps_per_img_val, f)

    with open(os.path.join(storage + '/' + dest_path, dataset, 'train', 'caps_per_img_new.json'), 'w') as f:
        json.dump(caps_per_img_train_new, f)

    with open(os.path.join(storage + '/' + dest_path, dataset, 'val', 'caps_per_img_new.json'), 'w') as f:
        json.dump(caps_per_img_val_new, f)

def main(args):
    #处理报告
    process_data(args.storage,
                 args.dataset,
                 args.karpathy_split_path,
                 args.min_freq,
                 args.dest_path,
                 args.max_caps_per_img)
    #提取图像特征
    if args.extract_image_features:

        encoder = Encoder(args.cnn_architecture)
        encoder.to(device)

        train_img_feats_path = args.storage + '/processed_data/' + args.dataset + '/train/feature'

        val_img_feats_path = args.storage + '/processed_data/' + args.dataset + '/val/feature'

        if len(os.listdir(train_img_feats_path)) <= 1:

            train_loader = DataLoader(
                ImageDataset(split_type='train',
                             dataset=args.dataset,
                             transform=data_transforms,
                             img_src_path=args.storage + '/' + args.image_path,
                             processed_data_path=args.storage + '/processed_data'),
                batch_size=args.batch_size, num_workers=1)

            for imgs, img_names in tqdm(train_loader):#tqdm可以在长循环中添加一个进度提示信息

                imgs = imgs.to(device)
                #print(img_names)
                img_feats = encoder(imgs)#图像特征编码，残差网络训练获得图像特征

                #print(img_names)
                for name in img_names:
                    for feats in img_feats:
                        np.savez(args.storage + '/processed_data/' + args.dataset + '/train/feature' + '/' + name.split('.')[0]+ '.npz', feat=feats.cpu().numpy())


        if len(os.listdir(val_img_feats_path)) <= 1:

            val_loader = DataLoader(
                ImageDataset(split_type='val',
                             dataset=args.dataset,
                             transform=data_transforms,
                             img_src_path=args.storage + '/' + args.image_path,
                             processed_data_path=args.storage + '/processed_data'),
                batch_size=args.batch_size, num_workers=1)

            for imgs, img_names in tqdm(val_loader):

                imgs = imgs.to(device)

                img_feats = encoder(imgs)

                for name in img_names:
                    for feats in img_feats:
                        np.savez(args.storage + '/processed_data/' + args.dataset + '/val/feature' + '/' + name.split('.')[0]+ '.npz', feat=feats.cpu().numpy())
'''
                for feats, name in zip(img_feats, img_names):
                    np.save(args.storage + '/processed_data/' + args.dataset + '/val/image_features/' +
                            args.cnn_architecture + '/' + name.split('.')[0] + '.npy', feats.cpu().numpy())
'''

if __name__ == "__main__":
    '''
    

    '''
    parser = argparse.ArgumentParser(description='Pre-process data')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--storage', type=str, default='.')
    parser.add_argument('--dataset', type=str, default='LGK')
    #parser.add_argument('--dataset', type=str, default='CXR')

    parser.add_argument('--cnn-architecture', type=str, default='resnet101')
    parser.add_argument('--karpathy-split-path', type=str, default='karpathy_splits')
    parser.add_argument('--min-freq', type=int, default=2)
    parser.add_argument('--dest-path', type=str, default='processed_data')
    parser.add_argument('--image-path', type=str, default='images')
    parser.add_argument('--extract-image-features', type=bool, default=True)
    parser.add_argument('--max_caps-per-img', type=int, default=1)
    #parser.add_argument('--max_caps-per-img', type=int, default=12)

    main(parser.parse_args())






