# coding=utf-8
import glob
import json
import jieba
import os
from getreports_CXR import train_val_split


def txtToJson(path):
    # 读取文件
    '''
    每张图片对应一条报告，最长报告数
    '''
    with open(path, 'r', encoding="utf-8") as file:
        result = []
        # 逐行读取
        i=0
        j=0
        maxlen=0
        for line in file:
            lst = line.strip().split()
            i+=1
            a=lst[6]
            #print(a)
            word=jieba.cut(a)
            k=[]
            for w in word:
                k.append(w)
            #print('每条报告token数：')
            #print(k,len(k))
            maxlen=max(maxlen,len(k))
            q={
                "tokens":k,
                "raw": a,
                "imgid": i,
                "sentids": j
            }
            item = {
                "sentids":[j],
                "imgid": i,
                "sentences":[q],
                "filename": lst[0]
            }
            j+=1
            name=lst[0].split('.')[0]+'.json'

            with open('.\LGK-labels'+'\\'+name, 'w+') as f:
                f.write(json.dumps(item))
        print('最大token长度：')
        print(maxlen)


def merge_LGK_json(path, path_merges):
    merges_file = os.path.join(path_merges, "dataset_LGK.json")
    result=[]
    r={}
    k=sorted(glob.glob(r".\LGK-labels\*.json"),key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
    print("总文件数：")
    print(len(k))
    for f in k:
        with open(f, "r") as infile:
            for line in infile.readlines():
                a=json.loads(line)
                result.append(a)
    r['images']=result
    r['dataset']='LGK'
    with open(merges_file, "w+") as outfile:
        json.dump(r, outfile)

if __name__ == '__main__':
    txtToJson('.\LGK.txt')
    train_val_split('.\LGK-labels')
    path_merges = ".\karpathy_splits"
    merge_LGK_json('.\LGK-labels', path_merges)














