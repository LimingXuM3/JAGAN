import glob
import nltk
import xmltodict
import os
import json
import shutil
from nltk import sent_tokenize, re
from sklearn import model_selection


def xml_to_JSON(xml):
    # 格式转换
    try:
        convertJson = xmltodict.parse(xml, encoding = 'utf-8')
        jsonStr = json.dumps(convertJson, indent=1)
        return jsonStr
    except Exception:
        print('something has occurred')
        pass

def read_xml_list(path):
    # 获取该文件夹下所有以.xml为后缀的文件
    file_list = os.listdir(path)
    read_list = []
    for i in file_list:
        a, b = os.path.splitext(i)
        if b == '.xml':
            read_list.append(i)
        else:
            continue
    fileIndex = []

    # 以下就是排序操作
    for i in range(0, len(read_list)):
        index = read_list[i].split(".")[0]
        fileIndex.append(int(index))

    for j in range(1, len(fileIndex)):
        for k in range(0, len(fileIndex) - 1):
            if fileIndex[k] > fileIndex[k + 1]:
                preIndex = fileIndex[k]
                preFile = read_list[k]
                fileIndex[k] = fileIndex[k + 1]
                read_list[k] = read_list[k + 1]
                fileIndex[k + 1] = preIndex
                read_list[k + 1] = preFile
    return read_list

def batch_convert(path):
    # 格式转换函数
    in_list = read_xml_list(path)

    for item in in_list:
        with open(path+'\\'+item, encoding = 'utf-8') as f:
            xml = f.read()
            converted_doc = xml_to_JSON(xml)

        new_name = item.rsplit('.xml')[0] + '.json'
        with open(path+'\\'+new_name, 'w+',encoding = 'utf-8') as f:
            f.write(converted_doc)

def read_json_list(path):
    # 获取该文件夹下所有以.json为后缀的文件
    file_list = os.listdir(path)
    read_list = []
    for i in file_list:
        a, b = os.path.splitext(i)
        if b == '.json':
            read_list.append(i)
        else:
            continue
    fileIndex = []

    # 以下就是排序操作
    for i in range(0, len(read_list)):
        index = read_list[i].split(".")[0]
        fileIndex.append(int(index))

    for j in range(1, len(fileIndex)):
        for k in range(0, len(fileIndex) - 1):
            if fileIndex[k] > fileIndex[k + 1]:
                preIndex = fileIndex[k]
                preFile = read_list[k]
                fileIndex[k] = fileIndex[k + 1]
                read_list[k] = read_list[k + 1]
                fileIndex[k + 1] = preIndex
                read_list[k + 1] = preFile
    return read_list

def read_png_list(path):
    # 获取该文件夹下所有以.png为后缀的文件
    file_list = os.listdir(path)
    read_list = []
    for i in file_list:
        a, b = os.path.splitext(i)
        if b == '.png':
            read_list.append(i)
        else:
            continue

    return read_list

def xmlvspng():
    # 判定两个文件夹内的文件是否相同
    file_path1 = '.\images\CXR'  # 已知 内容较少的文件夹
    file_path2 = '.\CXR-labels'

    f1 = []
    f2 = []
    for i in os.listdir(file_path1):
        a, b = os.path.splitext(i)
        if b == '.png':
            f1.append(a)
        else:
            continue

    for i in os.listdir(file_path2):
        a, b = os.path.splitext(i)
        if b == '.xml':
            f2.append(a)
        else:
            continue

    #将两个文件夹内的文件名不同的提出来
    # 匹配xml文件，删多余141份
    c=0
    for filename2 in f2:
        if filename2 not in f1:
            c+=1
            shutil.move(file_path2 + '\\' + filename2 + '.xml',
                        "D:\学习资料包\图像高级认知\Open-i\多余xml" + '\\' + filename2 + '.xml')  # 文件夹需要创建
    print('多余xml:')
    print(c)

def deljson():
    k = sorted(glob.glob(r".\CXR-labels\labels\*.json"),
               key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
    #print(len(k))
    count=0
    for f in k:
        with open(f, "r") as infile:
            for line in infile.readlines():
                a=json.loads(line)
                if a['sentences']==[]:
                    count+=1
                    shutil.move('.\CXR-labels\labels' + '\\' + f+'.json' ,
                                "D:\学习资料包\图像高级认知\Open-i\多余json" + '\\' + f+'.json')
    print('多余json:')
    print(count)

def tiqu(path):
    in_list = read_json_list(path)
    #读取指定数据写进json
    t = 0

    max_len=0#经过检查最大句子数为12
    maxlentoken=0
    maxlen=0
    minlen=12
    for i in in_list:
        input_file=open(path+'\\'+i, 'r',encoding='utf-8')
        item = json.load(input_file)
        my_dict1 = {}
        # 获取句子
        key2 = item['eCitation']['MeSH']
        s1=key2['major']
        if type(s1)==list:
            s='. '.join(s1)
        else:
            s=s1

        # 分割句子
        s2 = s.lower().strip()
        d2 = sent_tokenize(s2)

        max_len = max(max_len, len(d2))
        minlen = min(minlen, len(d2))
        for i in range(12 - len(d2)):
            d2.append(d2[i])

        s2 = {}
        d3 = []
        d4 = []
        for j in d2:
            if j == '.':
                j = 'None.'
            k = j
            if k[-1] == '.':
                k = k[:-1]
            k1=re.findall(r'[^/\s,/]+', k)
            #print(k1)
            s2['tokens'] =k1
            #print(s2['tokens'])
            #print(k)
            maxlentoken = max(maxlentoken, len(s2['tokens']))
            s2['raw'] = k
            s2['imgid'] = int(item['eCitation']['pmcId']['@id'])
            s2['sentid'] = t
            d4.append(t)
            d3.append(s2.copy())
            t += 1

        #print('单个token最大长度：')
        #print(maxlentoken)
        my_dict1['sentids'] = d4
        # 获取图片id
        my_dict1['imgid'] = int(item['eCitation']['pmcId']['@id'])
        my_dict1['sentences'] = d3
        # 获取图片名称
        my_dict1['filename']=item['eCitation']['pmcId']['@id']+'.png'

        new_name = str(my_dict1['imgid']) + '.json'

        with open('.\CXR-labels\labels' + '\\' + new_name, 'w+', encoding='utf-8') as f:
            f.write(json.dumps(my_dict1))

    print('最大句子数')
    print(max_len)
    print('最小句子数')
    print(minlen)

#划分训练集与验证集
def train_val_split(path):
    c=[]
    in_list = read_json_list(path)
    for i in in_list:
        c.append(i)
    c_train, c_val = model_selection.train_test_split(c, test_size=0.25)

    print('训练数')
    print(len(c_train))
    print('验证数')
    print(len(c_val))
    for k in c_train:

        input_file = open(path + '\\' + k, 'r', encoding='utf-8')
        item = json.load(input_file)
        item.update({'split': 'train'})
        with open(path + '\\' + k, 'w+', encoding='utf-8') as f:
            f.write(json.dumps(item))
    for k in c_val:

        input_file = open(path + '\\' + k, 'r', encoding='utf-8')
        item = json.load(input_file)
        item.update({'split': 'val'})
        with open(path + '\\' + k, 'w+', encoding='utf-8') as f:
            f.write(json.dumps(item))

#多个json合并
def merge_json(path,path_merges):
    merges_file = os.path.join(path_merges, "dataset_CXR.json")
    result=[]
    r={}

    k=sorted(glob.glob(r".\CXR-labels\labels\*.json"),key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
    print(len(k))

    count=0
    for f in k:
        with open(f, "r") as infile:
            for line in infile.readlines():
                a=json.loads(line)
                if a['sentences']==[]:
                    print('筛掉的空串：'+  str(a['imgid']))
                    count+=1
                else:
                    result.append(a)
    print('筛掉的空串：')
    print(count)
    r['images']=result
    r['dataset']='CXR'
    with open(merges_file, "w+") as outfile:
        json.dump(r, outfile)

if __name__ == '__main__':
    #nltk.download('punkt')
    xmlvspng()
    batch_convert('.\CXR-labels')
    tiqu('.\CXR-labels')
    deljson()
    train_val_split('.\CXR-labels\labels')
    path_merges = ".\karpathy_splits"
    merge_json('.\CXR-labels\labels',path_merges)






