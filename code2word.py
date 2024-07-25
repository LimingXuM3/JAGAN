import json

# 定义文件路径
path1 = './stats/LGK/gen/mle/result/'

# 打开文件,r是读取,encoding是指定编码格式
with open(path1 + 'result_val_7.json', 'r', encoding='utf-8') as fp:
    #print(type(fp))  # 输出结果是 <class '_io.TextIOWrapper'> 一个文件类对象

    # load()函数将fp(一个支持.read()的文件类对象，包含一个JSON文档)反序列化为一个Python对象
    data = json.load(fp)

    #print(type(data))  # 输出结果是 <class 'dict'> 一个python对象,json模块会根据文件类对象自动转为最符合的数据类型,所以这里是dict

print(
    "\n************************************************************************************************************\n")
print(data)
print(
    "\n************************************************************************************************************\n")

fp.close()
