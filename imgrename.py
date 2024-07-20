import os

from getreports_CXR import read_png_list

def rename(path):
    i_list = read_png_list('.\images\CXR')
    for i in i_list:
        try:
            k = i
            a = k.replace('CXR', '')
            a = a[:a.index('_')]
            print(a)
            old = path + '\\' + i
            print(old)
            new = path + '\\' + a + '.png'
            print(new)
            os.rename(old, new)
        except Exception:
            print('something has occurred')
            pass

rename('.\images\CXR')
