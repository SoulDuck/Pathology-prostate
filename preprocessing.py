#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from DataProvider import tf_padder

# 650, 650 이하의 이미지에 padding 을 붙인다.

def get_undersize(paths , height , width):
    ret_list = []
    for p in paths:
        h, w, ch = np.shape(np.asarray(Image.open(p)))
        if h < height and w < width:
            ret_list.append(p)
    return ret_list


if __name__ == '__main__':
    paths = glob.glob('/Users/seongjungkim/PycharmProjects/Pathology-prostate/Train_fg/*.png')
    target_paths = get_undersize(paths , 650 , 650)
    tf_padder(target_paths ,  650 , 650 , '/Users/seongjungkim/PycharmProjects/Pathology-prostate/patched_train_fg') # 작동확인





