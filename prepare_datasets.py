# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/3 11:59
# @Author  : LINYANZHEN
# @File    : prepare_datasets.py
import gzip
import os
import shutil
import tarfile
import time
import zipfile
import psutil
import sys
import h5py
import numpy as np
import wget
from PIL import Image
import imageio

link_list = {
    "set5": "https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip",
    "set14": "https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip",
    "BSD500": "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
    "Urban100": "https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip",
    "DIV2K_train_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_train_LR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    # "DIV2K_valid": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    # "ms_coco_train": "http://images.cocodataset.org/zips/train2014.zip",
    # "ms_coco_val": "http://images.cocodataset.org/zips/val2014.zip",
    # "ms_coco_test": "http://images.cocodataset.org/zips/test2014.zip",
}


def uncompress(src_file, output_dir=None):
    '''
    解压文件，默认解压到文件所在目录下

    :param src_file: 要解压的文件
    :param output_dir: 解压输出路径
    :return:
    '''

    # 获取文件后缀名，判断压缩格式
    file_name, file_format = os.path.splitext(src_file)
    # 创建解压路径
    if output_dir:
        os.mkdir(output_dir)
    else:
        file_path, _ = os.path.split(src_file)
        # output_dir = os.path.join(file_path, file_name)
        output_dir = file_path
        # os.mkdir(output_dir)
        print(output_dir)
    if file_format in ('.tgz', '.tar'):
        tar = tarfile.open(src_file)
        names = tar.getnames()
        for name in names:
            tar.extract(name, output_dir)
        tar.close()
    elif file_format == '.zip':
        zip_file = zipfile.ZipFile(src_file)
        for names in zip_file.namelist():
            zip_file.extract(names, output_dir)
        zip_file.close()
    elif file_format == '.gz':
        f_name = output_dir + '/' + os.path.basename(src_file)
        g_file = gzip.GzipFile(src_file)
        open(f_name, "w+").write(g_file.read())
        g_file.close()
    else:
        print('文件格式不支持或者不是压缩文件')
        return
    os.remove(src_file)

def download_datasets(dataset_path):
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    for name, link in link_list.items():
        print(name)
        print(link)
        output_path = os.path.join(dataset_path, name)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        file_name = wget.download(link, output_path)
        print(file_name)
        uncompress(file_name)



if __name__ == '__main__':
    print("开始下载数据集")
    dataset_path = "dataset"
    download_datasets(dataset_path)
    print("所有数据集下载完成")
