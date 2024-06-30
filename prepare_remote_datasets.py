# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2024/6/27 16:07
# @Author  : LINYANZHEN
# @File    : prepare_remote_datasets.py
import gzip
import os
import shutil
import ssl
import tarfile
import zipfile
import sys
import numpy as np
import wget
from PIL import Image
import imageio.v2
import tqdm

link_list = {
    # 遥感图像数据集
    # 多图像超分
    # "PROBA-V":"https://zenodo.org/records/6327426/files/probav_data.zip?download=1",
    # 单图像超分
    # "UCMerced":"http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip",
    # "WHURS19": "https://github.com/CAPTAIN-WHU/BED4RS/raw/main/datasets/WHU-RS19.zip",
    "RSSCN7": "https://github.com/palewithout/RSSCN7/archive/refs/heads/master.zip",
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


def unit_convert(size):
    unit_list = ["B", "KB", "MB", "GB"]
    unit_index = 0
    while size > 1024:
        unit_index += 1
        if unit_index > len(unit_list) - 1:
            unit_index = len(unit_list) - 1
            break
        size = size / 1024
    return "{:.3f}{}".format(size, unit_list[unit_index])


def bar_progress(current, total, width=80):
    # progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    progress_message = "Downloading: %d%% [%s / %s] " % (
        current / total * 100, unit_convert(current), unit_convert(total))
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


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
        file_name = wget.download(link, output_path, bar=bar_progress)
        # file_name = wget.download(link, output_path)
        print(file_name)
        uncompress(file_name)


def get_hr_list(dataset):
    hr_list = []
    if dataset in ["AID"]:
        for root, dirs, files in os.walk("dataset/{}".format(dataset, dataset)):
            for file in files:
                if "jpg" in file:
                    hr_list.append(os.path.join(root, file))

    elif dataset in ["RSSCN7"]:
        for root, dirs, files in os.walk("dataset/RSSCN7/RSSCN7-master".format(dataset, dataset)):
            for file in files:
                if "jpg" in file:
                    hr_list.append(os.path.join(root, file))
    return hr_list


def prepare_npy(hr_list, scale, save_path):
    print("开始准备{}倍超分辨率数据集".format(scale))
    lr_save_path = os.path.join(save_path, "lr")
    hr_save_path = os.path.join(save_path, "hr")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    os.mkdir(lr_save_path)
    os.mkdir(hr_save_path)
    i = 0
    process = tqdm.tqdm(hr_list)
    for hr in process:
        hr = Image.open(hr)
        hr_width = hr.width
        hr_height = hr.height
        # 计算出lr与hr的大小
        lr_width = hr_width // scale
        lr_height = hr_height // scale
        hr_width = lr_width * scale
        hr_height = lr_height * scale
        # 对图片进行裁剪并保存
        hr = hr.crop((0, 0, hr_width, hr_height))
        lr = hr.resize((lr_width, lr_height), Image.BICUBIC)
        lr.save(os.path.join(lr_save_path, "{}.png".format(i)))
        hr.save(os.path.join(hr_save_path, "{}.png".format(i)))
        # 使用imageio读取，并保存为npy文件加速后续读取
        lr = imageio.v2.imread(os.path.join(lr_save_path, "{}.png".format(i)))
        hr = imageio.v2.imread(os.path.join(hr_save_path, "{}.png".format(i)))
        np.save(os.path.join(lr_save_path, "{}.npy".format(i)), lr)
        np.save(os.path.join(hr_save_path, "{}.npy".format(i)), hr)
        i += 1


if __name__ == '__main__':
    # ssl._create_default_https_context = ssl._create_unverified_context
    # print("开始下载数据集")
    # dataset_path = "dataset"
    # download_datasets(dataset_path)
    # print("所有数据集下载完成")

    print("开始生成数据集")
    for dataset in ["RSSCN7"]:
        print(dataset)
        for scale in [4]:
            hr_list = get_hr_list(dataset)
            prepare_npy(hr_list, scale, "dataset/{}/x{}".format(dataset, scale))
    print("数据集准备完成")
