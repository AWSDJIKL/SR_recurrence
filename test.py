# -*- coding: utf-8 -*-
'''

'''
import math
import re
import ssl
import PIL.Image
# @Time    : 2021/12/23 10:42
# @Author  : LINYANZHEN
# @File    : test.py


import gdown
import wget

if __name__ == '__main__':
    link = "https://drive.google.com/file/d/1vVH7AsDMsLcd4benOo0SifFOc_u3qqFM/view?usp=drive_link"
    l = "https://drive.usercontent.google.com/download?id=1vVH7AsDMsLcd4benOo0SifFOc_u3qqFM&authuser=0&confirm=t&uuid=19d41d78-86f3-420d-9069-f8cae7c39d6d&at=ALoNOgnLv1nk2x4vJuNjQfu5K_sT%3A1747058304550"
    # l="https://drive.usercontent.google.com/download?id=1vVH7AsDMsLcd4benOo0SifFOc_u3qqFM&export=download&authuser=0&confirm=t&uuid=a49ccbeb-e6f8-48a7-898b-5ad1e0f689ba&at=ALoNOgn2OyruE8DtEi5FhT50KoSH%3A1747058606298"
    ssl._create_default_https_context = ssl._create_unverified_context
    file_name = gdown.download(link, "dataset/Set14.zip",
                               quiet=False, fuzzy=True)
    # file_name = wget.download(link, "dataset")
    print(file_name)
    pass
