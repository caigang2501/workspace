# -*- coding: utf-8 -*-
"""
  @Author: zzn 
  @Date: 2019-11-12 11:04:20 
  @Last Modified by:   zzn 
  @Last Modified time: 2019-11-12 11:04:20 
"""
import os
import sys

import torch

from predictor import Predictor


def predect(img_path):
    img_name = os.path.basename(img_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = Predictor(device=device)

    img = predictor.read_img(img_path)
    x = predictor.process_img(img)
    predictions = predictor.predict(x)
    img = predictor.display_boxes(img, predictions)
    img.save('demo/{}'.format(img_name))
    # img.show()


if __name__ == '__main__':
    # img_path = 'data_cartoon/imgs_cartoon/anime_0001.jpg'
    img_path = 'data_helmet/nohelmet_imgs/'
    for file_name in os.listdir(img_path)[:10]:
        predect(img_path+file_name)