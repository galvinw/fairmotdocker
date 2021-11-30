from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from trackx import eval_seq

import cv2
import numpy as np
import requests

import torch
import os
from opts import opts

IP_ADDRESS = '10.0.2.104:7000'
LOGIN_API = 'http://10.0.2.104:7000/auth/login'

try:
    PARAMS = {'password': int(1234), 'username': 'USER'}
    r = requests.post(url = LOGIN_API, data = PARAMS)
    print(r)

except Exception as e:
    print(e)


def tuna(opt):
    eval_seq(opt,IP_ADDRESS)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    tuna(opt)

