from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq, eval_prop

import cv2
import numpy as np



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
opt = opts().init()

eval_prop(opt)
