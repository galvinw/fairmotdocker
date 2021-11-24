import logging
import os
import os.path as osp
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
from lib.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.timer import Timer

import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

import requests
import itertools
import time

def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular

    return img, ratio, dw, dh

# def eval_seq(save_dir=None, show_image=True, frame_rate=50000000):

#     tracker = JDETracker("Muna", frame_rate=frame_rate)
#     timer = Timer()
#     results = []
#     frame_id = 0

#     cap = cv2.VideoCapture('1.mp4')
#     frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
#     vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     vn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     img_size=(1088, 608)
#     width = img_size[0]
#     height = img_size[1]
#     count = 0
#     w, h = 1280, 720
#     print('Length of the video: {:d} frames'.format(vn))
#     frame_id = 0
#     min_box_area = 100

#     while cap.isOpened():
#         count = count + 1
#         res, img0 = cap.read()
#         assert img0 is not None, 'Failed to load frame {:d}'.format(count)
#         #img0 = cv2.resize(img0, (w, h))
#         img, _, _, _ = letterbox(img0, height=height, width=width)
#         img = img[:, :, ::-1].transpose(2, 0, 1)
#         img = np.ascontiguousarray(img, dtype=np.float32)
#         img /= 255.0
#         if frame_id % 20 == 0:
#             logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
#         frame_id = frame_id + 1
#         # run tracking
#         timer.tic()
#         blob = torch.from_numpy(img).cuda().unsqueeze(0)
#         online_targets = tracker.update(blob, img0)
#         online_tlwhs = []
#         online_ids = []
#         #online_scores = []
#         for t in online_targets:
#             tlwh = t.tlwh
#             tid = t.track_id
#             vertical = tlwh[2] / tlwh[3] > 1.6
#             if tlwh[2] * tlwh[3] > min_box_area and not vertical:
#                 online_tlwhs.append(tlwh)
#                 online_ids.append(tid)
#                 #online_scores.append(t.score)
#         timer.toc()
#         # save results
#         results.append((frame_id + 1, online_tlwhs, online_ids))
#         #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
#         if 1:
#             online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id,
#                                           fps=1. / timer.average_time)
#         if 1:
#             cv2.imshow('online_im', online_im)
#             cv2.waitKey(1)
#         if save_dir is not None:
#             cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
#         frame_id += 1

muna = 0
def eval_seq(IP_ADDRESS='10.0.2.104:7000', save_dir=None, show_image=True, frame_rate=30):

    camera_ip_address = []
    camera_api = "http://" + str(IP_ADDRESS)+ "/api/cameras"

    r = requests.get(camera_api)

    for i in (r.json()['data']):
        camera_ip_address.append(i['camera_ip_address'])

    print(camera_ip_address)

    tracker = JDETracker("Muna", frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    min_box_area = 100
    global muna


    for camera_addr in itertools.cycle(camera_ip_address):
        print(camera_addr)
        camera_addr = camera_addr + "&?resolution=1280"
        cam = cv2.VideoCapture(camera_addr)
        img_size=(1088, 608)
        width = img_size[0]
        height = img_size[1]
        count = 0
        w, h = 1280, 720
        #frame_id = 0    
        prev_time = time.time()
        while True:
            print(muna)
            muna += 1    
            muna = muna + 1
            cur_time = time.time()
            if (cur_time - prev_time) < 4:
                count = count + 1
                res, img0 = cam.read()
                assert img0 is not None, 'Failed to load frame {:d}'.format(count)
                img0 = cv2.resize(img0, (w, h))
                img, _, _, _ = letterbox(img0, height=height, width=width)
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img, dtype=np.float32)
                img /= 255.0
                if frame_id % 20 == 0:
                    logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
                frame_id = frame_id + 1
                # run tracking
                timer.tic()
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
                online_targets = tracker.update(blob, img0)
                online_tlwhs = []
                online_ids = []
                #online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        #online_scores.append(t.score)
                timer.toc()
                # save results
                results.append((frame_id + 1, online_tlwhs, online_ids))
                if 1:
                    online_im = vis.plot_tracking(img0,muna, online_tlwhs, online_ids, frame_id=0,
                                                fps=1. / timer.average_time)
                if 1:
                    cv2.imshow('online_im', online_im)
                    cv2.waitKey(1)

                frame_id += 1
                muna = muna + 1
            else:
                break
            
        cam.release()

eval_seq()

