from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
import os.path as osp
import cv2
# import logging
import motmetrics as mm
import numpy as np
import torch
import itertools
from openpifpaf import Predictor

import time
from datetime import datetime
import base64
import requests
from src.lib.tracker.multitracker import JDETracker
from src.lib.tracking_utils import visualization as vis
from src.lib.tracking_utils.log import logger
from src.lib.tracking_utils.timer import Timer
from src.lib.tracking_utils.evaluation import Evaluator
# import datasets.dataset.jde as datasets

from src.lib.tracking_utils.utils import mkdir_if_missing
from src.lib.opts import options





def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
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

def eval_prop():
    opt = options().init()
    opt.task = 'mot'
    f = open("/config/cameras.txt", "r")
    camera_list = f.readlines()
    f.close()
    tracker = JDETracker(opt, frame_rate=30)
    predictor_pifpaf =  Predictor(checkpoint='shufflenetv2k30')



    for element in itertools.cycle(camera_list):
        print(element)
        element = element.split(",")
        cameraName = element[0]
        cameraIP = element[1]
        threshold = element[2]
        lat = element[3]
        longi = element[4]
        camera_shift_time = int(element[6])
        prev_time = time.time()

        if ".mp4" in cameraIP:
            print(f"Using demo {cameraIP} files...")
            cap = cv2.VideoCapture(cameraIP)
        else:
            cap = cv2.VideoCapture(cameraIP)

        timer = Timer()
        results = []
        frame_id = 0
        while True:
            if time.time() - prev_time > camera_shift_time:
                prev_time = time.time()
                break
            res, img0 = cap.read()  # BGR
            # assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
            img0 = cv2.resize(img0, (1920, 1080))
            img, _, _, _ = letterbox(img0, height=1088, width =608)
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0

            predictions, gt_anns, meta = predictor_pifpaf.numpy_image(img0)
        
            timer.tic()
            if opt.device == torch.device('cpu'):
                blob = torch.from_numpy(img).unsqueeze(0)
            else:
                blob = torch.from_numpy(img).cuda().unsqueeze(0)

            online_targets = tracker.update(blob, img0)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            timer.toc()
            results.append((frame_id + 1, online_tlwhs, online_ids))
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                            fps=1. / timer.average_time)
            # cv2.imshow('online_im', online_im)
            # cv2.waitKey(1)

            cv2.imwrite(f'online_im{frame_id}.jpg', online_im)

            frame_id += 1

            my_date = datetime.now()

            # print(len(predictions))
            # url = 'http://web:8000/add_zone_status/'
            
            # zone_status_obj = {
            #             "id": 0,
            #             "create_at": "2021-11-02T16:08:46.505Z",
            #             "zone_id": 1,
            #             "number": len(predictions)
            # }
                       
            # x = requests.post(url,json=zone_status_obj,headers={"content-type":"application/json","accept":"application/json"})

            # Zone_Status.objects.get_or_create(zone_id=1,number=int(len(predictions)))
            # if int(threshold) < len(predictions):
            #     threshold = threshold + 1
            #     element[2] = str(threshold)
            #     url = 'http://145.12.244.4:4011/api/va/crowdcount'
            #     if time.time() -  timeofcrowdcount > 60:
            #         print("Sending Crowd Alert")
            #         x = requests.post(url,json=crowdCount_obj,headers={"content-type":"application/json","x-api-key":"6cf06df6bb9d4bee82c6a965c166c973"})
            #         timeofcrowdcount = time.time()



            for i, entity_id in enumerate(online_ids):
                imx = np.ascontiguousarray(np.copy(img0))
                im_h, im_w = imx.shape[:2]

                x1, y1, w, h = online_tlwhs[i]
                x1,y1,w,h = int(x1), int(y1), int(w), int(h)
                person_interest = imx[y1:y1+h, x1:x1+w]

                retval, buffer = cv2.imencode('.jpg', person_interest)
                jpg_as_text = base64.b64encode(buffer)

                jpg_as_text = base64.b64encode(buffer)
                    
                img_bytes = len(jpg_as_text) * 3/4 

                tracking_obj = {
                        "camName": cameraName,
                        "alertTime": my_date.isoformat(),
                        "subjectId": entity_id,
                        "location": {
                            "latInDegrees": lat,
                            "lonInDegrees": longi
                        },
                        "imagePayload":{
                            "fileName": str(frame_id) + ".jpg",
                            "data": jpg_as_text,
                            "mimeType": "image/jpeg",
                            "size": img_bytes,
                            "confidence": "1"
                        },
                        "createInfo":{
                            "dateTime": my_date.isoformat(),
                            "sourceSystemId": "ARMY",
                            "action": "CREATE",
                            "userId": "VA System",
                            "username": "VA System",
                            "agency": "OTHERS"
                        },
                        "updateInfo":{
                            "dateTime": my_date.isoformat(),
                            "sourceSystemId": "ARMY",
                            "action": "CREATE",
                            "userId": "VA System",
                            "username": "VA System",
                            "agency": "OTHERS"
                        }
                    }

eval_prop()




            


            


            



            



