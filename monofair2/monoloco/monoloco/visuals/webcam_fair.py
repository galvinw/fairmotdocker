# pylint: disable=W0212
"""
Webcam demo application

Implementation adapted from https://github.com/vita-epfl/openpifpaf/blob/master/openpifpaf/webcam.py

"""

import time
import logging
import os

import torch
import matplotlib.pyplot as plt
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None
import requests
import itertools

import openpifpaf
from openpifpaf import decoder, network, visualizer, show, logger
from openpifpaf import datasets

from ..visuals import Printer
from ..network import Loco, preprocess_pifpaf, load_calibration
from ..predict import download_checkpoints

import numpy as np
from fairmot.src.track import eval_prop as fairmot
from fairmot.src.lib.opts import options
from fairmot.src.lib.tracker.multitracker import JDETracker
from fairmot.src.lib.tracking_utils import visualization as vis
from fairmot.src.lib.tracking_utils.timer import Timer

LOG = logging.getLogger(__name__)
BASE_URL = 'http://web:8000'

def factory_from_args(args):
    # Model
    dic_models = download_checkpoints(args)
    args.checkpoint = dic_models['keypoints']

    logger.configure(args, LOG)  # logger first

    assert len(args.output_types) == 1 and 'json' not in args.output_types

    # Devices
    args.device = torch.device('cpu')
    args.pin_memory = False
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("CUDA is activated")
        args.pin_memory = True
    else:
        print("CUDA is not detected. CPU is utilized.")
    LOG.debug('neural network device: %s', args.device)

    # Add visualization defaults
    if not args.output_types:
        args.output_types = ['multi']

    args.figure_width = 10
    args.dpi_factor = 1.0

    args.z_max = 10
    args.show_all = True
    args.no_save = True
    args.batch_size = 1

    if args.long_edge is None:
        # args.long_edge = 144
        args.long_edge = 864
    # Make default pifpaf argument
    args.force_complete_pose = True
    LOG.info("Force complete pose is active")

    # Configure
    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args, dic_models

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

def read_camera_config(camera):
    print(camera)
    camera = camera.strip().split(",")

    if len(camera) < 7: return False

    cam_info = {
        'cameraName': camera[0],
        'cameraIP': camera[1],
        'threshold': camera[2],
        'lat': camera[3],
        'longi': camera[4],
        'camera_shift_time': int(camera[6])
    }
    return cam_info

def tlwh_to_tlbr(tlwh):
    ''' 
    A function to convert a bounding box coordinates with tlwh (FairMOT) format to tlbr (monoloco)
    tlwh = top left width height, tlbr = top left bottom right
    '''
    tl_x = tlwh[0]              # x coordinate of top left of bounding box
    tl_y = tlwh[1]              # y coordinate of top left of bounding box
    width = tlwh[2]
    height = tlwh[3]

    tlbr = [0,0,0,0]
    tlbr[0] = tl_x              # x coordinate of top left of bounding box
    tlbr[1] = tl_y              # y coordinate of top left of bounding box
    tlbr[2] = tl_x + width      # x coordinate of bottom right of bounding box
    tlbr[3] = tl_y + height     # y coordinate of bottom right of bounding box

    return tlbr

def tlbr_to_tlwh(tlbr):
    ''' 
    A function to convert a bounding box coordinates with tlbr (monoloco) format to tlwh (FairMOT)
    tlwh = top left width height, tlbr = top left bottom right
    '''
    tl_x = tlbr[0]              # x coordinate of top left of bounding box
    tl_y = tlbr[1]              # y coordinate of top left of bounding box
    br_x = tlbr[2]
    br_y = tlbr[3]

    tlwh = [0,0,0,0]
    tlwh[0] = tl_x              # x coordinate of top left of bounding box
    tlwh[1] = tl_y              # y coordinate of top left of bounding box
    tlwh[2] = br_x - tl_x       # x coordinate of bottom right of bounding box
    tlwh[3] = br_y - tl_y       # y coordinate of bottom right of bounding box

    return tlwh

def webcam(args):
    assert args.mode in 'mono'
    assert cv2
    
    # ---- FairMOT init
    opt = options().init()
    tracker = JDETracker(opt, frame_rate=30)
    # ---- Monoloco init
    args, dic_models = factory_from_args(args)
    # Load Models
    net = Loco(model=dic_models[args.mode], mode=args.mode, device=args.device,
               n_dropout=args.n_dropout, p_dropout=args.dropout)

    f = open("/config/cameras.txt", "r")
    camera_list = f.readlines()
    f.close()

    # for openpifpaf predictions
    predictor = openpifpaf.Predictor(checkpoint=args.checkpoint)

    for camera in itertools.cycle(camera_list):
        camera = read_camera_config(camera)
        if not camera: continue

        try:
            print(f"Reading: {camera['cameraIP']}")
            cam = cv2.VideoCapture(camera['cameraIP'])

            visualizer_mono = None
            
            fairmot_results = []
            frame_id = 0

            timer = Timer()

            while True:
                start = time.time()

                ret, frame = cam.read()
                image = cv2.resize(frame, (1920, 1080))
                # scale = (args.long_edge)/frame.shape[0]
                # image = cv2.resize(frame, None, fx=scale, fy=scale)
                # skip_frame = 5
                # if frame_id % skip_frame != 0:
                #     frame_id += 1
                #     continue

                ############# RE-ID (FairMOT) ############# 
                img, _, _, _ = letterbox(image, height=1088, width=608)
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img, dtype=np.float32)
                img /= 255.0

                print(f"opt device is {opt.device}")

                if opt.device == torch.device('cpu'):
                    blob = torch.from_numpy(img).unsqueeze(0)
                    print("CPU is used")
                else:
                    blob = torch.from_numpy(img).cuda().unsqueeze(0)
                    print("CUDA is used")

                online_targets = tracker.update(blob, image)
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
                fairmot_results.append((frame_id + 1, online_tlwhs, online_ids))
                # ''' Output analyzed photos
                online_im = vis.plot_tracking(image, online_tlwhs, online_ids, frame_id=frame_id,
                                                fps=1. / timer.average_time)
                # '''

                ############# Convert Front View -> Bird View (Monoloco) ############# 
                height, width, _ = image.shape
                LOG.debug('resized image size: {}'.format(image.shape))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image)

                data = datasets.PilImageList(
                    [pil_image], preprocess=predictor.preprocess)

                data_loader = torch.utils.data.DataLoader(
                    data, batch_size=1, shuffle=False,
                    pin_memory=False, collate_fn=datasets.collate_images_anns_meta)

                for (_, _, _) in data_loader:

                    for idx, (preds, _, _) in enumerate(predictor.dataset(data)):

                        if idx == 0:
                            pifpaf_outs = {
                                'pred': preds,
                                'left': [ann.json_data() for ann in preds],
                                'image': image}

                # if not ret:
                #     break
                # key = cv2.waitKey(1)
                # if key % 256 == 27:
                #     # ESC pressed
                #     LOG.info("Escape hit, closing...")
                #     break

                kk = load_calibration(args.calibration, pil_image.size, focal_length=args.focal_length)
                boxes, keypoints = preprocess_pifpaf(
                    pifpaf_outs['left'], (width, height), min_conf=0.1)

                    pifpaf_outs['left'], (width, height), min_conf=0.3)

                dic_out = net.forward(keypoints, kk)
                dic_out = net.post_process(dic_out, boxes, keypoints, kk)

                camera_to_person_xyz = dic_out['xyz_pred']
                
                post_data(online_ids, camera_to_person_xyz, frame_id)

                # if 'social_distance' in args.activities:
                #     dic_out = net.social_distance(dic_out, args)
                # if 'raise_hand' in args.activities:
                #     dic_out = net.raising_hand(dic_out, keypoints)

                # if visualizer_mono is None:  # it is, at the beginning
                #     visualizer_mono = Visualizer(kk, args)(pil_image)  # create it with the first image
                #     visualizer_mono.send(None)

                # print(f"\n============== dic_out : {dic_out['xyz_pred']}\n")

                # print(f"Identified {len(dic_out['xyz_pred'])} people")
                print(f"xyz_pred: {dic_out['xyz_pred']}")

                    
                # ''' Output analyzed photos
                try:
                    path = 'output'
                    cv2.imwrite(os.path.join(path ,f'output_{frame_id}.jpg'), online_im)
                except:
                    print(f"Unable to write output for 'output_{frame_id}.jpg'")

                try:
                    cv2.imshow('online_im', online_im)
                    cv2.waitKey(1)
                except:
                    print(f"imshow could not work")

                # '''

                # cv2.imwrite(f'monoloco{frame_id}.jpg', image)

                LOG.debug(dic_out)
                frame_id += 1

                # visualizer_mono.send((pil_image, dic_out, pifpaf_outs))

                end = time.time()
                LOG.info("run-time: {:.2f} ms".format((end-start)*1000))

            # cam.release()

            # cv2.destroyAllWindows()
        except Exception as e:
            print("Re-reading camera feed...")
            print(e)
            continue

def post_data(online_ids, camera_to_person_xyz, frame_id):
    zone_status_obj = {
                "zone_id": 1,
                "number": len(online_ids)
            }
    try:
        x = requests.post(f"{BASE_URL}/add_zone_status/",json=zone_status_obj,headers={"content-type":"application/json","accept":"application/json"})
        print(f"POST /add_zone_status successfully")
    except:
        print(f"no POST /add_zone_status")
    
    if len(online_ids) > 0 and len (camera_to_person_xyz) > 0:
        for idx, id in enumerate(online_ids):

            person_id_obj = {
                "id": id,
                "name": f"Person {id}"
            }
            try:
                a = requests.post(f"{BASE_URL}/add_person/",
                    json=person_id_obj,headers={"content-type":"application/json",
                    "accept":"application/json"})
                print(f"POST /add_person/")
            except:
                print(f"no POST /add_person/")
            
            if not camera_to_person_xyz[idx]:
                camera_to_person_xyz[idx] = [0,0,0]
            else:
                x = camera_to_person_xyz[idx][0]
                # y = camera_to_person_xyz[idx][1]
                z = camera_to_person_xyz[idx][2]

            person_instance_obj = {
                "id": id,
                "name": f"Person {id}",
                "frame_id": frame_id,
                "x": x,
                "z": z
            }
        
            try:
                b = requests.post(f"{BASE_URL}/add_person_instance/",
                    json=person_instance_obj,headers={"content-type":"application/json",
                    "accept":"application/json"})
                print(f"POST /add_person_instance/")
            except:
                print(f"no POST /add_person_instance/")

class Visualizer:
    def __init__(self, kk, args):
        self.kk = kk
        self.args = args

    def __call__(self, first_image, fig_width=1.0, **kwargs):
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (fig_width, fig_width *
                                 first_image.size[0] / first_image.size[1])

        printer = Printer(first_image, output_path="",
                          kk=self.kk, args=self.args)

        figures, axes = printer.factory_axes(None)

        for fig in figures:
            fig.show()

        while True:
            image, dic_out, pifpaf_outs = yield

            # Clears previous annotations between frames
            axes[0].patches = []
            axes[0].lines = []
            axes[0].texts = []
            if len(axes) > 1:
                axes[1].patches = []
                axes[1].lines = [axes[1].lines[0], axes[1].lines[1]]
                axes[1].texts = []

            if dic_out and dic_out['dds_pred']:
                printer._process_results(dic_out)
                printer.draw(figures, axes, image, dic_out, pifpaf_outs['left'])
                mypause(0.01)


def mypause(interval):
    manager = plt._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)
