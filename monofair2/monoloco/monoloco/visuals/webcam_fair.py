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
from ..utils.iou import calculate_iou

import numpy as np
from fairmot.src.track import eval_prop as fairmot
from fairmot.src.lib.opts import options
from fairmot.src.lib.tracker.multitracker import JDETracker
from fairmot.src.lib.tracking_utils import visualization as vis
from fairmot.src.lib.tracking_utils.timer import Timer

from monoloco.monoloco.utils.rest import create_person_instances, get_cameras

LOG = logging.getLogger(__name__)

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

def merge_fairmot_and_monoloco_data(online_ids, online_tlwhs, dic_out, track_status_obj, acceptable_iou=0.30):
    ''' A function that combines FairMOT RE-ID data and Monoloco xyz values (distance from camera to person) '''

    monofair_dic_out = {
        "total_person": 0,
        "active_person_ids": [],
        "bboxes_tlwh": [],
        "bboxes_tlbr": [],
        "ious": [],
        "status": [],
        "xyz_preds": [],
    }
    
    for fair_id, fair_tlwh in enumerate(online_tlwhs):
        fair_tlbr = tlwh_to_tlbr(fair_tlwh)
        ious = []
        for mono_id, mono_tlbr in enumerate(dic_out["boxes"]):
            # Future optimization: If x coordinate of mono_tlbr does not lie within x coordinate of fair_tlbr, skip straight away
            iou = calculate_iou(fair_tlbr, mono_tlbr[0:4])
            ious.append(iou)

        if not ious:
            highest_iou = 0
        else:
            highest_iou = max(ious)

        monofair_dic_out["total_person"] += 1
        track_id = online_ids[fair_id]
        monofair_dic_out["active_person_ids"].append(track_id)
        
        if highest_iou > acceptable_iou:
            idx = ious.index(highest_iou)

            x1 = min(fair_tlbr[0], dic_out["boxes"][idx][0])
            y1 = min(fair_tlbr[1], dic_out["boxes"][idx][1])
            x2 = max(fair_tlbr[2], dic_out["boxes"][idx][2])
            y2 = max(fair_tlbr[3], dic_out["boxes"][idx][3])
            monofair_tlbr = [x1, y1, x2, y2]

            monofair_dic_out["bboxes_tlwh"].append(tlbr_to_tlwh(monofair_tlbr))
            monofair_dic_out["bboxes_tlbr"].append(monofair_tlbr)
            monofair_dic_out["ious"].append(highest_iou)
            monofair_dic_out["xyz_preds"].append(dic_out["xyz_pred"][idx])
        else:
            monofair_dic_out["bboxes_tlwh"].append(fair_tlwh)
            monofair_dic_out["bboxes_tlbr"].append(fair_tlbr)
            # Negative iou and xyz_pred indicate that FairMOT is able to track the person but monoloco can't
            monofair_dic_out["ious"].append(-1)
            monofair_dic_out["xyz_preds"].append([-100,-100,-100])

        if track_id in track_status_obj["Activated"]:
            monofair_dic_out["status"].append("Activated")
        elif track_id in track_status_obj["Refind"]:
            monofair_dic_out["status"].append("Refind")
        # elif track_id in track_status_obj["Lost"]:
        #     monofair_dic_out["status"].append("Lost")
        # elif track_id in track_status_obj["Removed"]:
        #     monofair_dic_out["status"].append("Removed")
        # NOTE: There will be no person_instance with status "Lost" and "Removed" since both IDs are not included in JDE Tracker.update return value
        # If Lost and removed needed to be included in the future, need to append the track_id to active_person_ids

    return monofair_dic_out


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

    camera_list = get_cameras()

    # for openpifpaf predictions
    predictor = openpifpaf.Predictor(checkpoint=args.checkpoint)
    
    frame_id = 0
    skipped_frame_id = 0
    loop_id = 0
    for camera in itertools.cycle(camera_list):

        try:
            print(f"Reading: {camera['connection_string']}")
            cam = cv2.VideoCapture(camera['connection_string'])

            # visualizer_mono = None
            
            # fairmot_results = []

            timer = Timer()
            # start_video_time = time.time()

            while True:
                start = time.time()

                ret, frame = cam.read()
                
                image = cv2.resize(frame, (1920, 1080))
                # scale = (args.long_edge)/frame.shape[0]
                # image = cv2.resize(frame, None, fx=scale, fy=scale)
            
                # Only run every nth frame
                # frames_to_skip = 5
                # if skipped_frame_id % frames_to_skip != 0:
                #     skipped_frame_id += 1
                #     continue

                # Skip n frames at the beginning
				# if frame_id < 470:
				#     print(f'Skipping frame {frame_id}')
				#     frame_id += 1
				#     continue

                ############# RE-ID (FairMOT) ############# 
                img, _, _, _ = letterbox(image, height=1088, width=608)
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img, dtype=np.float32)
                img /= 255.0

                # print(f"opt device is {opt.device}")

                if opt.device == torch.device('cpu'):
                    blob = torch.from_numpy(img).unsqueeze(0)
                    # print("CPU is used")
                else:
                    blob = torch.from_numpy(img).cuda().unsqueeze(0)
                    # print("CUDA is used")

                online_targets, track_status_obj = tracker.update(blob, image)
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
                # fairmot_results.append((frame_id + 1, online_tlwhs, online_ids))

                ''' Output FairMOT analyzed photos
                online_im = vis.plot_tracking(image, online_tlwhs, online_ids, frame_id=frame_id,
                                                fps=1. / timer.average_time)
                '''

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
                    pifpaf_outs['left'], (width, height), min_conf=0.3)

                dic_out = net.forward(keypoints, kk)
                dic_out = net.post_process(dic_out, boxes, keypoints, kk)

                ############# Combine FairMOT and Monoloco based on IOU #############
                monofair_dic_out = merge_fairmot_and_monoloco_data(online_ids, online_tlwhs, dic_out, track_status_obj, acceptable_iou=0.30)
                create_person_instances(monofair_dic_out, camera_id=camera["id"], frame_id=frame_id)
                 
                # ''' Output monofair analyzed photos
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                online_im = vis.plot_tracking(image, monofair_dic_out["bboxes_tlwh"], monofair_dic_out["active_person_ids"], frame_id=frame_id,
                                fps=1. / timer.average_time)
                # '''
                # print(f'monofair_dic_out : {monofair_dic_out}')
                

                # if 'social_distance' in args.activities:
                #     dic_out = net.social_distance(dic_out, args)
                # if 'raise_hand' in args.activities:
                #     dic_out = net.raising_hand(dic_out, keypoints)

                # if visualizer_mono is None:  # it is, at the beginning
                #     visualizer_mono = Visualizer(kk, args)(pil_image)  # create it with the first image
                #     visualizer_mono.send(None)

                # print(f"\n============== dic_out : {dic_out['xyz_pred']}\n")

                # print(f"Identified {len(dic_out['xyz_pred'])} people")
                # print(f"xyz_pred: {dic_out['xyz_pred']}")

                    
                # ''' Output analyzed photos
                try:
                    path = 'output'
                    cv2.imwrite(os.path.join(path ,f'output_{frame_id}_{loop_id}.jpg'), online_im)
                except:
                    print(f"Unable to write output for 'output_{frame_id}.jpg'")

                try:
                    # online_im = resize_with_aspect_ratio(online_im, width=1000)
                    cv2.imshow('online_im', online_im)
                    cv2.waitKey(1)
                except:
                    print(f"imshow could not work")

                # cv2.imwrite(f'monoloco{frame_id}.jpg', image)
                # '''


                LOG.debug(dic_out)
                frame_id += 1
                skipped_frame_id += 1

                # visualizer_mono.send((pil_image, dic_out, pifpaf_outs))

                end = time.time()
                LOG.info("run-time: {:.2f} ms".format((end-start)*1000))

            # cam.release()

            # cv2.destroyAllWindows()
        except Exception as e:
            print("Re-reading camera feed...")
            print(e)
            loop_id += 1

            # Video mode
            # end_video_time = time.time()
            # LOG.info("Total video run-time: {:.2f} s".format(end_video_time-start_video_time))
            # break

            continue

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]

	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))

	resized_image = cv2.resize(image, dim, interpolation=inter)

	return resized_image

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