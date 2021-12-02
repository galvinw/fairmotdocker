# pylint: disable=W0212
"""
Webcam demo application

Implementation adapted from https://github.com/vita-epfl/openpifpaf/blob/master/openpifpaf/webcam.py

"""

import time
import logging

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
        args.pin_memory = True
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

def webcam(args):

    assert args.mode in 'mono'
    assert cv2

    args, dic_models = factory_from_args(args)

    # Load Models
    net = Loco(model=dic_models[args.mode], mode=args.mode, device=args.device,
               n_dropout=args.n_dropout, p_dropout=args.dropout)

    f = open("/config/cameras.txt", "r")
    camera_list = f.readlines()

    f.close()

    # for openpifpaf predicitons
    predictor = openpifpaf.Predictor(checkpoint=args.checkpoint)

    # flag = 0
    for element in itertools.cycle(camera_list):
        print(element)
        element = element.strip().split(",")
        if len(element) < 7:
            print(f"Invalid line at cameras.txt : {element}")
            continue
        cameraName = element[0]
        cameraIP = element[1]
        threshold = element[2]
        lat = element[3]
        longi = element[4]
        camera_shift_time = int(element[6])
        prev_time = time.time()
        
        try:
            print(f"Reading: {cameraIP}")
            cam = cv2.VideoCapture(cameraIP)
            # cam = cv2.VideoCapture(args.camera)
        except Exception as e:
            print(f"Unable to read camera")
            continue

        if (cam.isOpened() == False):
            print("Camera feed is not running...")
            continue

        visualizer_mono = None
        
        frame_id = 0

        while True:
            start = time.time()

            try:
                ret, frame = cam.read()
            except Exception as e:
                print(f"Unable to read frame")
                continue    

            if ret:
                print(f"monoloco type frame : {type(frame)}")
                image = cv2.resize(frame, (1920, 1080))
            else:
                continue

            '''  Manual loop system
                flag = 0
            else:
                flag += 1
                if flag < 10:
                    print(f"Unable to resize, skipping frame...")
                    continue
                else:
                    print(f"Unable to resize frames after multiple attempts, skipping to next camera...")
                    break
            '''

            # scale = (args.long_edge)/frame.shape[0]
            # image = cv2.resize(frame, None, fx=scale, fy=scale)
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

            if not ret:
                break
            key = cv2.waitKey(1)
            if key % 256 == 27:
                # ESC pressed
                LOG.info("Escape hit, closing...")
                break

            kk = load_calibration(args.calibration, pil_image.size, focal_length=args.focal_length)
            boxes, keypoints = preprocess_pifpaf(
                pifpaf_outs['left'], (width, height))


            dic_out = net.forward(keypoints, kk)
            dic_out = net.post_process(dic_out, boxes, keypoints, kk)

            if 'social_distance' in args.activities:
                dic_out = net.social_distance(dic_out, args)
            if 'raise_hand' in args.activities:
                dic_out = net.raising_hand(dic_out, keypoints)
            if visualizer_mono is None:  # it is, at the beginning
                visualizer_mono = Visualizer(kk, args)(pil_image)  # create it with the first image
                visualizer_mono.send(None)

            # print(f"\n============== dic_out : {dic_out['xyz_pred']}\n")

            print(f"Identified {len(dic_out['xyz_pred'])} people")
            print(f"xyz_pred: {dic_out['xyz_pred']}")

            ################## POST DATA ################## 
            # '''
            BASE_URL = 'http://web:8000'

            camera_to_person_xyz = dic_out['xyz_pred']
    
            if len(camera_to_person_xyz) > 0:
                for id, xyz in enumerate(camera_to_person_xyz):
                    x = xyz[0]
                    # y = xyz[1]
                    z = xyz[2]

                    person_instance_obj = {
                                "id": 0,
                                "name": f"Person {id+1}",
                                "frame_id": frame_id,
                                "x": x,
                                "z": z
                    }
                    url = f"{BASE_URL}/patch_person_instance/"
                    # url = f"{BASE_URL}/patch_person_instance/{frame_id}/Person {id}"
                    try:
                        x = requests.patch(url,json=person_instance_obj,headers={"content-type":"application/json","accept":"application/json"})
                        print(f"POST /patch_person_instance")
                    except:
                        print(f"no POST /patch_person_instance")
                        continue
            # '''
            ############################################### 
                
            ''' Output analyzed photos
            cv2.imwrite(f'monoloco{frame_id}.jpg', image)
            '''

            LOG.debug(dic_out)
            frame_id += 1

            # visualizer_mono.send((pil_image, dic_out, pifpaf_outs))

            end = time.time()
            LOG.info("run-time: {:.2f} ms".format((end-start)*1000))

        # cam.release()

        # cv2.destroyAllWindows()


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
