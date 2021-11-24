
"""
Monoloco class. From 2D joints to real-world distances
"""

import logging
from collections import defaultdict
import math

import torch

from ..utils import get_iou_matches, reorder_matches, get_keypoints, pixel_to_camera, xyz_from_distance
from .process import preprocess_monoloco, unnormalize_bi, extract_outputs_mono
from .architectures import LinearModel


class MonoLoco:

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    INPUT_SIZE = 34
    OUTPUT_SIZE = 9
    LINEAR_SIZE = 256
    N_SAMPLES = 100
    AV_W = 0.68
    AV_L = 0.75
    AV_H = 1.72
    WLH_STD = 0.1

    def __init__(self, model, device=None, n_dropout=0, p_dropout=0.2):

        if not device:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.n_dropout = n_dropout
        self.epistemic = bool(self.n_dropout > 0)
        
        # if the path is provided load the model parameters
        if isinstance(model, str):
            model_path = model
            self.model = LinearModel(p_dropout=p_dropout, input_size=self.INPUT_SIZE, output_size=self.OUTPUT_SIZE,
                                     linear_size=self.LINEAR_SIZE)
            self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

        # if the model is directly provided
        else:
            self.model = model
        self.model.eval()  # Default is train
        self.model.to(self.device)

    def forward(self, keypoints, kk):
        """forward pass of monoloco network"""
        if not keypoints:
            return None

        with torch.no_grad():
            keypoints = torch.tensor(keypoints).to(self.device)
            kk = torch.tensor(kk).to(self.device)

            inputs = preprocess_monoloco(keypoints, kk)
            outputs = self.model(inputs)
            dic_out = extract_outputs_mono(outputs)
        return dic_out

    @staticmethod
    def post_process(dic_in, boxes, keypoints, kk, dic_gt=None, iou_min=0.3, reorder=True):
        """Post process monoloco to output final dictionary with all information for visualizations"""

        dic_out = defaultdict(list)
        if dic_in is None:
            return dic_out

        if dic_gt:
            boxes_gt = dic_gt['boxes']
            dds_gt = [el[3] for el in dic_gt['ys']]
            matches = get_iou_matches(boxes, boxes_gt, iou_min=iou_min)
            print("found {} matches with ground-truth".format(len(matches)))
        else:
            matches = [(idx, idx) for idx, _ in enumerate(boxes)]  # Replicate boxes

        if reorder:
            matches = reorder_matches(matches, boxes, mode='left_right')
        uv_shoulders = get_keypoints(keypoints, mode='shoulder')
        uv_heads = get_keypoints(keypoints, mode='head')
        uv_centers = get_keypoints(keypoints, mode='center')
        xy_centers = pixel_to_camera(uv_centers, kk, 1)

        # Match with ground truth if available
        for idx, idx_gt in matches:
            dd_pred = float(dic_in['d'][idx])
            ale = float(dic_in['bi'][idx])
            var_y = float(0)  # Varss to remove TODO
            dd_real = dds_gt[idx_gt] if dic_gt else dd_pred
            angle = float(dic_in['yaw'][1][idx])  # Original angle

            kps = keypoints[idx]
            box = boxes[idx]
            uu_s, vv_s = uv_shoulders.tolist()[idx][0:2]
            uu_c, vv_c = uv_centers.tolist()[idx][0:2]
            uu_h, vv_h = uv_heads.tolist()[idx][0:2]
            uv_shoulder = [round(uu_s), round(vv_s)]
            uv_center = [round(uu_c), round(vv_c)]
            uv_head = [round(uu_h), round(vv_h)]
            xyz_real = xyz_from_distance(dd_real, xy_centers[idx])
            xyz_pred = xyz_from_distance(dd_pred, xy_centers[idx])
            dic_out['boxes'].append(box)
            dic_out['boxes_gt'].append(boxes_gt[idx_gt] if dic_gt else boxes[idx])
            dic_out['dds_real'].append(dd_real)
            dic_out['dds_pred'].append(dd_pred)
            dic_out['stds_ale'].append(ale)
            dic_out['stds_epi'].append(var_y)
            dic_out['xyz_real'].append(xyz_real.squeeze().tolist())
            dic_out['xyz_pred'].append(xyz_pred.squeeze().tolist())
            dic_out['uv_kps'].append(kps)
            dic_out['uv_centers'].append(uv_center)
            dic_out['uv_shoulders'].append(uv_shoulder)
            dic_out['uv_heads'].append(uv_head)
            dic_out['angles'].append(angle)

        return dic_out

    def process_outputs(self, outputs):
        """Convert the output to xyz, orientation and wlh"""
        xyz = outputs[:, 0:3]
        bi = unnormalize_bi(outputs[:, 2:4])
        const = [self.AV_W, self.AV_L, self.AV_H]
        wlh = outputs[:, 4:7] * self.WLH_STD + torch.tensor(const).view(1, -1).to(self.device)
        yaw = torch.atan2(outputs[:, 7:8], outputs[:, 8:9]) * 180 / math.pi
        correction = torch.atan2(xyz[:, 0:1], xyz[:, 2:3]) * 180 / math.pi
        yaw = yaw + correction
        return xyz, bi, yaw, wlh
