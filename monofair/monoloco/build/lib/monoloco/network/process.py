
import json

import numpy as np
import torch
import torchvision

from ..utils import get_keypoints, pixel_to_camera, back_correct_angles


def preprocess_monoloco(keypoints, kk):

    """ Preprocess batches of inputs
    keypoints = torch tensors of (m, 3, 17)  or list [3,17]
    Outputs =  torch tensors of (m, 34) in meters normalized (z=1) and zero-centered using the center of the box
    """
    if isinstance(keypoints, list):
        keypoints = torch.tensor(keypoints)
    if isinstance(kk, list):
        kk = torch.tensor(kk)
    # Projection in normalized image coordinates and zero-center with the center of the bounding box
    uv_center = get_keypoints(keypoints, mode='center')
    xy1_center = pixel_to_camera(uv_center, kk, 10)
    xy1_all = pixel_to_camera(keypoints[:, 0:2, :], kk, 10)
    # kps_norm = xy1_all - xy1_center.unsqueeze(1)  # (m, 17, 3) - (m, 1, 3)
    kps_norm = xy1_all
    kps_out = kps_norm[:, :, 0:2].reshape(kps_norm.size()[0], -1)  # no contiguous for view
    return kps_out


def factory_for_gt(im_size, name=None, path_gt=None):
    """Look for ground-truth annotations file and define calibration matrix based on image size """

    try:
        with open(path_gt, 'r') as f:
            dic_names = json.load(f)
        print('-' * 120 + "\nGround-truth file opened")
    except (FileNotFoundError, TypeError):
        print('-' * 120 + "\nGround-truth file not found")
        dic_names = {}

    try:
        kk = dic_names[name]['K']
        dic_gt = dic_names[name]
        print("Matched ground-truth file!")
    except KeyError:
        dic_gt = None
        x_factor = im_size[0] / 1600
        y_factor = im_size[1] / 900
        pixel_factor = (x_factor + y_factor) / 2   # TODO remove and check it
        if im_size[0] / im_size[1] > 2.5:
            kk = [[718.3351, 0., 600.3891], [0., 718.3351, 181.5122], [0., 0., 1.]]  # Kitti calibration
        else:
            kk = [[1266.4 * pixel_factor, 0., 816.27 * x_factor],
                  [0, 1266.4 * pixel_factor, 491.5 * y_factor],
                  [0., 0., 1.]]  # nuScenes calibration

        print("Using a standard calibration matrix...")

    return kk, dic_gt


def laplace_sampling(outputs, n_samples):

    # np.random.seed(1)
    mu = outputs[:, 0]
    bi = torch.abs(outputs[:, 1])

    # Analytical
    # uu = np.random.uniform(low=-0.5, high=0.5, size=mu.shape[0])
    # xx = mu - bi * np.sign(uu) * np.log(1 - 2 * np.abs(uu))

    # Sampling
    cuda_check = outputs.is_cuda
    if cuda_check:
        get_device = outputs.get_device()
        device = torch.device(type="cuda", index=get_device)
    else:
        device = torch.device("cpu")

    laplace = torch.distributions.Laplace(mu, bi)
    xx = laplace.sample((n_samples,)).to(device)

    return xx


def unnormalize_bi(loc):
    """
    Unnormalize relative bi of a nunmpy array
    Input --> tensor of (m, 2)
    """
    assert loc.size()[1] == 2, "size of the output tensor should be (m, 2)"
    bi = torch.exp(loc[:, 1:2]) * loc[:, 0:1]

    return bi


def preprocess_pifpaf(annotations, im_size=None, enlarge_boxes=True):
    """
    Preprocess pif annotations:
    1. enlarge the box of 10%
    2. Constraint it inside the image (if image_size provided)
    """

    boxes = []
    keypoints = []
    dummy = 1 if enlarge_boxes else 2.3  # TODO fix for social
    for dic in annotations:
        kps = prepare_pif_kps(dic['keypoints'])
        box = dic['bbox']
        try:
            conf = dic['score']
            # Enlarge boxes
            delta_h = (box[3]) / (10 * dummy)
            delta_w = (box[2]) / (5 * dummy)
            # from width height to corners
            box[2] += box[0]
            box[3] += box[1]

        except KeyError:
            all_confs = np.array(kps[2])
            score_weights = np.ones(17)
            score_weights[:3] = 3.0
            score_weights[5:] = 0.1
            conf = np.sum(score_weights * np.sort(all_confs)[::-1])
            conf = float(np.mean(all_confs))
            # Add 15% for y and 20% for x
            delta_h = (box[3] - box[1]) / (7 * dummy)
            delta_w = (box[2] - box[0]) / (3.5 * dummy)
            assert delta_h > -5 and delta_w > -5, "Bounding box <=0"

        box[0] -= delta_w
        box[1] -= delta_h
        box[2] += delta_w
        box[3] += delta_h

        # Put the box inside the image
        if im_size is not None:
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(box[2], im_size[0])
            box[3] = min(box[3], im_size[1])

        box.append(conf)
        boxes.append(box)
        keypoints.append(kps)

    return boxes, keypoints


def prepare_pif_kps(kps_in):
    """Convert from a list of 51 to a list of 3, 17"""

    assert len(kps_in) % 3 == 0, "keypoints expected as a multiple of 3"
    xxs = kps_in[0:][::3]
    yys = kps_in[1:][::3]  # from offset 1 every 3
    ccs = kps_in[2:][::3]

    return [xxs, yys, ccs]


def image_transform(image):

    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize, ])
    return transforms(image)


def extract_outputs_mono(outputs, tasks=None, mono=False):
    """
    Extract the outputs for multi-task training and predictions
    Inputs:
        tensor (m, 10) or (m,9) if monoloco
    Outputs:
         - if tasks are provided return ordered list of raw tensors
         - else return a dictionary with processed outputs
    """
    dic_out = {'xy': outputs[:, 0:2], 'z': outputs[:, 2:4], 'hwl': outputs[:, 4:7], 'ori': outputs[:, 7:9]}

    # Multi-task training
    if tasks is not None:
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_out[task] for task in tasks]

    # Preprocess the tensor
    else:
        AV_W, AV_L, AV_H, WLH_STD = 0.68, 0.75, 1.72, 0.1
        bi = unnormalize_bi(dic_out['z'])

        dic_out = {key: el.detach().cpu() for key, el in dic_out.items()}
        dic_out['xyz'] = torch.cat((dic_out['xy'], dic_out['z'][:, 0:1]), dim=1)
        dd = torch.norm(dic_out['xyz'], p=2, dim=1).view(-1, 1)

        dic_out.pop('z'), dic_out.pop('xy')
        dic_out['d'], dic_out['bi'] = dd, bi

        yaw_pred = torch.atan2(dic_out['ori'][:, 0:1], dic_out['ori'][:, 1:2])
        yaw_orig = back_correct_angles(yaw_pred, dic_out['xyz'])

        # dic_out['wlh'] = dic_out['wlh'] * WLH_STD + torch.tensor([AV_W, AV_L, AV_H]).view(1, -1)
        dic_out['yaw'] = (yaw_pred, yaw_orig)  # alpha, ry
        return dic_out


def extract_labels(labels, tasks=None):

    dic_gt_out = {'dd': labels[:, 0:1], 'xy': labels[:, 1:3], 'loc': labels[:, 3:4], 'wlh': labels[:, 4:7],
                  'ori': labels[:, 7:9]}
    if tasks is None:
        return dic_gt_out
    else:
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_gt_out[task] for task in tasks]
