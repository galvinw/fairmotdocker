import time
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from ..visuals import Printer
from ..network import PifPaf, MonoLoco
from ..network.process import preprocess_pifpaf, factory_for_gt, image_transform


def webcam(args):
    args.device = torch.device('cpu')
    # if torch.cuda.is_available():
    #     args.device = torch.device('cuda')
    pifpaf = PifPaf(args)
    monoloco = MonoLoco(model=args.model, device=args.device)


    cap = cv2.VideoCapture("1.mp4")
    while cap.isOpened:
        _, image = cap.read()
        image = cv2.resize(image, (1280,720))
        height, width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image_cpu = image_transform(image.copy())
        processed_image = processed_image_cpu.contiguous().to(args.device, non_blocking=True)
        fields = pifpaf.fields(torch.unsqueeze(processed_image, 0))[0]
        _, _, pifpaf_out = pifpaf.forward(image, processed_image_cpu, fields)

        pil_image = Image.fromarray(image)
        intrinsic_size = [xx * 1.3 for xx in pil_image.size]
        kk, dict_gt = factory_for_gt(intrinsic_size)
        if pifpaf_out:
            boxes, keypoints = preprocess_pifpaf(pifpaf_out, (width, height))
            outputs = monoloco.forward(keypoints, kk)
            dic_out = monoloco.post_process(outputs, boxes, keypoints, kk, dic_gt=None, reorder=False)
            #Finding Social Distancing
            angles = dic_out['angles']
            xz_centers = [[xx[0], xx[2]] for xx in dic_out['xyz_pred']]
            for idx, _ in enumerate(dic_out['xyz_pred']):
                if social_distance(xz_centers, angles, idx):
                    print("Not Following Social Distancing Rules")
                else:
                    print("Following Social Distancing Rules")
        cv2.imshow("Frame", image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

def check_social_distance(xxs, zzs, angles):
    """
    Violation if same angle or ine in front of the other
    Obtained by assuming straight line, constant velocity and discretizing trajectories
    """
    min_distance = 0.5
    theta0 = angles[0]
    theta1 = angles[1]
    steps = np.linspace(0, 2, 20)  # Discretization 20 steps in 2 meters
    xs0 = [xxs[0] + step * math.cos(theta0) for step in steps]
    zs0 = [zzs[0] - step * math.sin(theta0) for step in steps]
    xs1 = [xxs[1] + step * math.cos(theta1) for step in steps]
    zs1 = [zzs[1] - step * math.sin(theta1) for step in steps]
    distances = [math.sqrt((xs0[idx] - xs1[idx]) ** 2 + (zs0[idx] - zs1[idx]) ** 2) for idx, _ in enumerate(xs0)]
    if np.min(distances) <= max(distances[0] / 1.5, min_distance):
        return True
    return False

def social_distance(centers, angles, idx, threshold=2.5):
    """
    return flag of alert if social distancing is violated
    """
    xx = centers[idx][0]
    zz = centers[idx][1]
    angle = angles[idx]
    distances = [math.sqrt((xx - centers[i][0]) ** 2 + (zz - centers[i][1]) ** 2) for i, _ in enumerate(centers)]
    sorted_idxs = np.argsort(distances)

    for i in sorted_idxs[1:]:

        # First check for distance
        if distances[i] > threshold:
            return False

        # More accurate check based on orientation and future position
        elif check_social_distance((xx, centers[i][0]), (zz, centers[i][1]), (angle, angles[i])):
            return True

    return False
