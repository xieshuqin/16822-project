import sys
sys.path.append('./DEKR')  # nopep8
sys.path.append('./DEKR/tools')  # nopep8
sys.path.append('./DEKR/lib')  # nopep8
from DEKR.tools.inference_demo import get_pose_estimation_prediction
from DEKR.lib.config import cfg, update_config
from DEKR.lib import models
import DEKR.lib.models.hrnet_dekr
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch


class Args(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.opts = []


def create_2d_pose_model(cfg_name='DEKR/experiments/coco/inference_demo_coco.yaml'):
    args = Args(cfg_name)
    update_config(cfg, args)
    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    MODEL_FILE = 'dekr_models/pose_dekr_hrnetw32_coco.pth'
    if MODEL_FILE:
        print('=> loading model from {}'.format(MODEL_FILE))
        pose_model.load_state_dict(torch.load(
            MODEL_FILE), strict=False)
    else:
        raise ValueError('expected model defined in config at TEST.MODEL_FILE')

    pose_model.cuda()
    pose_model.eval()

    return pose_model


def infer_2d_pose(pose_model, image, visthre=0.0):
    """
    Expect image to be h x w x c (rgb) order.
    """
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    pose_preds = get_pose_estimation_prediction(
        cfg, pose_model, image, visthre, transforms=pose_transform)

    return pose_preds


def pose_to_bbox(poses, image):
    scale = 1.2
    h, w, c = image.shape
    bboxes = []
    for pose in poses:
        xmin, ymin, xmax, ymax = 10000, 10000, 0, 0
        for p in pose:
            xmin = min(xmin, p[0])
            ymin = min(ymin, p[1])
            xmax = max(xmax, p[0])
            ymax = max(ymax, p[1])
        xcenter, ycenter = (xmin + xmax) / 2, (ymin + ymax) / 2
        width, height = xmax - xmin, ymax - ymin
        xmin, xmax = max(xcenter - 0.5 * scale * width,
                         0), min(xcenter + 0.5 * scale * width, w - 1)
        ymin, ymax = max(ycenter - 0.5 * scale * height,
                         0), min(ycenter + 0.5 * scale * height, h - 1)
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes


if __name__ == '__main__':
    image = torch.randn((512, 512, 3), dtype=torch.float)
    pose_model = create_2d_pose_model()
    pose_preds = infer_2d_pose(pose_model, image)
    bboxes = pose_to_bbox(pose_preds, image)
