from stacked_hourglass import HumanPosePredictor, hg2
from stacked_hourglass.datasets.mpii import MPII_JOINT_NAMES
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

from ransac_triangulation.ransac_triangulation import ransac_triangulaton

hg_model = hg2(pretrained=True)
predictor = HumanPosePredictor(hg_model, device='cuda')
convert_tensor = transforms.ToTensor()


def estimate_pose(img, xcenter, ycenter, bbox_width, bbox_height):
    img_width, img_height = img.size
    xmin, xmax = int(max(xcenter - 0.5 * bbox_width, 0)), int(min(xcenter + 0.5 * bbox_width, img_width))
    ymin, ymax = int(max(ycenter - 0.5 * bbox_height, 0)), int(min(ycenter + 0.5 * bbox_height, img_height))
    img_crop = img.crop((xmin, ymin, xmax, ymax))
    img_crop = convert_tensor(img_crop)
    joints = predictor.estimate_joints(img_crop, flip=True).cpu().numpy()
    joints[:, 0] += xmin
    joints[:, 1] += ymin
    return joints


# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*17
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[4]  = 'LHip'
H36M_NAMES[5]  = 'LKnee'
H36M_NAMES[6]  = 'LFoot'
H36M_NAMES[7] = 'Spine'
H36M_NAMES[8] = 'Thorax'
H36M_NAMES[9] = 'Neck/Nose'
H36M_NAMES[10] = 'Head'
H36M_NAMES[11] = 'LShoulder'
H36M_NAMES[12] = 'LElbow'
H36M_NAMES[13] = 'LWrist'
H36M_NAMES[14] = 'RShoulder'
H36M_NAMES[15] = 'RElbow'
H36M_NAMES[16] = 'RWrist'

# Stacked Hourglass produces 16 joints. These are the names.
MPII_NAMES = ['']*16
MPII_NAMES[0]  = 'RFoot'
MPII_NAMES[1]  = 'RKnee'
MPII_NAMES[2]  = 'RHip'
MPII_NAMES[3]  = 'LHip'
MPII_NAMES[4]  = 'LKnee'
MPII_NAMES[5]  = 'LFoot'
MPII_NAMES[6]  = 'Hip'
MPII_NAMES[7]  = 'Thorax'
MPII_NAMES[8]  = 'Neck/Nose'
MPII_NAMES[9]  = 'Head'
MPII_NAMES[10] = 'RWrist'
MPII_NAMES[11] = 'RElbow'
MPII_NAMES[12] = 'RShoulder'
MPII_NAMES[13] = 'LShoulder'
MPII_NAMES[14] = 'LElbow'
MPII_NAMES[15] = 'LWrist'

H36M_TO_MPII_PERM = np.array([H36M_NAMES.index(h) for h in MPII_NAMES if h != '' and h in H36M_NAMES])


def transform_hg_pose_to_h36m_pose(pose):
    pose_h36m = np.zeros((len(H36M_NAMES), 2))
    pose_h36m[H36M_TO_MPII_PERM] = pose
    return pose_h36m


def undistort(pose, cam):
    pose = pose.reshape((pose.shape[0], 1, 2))
    pose = cv2.undistortPoints(pose, cam.camera_matrix, cam.dist_coeffs)
    return pose.reshape((pose.shape[0], 2))