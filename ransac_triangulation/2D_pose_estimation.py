# Generate the pose pickle

from stacked_hourglass import HumanPosePredictor, hg2
from stacked_hourglass.datasets.mpii import MPII_JOINT_NAMES
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

#Run inference
img = Image.open("./4.jpg")
convert_tensor = transforms.ToTensor()
my_image_tensor = convert_tensor(img)
model = hg2(pretrained=True)
predictor = HumanPosePredictor(model, device='cuda')
joints = predictor.estimate_joints(my_image_tensor, flip=True)


joint_names = []
for i, name in enumerate(MPII_JOINT_NAMES):
    joint_names.append(name)
    #print(i, name)

pose_xy = []
for joint in joint_names:
    x = joints[MPII_JOINT_NAMES.index(joint)][0].item()
    y = joints[MPII_JOINT_NAMES.index(joint)][1].item()
    pose_xy.append([x,y])

#x = joints[MPII_JOINT_NAMES.index('right_elbow')][0].item()
#y = joints[MPII_JOINT_NAMES.index('right_elbow')][1].item()
#draw = cv2.circle(np.array(img), (int(x),int(y)) ,20, (255,0,0), 1)
#cv2.imwrite('test.jpg',draw)

import pickle
with open('pose4.pickle', 'wb') as handle:
    pickle.dump(pose_xy, handle, protocol=pickle.HIGHEST_PROTOCOL)

#print('Right elbow location: ', joints[MPII_JOINT_NAMES.index('right_elbow')])

