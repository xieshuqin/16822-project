# Generate the pose pickle

from stacked_hourglass import HumanPosePredictor, hg2
from stacked_hourglass.datasets.mpii import MPII_JOINT_NAMES
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import copy

#Run inference
img = Image.open("./3.jpg")
convert_tensor = transforms.ToTensor()
my_image_tensor = convert_tensor(img)
model = hg2(pretrained=True)
predictor = HumanPosePredictor(model, device='cuda')
joints = predictor.estimate_joints(my_image_tensor, flip=True)


joint_names = []
for i, name in enumerate(MPII_JOINT_NAMES):
    joint_names.append(name)
    print(i, name)

pose_xy = []
draw = copy.deepcopy(np.array(img))
for joint in joint_names:
    x = joints[MPII_JOINT_NAMES.index(joint)][0].item()
    y = joints[MPII_JOINT_NAMES.index(joint)][1].item()

    pose_xy.append([x,y])
    draw = cv2.circle(np.array(draw), (int(x),int(y)) ,5, (0,0,255), 5)

#import pdb; pdb.set_trace()
#x = joints[MPII_JOINT_NAMES.index('right_wrist')][0].item()
#y = joints[MPII_JOINT_NAMES.index('right_wrist')][1].item()
#draw = cv2.circle(np.array(img), (int(x),int(y)) ,5, (0,0,255), 5)
cv2.imwrite('check3.jpg',draw)

import pickle
with open('pose3.pickle', 'wb') as handle:
    pickle.dump(pose_xy, handle, protocol=pickle.HIGHEST_PROTOCOL)

#print('Right elbow location: ', joints[MPII_JOINT_NAMES.index('right_elbow')])


    """
    #2.jpg - human annotate
    if (joint == 'right_ankle'):
        x = 531
        y = 558
    if (joint == 'right_knee'):
        x = 524
        y = 492
    if (joint == 'right_hip'):
        x = 513
        y = 416
    if (joint == 'left_hip'):
        x = 567
        y = 418
    if (joint == 'left_knee'):
        x = 568
        y = 485
    if (joint == 'left_ankle'):
        x = 564
        y = 567
    if (joint == 'pelvis'):
        x = 541
        y = 410
    if (joint == 'spine'):
        x = 535
        y = 313
    if (joint == 'neck'):
        x = 537
        y = 289
    if (joint == 'head_top'):
        x = 532
        y = 242
    if (joint == 'right_wrist'):
        x = 489
        y = 434
    if (joint == 'right_elbow'):
        x = 497
        y = 372 
    if (joint == 'right_shoulder'):
        x = 500
        y = 316 
    if (joint == 'left_shoulder'):
        x = 567
        y = 318 
    if (joint == 'left_elbow'):
        x = 583
        y = 374 
    if (joint == 'left_wrist'):
        x = 588
        y = 432 
    
    
    #4.jpg - human annotate
    if (joint == 'right_ankle'):
        x = 460
        y = 565
    if (joint == 'right_knee'):
        x = 465
        y = 476
    if (joint == 'right_hip'):
        x = 472
        y = 374
    if (joint == 'left_hip'):
        x = 532
        y = 382
    if (joint == 'left_knee'):
        x = 512
        y = 474
    if (joint == 'left_ankle'):
        x = 504
        y = 569
    if (joint == 'pelvis'):
        x = 497
        y = 380
    if (joint == 'spine'):
        x = 505
        y = 249
    if (joint == 'neck'):
        x = 499
        y = 219
    if (joint == 'head_top'):
        x = 500
        y = 143
    if (joint == 'right_wrist'):
        x = 435
        y = 388
    if (joint == 'right_elbow'):
        x = 440
        y = 320
    if (joint == 'right_shoulder'):
        x = 450
        y = 250
    if (joint == 'left_shoulder'):
        x = 543
        y = 247 
    if (joint == 'left_elbow'):
        x = 550
        y = 317 
    if (joint == 'left_wrist'):
        x = 567
        y = 380 
    """

    # if (joint == 'left_wrist'):
    #     x = 587
    #     y = 434 