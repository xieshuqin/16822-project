# Generate the pose pickle

from stacked_hourglass import HumanPosePredictor, hg2
from stacked_hourglass.datasets.mpii import MPII_JOINT_NAMES
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import copy


number = 2

#Run inference
img = Image.open("./" + str(number) + ".jpg")
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
cv2.imwrite('check' + str(number) + '.jpg',draw)

import pickle
with open('pose' + str(number) +'.pickle', 'wb') as handle:
    pickle.dump(pose_xy, handle, protocol=pickle.HIGHEST_PROTOCOL)

#print('Right elbow location: ', joints[MPII_JOINT_NAMES.index('right_elbow')])

    """
   #Set 3 - 0.jpg - human annotate
    if (joint == 'right_ankle'):
        x = 501
        y = 615
    if (joint == 'right_knee'):
        x = 484
        y = 539
    if (joint == 'right_hip'):
        x = 495
        y = 463
    if (joint == 'left_hip'):
        x = 452
        y = 460
    if (joint == 'left_knee'):
        x = 457
        y = 541
    if (joint == 'left_ankle'):
        x = 466
        y = 625
    if (joint == 'pelvis'):
        x = 478
        y = 460
    if (joint == 'spine'):
        x = 485
        y = 349
    if (joint == 'neck'):
        x = 481
        y = 328
    if (joint == 'head_top'):
        x = 486
        y = 276
    if (joint == 'right_wrist'):
        x = 514
        y = 461
    if (joint == 'right_elbow'):
        x = 523
        y = 418 
    if (joint == 'right_shoulder'):
        x = 517
        y = 350
    if (joint == 'left_shoulder'):
        x = 442
        y = 357 
    if (joint == 'left_elbow'):
        x = 434
        y = 413 
    if (joint == 'left_wrist'):
        x = 410
        y = 460 

    Set 3 - 1.jpg
    if (joint == 'right_ankle'):
        x = 522
        y = 563
    if (joint == 'right_knee'):
        x = 522
        y = 500
    if (joint == 'right_hip'):
        x = 529
        y = 420
    if (joint == 'left_hip'):
        x = 576
        y = 422
    if (joint == 'left_knee'):
        x = 573
        y = 486
    if (joint == 'left_ankle'):
        x = 559
        y = 564
    if (joint == 'pelvis'):
        x = 553
        y = 423
    if (joint == 'spine'):
        x = 554
        y = 312
    if (joint == 'neck'):
        x = 552
        y = 298
    if (joint == 'head_top'):
        x = 548
        y = 244
    if (joint == 'right_wrist'):
        x = 498
        y = 422
    if (joint == 'right_elbow'):
        x = 513
        y = 373 
    if (joint == 'right_shoulder'):
        x = 520
        y = 316 
    if (joint == 'left_shoulder'):
        x = 584
        y = 316 
    if (joint == 'left_elbow'):
        x = 591
        y = 378 
    if (joint == 'left_wrist'):
        x = 586
        y = 426 

    
   #Set 3 - 2.jpg - human annotate
    if (joint == 'right_ankle'):
        x = 650
        y = 549
    if (joint == 'right_knee'):
        x = 650
        y = 468
    if (joint == 'right_hip'):
        x = 644
        y = 396
    if (joint == 'left_hip'):
        x = 589
        y = 390
    if (joint == 'left_knee'):
        x = 600
        y = 465
    if (joint == 'left_ankle'):
        x = 612
        y = 551
    if (joint == 'pelvis'):
        x = 611
        y = 391
    if (joint == 'spine'):
        x = 609
        y = 279
    if (joint == 'neck'):
        x = 612
        y = 258
    if (joint == 'head_top'):
        x = 613
        y = 209
    if (joint == 'right_wrist'):
        x = 676
        y = 395
    if (joint == 'right_elbow'):
        x = 661
        y = 349 
    if (joint == 'right_shoulder'):
        x = 651
        y = 282
    if (joint == 'left_shoulder'):
        x = 578
        y = 287 
    if (joint == 'left_elbow'):
        x = 578
        y = 349 
    if (joint == 'left_wrist'):
        x = 593
        y = 388
    
    #Set 3 - 3.jpg - human annotate
    if (joint == 'right_ankle'):
        x = 455
        y = 569
    if (joint == 'right_knee'):
        x = 489
        y = 491
    if (joint == 'right_hip'):
        x = 470
        y = 380
    if (joint == 'left_hip'):
        x = 536
        y = 379
    if (joint == 'left_knee'):
        x = 521
        y = 471
    if (joint == 'left_ankle'):
        x = 492
        y = 566
    if (joint == 'pelvis'):
        x = 507
        y = 382
    if (joint == 'spine'):
        x = 505
        y = 238
    if (joint == 'neck'):
        x = 506
        y = 220
    if (joint == 'head_top'):
        x = 507
        y = 151
    if (joint == 'right_wrist'):
        x = 450
        y = 393
    if (joint == 'right_elbow'):
        x = 447
        y = 325 
    if (joint == 'right_shoulder'):
        x = 455
        y = 241 
    if (joint == 'left_shoulder'):
        x = 547
        y = 250 
    if (joint == 'left_elbow'):
        x = 555
        y = 320 
    if (joint == 'left_wrist'):
        x = 581
        y = 387 
    """