import copy
import sys

import single_pose

sys.path.append('./EpipolarPose')
sys.path.append('./EpipolarPose/lib')

import pickle
import numpy as np
from PIL import Image

from EpipolarPose.lib.utils import prep_h36m

# split our dataset image name lists into four cameras
# example name: S7_Walking_1.54138969_000129.jpg
camera_name_to_id = {54138969: 1, 55011271: 2, 58860488: 3, 60457274: 4}
camera_id_to_name = {v: k for k, v in camera_name_to_id.items()}
camera_seq_names = {1: [], 2: [], 3: [], 4: []}
camera_frame_ids = {1: [], 2: [], 3: [], 4: []}
with open('data/h36m/annot/train_images.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip('\n')
        camera_name = int(line.split('.')[1][:8])
        camera_id = camera_name_to_id[camera_name]
        seq_name = line.split('.')[0]
        frame_id = int(line[-10:-4])
        camera_seq_names[camera_id].append(seq_name)
        camera_frame_ids[camera_id].append(frame_id)

for k in camera_seq_names.keys():
    camera_seq_names[k] = np.array(camera_seq_names[k], dtype=object)
    camera_frame_ids[k] = np.array(camera_frame_ids[k])

# sanity check to ensure seq_names and frame ids across four cameras are the same
for i in range(1, 4):
    assert np.all(camera_seq_names[i] == camera_seq_names[i+1])
    assert np.all(camera_frame_ids[i] == camera_frame_ids[i+1])

# for each image in the original pickle, find the nearest available image
# on our current available images, then run our approach to generate label.
with open('EpipolarPose/data/h36m/anno/train-fs.pkl', 'rb') as f:
    gt_anno = pickle.load(f)

num_annotation = len(gt_anno[1])
our_anno = {1: [], 2: [], 3: [], 4: []}
for i in range(num_annotation):
    # example name: images/S7/Walking_1.54138969/000129.jpg
    gt_imgname = gt_anno[1][i]['image'].split('/')
    frame_id = int(gt_imgname[-1][:6])
    seq_name = '_'.join([gt_imgname[1], gt_imgname[2].split('.')[0]])
    same_seq_idxs = np.where(camera_seq_names[1] == seq_name)
    closest_idx = same_seq_idxs[np.argmin(np.abs(camera_frame_ids[1][same_seq_idxs] - frame_id))]
    closest_frame_id = camera_frame_ids[1][closest_idx]

    # we have found the closest frame id available, let's run RANSAC and StackHourGlass to infer its 3d pose
    # Note that we need to do a mapping from Hourglass model to H36M keypoint order
    poses2d = []
    cam_matrices = []
    for camera_id in range(1, 5):
        cam_name = str(camera_id_to_name[camera_id])
        imgname = f'images/{gt_imgname[1]}/{seq_name}.{cam_name}_{closest_frame_id:06d}.jpg'
        img = Image.open(imgname)
        ann = gt_anno[camera_id][i]
        pose2d = single_pose.estimate_pose(img, ann['center_x'], ann['center_y'], ann['width'], ann['height'])
        pose2d_h36m = single_pose.transform_hg_pose_to_h36m_pose(pose2d)
        poses2d.append(pose2d_h36m)
        cam_matrices.append(ann['cam'].projection_matrix)
    # RANSAC triangulation
    pose3d = single_pose.ransac_triangulaton(poses2d, cam_matrices)

    # Once we have the 3d pose, generate annotation for each camera
    for camera_id in range(1, 5):
        R, T, f, c, k, p, cam_name = gt_anno[camera_id][i]['cam'].cam_params
        rect2d_l, rect2d_r, rect2d_t, rect2d_b, pt_2d, pt_3d, vis, pelvis3d = \
            prep_h36m.from_worldjt_to_imagejt(17, R, pose3d, T, f, c, 2000, 2000, False)
        new_anno = copy.deepcopy(gt_anno[camera_id][i])
        new_anno['image'] = f'images/{gt_imgname[1]}/{seq_name}.{cam_name}_{closest_frame_id:06d}.jpg'
        new_anno['joints_3d'] = pt_2d
        new_anno['joints_3d_cam'] = pt_3d
        new_anno['pelvis'] = pelvis3d
        our_anno[camera_id].append(new_anno)

with open('train-our-ss.pkl', 'wb') as f:
    pickle.dump(our_anno, f, protocol=4)