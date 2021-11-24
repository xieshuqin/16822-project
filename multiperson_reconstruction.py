import torch
import torch.nn as nn
import numpy as np
from pose2d import create_2d_pose_model, infer_2d_pose, pose_to_bbox
#from reid import create_reid_model, reid_people
import os
#from single_pose3d import ransac_triangulation
from IPython import embed
import json
import cv2


class CMUPanopticDataset(torch.utils.data.Dataset):
    def __init__(self, path, cameras):
        self.cameras = sorted(cameras)
        self.images = {}
        for c in cameras:
            self.images[c] = sorted(
                [path+'/hdImgs/'+p for p in os.listdir(path+'/hdImgs/'+c)])
        self.calib = {}
        for camera in json.load(open(path+'/calibration_'+os.path.split(path)[-1]+'.json'))['cameras']:
            if camera['name'] not in cameras:
                continue
            calib = {}
            calib['M'] = camera['K']@np.concatenate(
                (camera['R'], camera['t']), 1)
            calib['distCoef'] = np.array(camera['distCoef'])
            self.calib[camera['name']] = calib

    def __len__(self):
        return len(next(iter(self.images.values())))

    def __getitem__(self, index):
        M = []
        distCoef = []
        images = []
        for c in self.cameras:
            M.append(self.calib[c]['M'])
            distCoef.append(self.calib[c]['distCoef'])
            images.append(cv2.imread(self.images[c][index]))
        return {'M': M, 'distCoef': distCoef, 'images': images}


def main():
    dataset = CMUPanopticDataset(
        'panoptic-toolbox/171204_pose1_sample', ['00_16', '00_21'])
    embed()
    pose_model = create_2d_pose_model()
    reid_model = create_reid_model()

    for blob_in in dataloader:
        images = blob_in['images']
        camera_matrices = blob_in['camera_matrices']

        # run 2d pose estimation
        poses_2d = []
        bboxes = []
        for image in images:
            poses_2d_one_image = infer_2d_pose(pose_model, image)
            poses_2d.append(poses_2d_one_image)
            bboxes_one_image = pose_to_bbox(poses_2d_one_image, image)
            bboxes.append(bboxes_one_image)

        # run reid
        people_bbox_ids = reid_people(reid_model, images, bboxes)

        # run RANSAC and Triangulation to generate 3D pose
        poses_3d = []
        for one_person_bbox_ids in people_bbox_ids:
            one_person_poses_2d = [poses_2d[image_id][bbox_id]
                                   for (image_id, bbox_id) in one_person_bbox_ids]
            one_person_camera_matrices = [camera_matrices[image_id] for (
                image_id, bbox_id) in one_person_bbox_ids]
            one_person_poses_3d = ransac_triangulation(
                one_person_poses_2d, one_person_camera_matrices)
            poses_3d.append(one_person_poses_3d)

    return poses_3d


if __name__ == '__main__':
    main()
