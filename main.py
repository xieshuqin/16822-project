import torch
import torch.nn as nn

import numpy as np
from pose2d import create_2d_pose_model, infer_2d_pose, pose_to_bbox
from reid import create_reid_model, reid_people
from single_pose3d import ransac_triangulation


def main():
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
            one_person_poses_2d = [poses_2d[image_id][bbox_id] for (image_id, bbox_id) in one_person_bbox_ids]
            one_person_camera_matrices = [camera_matrices[image_id] for (image_id, bbox_id) in one_person_bbox_ids]
            one_person_poses_3d = ransac_triangulation(one_person_poses_2d, one_person_camera_matrices)
            poses_3d.append(one_person_poses_3d)

    return poses_3d


if __name__ == '__main__':
    main()