import torch
import torch.nn as nn
import numpy as np
from pose2d import create_2d_pose_model, infer_2d_pose, pose_to_bbox
from reid import create_reid_model, reid_people
import os
#from single_pose3d import ransac_triangulation
from IPython import embed
import json
import cv2
import matplotlib.pyplot as plt


class CMUPanopticDataset(torch.utils.data.Dataset):
    def __init__(self, path, cameras):
        self.cameras = sorted(cameras)
        self.images = {}
        for c in cameras:
            self.images[c] = sorted(
                [path+'/hdImgs/'+c+'/'+p for p in os.listdir(path+'/hdImgs/'+c) if '.jpg.' not in p])
        assert len({len(self.images[c]) for c in cameras}) == 1
        self.calib = {}
        for camera in json.load(open(path+'/calibration_'+os.path.split(path)[-1]+'.json'))['cameras']:
            if camera['name'] not in cameras:
                continue
            calib = {}
            calib['K'] = camera['K']
            calib['R'] = camera['R']
            calib['t'] = camera['t']
            calib['distCoef'] = np.array(camera['distCoef'])
            self.calib[camera['name']] = calib

    def __len__(self):
        return len(next(iter(self.images.values())))

    def __getitem__(self, index):
        K = []
        R = []
        t = []
        distCoef = []
        images = []
        img_paths = []
        for c in self.cameras:
            K.append(self.calib[c]['K'])
            R.append(self.calib[c]['R'])
            t.append(self.calib[c]['t'])
            distCoef.append(self.calib[c]['distCoef'])
            img_paths.append(self.images[c][index])
            images.append(cv2.imread(img_paths[-1]))
        return {'K': K, 'R': R, 't': t, 'distCoef': distCoef, 'images': images, 'img_paths': img_paths}


def main():
    dataset = CMUPanopticDataset('160422_ultimatum1_small', [
                                 '00_16', '00_18', '00_19'])
    dataloader = torch.utils.data.DataLoader(dataset)
    pose_model = create_2d_pose_model()
    reid_model = create_reid_model()

    for blob_in in dataloader:
        images = [i[0] for i in blob_in['images']]
        K = [m[0] for m in blob_in['K']]
        R = [m[0] for m in blob_in['R']]
        t = [m[0] for m in blob_in['t']]
        paths = [i[0] for i in blob_in['img_paths']]

        # run 2d pose estimation
        poses_2d = []
        bboxes = []
        for image, path in zip(images, paths):
            image_numpy = image.numpy().copy()
            poses_2d_one_image = infer_2d_pose(
                pose_model, image.numpy()[:, :, ::-1], visthre=0.1)
            for pose in poses_2d_one_image:
                color = np.random.randint(0, 256, 3)
                for p in pose:
                    cv2.circle(image_numpy, (int(p[0]), int(
                        p[1])), radius=8, color=tuple(map(int, color)), thickness=-1)
            poses_2d.append(poses_2d_one_image)
            bboxes_one_image = pose_to_bbox(poses_2d_one_image, image)
            for idx, box in enumerate(bboxes_one_image):
                color = np.random.randint(0, 256, 3)
                color = tuple(map(int, color))
                cv2.rectangle(
                    image_numpy, box[:2], box[2:], color, thickness=4)
                cv2.putText(image_numpy, str(idx),
                            (box[0]-8, box[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness=4)
            cv2.imwrite(path+'.pose2d.jpg', image_numpy)
            bboxes.append(bboxes_one_image)

        # run reid
        people_bbox_ids = reid_people(reid_model, images, bboxes)

        # run RANSAC and Triangulation to generate 3D pose
        poses_3d = []
        for person_index, one_person_bbox_ids in enumerate(people_bbox_ids):
            person_images = [images[image_id]
                             for (image_id, bbox_id) in one_person_bbox_ids]
            person_image_paths = [paths[image_id]
                                  for (image_id, bbox_id) in one_person_bbox_ids]
            person_boxes = [bboxes[image_id][bbox_id]for(
                image_id, bbox_id) in one_person_bbox_ids]
            person_camera_indices = [i[0]for i in one_person_bbox_ids]
            person_poses_2d = [poses_2d[image_id][bbox_id]
                               for (image_id, bbox_id) in one_person_bbox_ids]
            person_K = [K[image_id]
                        for (image_id, bbox_id) in one_person_bbox_ids]
            person_R = [R[image_id]
                        for (image_id, bbox_id) in one_person_bbox_ids]
            person_t = [t[image_id]
                        for (image_id, bbox_id) in one_person_bbox_ids]
            person_color = tuple(map(int, np.random.randint(0, 256, 3)))
            for person_image, person_image_path, person_pose_2d, person_box in zip(person_images, person_image_paths, person_poses_2d, person_boxes):
                person_visualization = person_image.numpy().copy()
                for p in person_pose_2d:
                    cv2.circle(person_visualization, (int(p[0]), int(
                        p[1])), radius=8, color=tuple(map(int, color)), thickness=-1)
                cv2.rectangle(person_visualization,
                              person_box[:2], person_box[2:], color, thickness=4)
                cv2.imwrite(person_image_path+'.person%d.jpg' %
                            person_index, person_visualization)

            # one_person_poses_3d = ransac_triangulation(
            #    one_person_poses_2d, one_person_camera_matrices)
            # poses_3d.append(one_person_poses_3d)
        break

    return poses_3d


if __name__ == '__main__':
    main()
