import numpy as np
from IPython import embed
import cv2
import random
import sys
sys.path.append('panoptic-toolbox/python')  # nopep8
from panutils import projectPoints


def solve_task(task):
    def make_constraints(m, p):
        return [
            p[1]*m[2]-m[1],
            m[0]-p[0]*m[2]
        ]

    def svd_triangulate(c):
        u, s, vt = np.linalg.svd(c)
        return (vt[-1]/vt[-1, -1])[:3]
    c = []
    for K, R, t, distCoef, p in task:
        p = K@np.concatenate((cv2.undistortPoints(p, K, distCoef)[0, 0], [1]))
        M = K@np.concatenate((R, t), 1)
        c += make_constraints(M, p)
    return svd_triangulate(c)


def reproject(K, R, t, distCoef, P):
    p = projectPoints(np.matrix(P[:, None]), np.matrix(
        K), np.matrix(R), t.reshape((3, 1)), distCoef)
    return p[0].item(), p[1].item()


def reprojection_error(task, P):
    error = []
    for K, R, t, distCoef, p in task:
        p_ = reproject(K, R, t, distCoef, P)
        error.append(((np.array(p)-np.array(p_))**2).sum())
    return error


def ransac_triangulation(person_poses_2d, person_K, person_R, person_t, person_distCoef, num_iter=256, thres=128):
    tasks = [[]for i in range(len(person_poses_2d[0]))]
    for pose_2d, K, R, t, distCoef in zip(person_poses_2d, person_K, person_R, person_t, person_distCoef):
        K = K.numpy()
        R = R.numpy()
        t = t.numpy()
        distCoef = distCoef.numpy()
        for i, p in enumerate(pose_2d):
            tasks[i] += [(K, R, t, distCoef, p)]
    pose_3d = []
    for task in tasks:
        print('before ransac:', np.mean(
            reprojection_error(task, solve_task(task))))
        best_P = None
        best_cnt = None
        best_error = None
        for i in range(num_iter):
            sample = random.sample(task, 2)
            this_P = solve_task(sample)
            this_cnt = (np.array(reprojection_error(
                task, this_P)) < thres).sum()
            this_error = np.mean([t for t in reprojection_error(
                task, this_P) if t < thres])
            if best_cnt is None or this_cnt > best_cnt or (this_cnt == best_cnt and this_error < best_error):
                best_cnt = this_cnt
                best_P = this_P
                best_error = this_error
        best_task = [task[i]for i in np.argwhere(
            np.array(reprojection_error(task, best_P)) < thres)[:, 0]]
        print(len(task), len(best_task))
        if len(best_task) < 2:
            print('oops')
            pose_3d.append(None)
            continue
        best_P = solve_task(best_task)
        print('after ransac:', np.mean(
            reprojection_error(best_task, best_P)))
        pose_3d.append(best_P)
    reprojection = []
    for pose_2d, K, R, t, distCoef in zip(person_poses_2d, person_K, person_R, person_t, person_distCoef):
        K = K.numpy()
        R = R.numpy()
        t = t.numpy()
        distCoef = distCoef.numpy()
        r = []
        for P in pose_3d:
            if P is None:
                r.append(None)
            else:
                r.append(reproject(K, R, t, distCoef, P))
        reprojection.append(r)
    return pose_3d, reprojection
