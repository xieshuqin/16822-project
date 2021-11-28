import numpy as np
from IPython import embed
import cv2
import random
import sys
sys.path.append('panoptic-toolbox/python')  # nopep8
from panutils import projectPoints
import torch


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


def quality(pose_3d):
    for p in pose_3d:
        if p is None:
            return -1
    pairs = [
        [[5, 7], [6, 8]],
        [[7, 9], [8, 10]],
        [[5, 11], [6, 12]],
        [[11, 13], [12, 14]],
        [[13, 15], [14, 16]]
    ]

    def dis(u, v):
        return ((pose_3d[u]-pose_3d[v])**2).sum()**0.5
    qu = 0
    for p, q in pairs:
        qu += (dis(*p)-dis(*q))**2
    return qu


def myProjectPoints(X, K, R, t, Kd):
    if type(X) == torch.Tensor:
        R = torch.tensor(R)
        t = torch.tensor(t)
    x = R@X + t[:, 0]
    x = x/x[2]
    r = x[0]*x[0] + x[1]*x[1]
    x0 = x[0]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + \
        2*Kd[2]*x[0]*x[1] + Kd[3]*(r + 2*x[0]*x[0])
    x1 = x[1]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + \
        2*Kd[3]*x[0]*x[1] + Kd[2]*(r + 2*x[1]*x[1])
    x0 = K[0, 0]*x0 + K[0, 1]*x1 + K[0, 2]
    x1 = K[1, 0]*x0 + K[1, 1]*x1 + K[1, 2]
    if type(X) == torch.Tensor:
        x = torch.stack((x0, x1))
    else:
        x = np.array([x0, x1])
    return x


def myReproject(K, R, t, distCoef, P):
    p = myProjectPoints(P, K, R, t, distCoef)
    return p


def myReprojection_error(task, P):
    error = []
    for K, R, t, distCoef, p in task:
        p_ = myReproject(K, R, t, distCoef, P)
        p = np.array(p)
        if type(p_) == torch.Tensor:
            p = torch.tensor(p)
        error.append(((p-p_)**2).sum())
    return error


def recompute_reprojection_error(tasks, pose_3d):
    e = []
    for task, P in zip(tasks, pose_3d):
        re = myReprojection_error(task, P)
        re = sum(re)/len(re)
        e.append(re)
    return sum(e)/len(e)


def joint_opt(pose_3d, tasks):
    for p in pose_3d:
        if p is None:
            print('skipping joint opt for incomplete pose')
            return pose_3d
    print("pose quality:", quality(pose_3d))
    print("reprojection error:", recompute_reprojection_error(tasks, pose_3d))
    pose_3d = torch.tensor(pose_3d, requires_grad=True)
    for i in range(32):
        loss = quality(pose_3d)+recompute_reprojection_error(tasks, pose_3d)
        loss.backward()
        pose_3d.data -= 0.01*pose_3d.grad
        print("pose quality:", quality(pose_3d))
        print("reprojection error:", recompute_reprojection_error(tasks, pose_3d))
    return pose_3d.detach().numpy()


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
    best_tasks = []
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
            best_tasks.append(None)
            continue
        best_P = solve_task(best_task)
        print('after ransac:', np.mean(
            reprojection_error(best_task, best_P)))
        pose_3d.append(best_P)
        best_tasks.append(best_task)
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
    pose_3d = joint_opt(pose_3d, best_tasks)
    return pose_3d, reprojection
