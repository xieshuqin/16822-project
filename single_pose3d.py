import numpy as np
from IPython import embed


def ransac_triangulation(person_poses_2d, person_K, person_R, person_t, person_distCoef):
    def make_constraints(m, p):
        return [
            p[1]*m[2]-m[1],
            m[0]-p[0]*m[2]
        ]
    constraints = [[]for i in range(len(person_poses_2d[0]))]
    for pose_2d, K, R, t, distCoef in zip(person_poses_2d, person_K, person_R, person_t, person_distCoef):
        K = K.numpy()
        R = R.numpy()
        t = t.numpy()
        distCoef = distCoef.numpy()
        M = K@np.concatenate((R, t), 1)
        for i, p in enumerate(pose_2d):
            constraints[i] += make_constraints(M, p)
    pose_3d = []
    error = []
    for c in constraints:
        u, s, vt = np.linalg.svd(c)
        error.append(s[-1])
        pose_3d.append((vt[-1]/vt[-1, -1])[:3])
    reprojection = []
    for pose_2d, K, R, t, distCoef in zip(person_poses_2d, person_K, person_R, person_t, person_distCoef):
        K = K.numpy()
        R = R.numpy()
        t = t.numpy()
        distCoef = distCoef.numpy()
        M = K@np.concatenate((R, t), 1)
        r = []
        for P in pose_3d:
            p = M@np.concatenate((P, [1]))
            r.append((p[0]/p[-1], p[1]/p[-1]))
        reprojection.append(r)
    print('mean error:', np.mean(error))
    return pose_3d, reprojection
