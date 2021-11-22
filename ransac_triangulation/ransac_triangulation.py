#Get 2d pose positions
#   - 
#Get Camera Matrix 
#Triangulation
#Apply Ransac

import numpy as np
import json
import pickle
from matplotlib import pyplot as plt
import cv2

# draw the body keypoint and lims
def show3Dpose(channels, ax, radius=40, mpii=1, lcolor='#ff0000', rcolor='#0000ff'):
    vals = channels

    if mpii == 0: # h36m with mpii joints
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                       [5, 6], [0, 8], [8, 9], [9, 10],
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    elif mpii == 1: # only mpii
        connections = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7],
                       [7, 8], [8, 9], [7, 12], [12, 11], [11, 10], [7, 13], [13, 14], [14, 15]]
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
    else: # default h36m
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                       [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    for ind, (i,j) in enumerate(connections):
        if (np.isnan(vals[i,2]) or  np.isnan(vals[j,2])):  #i.e. not detected
            pass
        else:
            x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
            ax.plot(x, y, z, lw=2, c=lcolor if LR[ind] else rcolor)

    #RADIUS = radius  # space around the subject
    #if mpii == 1:
    #    xroot, yroot, zroot = vals[6, 0], vals[6, 1], vals[6, 2]
    #else:
    #    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    #ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    #ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    #ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def readPoses(pose1_json_path, pose2_json_path, pose3_json_path, pose4_json_path):
    poses = []
    with open(pose1_json_path, 'rb') as handle:
        pose1 = pickle.load(handle)
    with open(pose2_json_path, 'rb') as handle:
        pose2 = pickle.load(handle)
    with open(pose3_json_path, 'rb') as handle:
        pose3 = pickle.load(handle)
    with open(pose4_json_path, 'rb') as handle:
        pose4 = pickle.load(handle)

    poses.append(pose1)
    poses.append(pose2)
    poses.append(pose3)
    poses.append(pose4)
    return poses

def readAndGenerateCameraMatrice(camera_json_path):
    with open(camera_json_path) as f:
        camera_dict = json.loads(f.read())
    camera_matrices = []
    
    for i in range(1,5):
        idx = str(i)
        R = np.array(camera_dict[idx]['R'])
        t = np.array(camera_dict[idx]['t']).reshape(3,1)
        Rt = np.concatenate((R, t), axis=1)

        f = camera_dict[idx]['f']
        c = camera_dict[idx]['c']
        K = np.array([[f[0],0,c[0]], [0 , f[1], c[1]], [0,0,1]])

        M = K @ Rt
        camera_matrices.append(M)
    return camera_matrices

#input: list of dictionary 
def ransac_triangulaton(poses_2d, camera_matrices, threshold=0.5):
    
    #Per point triangulate?
    return


'''
Q3.3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(pts1, C1, pts2, C2):
    # Replace pass by your implementation

    #convert into numpy for calculation
    pts1, pts2 = np.array(pts1), np.array(pts2)

    #init
    num_of_points= pts1.shape[0]
    P = np.empty([num_of_points, 3])

    #Estimate the P
    for i in range(num_of_points):
        if (pts1[i] is None or pts2[i] is None): #i.e. the pose estimation network cannot detect the point
            p = np.array([None,None,None])
        else:
            x1, y1, x2, y2 = pts1[i][0] , pts1[i][1] , pts2[i][0] , pts2[i][1]
            r1 = y1*C1[2] - C1[1]
            r2 = C1[0]    - x1*C1[2]
            r3 = y2*C2[2] - C2[1]
            r4 = C2[0]    - x2*C2[2]
            A = np.vstack((r1,r2,r3,r4))

            #SVD , pick best and normalize
            U, S, V = np.linalg.svd(A, full_matrices=True)
            p = V[-1]
            p = p/p[3]
            
        #Record the points
        P[i, :] = p[0:3]

    return P


def triangulate_cv2(pts1, C1, pts2, C2):
    pts1 = np.array(pts1).astype(np.float32).T
    pts2 = np.array(pts2).astype(np.float32).T
    pts4d = cv2.triangulatePoints(C1, C2, pts1, pts2)
    points3d = (pts4d[:3, :]/pts4d[3, :]).T
    return points3d

if __name__ == '__main__':
    camera_matrices = readAndGenerateCameraMatrice('subject1_camera.json')
    poses = readPoses('pose1.pickle','pose2.pickle','pose3.pickle','pose4.pickle')
    #P = triangulate_cv2(poses[1], camera_matrices[1], poses[3], camera_matrices[3])
    #P = triangulate_cv2(poses[0], camera_matrices[0], poses[2], camera_matrices[2])

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(projection='3d')
    show3Dpose(P, ax, radius=128)
    plt.savefig('triangulate.png')