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
    
    for i in range(0,4):
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


# Initialize consts to be used in iterative_LS_triangulation()
iterative_LS_triangulation_C = -np.eye(2, 3)
def iterative_LS_triangulation(u1, P1, u2, P2, tolerance=3.e-5):
    """
    Iterative (Linear) Least Squares based triangulation.
    From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
    Relative speed: 0.025

    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    "tolerance" is the depth convergence tolerance.

    Additionally returns a status-vector to indicate outliers:
        1: inlier, and in front of both cameras
        0: outlier, but in front of both cameras
        -1: only in front of second camera
        -2: only in front of first camera
        -3: not in front of any camera
    Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).

    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    """
    #convert into numpy for calculation
    u1, u2 = np.array(u1), np.array(u2)

    A = np.zeros((4, 3))
    b = np.zeros((4, 1))

    # Create array of triangulated points
    x = np.empty((4, len(u1)))
    x[3, :].fill(1)  # create empty array of homogenous 3D coordinates
    x_status = np.empty(len(u1), dtype=int)

    # Initialize C matrices
    C1 = np.array(iterative_LS_triangulation_C)
    C2 = np.array(iterative_LS_triangulation_C)

    for xi in range(len(u1)):
        #import pdb; pdb.set_trace()
        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[xi, :]
        C2[:, 2] = u2[xi, :]

        # Build A matrix
        A[0:2, :] = C1.dot(P1[0:3, 0:3])  # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])  # C2 * R2

        # Build b vector
        b[0:2, :] = C1.dot(P1[0:3, 3:4])  # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])  # C2 * t2
        b *= -1

        # Init depths
        d1 = d2 = 1.

        for i in range(10):  # Hartley suggests 10 iterations at most
            # Solve for x vector
            cv2.solve(A, b, x[0:3, xi:xi + 1], cv2.DECOMP_SVD)

            # Calculate new depths
            d1_new = P1[2, :].dot(x[:, xi])
            d2_new = P2[2, :].dot(x[:, xi])

            if abs(d1_new - d1) <= tolerance and \
                            abs(d2_new - d2) <= tolerance:
                break

            # Re-weight A matrix and b vector with the new depths
            A[0:2, :] *= 1 / d1_new
            A[2:4, :] *= 1 / d2_new
            b[0:2, :] *= 1 / d1_new
            b[2:4, :] *= 1 / d2_new

            # Update depths
            d1 = d1_new
            d2 = d2_new

        # Set status
        x_status[xi] = (i < 10 and  # points should have converged by now
                        (d1_new > 0 and d2_new > 0))  # points should be in front of both cameras
        if d1_new <= 0: x_status[xi] -= 1
        if d2_new <= 0: x_status[xi] -= 2

    return x[0:3, :].T.astype(np.float32), x_status

#input: list of poses [pose_0, pose_1, …, pose_n]  -- Each pose: [(x1, y1), (x2, y2), …, (xm, ym)] for m joints
#       list of camera matrices - [cam1, cam2, cam3, cam4] -- each camaera is a numpy array 3x4
def ransac_triangulaton(poses_2d, camera_matrices, threshold=20):
    debug = True

    #init 
    best_joints_3Dpts = None
    best_joint_inliers = 0
    best_errors = float("inf")
    best_combination = None
    
    #Generate all the combinations
    camera_num = len(camera_matrices)
    all_camera_combinations = []
    for i in range(camera_num):
        for j in range(i + 1, camera_num):
            all_camera_combinations.append([i,j])

    #debug
    if debug:
        all_errors = []
        all_inlier_num = []
    
    #Pick a combination of 2 cameras to triangulate points
    for combination in all_camera_combinations:
        #init
        M1, M2  = camera_matrices[combination[0]], camera_matrices[combination[1]]
        pose1, pose2 = poses_2d[combination[0]], poses_2d[combination[1]]
        cur_joint_inliers = 0
        cur_allCam_erros = 0

        #Triangulate the points
        cur_joints_3Dpts, vis = iterative_LS_triangulation(pose1, M1, pose2, M2)
        #cur_joints_3Dpts = triangulate_cv2(pose1, M1, pose2, M2)

        if debug:
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(projection='3d')
            show3Dpose(cur_joints_3Dpts, ax, radius=128)
            plt.savefig('combination_' + str(combination[0]) + ',' + str(combination[1]) + str('_triangulate.png'))

        #Back project to the 2D poses
        cur_joints_4Dpts = np.hstack((cur_joints_3Dpts, np.ones((1,16)).T))
        for cam_idx in range(len(camera_matrices)):
            #Debug 
            if debug:
                img = cv2.imread(str(cam_idx) + '.jpg')
            
            cur_cam_error = 0
            #Per joint comparison
            for joint_idx in range(len(cur_joints_4Dpts)):
                projected_2Dpt = camera_matrices[cam_idx] @ cur_joints_4Dpts[joint_idx] # M @ joint
                projected_2Dpt = projected_2Dpt/projected_2Dpt[2] #Normalized to homo. coordinates

                #Calculate the Euclidean distance and use that as the error/diff
                dist = np.linalg.norm(projected_2Dpt[:2] - np.array(poses_2d[cam_idx][joint_idx])) #the projected point - the pose estimated point

                #Increase inliner number if projected joint has Euclidean distance smaller than the threshold (with the estimated pose joint)
                if (dist < threshold):
                    cur_joint_inliers += 1

                cur_cam_error += dist
        
                #debug
                if debug:
                    x = projected_2Dpt[0]
                    y = projected_2Dpt[1]
                    img = cv2.circle(np.array(img), (int(x),int(y)) ,5, (0,0,255), 5)

            #Accumulate the error for all the cameras (in this case: 4 cameras)
            cur_allCam_erros += cur_cam_error

            #debug
            if debug:
                cv2.imwrite('combination_' + str(combination[0]) + ',' + str(combination[1]) + '_cam' + str(cam_idx) + '_back_projected.jpg', img)
                all_inlier_num.append(cur_joint_inliers)

        all_errors.append(cur_allCam_erros)
        
        
        #Record the best triangulation
        if (cur_joint_inliers > best_joint_inliers): ##or (cur_joint_inliers == best_joint_inliers and cur_allCam_erros < best_errors)):
            best_errors = cur_allCam_erros
            best_joints_3Dpts = cur_joints_3Dpts
            best_joint_inliers = cur_joint_inliers
            best_combination = combination

    #import pdb; pdb.set_trace()
    return best_combination


'''
From 16720 HW
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
    poses = readPoses('pose0.pickle','pose1.pickle','pose2.pickle','pose3.pickle')
    #import pdb; pdb.set_trace()
    #P, status = iterative_LS_triangulation(poses[0], camera_matrices[0], poses[3], camera_matrices[3])
    #P, status = iterative_LS_triangulation(poses[2], camera_matrices[2], poses[3], camera_matrices[3])
    #P = triangulate(poses[1], camera_matrices[1], poses[3], camera_matrices[3])
    #P = triangulate_cv2(poses[1], camera_matrices[1], poses[3], camera_matrices[3])
    #P = triangulate_cv2(poses[0], camera_matrices[0], poses[2], camera_matrices[2])

    ransac_triangulaton(poses, camera_matrices) 
    with open('P.pickle', 'wb') as handle:
        pickle.dump(P, handle)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(projection='3d')
    show3Dpose(P, ax, radius=128)
    plt.savefig('triangulate.png')