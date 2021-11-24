import pickle as pkl
from os.path import exists
import numpy as np
import os
import math
import shutil
import sys

sys.path.append('./EpipolarPose')
sys.path.append('./EpipolarPose/lib')

file_name = '/home/ubuntu/16822-project/EpipolarPose/data/h36m/annot/valid.pkl'
with open(file_name, 'rb') as anno_file:
    gt_anno = pkl.load(anno_file)

SEQ_NAME_MAPPING = {
    'S9_Photo': 'S9_TakingPhoto', 
    'S9_Photo_1': 'S9_TakingPhoto_1', 
    'S9_WalkDog': 'S9_WalkingDog',
    'S9_WalkDog_1': 'S9_WalkingDog_1',
    'S11_Photo': 'S11_TakingPhoto', 
    'S11_Photo_1': 'S11_TakingPhoto_1', 
    'S11_WalkDog': 'S11_WalkingDog',
    'S11_WalkDog_1': 'S11_WalkingDog_1',   
}

SEQ_NAME_LIST = ['S9_Photo', 'S9_Photo_1', 'S9_WalkDog', 'S9_WalkDog_1', 'S11_Photo', 'S11_Photo_1', 'S11_WalkDog', 'S11_WalkDog_1']

#Construct a dictionary for finding the closest index
dict_ActionImageIdx = {} 
for r, d, f in os.walk('/home/ubuntu/16822-project/data/h36m/images'):
    for file in f:
        for name in SEQ_NAME_LIST: #Replace the name to use the notation of the dataset that we currently have
            if name in file:
                file.replace(name, SEQ_NAME_MAPPING[name])

        if len(file.split('_')) == 3:
            subject_cam_sequence_name = file.split('_')[0] + '_' + file.split('_')[1]
        else:
            subject_cam_sequence_name = file.split('_')[0] + '_' + file.split('_')[1] + '_' + file.split('_')[2] #S1_TakingPhoto_1.60457274_000121.jpg

        idx = file.split('_')[-1][:6]
        if subject_cam_sequence_name in dict_ActionImageIdx:
            dict_ActionImageIdx[subject_cam_sequence_name].append(int(idx))
        else:
            dict_ActionImageIdx[subject_cam_sequence_name] = []
            dict_ActionImageIdx[subject_cam_sequence_name].append(int(idx))

#Sort the index 
debug_2 = set()
for k in dict_ActionImageIdx:
    dict_ActionImageIdx[k] = np.sort(np.array(dict_ActionImageIdx[k]))
    #if 'S9' in k or 'S11' in k:
    #    debug_2.add(k)

#A method to find the neareset index 
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


##############################
# Start searching the folder #
##############################

debug_list = set()
image_stored_path = "/home/ubuntu/16822-project/data/h36m/"
#Read all the images name and find if it exists
for i in range(len(gt_anno)):
    image_path = gt_anno[i]['image']
    subject = image_path.split('/')[1]
    cam_action_name = image_path.split('/')[-2] + "/" + image_path.split('/')[-1]
    cam_action_name = cam_action_name.replace('/','_')
    image_path = "images/" + subject + "/" + subject + "_" + cam_action_name
    
    #replace the name to match the dataset
    for name in SEQ_NAME_LIST:
        if name in image_path:
            image_path.replace(name, SEQ_NAME_MAPPING[name]) 
    
    #Check if the image exists in our current dataset
    if exists(image_stored_path + image_path):
        continue
    else:
        import pdb; pdb.set_trace()
        print("Missing : " + image_path)
        print("Copying the most recent/closest image")

        #Find the name and the idx
        idx = int(image_path.split('.')[-2].split('_')[-1])
        subject_cam_sequence_name = image_path.split('/')[-2] + '_' + image_path.split('/')[-1]
        shift = len(subject) + 1
        subject_cam_sequence_name = subject_cam_sequence_name[shift:] #remove the duplicate S12/S1 etc
        
        if len(subject_cam_sequence_name.split('_')) == 3:
            subject_cam_sequence_name = subject_cam_sequence_name.split('_')[0] + '_' + subject_cam_sequence_name.split('_')[1]
        else:
            subject_cam_sequence_name = subject_cam_sequence_name.split('_')[0] + '_' + subject_cam_sequence_name.split('_')[1] + '_' + subject_cam_sequence_name.split('_')[2] #S1_TakingPhoto_1.60457274_000121.jpg
        
        #Find the closest index
        nearest_idx = find_nearest(dict_ActionImageIdx[subject_cam_sequence_name], idx)
        
        #Start copying
        padded_idx = str(nearest_idx).zfill(6)  
        src = (image_stored_path + image_path)[:-10] + padded_idx + '.jpg' #replace the last idx
        dst = image_stored_path + image_path
        shutil.copyfile(src, dst)
        
