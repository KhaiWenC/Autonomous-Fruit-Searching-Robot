
############### run this scrpt using the command: ###################################
#-------------- python generateGroundTruth.py <output file name> -------------------#
############### without the angle brackets ##########################################

import numpy as np
import json
from pathlib import Path
import os

def load_markers(fname='marker_pose_after_align.txt'):
    base_dir = Path('./')
    aruco_true_pos = np.empty([10, 2])

    with open(base_dir/fname,'r') as f:
        try:
            gt_dict = json.load(f)
        except ValueError as e:
            with open(self.fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id - 1][0] = x
                    aruco_true_pos[marker_id - 1][1] = y
                    
    return aruco_true_pos


def load_fruits(fname='targets.txt'):
    fruit_list = []
    obs_cor = []
    base_dir = Path('./')
    with open(base_dir/'lab_output/'/fname,'r') as map_file:
        map_attributes = json.load(map_file)
    key = [x for x in map_attributes.keys()]
    for k in range(len(key)):
        res = ""
        for j in key[k]:
            if j.isalpha():
                res="".join([res,j])
        fruit_list.append(res)
        obs_fruit_x = map_attributes[key[k]]['x']
        obs_fruit_y = map_attributes[key[k]]['y']                            
        obs_fruit_x = round(obs_fruit_x, 1)
        obs_fruit_y = round(obs_fruit_y, 1)
        obs_cor.append([obs_fruit_x, obs_fruit_y])
        return fruit_list, obs_cor
		
def save(fname="M5_true_map.txt"):
    aruco_true_pos = load_markers()
    fruit_list, obs_cor = load_fruits()
    base_dir = Path('./')
    d = {}
    for i in range(10):
        d['aruco' + str(i+1) + '_0'] = {'x': round(aruco_true_pos[i][0], 1), 'y':round(aruco_true_pos[i][1], 1)}
    for i in range(len(fruit_list)):     
        d[fruit_list[i] + '_0'] = {'x': obs_cor[i][0], 'y':obs_cor[i][1]}
    map_attributes = d
    with open(base_dir/fname,'w') as map_file:
        json.dump(map_attributes, map_file, indent=2)
    print("True Map Generated!!!")
            

if __name__ == "__main__":
    save()
