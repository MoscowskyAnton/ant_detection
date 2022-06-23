import json
import numpy as np
import glob
import os
import shutil
import argparse
import numpy as np

def read_json(root_path, path, max_obj = 20):
    
    keypoints_path = root_path + '/keypoints'
    bboxes_path = root_path + '/bboxes'
    dir_size = len(glob.glob(root_path + '/images' + '/*'))
    for i in [keypoints_path, bboxes_path]:
        if not os.path.exists(i):
            os.mkdir(i)
        else:
            shutil.rmtree(i)
            os.mkdir(i)
    
    #head_list = [] # shape [N_im, max_obj, 2]
    #abdomen_list = [] # shape [N_im, max_obj, 2]
    #bboxes_list = [] # shape [N_im, max_obj, 4]
    
    with open(path) as f:
        data = json.load(f)
    for img in data:
        number = int(img['img'][img['img'].rfind('e') + 1 : img['img'].rfind('.')])
        single_head_list = [0] * max_obj
        single_abdomen_list = [0] * max_obj
        count_head = 0
        count_abdimen = 0
        
        if not 'kp-1' in img:
            print(f"Skipped {img['img']}")
            continue
        for kps in img['kp-1']:
            label = kps['keypointlabels'][0]
            kp_x = kps['x']
            kp_y = kps['y']
            
            if label == 'Abdomen':
                single_abdomen_list[count_abdimen] = [conv_x(kp_x), conv_y(kp_y)]
                count_abdimen += 1
            elif label == 'Head':
                single_head_list[count_head] = [conv_x(kp_x), conv_y(kp_y)]
                count_head += 1
                
        #head_list.append(single_head_list)
        #abdomen_list.append(single_abdomen_list)
        
        single_bboxes_list = [0] * max_obj
        for i, bboxes in enumerate(img['label']):
            xmin = bboxes['x']
            ymin = bboxes['y']
            xmax = bboxes['x'] + bboxes['width']
            ymax = bboxes['y'] + bboxes['height']
            
            if bboxes['rectanglelabels'][0] == 'Ant':
                single_bboxes_list[i] = [conv_x(xmin), conv_y(ymin), conv_x(xmax), conv_y(ymax)]
        
        
        # ТУТ СОПОСТАВЛЕНИЕ И СОХРАНЕНИЕ ПО ИМЕНИ ИЗОБРАЖЕНИЯ + объединение
        compared_h, compared_a, _ = correct_comparison(single_head_list, single_abdomen_list, single_bboxes_list)
        single_keypoints_list = join_keypoints(compared_h, compared_a)
        bb_filename = bboxes_path + '/bbox' + str(number) + '.txt'
        write_txt(single_bboxes_list, bb_filename)
        k_filename = keypoints_path + '/keypoint' + str(number) + '.txt'
        write_txt(single_keypoints_list, k_filename)
        print(f'№ {number}, ants: {np.count_nonzero(single_bboxes_list)}')
        #bboxes_list.append(single_bboxes_list)
    
    #print(f'bboxes_list {len(bboxes_list), len(bboxes_list[0]), len(bboxes_list[0][0])}')
    #print(f'\nhead_list {len(head_list), len(head_list[0]), len(head_list[0][0])}')
    #print(f'\nabdomen_list {len(abdomen_list), len(abdomen_list[0]), len(abdomen_list[0][0])}')
    
    return 0


def correct_comparison(h, a, b):
    #N_im = len(h)
    N_ob = len(h)
    single_im_h = [0] * N_ob
    single_im_a = [0] * N_ob
    for j in range(N_ob):
        if b[j] == 0:
            break
        else:
            xmin, ymin, xmax, ymax = b[j][0], b[j][1], b[j][2], b[j][3]
            for g in range(N_ob):
                if h[g] != 0:
                    x_h, y_h = h[g][0], h[g][1]
                    if (xmin < x_h < xmax) and (ymin < y_h < ymax):
                        single_im_h[j] = [x_h, y_h]
                        
            for g in range(N_ob):
                if a[g] != 0:
                    x_a, y_a = a[g][0], a[g][1]
                    if (xmin < x_a < xmax) and (ymin < y_a < ymax):
                        single_im_a[j] = [x_a, y_a]
    
    return single_im_h, single_im_a, b
        
 
def write_txt(list_of_lists, filename):
    str_list = []
    for i in list_of_lists:
        if i != 0:
            int_s = [int(a) for a in i]
            s = ' '.join(map(str, int_s)) + "\n"
            str_list.append(s)
    with open(filename, 'w') as file:
        file.writelines(str_list)
        file.close()
        
def conv_x(old):
    old_min = new_min = 0
    old_range = 100 - 0  
    new_range = 1920 - 0 # 1920 - 0
    
    converted = (((old - old_min) * new_range) / old_range) + new_min
    return converted

def conv_y(old):
    old_min = new_min = 0
    old_range = 100 - 0  
    new_range = 1080 - 0 # 1080 - 0
    
    converted = (((old - old_min) * new_range) / old_range) + new_min
    return converted
        
def join_keypoints(head, abdomen):
    single_im_k = []
    for j in range(len(head)):
        if abdomen[j] != 0:
            x_a, y_a = abdomen[j][0], abdomen[j][1]
            x_h, y_h = head[j][0], head[j][1]
            single_im_k.append([x_a, y_a, x_h, y_h])
                
        else:
            single_im_k.append(0)
    return single_im_k
    
 
def create_dataset(root_path, json_path):
    keypoints_path = root_path + '/keypoints'
    bboxes_path = root_path + '/bboxes'
    dir_size = len(glob.glob(root_path + '/images' + '/*'))
    for i in [keypoints_path, bboxes_path]:
        if not os.path.exists(i):
            os.mkdir(i)
        else:
            shutil.rmtree(i)
            os.mkdir(i)
    
    head_list, abdomen_list, bboxes_list = read_json(json_path)
    head_list, abdomen_list, bboxes_list = correct_comparison(head_list, abdomen_list, bboxes_list)
    keypoints = join_keypoints(head_list, abdomen_list)
    real_im_number = 0
    for i in reversed(range(dir_size)):
        bb_filename = bboxes_path + '/bbox' + str(real_im_number) + '.txt'
        write_txt(bboxes_list[i], bb_filename)
        k_filename = keypoints_path + '/keypoint' + str(real_im_number) + '.txt'
        write_txt(keypoints[i], k_filename)
        real_im_number += 1
        
        
if __name__ == '__main__':
    #root_path - is a forder, where folger with images and a json file with annotation lies. it will create there two folders for bboxes amd keypoints txt files.
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', nargs='?', default='/home/ubuntu/ant_detection/TRAIN_on_real', help="Specify directory to create dataset", type=str)
    parser.add_argument('json_path', nargs='?', default='/home/ubuntu/ant_detection/TRAIN_on_real/new.json', help="Specify path to json file", type=str)
    args = parser.parse_args()
    ROOT = args.root_path
    JSON = args.json_path
    read_json(ROOT, JSON)
    #create_dataset(ROOT, JSON)
