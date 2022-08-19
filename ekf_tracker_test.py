import yaml
import argparse
from RCNN_overlay_test import read_yaml
import matplotlib.pyplot as plt
import numpy as np

ARROW_LEN = 50
D_ANT_COLOR = 'k'
D_ANT_SYM = 'o'
ANT_SCORE_MIN = 0.0

def proceed_frame(frame, H, ax):
    ants = get_ants(frame)
    
    plot_ants(ax, ants, H)

def get_ants(frame):
    ants = []
    for k, v in frame.items():
        for kp, score in zip(v['keypoints'], v['bboxes_scores']):
            if score < ANT_SCORE_MIN:
                continue
            cx = (kp[0][0] + kp[1][0])/2
            cy = (kp[0][1] + kp[1][1])/2            
            a = np.arctan2(kp[1][1]-kp[0][1], kp[1][0]-kp[0][0])
            ant = [score, cx, cy, a, 0, 0]
            ants.append(ant)
    return np.array(ants)

def plot_ants(ax, ants, H):            
    ants = get_ants(frame)                                
    for i in range(ants.shape[0]):
        ax.plot(ants[i,1], H-ants[i,2], D_ANT_COLOR+D_ANT_SYM, alpha = ants[i,0])
        ax.arrow(ants[i,1], H-ants[i,2], ARROW_LEN * np.cos(ants[i,3]), ARROW_LEN * np.sin(ants[i,4]), color = D_ANT_COLOR, alpha = ants[i,0])
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', nargs='?', default='//home/anton/Projects/ant_detection/videos/short.yml', help="Full path to yaml-file with ant data", type=str)
    args = parser.parse_args()
    print(f"Loading data from {args.yaml_path}...")
    ANT_DATA = read_yaml(args.yaml_path)    
    #print(d.keys() for d in ANT_DATA['frames'])
            
    fig, ax = plt.subplots()    
    
    for frame in ANT_DATA['frames']:
        ax.clear()
        ax.set_title(f"Frame {list(frame.keys())[0]}")
        ax.set_xlim(0, ANT_DATA['weight'])
        ax.set_ylim(0, ANT_DATA['height'])
        proceed_frame(frame, ANT_DATA['height'], ax)
        plt.pause(0.1)
    
