import torchvision
from overlay_intersection import crop_one_im, read_boxes, resize_bboxes_kps 
import numpy as np
import argparse
from RCNN_model import get_model
import cv2
import os
import glob
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import yaml
import time
from collections import OrderedDict
    
def get_out_kp_bb(out, left_x, left_y, conf_threshold, iou_threshold):
    # Функция для маштабирования предказанных координат на новый диапазон
    scores = out[0]['scores'].detach().cpu().numpy()
    high_scores_idxs = np.where(scores > conf_threshold)[0].tolist() # Indexes of boxes with scores > 0.7
    post_nms_idxs = torchvision.ops.nms(out[0]['boxes'][high_scores_idxs], out[0]['scores'][high_scores_idxs], iou_threshold).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)
    
    my_scores = out[0]['scores'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()
    
    keypoints = []
    bboxes = []
    
    for kps in out[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        int_kps = [list(map(int, kp[:2])) for kp in kps]
        x_a, y_a = int_kps[0][0], int_kps[0][1]
        x_h, y_h = int_kps[1][0], int_kps[1][1]
        keypoints.append([[x_a + left_x, y_a + left_y], [x_h + left_x, y_h + left_y]])

    for bbox in out[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        xmin, ymin, xmax, ymax = map(int, bbox.tolist())
        bboxes.append([xmin + left_x, ymin + left_y, xmax + left_x, ymax + left_y])
    
    return keypoints, bboxes, my_scores


def visualize(image, bboxes, keypoints, scores, image_original=None, bboxes_original=None, keypoints_original=None, show_flag = True):
    # Рисует на изображении предсказанные и настоящие боксы и ключевые точки.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # !!!!! ВОЗМОЖНО НУЖНО УДАЛИТЬ !!!!!
    fontsize = 12
    keypoints_classes_ids2names = {0: 'A', 1: 'H'}
    for idx, bbox in enumerate(bboxes):
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (255,0,0), 2)
        org = (bbox[0] - 3, bbox[1] - 3)
        image = cv2.putText(image.copy(), str(round(scores[idx], 2)), org , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            overlay = image.copy()
            overlay = cv2.circle(overlay, tuple(kp), 2, (255,0,0), 10)
            # try to make transparent
            alpha = 0.5
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    if image_original is None and keypoints_original is None:
        if show_flag:
            plt.figure(figsize=(40,40))
            plt.imshow(image)
            plt.show(block=True)
        else:
            return image

    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)
        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                orig_overlay = image_original.copy()
                orig_overlay = cv2.circle(orig_overlay, tuple(kp), 2, (0,255,0), 10)
                alpha = 0.5
                image_original = cv2.addWeighted(orig_overlay, alpha, image_original, 1 - alpha, 0)
                #image_original = cv2.circle(image_original, tuple(kp), 2, (0,255,0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                
        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original annotations', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Predicted annotations', fontsize=fontsize)
        plt.show(block=True)
        
        
def intersection_over_union(boxA, boxB):
    #Считает IoU для двух боксов
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    iou = round(iou, 2)
    return iou  
    
 
def chose_right_bbox(bbox, keypoints, scores):
    #Отбор не совпадающих с другим изображением боксов
    new_bb = []
    new_kp = []
    new_scores = []
    for i in range(len(bbox)):
        if type(bbox[i]) != int:
            new_bb.append(bbox[i])
            new_kp.append(keypoints[i])
            new_scores.append(scores[i])
    return new_bb, new_kp, new_scores
            

def show_video(path):
    #Показ видео
    cap = cv2.VideoCapture(path)
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("video.mp4",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    
    
def selection(boxA, boxB, kpA, kpB, scA, scB, nms_threshold):
    #Функция, фильтрующая накладывающиеся боксы одного объекта разных частей изображения
    if len(boxA) == 0 or len(boxB) == 0:
        if len(boxA) == 0 and len(boxB) == 0:
            return [], [], []
        elif len(boxA) == 0:
            return boxB, kpB, scB
        elif len(boxB) == 0:
            return boxA, kpA, scA
    else:
        #Матрица Джакардова индекса для всех элементов двух частей изображения
        iou_matrix = [[0 for i in range(len(boxB))] for j in range(len(boxA))]

        for i in range(len(iou_matrix)):
            for j in range(len(iou_matrix[0])):
                iou_matrix[i][j] = intersection_over_union(boxA[i], boxB[j])
        
        mean_bb = []
        mean_kp = []
        mean_sc = []
        
        while max(map(max, iou_matrix)) >= nms_threshold:
            for i in range(len(boxA)):
                for j in range(len(boxB)):
                    if iou_matrix[i][j] == max(map(max, iou_matrix)) and iou_matrix[i][j] >= nms_threshold:
                        #print(f'iou: {iou_matrix[i][j]}, \nboxA[i]: {boxA[i]}, \nboxB[j]: {boxB[j]}')
                        #Удаляем совпадающие боксы и заменяем их средним значением
                        xmin = int((boxA[i][0] + boxB[j][0]) / 2)
                        ymin = int((boxA[i][1] + boxB[j][1]) / 2)
                        xmax = int((boxA[i][2] + boxB[j][2]) / 2)
                        ymax = int((boxA[i][3] + boxB[j][3]) / 2)
                        mean_bb.append([xmin, ymin, xmax, ymax])
                        #То же самое с ключевыми точками
                        x_a = int((kpA[i][0][0] + kpB[j][0][0]) / 2)
                        y_a = int((kpA[i][0][1] + kpB[j][0][1]) / 2)
                        x_h = int((kpA[i][1][0] + kpB[j][1][0]) / 2)
                        y_h = int((kpA[i][1][1] + kpB[j][1][1]) / 2)
                        mean_kp.append([[x_a, y_a], [x_h, y_h]])
                        
                        mean_sc.append((scA[i] + scB[j]) / 2)
                        
                        boxA[i] = -1
                        boxB[j] = -1
                        #"Зануляем" соответствущие элементы в матрице
                        iou_matrix[i] = [-1] * len(boxB)
                        for row in iou_matrix:
                            row[j] = -1
        
        #Отбираем несовпадающие боксы и ключевые точки
        #print(f'boxB {boxB}, \nkpB {kpB}')
        #print(f'boxA {boxA}, \nkpA {kpA}')
        new_boxB, new_kpB, new_scoresB = chose_right_bbox(boxB, kpB, scB)
        new_boxA, new_kpA, new_scoresA = chose_right_bbox(boxA, kpA, scA)
        #print(f'new_boxB {new_boxB}, \nnew_kpB {new_kpB}')
        bb = []
        kp = []
        sc = []
        
        if len(new_boxB) == 0 and len(new_boxA) == 0:
            bb = []
            kp = []
            sc = []
        elif len(new_boxB) == 0:
            bb = new_boxA
            kp = new_kpA
            sc = new_scoresA
        elif len(new_boxA) == 0:
            bb = new_boxB
            kp = new_kpB
            sc = new_scoresB
        else:
            bb = np.vstack([np.array(new_boxA), np.array(new_boxB)]) 
            kp = np.vstack([np.array(new_kpA), np.array(new_kpB)]) 
            sc = np.hstack([np.array(new_scoresA), np.array(new_scoresB)]) 
        
        
        #print(f'bb {bb}, \nkp {kp}')
        #Объединяем с новыми и несовпадающими боксами
        if len(mean_bb) != 0:
            if len(bb) == 0:
                bb = mean_bb
                kp = mean_kp
                sc = mean_sc
            else:
                bb = np.vstack([bb, np.array(mean_bb)])
                bb = bb.tolist()
                kp = np.vstack([kp, np.array(mean_kp)]) 
                kp = kp.tolist()
                sc = np.hstack([sc, np.array(mean_sc)])
                sc = sc.tolist()
        
        #print(f'bb {bb.tolist()}, \nkp {kp.tolist()}')
        return bb, kp, sc
    
    
def iou_filter(kp_l1, bb_l1, sc_l1, kp_l2, bb_l2, sc_l2, kp_r1, bb_r1, sc_r1, kp_r2, bb_r2, sc_r2, c_w, c_h, delta_w, delta_h, nms_threshold):
    #Функция, объединяющая предсказанные ключевые точки и боксы
    
    #Объединение двух верхних частей
    sel_l1_r1_b, sel_l1_r1_kp, sel_l1_r1_sc = selection(bb_l1, bb_r1, kp_l1, kp_r1, sc_l1, sc_r1, nms_threshold)
    #Объединение двух нижних частей
    sel_l2_r2_b, sel_l2_r2_kp, sel_l2_r2_sc = selection(bb_l2, bb_r2, kp_l2, kp_r2, sc_l2, sc_r2, nms_threshold)
    #Объединение двух получившихся частей
    sel_all_b, sel_all_kp, sel_all_sc = selection(sel_l1_r1_b, sel_l2_r2_b, sel_l1_r1_kp, sel_l2_r2_kp, sel_l1_r1_sc, sel_l2_r2_sc, nms_threshold)
    
    return(sel_all_b, sel_all_kp, sel_all_sc)
        
    
def one_image_test(im_path, model, device, flag, conf_threshold, nms_threshold, iou_threshold, delta_w, delta_h, show_flag = True):
    #Функция показывающая предсказывания модели на отдельном изображении (Для модели, что делит на 4 исходное изображение)
    if type(im_path) == str:
        img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    else:
        img = im_path
        
    orig_bb = []
    orig_kp = []
    #Получаем настоящие координаты для целого изображения
    if flag:
        number = im_path[im_path.rfind('e') + 1 : im_path.rfind('.')]
        im_root = im_path[:im_path.rfind('/')]
        test_root = im_root[:im_root.rfind('/') + 1]
        b_path = test_root + 'bboxes/bbox' + number + '.txt'
        orig_bb = read_boxes(b_path)
        k_path = test_root + 'keypoints/keypoint' + number + '.txt'
        kp = read_boxes(k_path)
        for i in kp:
            orig_kp.append([[i[0], i[1]], [i[2], i[3]]])
            
    #Разрезаем изображение на 4    
    left_1, left_2, right_1, right_2, c_w, c_h = crop_one_im(img, delta_w, delta_h)
    
    with torch.no_grad():
        model.to(device)
        model.eval()
        #data = torch.tensor([left_1, left_2, right_1, right_2])
        #data = data.to(device)
        out_l1 = model([F.to_tensor(cv2.cvtColor(left_1, cv2.COLOR_BGR2RGB)).to(device)])
        out_l2 = model([F.to_tensor(cv2.cvtColor(left_2, cv2.COLOR_BGR2RGB)).to(device)])
        out_r1 = model([F.to_tensor(cv2.cvtColor(right_1, cv2.COLOR_BGR2RGB)).to(device)])
        out_r2 = model([F.to_tensor(cv2.cvtColor(right_2, cv2.COLOR_BGR2RGB)).to(device)])
    #Получаем предсказанные значения для каждой из 4 картинок, с пороговыми NMS и IOU        
    kp_l1, bb_l1, sc_l1 = get_out_kp_bb(out_l1, 0, 0, conf_threshold, iou_threshold)
    kp_l2, bb_l2, sc_l2 = get_out_kp_bb(out_l2, 0, c_h - delta_h, conf_threshold, iou_threshold)
    kp_r1, bb_r1, sc_r1 = get_out_kp_bb(out_r1, c_w - delta_w, 0, conf_threshold, iou_threshold)
    kp_r2, bb_r2, sc_r2 = get_out_kp_bb(out_r2, c_w - delta_w, c_h - delta_h, conf_threshold, iou_threshold)
    #Объединяем предсказанные значения    
    pred_b, pred_kp, pred_sc = iou_filter(kp_l1, bb_l1, sc_l1, kp_l2, bb_l2, sc_l2, kp_r1, bb_r1, sc_r1, kp_r2, bb_r2, sc_r2, c_w, c_h, delta_w, delta_h, nms_threshold)
    #Визуализация и таргетов, и предсказаний    
    if flag:
        visualize(img, pred_b, pred_kp, pred_sc, img, orig_bb, orig_kp, show_flag)
    #Визуализация только предсказаний
    else:
        if not show_flag:
            pr_im = visualize(img, pred_b, pred_kp, pred_sc, show_flag = show_flag)
            return pr_im, pred_b, pred_kp, pred_sc
        else:
            visualize(img, pred_b, pred_kp, pred_sc, show_flag = show_flag)
        

def batch_test(root, model, device, flag, conf_threshold, iou_threshold, nms_threshold, delta_w, delta_h, show_flag = True):
    #Функция показывающая предсказывания модели на пакете изображений (Для модели, что делит на 4 исходное изображение
    image_data_path = root + '/images'
    dir_size = len(glob.glob(image_data_path + '/*'))
    counter = 0
    for f in os.scandir(image_data_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            print(f'Изображение №{counter + 1} из {dir_size}')
            image_path = f.path
            one_image_test(image_path, model, device, flag, conf_threshold, nms_threshold, iou_threshold, delta_w, delta_h, show_flag)
            counter += 1
    

def full_video(filename, model, device, targets, conf_threshold, nms_threshold, iou_threshold, delta_w, delta_h):
    #Тестироване модели на видеофайле
    cap = cv2.VideoCapture(filename)
    targets = False
    #Подготовка файла записи
    new_filename = filename[:filename.rfind('/')] + '/predicted.mp4'
    yml_filename = filename[:filename.rfind('/')] + '/predicted.yml'
    
    yml_data = []
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    with open(yml_filename, 'w') as f:
        #data = yaml.dump({'name': filename}, f)
        yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
        yaml.dump(OrderedDict({'name': filename, 'FPS': fps, 'weight': w, 'height': h}), f)
        
        
    size = (int(w), int(h))
    out = cv2.VideoWriter(new_filename, fourcc, fps, size, True)
    #Открываем файл
    while not cap.isOpened():
        cap = cv2.VideoCapture(filename)
        cv2.waitKey(1000)
        print("Openning the file...")

    #pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    maxim_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    while True:
        flag, frame = cap.read()
        if flag:
            #Получаем кадр, рисуем на нем предсказания и записываем в видеоряд
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            pred_im, pred_b, pred_kp, pred_sc = one_image_test(frame, model, device, targets, conf_threshold, nms_threshold, iou_threshold, delta_w, delta_h, False)
            yml_data.append(OrderedDict({int(pos_frame): {'bboxes': pred_b, 'bboxes_scores': pred_sc, 'keypoints': pred_kp}}))
            
            #with open(yml_filename, 'a') as f:
                #my_dict = OrderedDict({'frame': pos_frame, 'values': {'bboxes': pred_b, 'bboxes_scores': pred_sc, 'keypoints': pred_kp}})
                #yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
                #yaml.dump(my_dict, f)
            #write_pred_into_file(pos_frame, pred_b, pred_kp, pred_sc, yml_filename)
            out.write(pred_im)
            print(f'{pos_frame} frame from {maxim_frames}')
        else:
            #Читаем заново, если следующий кадр не готов
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            #Делаем задержку,чтобы следующий кадр только прочитался
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            #Если прочитали все кадры - выходим из цикла
            break
        
    #Пушим всю информацию о кадрах    
    with open(yml_filename, 'a') as f:
        yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
        data = OrderedDict({'frames': yml_data})
        yaml.dump(data, f)
        
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    
    show_video(new_filename)
    
    
def read_yaml(yml_filename):
    with open(yml_filename) as f:
        yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
        datas = list(yaml.safe_load_all(f))
        return datas[0]
    
    
def visualize_from_yml(yml_path, video_path, pred_video_path):
    data = read_yaml(yml_path)
    cap = cv2.VideoCapture(video_path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
        cv2.waitKey(1000)
        print("Openning the file...")
        
    maxim_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(w), int(h))
    out = cv2.VideoWriter(pred_video_path, fourcc, fps, size, True)
    
    while True:
        flag, frame = cap.read()
        if flag:
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            pred_bboxes = data['frames'][int(pos_frame)-1][pos_frame]['bboxes']
            pred_keypoints = data['frames'][int(pos_frame)-1][pos_frame]['keypoints']
            pred_scores = data['frames'][int(pos_frame)-1][pos_frame]['bboxes_scores']
            pred_image = visualize(frame, pred_bboxes, pred_keypoints, pred_scores, show_flag = False)
            out.write(pred_image)
            print(f'{pos_frame} frame from {maxim_frames}')
        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            #Если прочитали все кадры - выходим из цикла
            break
            
    out.release()
    cap.release()   
    show_video(pred_video_path)
    cv2.destroyAllWindows()
            
    
if __name__ == '__main__':       
    parser = argparse.ArgumentParser()
    #parser.add_argument('test_data_path', nargs='?', default='/home/ubuntu/ant_detection/TEST_ACC_DATA', help="Specify the path either to the folder with test images to test everything, or the path to a single image", type=str)
    #parser.add_argument('test_data_path', nargs='?', default='/home/ubuntu/ant_detection/TEST_ACC_DATA/images/0a302e52-image202.png', help="Specify the path either to the folder with test images to test everything, or the path to a single image", type=str)
    parser.add_argument('test_data_path', nargs='?', default='/home/ubuntu/ant_detection/videos/inputs/short.mp4', help="Specify the path either to the folder with test images to test everything, or the path to a single image", type=str)
    parser.add_argument('model_path', nargs='?', default='/home/ubuntu/ant_detection/crop_with_overlay/rcnn_models/20220804-115738/best_weights.pth', help="Specify weights path", type=str)
    parser.add_argument('draw_targets', nargs='?', default=False, help="True - will draw targets, False - will not", type=bool)
    parser.add_argument('conf_threshold', nargs='?', default=0.1, help="Confident threshold for boxes", type=float)
    parser.add_argument('nms_threshold', nargs='?', default=0.1, help="Non maximum suppression threshold for boxes", type=float)
    parser.add_argument('iou_threshold', nargs='?', default=0.2, help="IOU threshold for boxes", type=float)
    parser.add_argument('overlay_w', nargs='?', default=60, help="Num of pixels that x-axis images intersect", type=int)
    parser.add_argument('overlay_h', nargs='?', default=30, help="Num of pixels that y-axis images intersect", type=int)
    
    
    #visualize_from_yml('/home/ubuntu/ant_detection/videos/inputs/predicted.yml', '/home/ubuntu/ant_detection/videos/inputs/short.mp4', '/home/ubuntu/ant_detection/videos/inputs/annot_from_yml.mp4')
    
    #print('vse!!!!!!!!!!!!!')
    #time.sleep(20000)
    args = parser.parse_args()
    test_data_path = args.test_data_path
    model_path = args.model_path
    draw_targets = args.draw_targets
    conf_threshold = args.conf_threshold
    nms_threshold = args.nms_threshold
    iou_threshold = args.iou_threshold
    overlay_w = args.overlay_w
    overlay_h = args.overlay_h
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if torch.cuda.is_available():
        print('*****************************DEVICE: GPU*****************************')
    else:
        print('*****************************DEVICE: CPU*****************************')
        
    test_model = get_model(2, model_path)
    
    if test_data_path[-3:] == 'png':
        print('*********Обработка единственного изображения*********')
        one_image_test(test_data_path, test_model, device, draw_targets, conf_threshold, nms_threshold, iou_threshold, overlay_w, overlay_h)
    elif test_data_path[-3:] == 'mp4' or test_data_path[-3:] == 'MOV':
        draw_targets = False
        print('*********Обработка видеофайла*********')
        full_video(test_data_path, test_model, device, draw_targets, conf_threshold, nms_threshold, iou_threshold, overlay_w, overlay_h)
    else:
        batch_test(test_data_path, test_model, device, draw_targets, conf_threshold, nms_threshold, iou_threshold, overlay_w, overlay_h)
