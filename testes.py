import numpy as np

import sys
sys.path.insert(0, '../mean_average_precision')
from mean_average_precision.utils.bbox import jaccard

import operator


frames = np.load("./dados.txt.npy")

POS_CLASS = 0
POS_SCORE = 5
POS_OK = 6
POS_FINAL_CLASS = 7

TEMPORAL_QTD_FRAMES = 10
TEMPORAL_MIN_BOXES = 5
TEMPORAL_MIN_IOU = 0.1
TEMPORAL_MIN_TO_ADD = 3

linha = -1
last_box = None

last_frames = []
final_frames = []
for f1 in range(len(frames)):

    frame = frames[f1]

    # criando informação que indica se é para manter frame
    if (len(frame)>1):
        for box1 in range(len(frame[1])):
            frame[1][box1].append(True)

    #calculando IOU de janelas no mesmo frame
    if (len(frame)>1):
        for box1 in range(len(frame[1])):
            for box2 in range(box1+1,len(frame[1])):
                if (frame[1][box1][POS_OK] and frame[1][box2][POS_OK]):
                    box_a = np.array([frame[1][box1][1:5]])
                    box_b = np.array([frame[1][box2][1:5]])
                    iou = jaccard(box_a,box_b)

                    #se iou=0 fica ambos boxes, se iou > 0 fica com o de maior score
                    if (iou>0):
                        if (frame[1][box1][POS_SCORE]>frame[1][box2][POS_SCORE]):
                            frame[1][box2][POS_OK] = False
                        else:
                            frame[1][box1][POS_OK] = False

        #for box1 in range(len(frame[1])):
        #    for box2 in range(box1+1,len(frame[1])):
        #        print(frame[0],box1,box2,
        #            frame[1][box1][0],frame[1][box1][POS_SCORE],
        #            frame[1][box2][0],frame[1][box2][POS_SCORE], 
        #            frame[1][box1][0],frame[1][box1][POS_OK],
        #            frame[1][box2][0],frame[1][box2][POS_OK], 
        #            iou)

    new_frame = None

    # verificando temporal para saber se é para manter o box no frame e se a classe está correta
    if (len(frame)>1):
        for box1 in range(len(frame[1])):
            count = 0
            class_dict = {
                frame[1][box1][POS_CLASS]:1
                }

            if (len(last_frames)>0):
                if (frame[1][box1][POS_OK]):
                    for frame2 in last_frames:
                        if (len(frame2)>1):
                            for box2 in range(len(frame2[1])):
                                box_a = np.array([frame[1][box1][1:5]])
                                box_b = np.array([frame2[1][box2][1:5]])
                                iou = jaccard(box_a,box_b)
                                if (iou > TEMPORAL_MIN_IOU):
                                    count += 1
                                    if frame2[1][box2][POS_CLASS] in class_dict.keys():
                                        class_dict[frame2[1][box2][POS_CLASS]] += 1
                                    else:
                                        class_dict[frame2[1][box2][POS_CLASS]] = 1
                                        
            if (count > TEMPORAL_MIN_BOXES):
                frame[1][box1][POS_OK] = True
                #print(frame[0], box1, frame[1][box1][POS_OK], frame[1][box1][POS_CLASS], "=>", final_class, class_dict)
            else:
                frame[1][box1][POS_OK] = False
            
            final_class = max(class_dict.items(), key=operator.itemgetter(1))[0]
            frame[1][box1].append(final_class)

    else: #verificando se precisa adicionar um box no frame
    
        if (len(last_frames) > TEMPORAL_MIN_TO_ADD):
            if (len(last_frames[-1])>1):
                new_frame = list(last_frames[-1])

            elif (len(last_frames[-2])>1):
                new_frame = list(last_frames[-2])

    last_frames.append(frame)
    if (len(last_frames)>TEMPORAL_QTD_FRAMES):
        last_frames = last_frames[-TEMPORAL_QTD_FRAMES:]

    if (new_frame != None):
        final_frames.append(new_frame)
    else:
        final_frames.append(frame)


# SALVANDO VIDEO
if (True):

    import cv2
    from tqdm import tqdm
    from utils import draw_boxes
    import json

    config_path = "E:/rodney/configs/keras-yolo2/config.TinyYolo.json"
    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)


    image_path = "E:/rodney/NovosVideos/VID_20180903_124836555_HDR.mp4"
    video_out = image_path[:-4] + '_temporal' + image_path[-4:]
    video_reader = cv2.VideoCapture(image_path)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_writer =  cv2.VideoWriter(video_out,
                    cv2.VideoWriter_fourcc(*'mp4v'), 
                    30.0, 
                    (frame_w, frame_h),True)

    for i in tqdm(range(nb_frames)):
        _, image = video_reader.read()

        if len(final_frames[i])>1:
            boxes = []
            for box1 in range(len(final_frames[i][1])):
                if (final_frames[i][1][box1][POS_OK]):
                    final_frames[i][2][box1].label = final_frames[i][1][box1][POS_FINAL_CLASS]
                    boxes.append(final_frames[i][2][box1])

            image = draw_boxes(image, boxes, config['model']['labels'], 2, 1.1, -30)
        video_writer.write(np.uint8(image))

video_reader.release()
video_writer.release()  
