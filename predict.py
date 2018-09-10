#! /usr/bin/env python
import argparse
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
import xml.etree.ElementTree as ET

sys.path.insert(0, '../mean_average_precision')
from mean_average_precision.utils.bbox import jaccard

import copy
import operator
import logging


hasLog = False
LOG_FILENAME = 'predict.log'
if os.path.isfile(LOG_FILENAME):
    os.remove(LOG_FILENAME)
# logging.basicConfig(filename=LOG_FILENAME,
#                     format='%(asctime)s %(levelname)-8s %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     level=logging.DEBUG)
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG)

def log(*args):
    if hasLog:
        msg = ''.join(str(x)+"\t" for x in args)
        logging.debug(msg)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

argparser.add_argument(
    '-a',
    '--annotFile',
    help='annotation File')   

argparser.add_argument(
    "-s",
    "--SpatioTemporal", 
    help="Spatio-Temporal", 
    action="store_true")  

argparser.add_argument(
    "-t",
    "--turnVideo", 
    help="Turn Video", 
    action="store_true")     

argparser.add_argument(
    '-p',
    '--predictFile',
    help='predict file (np array format)')       

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input
    predict_file = args.predictFile
    temporal_predict = args.SpatioTemporal

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    turnVideo = args.turnVideo

    if image_path[-4:] == '.mp4':

        predicted_boxes = []

        last_frames = []
        TEMPORAL_QTD_FRAMES = 10
        TEMPORAL_MIN_BOXES = 5
        TEMPORAL_MIN_IOU = 0.1
        TEMPORAL_MIN_TO_ADD = 3

        if temporal_predict:
            video_out = image_path[:-4] + "_" + config['model']['backend'] +'_spatio-temporal' + image_path[-4:]
        else:
            video_out = image_path[:-4] + "_" + config['model']['backend'] +'_normal-detected' + image_path[-4:]


        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if turnVideo:
            frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer =  cv2.VideoWriter(video_out,
                        cv2.VideoWriter_fourcc(*'mp4v'), 
                        30.0, 
                        (frame_w, frame_h),True)

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            if turnVideo:
                image = np.rot90(image,3)
                image = image.copy() # Fix Bug np.rot90 

            raw_height, raw_width, _ = image.shape
            
            boxes = yolo.predict(image)

            log('Frame ', i, 'boxes', len(boxes))


            #==================================================
            # Spatio-Temporal
            if (temporal_predict):

                # Add information column to valid box (True/False)
                if len(boxes)>0:
                    for box in boxes:
                        box.keeps = True
                        box.final_class = box.get_label()

                # Verify two or more box in same place
                for box1 in range(len(boxes)):
                    for box2 in range(box1+1,len(boxes)):
                        if (boxes[box1].keeps and boxes[box2].keeps):

                            box = boxes[box1]
                            box_a = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height]])
                            box = boxes[box2]
                            box_b = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height]])
                            
                            iou = jaccard(box_a,box_b)

                            #If iou==0, then keeps both, else, keeps the box with the best score
                            if (iou>0):
                                if (boxes[box1].get_score()>boxes[box2].get_score()):
                                    if boxes[box2].keeps:
                                        log('Frame ', i, 'boxe 2 retirado')
                                    boxes[box2].keeps = False
                                else:
                                    if boxes[box1].keeps:
                                        log('Frame ', i, 'boxe 1 retirado')
                                    boxes[box1].keeps = False

                new_boxes = None

                # Verify spatio-temporal of all boxes
                if len(boxes)>0:
                    for box in boxes:
                        if box.keeps:
                            count = 0
                            class_dict = { box.get_label() : 1 }

                            if len(last_frames) > 0:
                                box_a = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height]])
                                for last_boxes in last_frames:
                                    for last_box in last_boxes:
                                        box_b = np.array([[last_box.xmin*raw_width, last_box.ymin*raw_height, last_box.xmax*raw_width, last_box.ymax*raw_height]])
                                        iou = jaccard(box_a,box_b)
                                        
                                        if (iou > TEMPORAL_MIN_IOU):
                                            count += 1
                                            log('Frame ', i, 'count', count)
                                            if last_box.get_label() in class_dict.keys():
                                                class_dict[last_box.get_label()] += 1
                                            else:
                                                class_dict[last_box.get_label()] = 1

                            if (count <= TEMPORAL_MIN_BOXES):
                                if box.keeps:
                                    log('Frame ', i, 'boxe 3 retirado')
                                box.keeps = False
                                    
                            # most used class
                            final_class = max(class_dict.items(), key=operator.itemgetter(1))[0]
                            box.final_class = final_class

                else:
                    if (len(last_frames) > TEMPORAL_MIN_TO_ADD):
                        if (len(last_frames[-1])>0):
                            new_boxes = list(last_frames[-1])
                            log('Frame ', i, 'box adicionado 4')

                        elif (len(last_frames[-2])>0):
                            new_boxes = list(last_frames[-2])
                            log('Frame ', i, 'box adicionado 5')
                           

                # Save last frame
                last_frames.append(boxes)
                if (len(last_frames)>TEMPORAL_QTD_FRAMES):
                    last_frames = last_frames[-TEMPORAL_QTD_FRAMES:]

                # box adjust 
                tmp_boxes = []
                if (new_boxes != None):
                    for box in new_boxes:
                        if (box.keeps):
                            b = copy.copy(box)
                            b.label = box.get_label()
                            tmp_boxes.append(b)
                else:
                    for box in boxes:
                        if (box.keeps):
                            b = copy.copy(box)
                            b.label = box.get_label()
                            tmp_boxes.append(b)

                boxes = tmp_boxes          


            #==================================================
            # predicted boxes
            if (predict_file):
                subbox = []
                if len(boxes) > 0:
                    raw_height, raw_width, _ = image.shape
                    for box in boxes:
                        subbox.append([box.get_label(), box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score ])
                    
                    predicted_boxes.append([i, subbox, boxes])
                else:
                    predicted_boxes.append([i])


            #==================================================
            log("Frame", i, "Boxes Finais", len(boxes))

            image = draw_boxes(image, boxes, config['model']['labels'], 2, 1.1, -30)
            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()  

        if (predict_file):
            np.save(predict_file, np.array(predicted_boxes))


    else:
        image = cv2.imread(image_path)

        if (args.annotFile != None):
            boxes_ann = []
            tree = ET.parse(args.annotFile)
            for elem in tree.iter():
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}
                    
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text

                            boxes_ann.append(obj)
                                
                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))
            
            for box in boxes_ann:
                cv2.rectangle(image, (box['xmin'],box['ymin']), (box['xmax'],box['ymax']), (255,0,0), 30)

        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'], 30, 4.5, 35)

        print(len(boxes), 'boxes are found')

        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
