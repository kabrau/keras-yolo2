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
  

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

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

    virar_video = False
    
    temporal_predict = True
    temporal_boxes = []


    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]
        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if virar_video:
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
            if virar_video:
                image = np.rot90(image,3)
                image = image.copy() # Fix Bug np.rot90 
            
            boxes = yolo.predict(image)

            if len(boxes) > 0:
                raw_height, raw_width, _ = image.shape
                pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score] for box in boxes])

                for box in boxes:
                    temporal_boxes.append([i, box.get_label(), box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score ])
                
            else:
                temporal_boxes.append([i])


            #temporal_boxes.append(boxes)
            #temporal_boxes = temporal_boxes[-3:]

            # if len(boxes) > 0:
            #     raw_height, raw_width, _ = image.shape
            #     pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score] for box in boxes])
            #     for box in boxes:
            #         iou = jaccard(pred_boxes[:,0:4], pred_boxes[:,0:4])
            #         print(iou)
                    

            # for box in temporal_boxes[0]:
            #     image_h, image_w, _ = image.shape
            #     xmin = int(box.xmin*image_w)
            #     ymin = int(box.ymin*image_h)
            #     xmax = int(box.xmax*image_w)
            #     ymax = int(box.ymax*image_h)
            #     label = config['model']['labels'][box.get_label()]
            #     #iou = jaccard(np.array([xmin, ymin, xmax, ymax]), np.array([xmin, ymin, xmax, ymax]))
            #     #[x1, y1, x2, y2]
            #     print(label, xmin, ymin, xmax, ymax)

            image = draw_boxes(image, boxes, config['model']['labels'], 2, 1.1, -30)

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()  

        for b in temporal_boxes:
            print(b)
        tf = np.array(temporal_boxes)
        np.save("./dados.txt", tf)


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
