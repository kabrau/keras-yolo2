import numpy as np

import sys
sys.path.insert(0, '../mean_average_precision')
from mean_average_precision.utils.bbox import jaccard

lista = np.load("./dados.txt.npy")



linha = -1
last_box = None

boxes = []
for box in lista:
    #if (box[0]!=linha):
        #print("NOVA")

    # if linha != -1:
    #     if (len(box)>1 and len(last_box)>1 ):
    #         box_a = np.array([box[2:6]])
    #         box_b = np.array([last_box[2:6]])
    #         print(box_a)
    #         print(box_b)
    #         iou = jaccard(box_a,box_b)
    #         print(iou)
    
    # last_box = box
    # linha = box[0]

    img = box[0]
    if (img>2):
        if (len(box)>1 and len(boxes[-1])>1 ):
            box_a = np.array([box[2:6]])
            box_b = np.array([boxes[-1][2:6]])
            iou = jaccard(box_a,box_b)
            print(box[1],box[6], iou)

    boxes.append(box)




