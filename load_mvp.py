import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import pandas as pd
from engine import train_one_epoch, evaluate
import utils
import transforms as T

#functions
def get_mask(coordenates, height, width):
    mask = np.zeros((height, width))
    mask[int(coordenates[1]):int(coordenates[1]) + int(coordenates[3]),
    int(coordenates[0]):int(coordenates[0]) + int(coordenates[2])] = 1
    return mask

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def predict_new_image(original_image, model,device='cpu', threshold=0.7):
    im = original_image.copy()
    im = im / 255
    im = torch.tensor(im.transpose(2, 0, 1), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model([im.to(device)])

    boxes = prediction[0]['boxes'].to('cpu').numpy().tolist()
    scores = prediction[0]['scores'].to('cpu').numpy().tolist()
    for i in range(len(boxes)):
        coordenadas = boxes[i]
        confianza = scores[i]
        if confianza > threshold:
            cv2.rectangle(original_image, (int(coordenadas[0]),
                                        int(coordenadas[1])),
                          (int(coordenadas[2]),
                           int(coordenadas[3])), (0, 255, 0), 6)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.show()