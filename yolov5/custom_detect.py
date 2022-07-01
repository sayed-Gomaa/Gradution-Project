import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import shutil

def detection(img,model,device):
    # # Load model
    img = letterbox(img, 640, 32, True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(img)
      
    half=False
    
    
    
    imgsz=[640, 640]
    #source = src#'/home/sayedgomaa/PyProject/GradutionProject/Yolo5/static/uploads/3.jpg'
    
    augment=False
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000
    classes=None
    agnostic_nms=False,  # class-agnostic NMS
    webcam = False  # batch_size >= 1
    detected_class=''

    #load model
    
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    #load image 
    #dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs=1

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0


    
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

            # Inference
            #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    pred = model(im, augment=augment )
    t3 = time_sync()
    dt[1] += t3 - t2

            # NMS
    pred = non_max_suppression(pred, conf_thres,0.25, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3
            # # Second-stage classifier (optional)
            # # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
            
            # Process predictions
    for i, det in enumerate(pred):  # per image
    
                #annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # # Rescale boxes from img_size to im0 size
                    # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for i in range(len(det)):
                        for c in det[:, -1].unique():
                            print(f' the >>>>>> {1}', c)
                            detected_class += names[int(c)] + ' / '  
                            # n = (det[:, -1] == c).sum()  # detections per class
                            # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    if len(detected_class)<2:detected_class='No Object Detected'
    return detected_class


def train_face():
    features = []
    labels = []
    path = 'D:/PyProject/GradutionProject/Yolo5/data/'
    fileNames = open('D:/PyProject/GradutionProject/Yolo5/names.txt', 'r')
    f1 = fileNames.read()
    poeple = f1.split("\n")
    fileNames.close()
    print(os.listdir(path))
    for person in os.listdir(path):
        for img_path in os.listdir(path + person):
            img = cv2.imread(path + person + "/" + img_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            features.append(img)
            label = poeple.index(person)
            # print("label= ",label)
            labels.append(label)
    features = np.array(features, dtype='object')
    labels = np.array(labels)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Train the Recognizer on the features list and the labels list
    face_recognizer.train(features, labels)
    face_recognizer.save('D:/PyProject/GradutionProject/Yolo5/face_trained.yml')
    print("model saved successfuly")
