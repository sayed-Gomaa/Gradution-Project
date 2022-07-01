from tkinter.font import names
from charset_normalizer import detect
from flask import Flask, render_template,jsonify, Response,  request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from yolov5.detect import run
from PIL import Image
import numpy as np
import os
import sys
import cv2
import cv2 as cv
from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend
from yolov5.custom_detect import detection ,train_face


__author__ = 'Sayed Gomaa'
__source__ = ''

app = Flask(__name__)


# --------------------------- initilize variable ----------------------------------------------------------------------------------- #
esp_img=['current image ']     #----------------------> store currect image from esp  in indx 0  EX esp_img[0]= image                #           
model_results=['currency result ', 'object result'] #->  store result of currency model in index 0 and object model in index 1       #
                                                                                                                                     #
currancy_class=['5Egp', '10Egp', '20Egp','50Egp', '100Egp', '200Egp'] #---------------------> contain names of classes of curreny    #
object_class=['Cup' ,'Shoes' ,'Chair' ,'Bed' ,'Spoon' ,'Bottle' ,'Door' ,'Toilet' ,'Towel'] #-> contain names of classes of curreny  #
device=''                                                                                                                            #
device = select_device(device)                                                                                                       #
dnn=False                                                                                                                            #
# ---------------------------------------------------------------------------------------------------------------------------------- #


# --------------------------- define Paths ---------------------------------------------------------------------------------------- #
# -paths for face model                                                                                                             #
names_path='D:/PyProject/GradutionProject/Yolo5/names.txt'                                                                          #
data_names_path="D:/PyProject/GradutionProject/Yolo5/data/"                                                                         #
haar_face_path='D:/PyProject/GradutionProject/Yolo5/haar_face.xml'                                                                  #
face_trained_path='D:/PyProject/GradutionProject/Yolo5/face_trained.yml'                                                            #
# -paths for currency and object model                                                                                              #
weights_currancy_path='GradutionProject/Yolo5/static/trained_model/best_currancy.pt'                                                #
weights_object_path='GradutionProject/Yolo5/static/trained_model/best_object.pt'                                                    #
data='GradutionProject/Yolo5/cfg.yaml'                                                                                              #
UPLOAD_FOLDER = 'GradutionProject/Yolo5/static/uploads'                                                                             #
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER                                                                                         #
#---------------------------------------------------------------------------------------------------------------------------------- #

# --------------------------- load models ----------------------------------------------------------------------------------------- #
# ---- load face modal                                                                                                              #
# load names of people that model train them                                                                                        #
fileNames = open(names_path, 'r')                                                                                                   #
f1 = fileNames.read()                                                                                                               #
poeple = f1.split("\n")                                                                                                             #
fileNames.close()                                                                                                                   #
# need to explain                                                                                                                   #
haar_cascade = cv.CascadeClassifier(haar_face_path)                                                                                 #
face_recognizer = cv.face.LBPHFaceRecognizer_create()                                                                               #
face_recognizer.read(face_trained_path)                                                                                             #
face_argument=[poeple,haar_cascade,face_recognizer,names_path,data_names_path]                                                      #                                                               #
# ---- load currency and object model                                                                                               #
currancy_model = DetectMultiBackend(weights_currancy_path, device=device, dnn=dnn, data=data)                                       #
object_model = DetectMultiBackend(weights_object_path, device=device, dnn=dnn, data=data)                                           #
#---------------------------------------------------------------------------------------------------------------------------------- #


@app.route("/about")
def about():
  return render_template("about.html")

@app.route("/<x>", methods=['POST'])
def index(x):
    return "dd"

@app.route("/detect/<x>")
def detected(x):
    """
    description : -detect function used to get response from flutter to send result of currency model or object model 
    parameter   : -x  variable use for if want currency result or object result  and it should equal { 'currency' , 'object' } 
    return      : -json file have result of currency or object model depend on x 
    """
    if x=='currency':
        return jsonify({ "result": model_results[0] })
    elif x=='object':
        return jsonify({ "result": model_results[1] })


@app.route('/upload', methods=['POST','GET'])
def upload():
  """
    description : -responsible to get images from ESP and update it in esp_image,
                  -enter image into two models and get result then update model_result
    parameter   : -   
    return      : -json file to confirm that image is recived
  """
  received = request
  img = None
  if received.files:
    print(received.files['imageFile'])
		# convert string of image data to uint8
    file  = received.files['imageFile']
    nparr = np.fromstring(file.read(), np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # esp_img[1]=esp_img[0]
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img_thresh_mean_c = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    esp_img[0]=img_thresh_mean_c
    # print(esp_img[0])
    currancy_result=detection(img, model=currancy_model, device=device)
    object_result=detection(img, model=object_model, device=device)
    print('>>: ', currancy_result ,'  ::<<<<<<<<<<<<<<<<<<<<<<' )
    print('>>:  ', object_result ,'  ::<<<<<<<<<<<<<<<<<<<<<<' )
    model_results[0]=currancy_result
    model_results[1]=object_result
    print('>>>:  ', model_results ,'  ::<<<<<<<<<<<<<<<<<<<<<<' )
  return  jsonify({ "result": 'done' })


@app.route("/face/<y>")
def video_feed(y):
    """
    description : -get order from flutter to detect name of person by detect his face or 
                  -store name of person and take images to its face to train model to identify it
    parameter   : -y varible for if want to detect face or store it, it should equal {'detct' ,'store'+'name' }  
    return      : -json file have name of person or 
                  -confirm that name is store  
    """
    poeple, haar_cascade, face_recognizer, names_path, data_names_path = face_argument
    img =esp_img[0]
    x=y
    if x == 'detect':
        print("start to detect face")
        # gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray=img
        faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 6)
        print(faces_rect)
        for (x, y, w, h) in faces_rect:
            faces_roi = gray[y:y + h, x:x + w]
            # cv.imshow(' Face', faces_roi)
            label, confidence = face_recognizer.predict(faces_roi)
            cv.putText(img, str(poeple[label] + "  " + str(int(confidence))), (x, y - 10),
                       cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            print("Face: ", str(poeple[label]))
            # jsonPrint(str(poeple[label]))
            return jsonify({
                "result": str(poeple[label])
            })
    elif x[0:4] == 'stor':
        print ('Start storing >>>>>>>>>>>>>>>>>>>>')
        n = x[5:]
        name = n
        if name in poeple:
            print("name in directory")
        else:
            try:
                os.mkdir(data_names_path + name)
                fileNames = open( names_path, 'a+')
                fileNames.write(name + "\n")
                fileNames.close()
            except OSError as e:
                pass
        counter = 1
        print("start to store  ",name)
        while counter <= 100:
            img =esp_img[0]
            # old_img =esp_img[1]
            print("still store in  ",counter )    
            # gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            gray =img
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces_rect:
                    faces_roi = gray[y:y + h, x:x + w]
                    cv.imwrite('D:/PyProject/GradutionProject/Yolo5/data/' + name + "/img" + str(counter) + ".jpg", faces_roi)
                    counter+=1
                    print("save image ", name, str(counter))
        train_face()             
        return jsonify({ "result": "name stored done" })            
                
    return jsonify({
        "result": "none3"
    })
    
if __name__ == '__main__':
  app.run(debug=True, host='192.168.1.6')