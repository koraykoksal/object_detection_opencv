import cv2
import numpy as np
import os

thres = 0.45  # Eşik değeri, nesnelerin tespiti için
nms_threshold = 0.5  # Non-Maximum Suppression eşiği

# Yerleşik webcam kullanımı için 0 indeksi
cap = cv2.VideoCapture(0) 

# Kamera ayarları (bu ayarlar bazı kameralarda farklı olabilir)
cap.set(3, 1280)  # Genişlik
cap.set(4, 720)   # Yükseklik
cap.set(10, 150)  # Parlaklık

# Class names okuma
classFile = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names'))
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Yol ayarları
configPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'))
weightsPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'frozen_inference_graph.pb'))

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, image = cap.read()
    if not success:
        break

    classIds, confs, bbox = net.detect(image, confThreshold=thres)

    if len(classIds) != 0:
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.putText(image, classNames[classIds[i][0]-1], (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basarak çık
        break

cap.release()
cv2.destroyAllWindows()
