import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import sys
import numpy as np
import dlib
import imutils

#HOG
def svmdetectperson(img):
    hog=cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    person,w= hog.detectMultiScale(img)
    return person
def is_inside(a,b):
    x1,y1,w1,h1=a
    x2,y2,w2,h2=b #judge b  is not include a
    return x1>x2 and y1>y2 and x1+w1<x2+w2 and y1+h1<y2+h2
def draw(img,a):
    x,y,w,h=a
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
#--------------------


img_counter = 0

HOST='192.168.31.224'
PORT=8000

server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

server.bind((HOST,PORT))
print('Socket bind complete')
server.listen(10)
print('Socket now listening')

conn,addr=server.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
while True:
    while len(data) < payload_size:
        print("Recv: {}".format(len(data)))
        data += conn.recv(4096)

    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
#.......----------------------------------------------
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    vis = frame.copy()
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor( 'shape_predictor_68_face_landmarks.dat')
    face_rects, scores, idx = detector.run(vis, 0)

    #取出偵測的結果
    for i, d in enumerate(face_rects):
      x1 = d.left()
      y1 = d.top()
      x2 = d.right()
      y2 = d.bottom()
      text = " %2.2f ( %d )" % (scores[i], idx[i])

      if scores[i]>0.8:
           #繪製出偵測人臉的矩形範圍
          cv2.rectangle(vis, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)
          #ractangle_area = x2*y2
         # print('ractangle_area:',ractangle_area)
         # avi=
          
          #0.X~1.X
          #標上人臉偵測分數與人臉方向子偵測器編號
          cv2.putText(vis, text, (x1, y1), cv2. FONT_HERSHEY_DUPLEX,
          0.7, ( 255, 255, 255), 1, cv2. LINE_AA)
     
          #給68特徵點辨識取得一個轉換顏色的frame
          landmarks_frame = cv2.cvtColor(vis, cv2. COLOR_BGR2RGB)

          #找出特徵點位置
          shape = predictor(landmarks_frame, d)
          #繪製68個特徵點
         # face_area = round(((shape.part(7).y - shape.part(19).y)/2 + ( shape.part(15).x -  shape.part(1).x)/2)*np.pi)
        #  face_area = face_area*avi
        #  print(face_area)
         # cv2.putText(vis, str(face_area),(x1, y1),cv2. FONT_HERSHEY_COMPLEX, 0.5,( 255, 0, 0), 1)
          
          for i in range( 26):
            cv2.circle(vis,(shape.part(i).x,shape.part(i).y), 3,( 0, 0, 255), 2)
            cv2.putText(vis, str(i),(shape.part(i).x,shape.part(i).y),cv2. FONT_HERSHEY_COMPLEX, 0.5,( 255, 0, 0), 1)
     
    #輸出到畫面
    cv2.imshow('ImageWindow',vis),
#.......--------------------------,---------------------------
    cv2.waitKey(1)
