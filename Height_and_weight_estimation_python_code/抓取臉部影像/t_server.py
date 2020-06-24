import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import sys
import numpy as np
import cmath
import argparse
import dlib
import imutils

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
BodyPartFin = False
  
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

WholePartFin = False
Face_catch =False

body_weight=0
face_weight=0
FaceWit=0
yy2=0
xx1=0
xx2=0
yy1=0

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
StardredWight =0
while True:
    receive_Boolean = False
    send_img = True
    while send_img:
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
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        
        vis = frame.copy()
        cv2.imshow('v',vis)
        cv2.waitKey(1)
        
        
        if(BodyPartFin == False):
            #body detect
            person=svmdetectperson(vis)
            filtered=[]
            CatchFea = False
            grab = False
            rect = [0,0,0,0]
            for i,p in enumerate(person):
                for j,p1 in enumerate(person):
                    if i!=j and is_inside(p,p1):
                        break 
                filtered.append(p)
            for p in filtered:
                initX = p[0]
                initY = p[1]
                XLen =p[2]
                YLen = p[3]   
                draw(vis,p)
                grab = True
                #grab = Ture or False cam has body bounding
            if(grab == True):
                grab =False
                #deliver to client frame stop
                signal = "call_frame_stop"
                signal = signal.encode()
                conn.sendall(signal)
                
                #process img
                #grab body
                #設定矩形區域  作為ROI         矩形區域外作為背景
                rect = (initX, initY, XLen,YLen)
                rect_copy = tuple(rect)
                rect = [0,0,0,0]
                #img.shape[:2]得到img的row 和 col ,
                # 得到和img尺寸一樣的掩模即mask ,然後用0填充
                mask = np.zeros(vis.shape[:2], np.uint8)

                #建立以0填充的前景和背景模型,  輸入必須是單通道的浮點型影象, 1行, 13x5 = 65的列 即(1,65)
                bgModel = np.zeros((1,65), np.float64)
                fgModel = np.zeros((1,65), np.float64)

                #呼叫grabcut函式進行分割,輸入影象img, mask,  mode為 cv2.GC_INIT_WITH-RECT
                cv2.grabCut(vis, mask, rect_copy, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

                ##呼叫grabcut得到rect[0,1,2,3],將0,2合併為0,   1,3合併為1  存放於mask2中
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

                #得到輸出影象
                GrabMat = vis * mask2[:, :, np.newaxis]
                cv2.imshow('cut',GrabMat)
                cv2.waitKey(1)
                
                GrabMat=cv2.cvtColor(GrabMat,cv2.COLOR_BGR2GRAY)
                roi = GrabMat[initY:(initY+YLen)+1,initX:(initX+XLen)+1]
                ret, thresh4 = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)
                kernel = np.ones((5,5),np.uint8) 
                thresh4 = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, kernel)
                thresh4 = cv2.morphologyEx(thresh4, cv2.MORPH_CLOSE, kernel)


                GrabMat[initY:(initY+YLen)+1,initX:(initX+XLen)+1] = thresh4
                #output GrayMat
                
                #cv2.imshow('g',GrabMat)
                
                
                flag=False
                for i in range(GrabMat.shape[0]-1):
                    for j in range(GrabMat.shape[1]-1):
                        if(GrabMat[i,j] !=0):
                            xx1=i
                            yy1=j
                            flag=True
                            break
                    if(flag == True):
                        break
                flag=False
                for i in range(GrabMat.shape[0]-1,0,-1):
                    for j in range(GrabMat.shape[1]-1):
                        if(GrabMat[i,j] !=0):#!=0
                            xx2=i
                            yy2=j
                            flag=True
                            break
                    if(flag == True):
                        break
                height = cmath.sqrt((xx1-xx2)**2 + (yy1-yy2)**2)
                height =  0.3971*int(height.real) + 112.57# 調整比率
 
                print('height:',height)
                #身高 height
                cv2.line(vis, (yy1, xx1), (yy2, xx2), (0,255,0))
                cv2.imshow('line',vis)
                cv2.waitKey(1)

                StardredWight =( height-150)*0.6+50
                
                #Area
                BodyArea =0
                for i in range(GrabMat.shape[0]-1):
                    for j in range(GrabMat.shape[1]-1):
                        if(GrabMat[i,j] !=0):
                            BodyArea = BodyArea +1
                print('area',BodyArea)

                Body_weight = StardredWight *0.5

                #4800 基準
                area_base = 4400
                if(BodyArea<area_base):
                    local = (abs(BodyArea-4500)/200)
                    for i in range(int(local)):
                        Body_weight = Body_weight *0.95
                else:
                    local = (abs(BodyArea-4500)/200)
                    for i in range(int(local)):
                        Body_weight = Body_weight /0.95
                print('local',local)
                print('Body_weight',Body_weight)
                BodyPartFin = True

                
                
                cv2.imshow('GrabMat',GrabMat)
                cv2.waitKey(1)

                result, vis = cv2.imencode('.jpg', vis, encode_param)
                data2 = pickle.dumps(frame, 0)
                size = len(data2)
                conn.sendall(struct.pack(">L", size) + data2)
                time.sleep(2)
            else:
                signal = "no"
                signal = signal.encode()
                conn.sendall(signal)
            # area exchange to body_weight
            '''
            
            '''
            #Output data => height , y1,x1,y2,x2(身高兩點)  ,body_weight(身重)
            
                    
           

            #process pitcute save data while final   Area 
            
            
        else:
           
            #face
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor( 'shape_predictor_68_face_landmarks.dat')
            face_rects, scores, idx = detector.run(vis, 0)
            for i, d in enumerate(face_rects):
              x1 = d.left()
              y1 = d.top() 
              x2 = d.right()
              y2 = d.bottom()
              text = " %2.2f ( %d )" % (scores[i], idx[i])
              
              if scores[i]>0.8 and idx[i] == 0:
                  Face_catch =True
                  signal = "call_frame_stop1"
                  signal = signal.encode()
                  conn.sendall(signal)
                  
                   #繪製出偵測人臉的矩形範圍
                  cv2.rectangle(vis, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)
                  
                  #0.X~1.X
                  #標上人臉偵測分數與人臉方向子偵測器編號
                  cv2.putText(vis, text, (x1, y1), cv2. FONT_HERSHEY_DUPLEX,
                  0.7, ( 255, 255, 255), 1, cv2. LINE_AA)
             
                  #給68特徵點辨識取得一個轉換顏色的frame
                  landmarks_frame = cv2.cvtColor(vis, cv2. COLOR_BGR2RGB)

                  #找出特徵點位置
                  shape = predictor(landmarks_frame, d)
                  #繪製68個特徵點
                  maxY = shape.part(0).y
                  minY = shape.part(0).y
                  maxX = shape.part(0).x
                  minX = shape.part(0).x
                  for i in range( 26):
                      if(shape.part(i).y >= maxY):
                          maxY = shape.part(i).y
                      if(shape.part(i).y <= minY):
                          minY= shape.part(i).y
                      if(shape.part(i).x >= maxX):
                          maxX = shape.part(i).x
                      if(shape.part(i).x <= minX):
                          minX= shape.part(i).x
                      cv2.circle(vis,(shape.part(i).x,shape.part(i).y), 3,( 0, 0, 255), 2)
                      cv2.putText(vis, str(i),(shape.part(i).x,shape.part(i).y),cv2. FONT_HERSHEY_COMPLEX, 0.5,( 255, 0, 0), 1)
                  
                  faceArea = int((maxY-minY)/2 * (maxX-minX) * 3.14)
                  
                
                  print('faceArea',faceArea)
                  face_weight = StardredWight *0.5


                 #4800 基準
                  face_base = 6000
                  if(faceArea<face_base):
                      local = (abs(faceArea-face_base)/100)
                      for i in range(int(local)):
                          face_weight = face_weight *0.95
                  else:
                      local = (abs(faceArea-face_base)/100)
                      for i in range(int(local)):
                          face_weight = face_weight /0.95
                
                  print('face_weight',face_weight)
             
                  
                 
                  cv2.putText(vis, str(face_weight),(x1, y1),cv2. FONT_HERSHEY_COMPLEX, 0.5,( 255, 0, 0), 1)
                  cv2.imshow('vis',vis)
                  cv2.waitKey(1)
                  WholePartFin =True
                  
                  print('deliver to client')
                  result, vis = cv2.imencode('.jpg', vis, encode_param)
                  data2 = pickle.dumps(frame, 0)
                  size = len(data2)
                  conn.sendall(struct.pack(">L", size) + data2)
                  time.sleep(2)
                  
            if(Face_catch ==False):
                signal = "no"
                signal = signal.encode()
                conn.sendall(signal)
                  # output FaceWit
            #------
        
       
        #waiting client sent to flag \
        if(WholePartFin):
            accept = conn.recv(1024)
            accept = accept.decode() 
            if accept == "go":
             
                #All proceess alredly Finished
                receive_Boolean = True
                send_img = False
            
            #while receive flag go down
            ####
        
    while receive_Boolean:
        print('asdas12')
        msg = str(height)
        msg = msg.encode()
        conn.sendall(msg)

        time.sleep(2)
        accept = conn.recv(1024)
        accept = accept.decode() 
        if accept == "1":
            msg = str(yy1)
            msg = msg.encode()
            conn.sendall(msg)
            accept = conn.recv(1024)
            accept = accept.decode()
            if accept == "2":
                msg = str(xx1)
                msg = msg.encode()
                conn.sendall(msg)
                accept = conn.recv(1)
                accept = accept.decode()
                if accept == "3":
                    msg = str(yy2)
                    msg = msg.encode()
                    conn.sendall(msg)
                    accept = conn.recv(1)
                    accept = accept.decode()
                    if accept == "4":
                        msg = str(xx2)
                        msg = msg.encode()
                        conn.sendall(msg)
                        accept = conn.recv(1)
                        accept = accept.decode()  
                        if accept == "5":
                            msg = str(Body_weight)
                            msg = msg.encode()
                            conn.sendall(msg)
                            accept = conn.recv(1)
                            accept = accept.decode() 
                            if accept == "6":
                                msg = str(face_weight)
                                msg = msg.encode()
                                conn.sendall(msg)
                                receive_Boolean = False
                                #break

server_socket.close()
