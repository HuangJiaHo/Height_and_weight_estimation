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
payload_size = struct.calcsize(">L")
data = b""
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.31.224',8000))

cam = cv2.VideoCapture(0)

cam.set(3, 320);
cam.set(4, 240);
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
count = 0
send_data_Boolean = True
receive_Boolean = False
recv_data_Boolean = False
A=[0]*2
i=0
while True:
    while send_data_Boolean:
        try:
            ret, frame = cam.read()
            result, frame = cv2.imencode('.jpg', frame, encode_param)
            frame_len = pickle.dumps(frame, 0)
            frame_len_size = len(frame_len)
            client_socket.sendall(struct.pack(">L", frame_len_size) + frame_len)
            print("photo sent")
            try:
                signal = client_socket.recv(16)               
                signal = signal.decode()
                print(signal)
            except:
                print("signal_recv_error!")
                
            if signal == "call_frame_stop":                        
                 send_data_Boolean = False                
                 recv_data_Boolean= True
            if signal == "call_frame_stop1":              
                 send_data_Boolean = False                
                 recv_data_Boolean= True
                 receive_Boolean = True  
         
        except:
            print("send_data_Boolean_error!")
        
    while recv_data_Boolean:
        while len(data) < payload_size:
            print("Recv: {}".format(len(data)))
            data += client_socket.recv(4096)

        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += client_socket.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        #frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        A[i]=frame
        cv2.imshow('client',frame)
        i= i + 1
        singal = "aaa"
        cv2.waitKey(1)
        
        send_data_Boolean = True
        recv_data_Boolean = False
     
        time.sleep(0.01)
        
    while receive_Boolean:
        recv_data_Boolean = False
        msg = "go"
        msg = msg.encode()
        client_socket.sendall(msg)

        height = client_socket.recv(1024)
        height = height.decode()
        if height != None:
            signal = "1"
            signal = signal.encode()
            client_socket.sendall(signal)
            
        y1 = client_socket.recv(10)
        y1 = y1.decode()
        if y1 != None:
            signal = "2"
            signal = signal.encode()
            client_socket.sendall(signal)
            
        x1 = client_socket.recv(1024)
        x1 = x1.decode()
        if x1 != None:
            signal = "3"
            signal = signal.encode()
            client_socket.sendall(signal)

        y2 = client_socket.recv(1024)
        y2 = y2.decode()
        if y2 != None: 
            signal = "4"
            signal = signal.encode()
            client_socket.sendall(signal)

        x2 = client_socket.recv(1024)
        x2 = x2.decode()
        if x2 != None:
            signal = "5"
            signal = signal.encode()
            client_socket.sendall(signal)

        
        body = client_socket.recv(1024)
        body = body.decode()
        if body != None:
            signal = "6"
            signal = signal.encode()
            client_socket.sendall(signal)

        face = client_socket.recv(1024)
        face = face.decode()        

        weight = float(body) + float(face)
        cv2.line(A[0],(int(y1),int(x1)),(int(y2),int(x2)),(0,255,0))
        cv2.imshow('1',A[0])
        cv2.imshow('2',A[1])
        cv2.waitKey(5)
        print("height:{} height:{}".format(round(float(height)), round(weight,2)))
        #print("height: ",height,body, face, round(weight,2))
client_socket.close()
