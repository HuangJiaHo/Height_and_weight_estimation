import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import sys
import numpy as np


HOST='127.0.0.1'
PORT=8000

server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

server.bind((HOST,PORT))
print('Socket bind complete')
server.listen(True)
print('Socket now listening')

conn,addr=server.accept()

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

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
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('SERVER',frame)
   
#......
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data2 = pickle.dumps(frame, 0)
    size = len(data2)
    conn.sendall(struct.pack(">L", size) + data2)
    cv2.waitKey(1)
    
 


