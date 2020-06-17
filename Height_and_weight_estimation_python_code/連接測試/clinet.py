import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import sys
import numpy as np


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 8000))
connection = client_socket.makefile('wb')

cam = cv2.VideoCapture(0)

cam.set(3, 320);
cam.set(4, 240);

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))

while True:
    ret, frame = cam.read()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
#    data = zlib.compress(pickle.dumps(frame, 0))
    data2 = pickle.dumps(frame, 0)
    size = len(data2)
    client_socket.sendall(struct.pack(">L", size) + data2)

#.......
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
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    cv2.imshow('client',frame)
    cv2.waitKey(1)
 
cam.release()

