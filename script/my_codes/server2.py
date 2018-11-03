#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import os

import cv2

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("192.168.8.103", 12345)) #if the clients/server are on different network you shall bind to ('', port)

s.listen(10)
c, addr = s.accept()
print('{} connected.'.format(addr))

# f = open("/home/niruhan/AlignedReID-Re-Production-Pytorch/script/my_codes/my_images/pedestrians.jpg", "rb")
# l = os.path.getsize("/home/niruhan/AlignedReID-Re-Production-Pytorch/script/my_codes/my_images/pedestrians.jpg")
# m = f.read(l)

img = cv2.imread('my_images/sutha.jpg')

c.sendall(img)
# f.close()
print("Done sending...")