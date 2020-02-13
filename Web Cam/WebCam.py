# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 08:37:39 2020

@author: Chirag
"""

# Capture the image by just when the WebCam is opened

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cv2
import tensorflow.keras as tf


def capture_image():
    cap = cv2.VideoCapture(0) # These numbers are for use of the web cam
    ''' i.e if we want to use the first WebCam of system then we write 0 : VideoCapture(0)
      and if we want to use second WebCam then we write 1 : VideoCapture(1)'''
    
    if cap.isOpened():
        ret, frame = cap.read()
        # print(ret)
        # print(frame)
    else:
        ret = False
    cap.release()
    return ret, frame

ret, frame = capture_image()

if ret:
    
    img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img1)
    plt.title("Camera Image-1")
    plt.xticks([])
    plt.yticks([])
    plt.show()

else:
    print("No Image Opened")
