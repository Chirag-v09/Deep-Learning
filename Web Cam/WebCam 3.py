# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:10:24 2020

@author: Chirag
"""

# To capture a particular image from the Real-Time video

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cv2

cap = cv2.VideoCapture(0)

while (True):
    
    ret, frame = cap.read()
    
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame)
plt.xticks([])
plt.yticks([])
plt.title('Capture')
plt.show()
