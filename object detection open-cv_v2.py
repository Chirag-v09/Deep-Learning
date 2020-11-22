import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
im = cv2.imread('cars_4.jpeg')
bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)
plt.imshow(output_image)
plt.show()
print('Number of cars in the image is '+ str(label.count('car')))


#############
from platform import python_version
import tensorflow
import keras
import cvlib as cv
import cv2

print('Python version: {}'.format(python_version()))
print('cvlib version: {}'.format(cv.__version__))
print('OpenCV version: {}'.format(cv2.__version__))
print('Tensorflow version: {}'.format(tensorflow.__version__))
print('Keras version: {}'.format(keras.__version__))


# Uncomment this to read image from local folder/directory
#im = cv2.imread('cars.jpeg')

# Comment the above line and Uncomment the following two lines if you are reading imgae from a url
#from skimage import io
#im = io.imread('http://www.samacharnama.com/wp-content/uploads/2019/06/third-party-insurance..687.png')
