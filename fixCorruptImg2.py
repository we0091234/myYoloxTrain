import os 
import cv2 
from PIL import Image

# Directory to search for images
dir_path = r'datasets/VOCdevkit/VOC2007/JPEGImages'

for path in os.listdir(dir_path):
    # Make sure to change the extension if it is nor 'jpg' ( for example 'JPG','PNG' etc..)
    if path.endswith('.jpg'):
      img_path = os.path.join(dir_path, path)
      try:
        img = Image.open(img_path)
        img.verify()
      except:
        print('Bad file: is %s'%img_path) # print out the names of corrupt files