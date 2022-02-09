import os 
import cv2 

# Directory to search for images
dir_path = r'datasets/VOCdevkit/VOC2007/JPEGImages'
i=0
def detect_and_fix(img_path, img_name):
    global i
    # detect for premature ending
    try:
        with open( img_path, 'rb') as im :
            im.seek(-2,2)
            if im.read() == b'\xff\xd9':
                print('Image OK :', img_name) 
                # pass
            else: 
                # fix image
                img = cv2.imread(img_path)
                cv2.imwrite( img_path, img)
                print('FIXED corrupted image :', img_name)       
                i=i+1
    except(IOError, SyntaxError) as e :
      print(e)
      print("Unable to load/write Image : {} . Image might be destroyed".format(img_path) )


for path in os.listdir(dir_path):
    # Make sure to change the extension if it is nor 'jpg' ( for example 'JPG','PNG' etc..)
    if path.endswith('.jpg'):
      img_path = os.path.join(dir_path, path)
      detect_and_fix( img_path=img_path, img_name = path)

print("Process Finished is %d"%i)