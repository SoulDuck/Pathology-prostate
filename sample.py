import numpy as np
from PIL import Image
img=Image.open('A.png')
np_img = np.asarray(img)

# Image shape
print np.shape(np_img)
print np.mean(np_img[:,:,0])
print np.mean(np_img[:,:,1])
print np.mean(np_img[:,:,2])


crop_img_size = (700,700)


class Cropper(object):
    def __init__(self , img_path , stride):














