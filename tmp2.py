import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
img=np.asarray(Image.open('/Users/seongjungkim/PycharmProjects/Pathology-prostate/Train_fg/E.png'))

shape=tf.shape(img)
img_tensor = tf.Variable(img)
padded_img_tensor=tf.image.resize_image_with_crop_or_pad(img_tensor , 700,700)
sess=tf.Session()
init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)
for i in range(5):
    cropped_tensor =  sess.run(padded_img_tensor)
    plt.imshow(cropped_tensor )
    plt.show()
