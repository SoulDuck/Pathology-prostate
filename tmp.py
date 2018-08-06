import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img=np.asarray(Image.open('A.png'))
A = tf.Variable(img)
A_crop_op = tf.random_crop(A ,(700 ,700 , 3 ))

sess=tf.Session()
init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)

coord =

A_crop = sess.run(A_crop_op)
plt.imshow(A_crop)
plt.show()



