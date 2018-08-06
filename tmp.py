import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from DataProvider import random_crop_shuffled_batch

img=np.asarray(Image.open('A.png'))
A = tf.Variable(img)
A_crop_op = tf.random_crop( A ,(700, 700 ,3))
tfrecord_path = 'tmp.tfrecord'
images_op , labels_op  , fnames_op = random_crop_shuffled_batch(tfrecord_path=tfrecord_path,
                                                        batch_size = 30 , crop_size = (700, 700, 3) , num_epoch= 10)
sess=tf.Session()
init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess , coord = coord)

images =sess.run(images_op)
coord.request_stop()
coord.join(threads)


