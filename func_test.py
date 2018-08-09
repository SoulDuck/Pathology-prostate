#-*- coding:utf-8 -*-
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from DataProvider import random_crop_shuffled_batch


# random_crop_shuffled_batch Test
# Crop Size 에 대한 조건이 달라지면 InvalidArgumentError 에러가 뜬다.
tfrecord_path = 'tmp.tfrecord'
images_op , labels_op  , fnames_op = random_crop_shuffled_batch(tfrecord_path=tfrecord_path,
                                                        batch_size = 1 , crop_size = (500, 500, 3) , num_epoch= 100)
sess=tf.Session()
init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess , coord = coord)

for i in range(100):
    images =sess.run(images_op)
    image=np.squeeze(images)
    plt.imshow(image)
    plt.show()

coord.request_stop()
coord.join(threads)


