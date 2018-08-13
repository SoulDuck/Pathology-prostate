#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
# Reconstruct Model
class Eval(object):
    def __init__(self):
        pass;
    # Restore Model
    def restore_model(self , model_path):
        print 'Restore Model at {}'.format(model_path)
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(meta_graph_or_file=model_path+'.meta') #example model path ./models/fundus_300/5/model_1.ckpt
        saver.restore(self.sess, save_path=model_path) # example model path ./models/fundus_300/5/model_1.ckpt

        self.x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
        self.y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
        self.pred_ = tf.get_default_graph().get_tensor_by_name('softmax:0')
        self.is_training =tf.get_default_graph().get_tensor_by_name('is_training:0')
        self.top_conv = tf.get_default_graph().get_tensor_by_name('top_conv:0')
        self.logits = tf.get_default_graph().get_tensor_by_name('logits:0')

        self.cam_label = tf.get_default_graph().get_tensor_by_name('cam_label:0')
        self.classmap = tf.get_default_graph().get_tensor_by_name('classmap:0')

    # Change Input placeholder to [ None , None , None , 3 ]
    def change_node(self):
        raise NotImplementedError
    def actmap(self , test_img , savepath):
        test_img = test_img if (np.ndim(test_img) !=3 ) else test_img.reshape([[1] + list(test_img) ]);
        classmap = self.sess.run([self.classmap],
                                 feed_dict={self.x_: test_img, self.is_training: False, self.cam_label: 0})
        classmap = np.asarray((map(lambda x: (x - x.min()) / (x.max() - x.min()), classmap)))  # -->why need this?
        classmap = np.squeeze(classmap)
        return classmap

if __name__ == '__main__':
    eval=Eval()
    eval.restore_model('saved_model/model.ckpt')
    img=np.asarray(Image.open('./sample.png').convert('RGB'))

    img=img.reshape([1] + list(np.shape(img)))
    classmap = eval.actmap(img, None)
    plt.imshow(classmap)
    plt.imsave('tmp_classmap.png' , classmap)
    plt.show()

