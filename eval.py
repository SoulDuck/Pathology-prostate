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
        return classmap
        """
        plt.imshow(cam_img, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest',vmin=0, vmax=1)
        # plt.show()
        cmap = plt.cm.jet
        plt.imsave('{}/actmap_abnormal_label_0.png'.format(save_dir), cmap(cam_img))
        cam_img = Image.open('{}/actmap_abnormal_label_0.png'.format(save_dir))
        ##임시로 한것이다 나중에 299 가 아닌 224로 고쳐진 코드가 있으면 지우자
        cam_img = cam_img.resize((224, 224), PIL.Image.ANTIALIAS)
        np_cam_img = np.asarray(cam_img)  # img 2 numpy
        np_cam_img = fundus_processing.add_padding(np_cam_img.reshape(1, 224, 224, -1), 299, 299)  # padding
        cam_img = Image.fromarray(
            np_cam_img.reshape([ori_img_h, ori_img_w, 4]).astype('uint8'))  # padding#numpy 2 img

        ori_img = Image.fromarray(ori_img.astype('uint8')).convert("RGBA")
        # cam_img = Image.fromarray(cam_img.astype('uint8')).convert("RGBA")
        overlay_img = Image.blend(ori_img, cam_img, 0.5)
        plt.imshow(overlay_img)
        plt.imsave('{}/overlay.png'.format(save_dir), overlay_img)
        # plt.show()
        plt.close();
        """
        """
        if test_imgs.shape[-1] == 1:  # grey
            plt.imshow(1 -img.reshape([test_imgs.shape[1], test_imgs.shape[2]]))
            plt.show()
        """
if __name__ == '__main__':
    eval=Eval()
    eval.restore_model('saved_model/model.ckpt')
    img=np.asarray(Image.open('./Pathology-prostate/patched_train_fg/A.png'))
    img=img.reshape([1] + list(np.shape(img)))
    classmap = eval.actmap(img, None)
    print np.shape(classmap)

