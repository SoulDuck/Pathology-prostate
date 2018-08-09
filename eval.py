import tensorflow as tf
import numpy as np


# Reconstruct Model
class Eval():
    def __init__(self):
        pass;
    # Restore Model
    def restore_model(self , model_path):
        print 'Restore Model at {}'.format(model_path)
        sess = tf.Session()
        saver = tf.train.import_meta_graph(meta_graph_or_file=model_path+'.meta') #example model path ./models/fundus_300/5/model_1.ckpt
        saver.restore(sess, save_path=model_path) # example model path ./models/fundus_300/5/model_1.ckpt

        self.x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
        self.y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
        self.pred_ = tf.get_default_graph().get_tensor_by_name('softmax:0')
        self.is_training_=tf.get_default_graph().get_tensor_by_name('is_training:0')
        self.top_conv = tf.get_default_graph().get_tensor_by_name('top_conv:0')
        self.logits = tf.get_default_graph().get_tensor_by_name('logits:0')
        self.cam_w = tf.get_default_graph().get_tensor_by_name('final/w:0')
        self.cam_b = tf.get_default_graph().get_tensor_by_name('final/b:0')

    # Change Input placeholder to [None , None ,None , 3 ]
    def change_node(self):
        pass ;
    # Get Activation Map






    # Visualization




    # Calculate

