import tensorflow as tf
import numpy as np
class DNN(object):
    def __init__(self , img_h , img_w , img_ch , n_classes ):
        ### define tensorflow placeholder ###
        self.n_classes = n_classes
        self.x_ = tf.placeholder(dtype=tf.float32, shape=[None, img_h , img_w , img_ch], name='x_')
        self.y_ = tf.placeholder(dtype=tf.float32, shape = [None ,self.n_classes ] , name='y_')
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.optimizer_name = 'adam'
        self.l2_weight_decay = False
        self.init_lr = 0.0001
        self.global_step = 0
        self.lr_decay_step = 1000
        self.max_iter = 10000
    def convolution2d(self, name, x, out_ch, k=3, s=2, padding='SAME'):
        def _fn():
            in_ch = x.get_shape()[-1]
            filter = tf.get_variable("w", [k, k, in_ch, out_ch],
                                     initializer=tf.contrib.layers.variance_scaling_initializer())
            bias = tf.Variable(tf.constant(0.1), out_ch, name='b')
            layer = tf.nn.conv2d(x, filter, [1, s, s, 1], padding) + bias
            layer = tf.nn.relu(layer, name='relu')
            if __debug__ == True:
                print 'layer name : ', name
                print 'layer shape : ', layer.get_shape()

            return layer

        if name is not None:
            with tf.variable_scope(name) as scope:
                layer = _fn()
        else:
            layer = _fn()
        return layer
    def batch_norm_layer(self , x, phase_train, scope_bn):
        with tf.variable_scope(scope_bn):
            n_out = int(x.get_shape()[-1])
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
            if len(x.get_shape()) == 4:  # for convolution Batch Normalization
                print 'BN for Convolution was applied'
                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            if len(x.get_shape()) == 2:  # for Fully Convolution Batch Normalization:
                print 'BN for FC was applied'
                batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed
    def gap(self, x):
        gap = tf.reduce_mean(x, (1, 2) , name='gap')
        return gap
    def fc_layer_to_clssses(self, layer , n_classes):
        #layer should be flatten
        assert len(layer.get_shape()) ==2
        in_ch=int(layer.get_shape()[-1])
        with tf.variable_scope('final') as scope:
            w = tf.get_variable('w', shape=[in_ch, n_classes], initializer=tf.random_normal_initializer(0, 0.01),
                                    trainable=True)
            b = tf.Variable(tf.constant(0.1), n_classes , name='b')
            logits = tf.matmul(layer, w, name='matmul') +b
        logits=tf.identity(logits , name='logits')
        return logits
    def affine(self,name, x, out_ch, keep_prob , is_training):
        def _fn(x):
            if len(x.get_shape()) == 4:
                batch, height, width, in_ch = x.get_shape().as_list()
                w_fc = tf.get_variable('w', [height * width * in_ch, out_ch],
                                       initializer=tf.contrib.layers.xavier_initializer())
                x = tf.reshape(x, (-1, height * width * in_ch))
            elif len(x.get_shape()) == 2:
                batch, in_ch = x.get_shape().as_list()
                w_fc = tf.get_variable('w', [in_ch, out_ch], initializer=tf.contrib.layers.xavier_initializer())
            else:
                print 'x n dimension must be 2 or 4 , now : {}'.format(len(x.get_shape()))
                raise AssertionError
            print x
            b_fc = tf.Variable(tf.constant(0.1), out_ch)
            layer = tf.matmul(x, w_fc) + b_fc

            layer = tf.nn.relu(layer)
            layer =self.dropout(layer ,is_training , keep_prob)
            print 'layer name :'
            print 'layer shape :', layer.get_shape()
            print 'layer dropout rate :', keep_prob
            return layer

        if name is not None:
            with tf.variable_scope(name) as scope:
                layer = _fn(x)
        else:
            layer = _fn(x)
        return layer
    def algorithm(self, logits):
        """
        :param y_conv: logits
        :param y_: labels
        :param learning_rate: learning rate
        :return:  pred,pred_self , cost , correct_pred ,accuracy
        """

        print "############################################################"
        print "#                     Optimizer                            #"
        print "############################################################"
        print 'optimizer option : sgd | adam | momentum | '
        print 'selected optimizer : ', self.optimizer_name
        print 'logits tensor Shape : {}'.format(logits.get_shape())
        print 'Preds tensor Shape : {}'.format(self.y_.get_shape())
        print 'Learning Rate initial Value : {}'.format(self.init_lr)
        print 'Learning Decay: {}'.format(self.lr_decay_step)
        print '# max_iter : {}'.format(self.max_iter)
        print 'L2 Weight Decay : {} '.format(self.l2_weight_decay)


        optimizer_dic = {'sgd': tf.train.GradientDescentOptimizer, 'adam': tf.train.AdamOptimizer,
                         'momentum': tf.train.MomentumOptimizer}

        self.pred_op = tf.nn.softmax(logits, name='softmax')
        self.pred_self_op = tf.argmax(self.pred_op, axis=1, name='pred_self')
        self.cost_op= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y_), name='cost')
        self.lr_op = tf.train.exponential_decay(self.init_lr, self.global_step, decay_steps=int(self.max_iter / self.lr_decay_step),
                                               decay_rate=0.96,
                                               staircase=False)
        # L2 Loss
        if not self.l2_weight_decay is 0.0:
            print 'L2 Loss is Applied'
            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')
            total_cost=self.cost_op + l2_loss * self.l2_weight_decay
        else:
            print 'L2 Loss is Not Applied'
            total_cost = self.cost_op
        # Select Optimizer
        if self.optimizer_name == 'momentum':
            self.train_op = optimizer_dic[self.optimizer_name](self.lr_op, use_nesterov=True).minimize(total_cost,
                                                                                                    name='train_op')

        else:
            self.train_op = optimizer_dic[self.optimizer_name](self.lr_op).minimize(total_cost,name='train_op')
        # Prediction Op , Accuracy Op
        self.correct_pred_op = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y_, 1), name='correct_pred')
        self.accuracy_op = tf.reduce_mean(tf.cast(self.correct_pred_op, dtype=tf.float32), name='accuracy')




class VGG(DNN):
    def __init__(self , model):

        super(VGG ,self ).__init__(500,500,3 , n_classes=2)
        print self.x_
        self.model = model
        self.use_BN = True
        self.logit_type = 'gap'
        self.build_graph()
        self.algorithm(self.logits)

    def build_graph(self ):
        ##### define conv connected layer #######
        image_size = int(self.x_.get_shape()[-2])
        n_classes = int(self.x_.get_shape()[-1])
        if self.model == 'vgg_11':
            print 'Model : {}'.format('vgg 11')
            conv_out_features = [64, 128, 256, 256, 512, 512, 512, 512]
            conv_kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
            conv_strides = [1, 1, 1, 1, 1, 1, 1, 1]
            before_act_bn_mode = [False, False, False, False, False, False, False, False, False]
            after_act_bn_mode = [False, False, False, False, False, False, False, False, False]
            if self.use_BN == True:
                before_act_bn_mode = [True, True, True, True, True, True, True, True]
            allow_max_pool_indices = [0, 1, 2, 3, 5, 7]

        elif self.model == 'vgg_13':
            conv_out_features = [64, 64, 128, 128, 256, 256, 512, 512, 512, 512]
            conv_kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            conv_strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            before_act_bn_mode = [False, False, False, False, False, False, False, False, False, False]
            after_act_bn_mode = [False, False, False, False, False, False, False, False, False, False]
            if self.use_BN == True:
                before_act_bn_mode = [True, True, True, True, True, True, True, True, True, True]
            allow_max_pool_indices = [1, 3, 5, 7, 9]

        elif self.model == 'vgg_16':
            conv_out_features = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            conv_kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            conv_strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            before_act_bn_mode = [False, False, False, False, False, False, False, False, False, False, False, False, False]
            after_act_bn_mode = [False, False, False, False, False, False, False, False, False, False, False, False, False]
            if self.use_BN == True:
                before_act_bn_mode = [True, True, True, True, True, True, True, True, True, True, True, True, True]

            allow_max_pool_indices = [1, 3, 6, 9, 12]

        elif self.model == 'vgg_19':
            conv_out_features = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
            conv_kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            conv_strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            before_act_bn_mode = [False, False, False, False, False, False, False, False, False, False, False, False, False,
                                  False, False, False]
            after_act_bn_mode = [False, False, False, False, False, False, False, False, False, False, False, False, False,
                                 False, False, False]
            if self.use_BN == True:
                before_act_bn_mode = [True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                      True, True, True]
            allow_max_pool_indices = [1, 3, 7, 9, 11, 15]
        else:
            print '[vgg_11 , vgg_13 , vgg_16 , vgg_19]'
            raise AssertionError

        ###VGG Paper ###
        """
        VGG-11 64 max 128 max 256 256 max 512 512 max 512 512 max 4096 4096 1000  
        VGG-11 64 LRN max 128 max 256 256 max 512 512 max 512 512 max 4096 4096 1000
        VGG-13 64 64 LRN max 128 128 max 256 256 max 512 512 max 512 512 max 4096 4096 1000
        VGG-16 64 64 LRN max 128 128 max 256 256 256 max 512 512 512 max 512 512 512 max 4096 4096 1000
        VGG-16 64 64 LRN max 128 128 max 256 256 256 max 512 512 512 max 512 512 512 max 4096 4096 1000
        VGG-16 64 64 LRN max 128 128 max 256 256 256 256 max 512 512 512 512 max 512 512 512 512 max 4096 4096 1000
    
        """
        print '###############################################################'
        print '#                            {}'.format(self.model), '                          #'
        print '###############################################################'
        layer = self.x_
        for i in range(len(conv_out_features)):
            with tf.variable_scope('conv_{}'.format(str(i))) as scope:
                # Apply Batch Norm
                if before_act_bn_mode[i] == True:
                    layer = self.batch_norm_layer(layer, self.is_training, 'before_BN')
                # Apply Convolution
                layer = self.convolution2d(name=None, x=layer, out_ch=conv_out_features[i], k=conv_kernel_sizes[i],
                                           s=conv_strides[i], padding="SAME")
                # Apply Max Pooling Indices
                if i in allow_max_pool_indices:
                    print 'max pooling layer : {}'.format(i)
                    layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    print layer
                layer = tf.nn.relu(layer)
                # Apply Batch Norm
                if after_act_bn_mode[i] == True:
                    layer = self.batch_norm_layer(layer, self.is_training, 'after_BN')
                # Apply Dropout layer=tf.nn.dropout(layer , keep_prob=conv_keep_prob)
                layer = tf.cond(self.is_training, lambda: tf.nn.dropout(layer, keep_prob=1.0), lambda: layer)

        self.top_conv = tf.identity(layer, name='top_conv')

        if self.logit_type == 'gap':
            layer = self.gap(self.top_conv)
            self.logits = self.fc_layer_to_clssses(layer, self.n_classes)

        elif self.logit_type == 'fc':
            fc_features = [4096, 4096]
            before_act_bn_mode = [False, False]
            after_act_bn_mode = [False, False]
            self.top_conv = layer
            for i in range(len(fc_features)):
                with tf.variable_scope('fc_{}'.format(str(i))) as scope:
                    print i
                    if before_act_bn_mode[i]:
                        layer = self.batch_norm_layer(layer, self.is_training, 'bn')
                    layer = self.affine(name=None, x=layer, out_ch=fc_features[i], keep_prob=0.5,
                                        is_training=self.is_training)
                    if after_act_bn_mode[i]:
                        layer = self.batch_norm_layer(layer, self.is_training, 'bn')
            self.logits = self.fc_layer_to_clssses(layer, self.n_classes)
        else:
            print '["fc", "gap"]'
            raise AssertionError
        return self.logits


if __name__ == '__main__':
    vgg=VGG('vgg_11')
    from DataProvider import random_crop_shuffled_batch
    from utils import cls2onehot
    # random_crop_shuffled_batch Test
    tfrecord_path = 'tmp.tfrecord'
    images_op, labels_op, fnames_op = random_crop_shuffled_batch(tfrecord_path=tfrecord_path,
                                                                 batch_size=20, crop_size=(500, 500, 3), num_epoch=30)
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(100):
        print i
        images , labels = sess.run([images_op , labels_op])
        print np.shape(images)
        labels =cls2onehot(labels ,2 )
        cost, _ = sess.run([vgg.cost_op, vgg.train_op],
                           feed_dict={vgg.x_: images, vgg.y_: labels, vgg.is_training: True})
    coord.request_stop()
    coord.join(threads)


