#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
import os  , sys , glob

def tf_writer(tfrecord_path, img_sources, labels , resize=None):
    """
    img source 에는 두가지 형태로 존재합니다 . str type 의 path 와
    numpy 형태의 list 입니다.
    :param tfrecord_path: e.g) './tmp.tfrecord'
    :param img_sources: e.g)[./pic1.png , ./pic2.png] or list flatted_imgs
    img_sources could be string , or numpy
    :param labels: 3.g) [1,1,1,1,1,0,0,0,0]
    :return:
    """
    if os.path.exists(tfrecord_path):
        print tfrecord_path + 'is exists'
        return

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    img_sources_labels = zip(img_sources, labels)
    for ind, (img_source, label) in enumerate(img_sources_labels ):
        try:
            msg = '\r-Progress : {0}'.format(str(ind) + '/' + str(len(img_sources_labels )))
            sys.stdout.write(msg)
            sys.stdout.flush()
            if isinstance(img_source , str): # img source  == str
                np_img = np.asarray(Image.open(img_source)).astype(np.int8)
                height = np_img.shape[0]
                width = np_img.shape[1]
                dirpath, filename = os.path.split(img_source)
                filename, extension = os.path.splitext(filename)
            elif type(img_source).__module__ == np.__name__: # img source  == Numpy
                np_img = img_source
                height , width = np.shape(img_source)[:2]
                filename = str(ind)
            else:
                raise AssertionError , "img_sources's element should path(str) or numpy"
            if not resize is None:
                np_img=np.asarray(Image.fromarray(np_img).resize(resize, Image.ANTIALIAS))
            raw_img = np_img.tostring() # ** Image to String **

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'raw_image': _bytes_feature(raw_img),
                'label': _int64_feature(label),
                'filename': _bytes_feature(tf.compat.as_bytes(filename))
            }))
            writer.write(example.SerializeToString())
        except IOError as ioe:
            if isinstance(img_source , str):
                print img_source
                print ioe
                exit()
            continue
        except TypeError as te:
            if isinstance(img_source , str):
                print img_source
                print te
                exit()
            continue
        except Exception as e:
            if isinstance(img_source , str):
                print img_source
                print e
                exit()
    writer.close()

def random_crop_shuffled_batch(tfrecord_path, batch_size, crop_size , num_epoch , min_after_dequeue=500):
    crop_height, crop_width = crop_size
    filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=num_epoch , name='filename_queue')
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       # Defaults are not specified since both keys are required.
                                       features={
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'raw_image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'filename': tf.FixedLenFeature([], tf.string)
                                       })
    image = tf.decode_raw(features['raw_image'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    filename = tf.cast(features['filename'], tf.string)

    image_shape = tf.stack([height , width , 3])  # image_shape shape is ..
    #image_size_const = tf.constant((resize_height, resize_width, 3), dtype=tf.int32)
    image = tf.reshape(image, image_shape)
    image = tf.random_crop(value = image, size = (crop_height,crop_width))
    image=tf.cast(image , dtype=tf.float32)
    images, labels, fnames = tf.train.shuffle_batch([image, label, filename], batch_size=batch_size, capacity=5000,
                                                    num_threads=1,
                                                    min_after_dequeue=min_after_dequeue)
    return images, labels , fnames

if __name__ == '__main__':
    fgs = glob.glob('/Users/seongjungkim/PycharmProjects/pathology/SS14-35488_B2/roi/Train/*.png')
    bgs = glob.glob('/Users/seongjungkim/PycharmProjects/pathology/SS14-35488_B2/bg/Train/*.png')
    print '# fg : {} , # bg : {} '.format(len(fgs) , len(bgs))
    # NORMAL = 0  ABNORMAL =1
    labels = np.append(np.ones(len(fgs)), np.zeros(len(bgs))).astype(np.int32)
    img_paths = fgs  + bgs
    tfrecord_path = './tmp.tfrecord'
    tf_writer(tfrecord_path=tfrecord_path , img_sources = img_paths , labels = labels )











