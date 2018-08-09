#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
import os  , sys , glob
import shutil


def reconstruct_tfrecord_rawdata(tfrecord_path, resize , ):
    debug_flag_lv0 = False
    debug_flag_lv1 = False
    if __debug__ == debug_flag_lv0:
        print 'debug start | batch.py | class tfrecord_batch | reconstruct_tfrecord_rawdata '

    print 'now Reconstruct Image Data please wait a second'
    print 'Resize {}'.format(resize)
    reconstruct_image = []
    # caution record_iter is generator
    record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)

    ret_img_list = []
    ret_lab_list = []
    ret_filename_list = []
    for i, str_record in enumerate(record_iter):
        msg = '\r -progress {0}'.format(i)
        sys.stdout.write(msg)
        sys.stdout.flush()
        example = tf.train.Example()
        example.ParseFromString(str_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
        label = int(example.features.feature['label'].int64_list.value[0])
        filename = (example.features.feature['filename'].bytes_list.value[0])
        image = np.fromstring(raw_image, dtype=np.uint8)
        image = image.reshape((height, width, -1))
        if not resize is None:
            image = np.asarray(Image.fromarray(image).resize(resize, Image.ANTIALIAS))
        ret_img_list.append(image)
        ret_lab_list.append(label)
        ret_filename_list.append(filename)

    ret_img = np.asarray(ret_img_list)
    ret_lab = np.asarray(ret_lab_list)
    if debug_flag_lv1 == True:
        print ''
        print 'images shape : ', np.shape(ret_img)
        print 'labels shape : ', np.shape(ret_lab)
        print 'length of filenames : ', len(ret_filename_list)
    return ret_img, ret_lab, ret_filename_list

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
                np_img = np.asarray(Image.open(img_source).convert('RGB')).astype(np.int8)
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


def tf_padder(img_paths , height , width  , save_folder):
    ret_images = []
    print '## TF Graph was Reset!! ##'
    tf.reset_default_graph()
    sess = tf.Session()
    img_tensor = tf.placeholder(dtype=tf.uint8, shape=[None, None , 3])
    padded_image_op = tf.image.resize_image_with_crop_or_pad(img_tensor ,  height  , width )
    import matplotlib.pyplot as plt
    for path in img_paths:
        img = np.asarray(Image.open(path)).astype(np.uint8)
        assert img.max > 1 , 'Image pixel range 0 ~ 255'
        padded_image = sess.run( fetches =  padded_image_op , feed_dict={img_tensor : img})
        if not save_folder is None:
            name = os.path.split(path)[-1]
            savepath = os.path.join(save_folder, name)
            Image.fromarray(padded_image).convert('RGB').save(savepath)

            #plt.imsave( savepath ,padded_image )
        ret_images.append(padded_image)

    return ret_images


def copy_images(src_paths , dst_dirname ):
    for path in src_paths:
        name= os.path.split(path)[-1]
        dst_path =os.path.join(dst_dirname , name)
        shutil.copy(path, dst_path)


def size_checker(paths , height, width):
    larger =[]
    smaller = []
    for path in paths:
        img=Image.open(path)
        w,h=img.size
        if w <= width and h <= height:
            smaller.append(path)
        else:
            larger.append(path)
    return smaller , larger

def image_channel_checker(paths , channel = 'RGB'):
    for path in paths:
        img = Image.open(path)
        print np.shape(img)




def random_crop_shuffled_batch(tfrecord_path, batch_size, crop_size , num_epoch , min_after_dequeue=10):
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

    image = tf.random_crop(value = image, size = crop_size)
    image=tf.cast(image , dtype=tf.float32)
    images, labels, fnames = tf.train.shuffle_batch([image, label, filename], batch_size=batch_size, capacity=200,
                                                    num_threads=1,
                                                    min_after_dequeue=min_after_dequeue)
    return images, labels , fnames

if __name__ == '__main__':
    # Paddding Test
    paths = glob.glob('./Train_fg/*.png')
    # 지정한 사이즈 보다 작은 paths 들을 분리합니다
    smaller , larger = size_checker(paths , 650 ,650)
    # 지정한 사이즈 보다 작으면 padding 을 붙입니다
    imgs = tf_padder(smaller  , height = 650 , width = 650, save_folder='patched_train_fg')
    # 만약 크면 그냥 저장합니다
    copy_images(larger , 'patched_train_fg')

    fgs = glob.glob('patched_train_fg/*.png')
    bgs = glob.glob('Train_bg/*.png')
    print '# fg : {} , # bg : {} '.format(len(fgs) , len(bgs))

    # NORMAL = 0  ABNORMAL =1
    img_paths = fgs *3 + bgs
    labels = np.append(np.ones(len(fgs)*3), np.zeros(len(bgs))).astype(np.int32)

    print len(img_paths)
    print len(labels)

    #image_channel_checker(img_paths)
    tfrecord_path = './tmp.tfrecord'
    tf_writer(tfrecord_path=tfrecord_path , img_sources = img_paths , labels = labels )
    images, labels ,fnames = reconstruct_tfrecord_rawdata(tfrecord_path, None)









