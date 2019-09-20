import tensorflow as tf
import os
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet

def cv_imread(path):
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return cv_img


sess = tf.Session()
# saver = tf.train.import_meta_graph('./model/InsightFace_iter_20001.ckpt.meta')
# saver.restore(sess, './model/InsightFace_iter_20001.ckpt')
# images = sess.graph.get_tensor_by_name("img_inputs:0")
# labels = sess.graph.get_tensor_by_name("img_labels:0")
# dropout_rate = sess.graph.get_tensor_by_name("dropout_rate:0")
# output = sess.graph.get_tensor_by_name("resnet_v1_50/E_BN2/Identity_2:0")

images = tf.placeholder(name='img_inputs', shape=[None, 112, 112, 3], dtype=tf.float32)
labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
net = get_resnet(images, 50, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
output = net.outputs

saver = tf.train.Saver()
saver.restore(sess, './model/InsightFace_iter_best_710000.ckpt')


embedding_array = np.zeros((278, 10, 512))
label_array = np.zeros((278, 1))
file_list = 'D:/Project/face/INSIGHT_FACE/datasets/CROP_0.95'

classes = [path for path in os.listdir(file_list) if os.path.isdir(os.path.join(file_list, path))]

label_strings = [name for name in classes if os.path.isdir(os.path.join(file_list, name))]

list = os.listdir(file_list)
for i in range(0, len(list)):
    path = os.listdir(os.path.join(file_list, list[i]))
    for j in range(0, 10):
        filename = os.path.join(os.path.join(file_list, list[i]), path[j])
        img = cv_imread(filename)
        if img is None:
            break
        img = cv2.resize(img, (112, 112))
        orring = np.fliplr(img)

        rgb_face = np.expand_dims(img, 0)
        rgb_face = rgb_face.astype(np.float32)
        rgb_face -= 127.5
        rgb_face *= 0.0078125
        f1 = sess.run(output, {images: rgb_face, dropout_rate:1.0})

        orrrgb_face = np.expand_dims(orring, 0)
        orrrgb_face = orrrgb_face.astype(np.float32)
        orrrgb_face -= 127.5
        orrrgb_face *= 0.0078125
        f2 = sess.run(output, {images: orrrgb_face, dropout_rate: 1.0})

        f1 = f1+f2
        f1 = sklearn.preprocessing.normalize(f1)

        embedding_array[i, j, :] = f1
label_strings = np.array(label_strings)
np.save('./embedding_array.npy', embedding_array)
np.save('./label_array.npy', label_strings)