import tensorflow as tf
import numpy as np
import cv2
import os
import sklearn
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet

def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]

def cv_imread(path):
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return cv_img


if __name__ == '__main__':
    sess = tf.Session()

    # saver = tf.train.import_meta_graph('./model/InsightFace_iter_20001.ckpt.meta')
    # saver.restore(sess, './model/InsightFace_iter_20001.ckpt')
    # graph = tf.get_default_graph()  # 获得默认的图
    # input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    faces = np.load('./embedding_array.npy')
    mylabels = np.load('./label_array.npy')

    # images = sess.graph.get_tensor_by_name("img_inputs:0")
    # dropout_rate = sess.graph.get_tensor_by_name("dropout_rate:0")
    # output = sess.graph.get_tensor_by_name("resnet_v1_50/E_BN2/Identity_2:0")

    images = tf.placeholder(name='img_inputs', shape=[None, 112, 112, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = get_resnet(images, 50, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
    output = net.outputs

    saver = tf.train.Saver()
    saver.restore(sess, './model/InsightFace_iter_20001.ckpt')


    k = 1.55
    v = 8
    file_list = os.walk('D:/Project/face/INSIGHT_FACE/datasets/CROP_0.56')
    print('*' * 60)
    for temp in file_list:
        for filename in temp[2]:
            file = os.path.join(temp[0], filename)
            # file = 'D:/Project/face/INSIGHT_FACE/datasets/CROP_0.56/angelababy/34.png'
            img = cv_imread(file)
            # if img is None:
            #     break
            img = cv2.resize(img, (112, 112))
            orring = np.fliplr(img)


            rgb_face = np.expand_dims(img, 0)
            rgb_face = rgb_face.astype(np.float32)
            rgb_face -= 127.5
            rgb_face *= 0.0078125
            f1 = sess.run(output, {images: rgb_face, dropout_rate: 1.0})

            orrrgb_face = np.expand_dims(orring, 0)
            orrrgb_face = orrrgb_face.astype(np.float32)
            orrrgb_face -= 127.5
            orrrgb_face *= 0.0078125
            f2 = sess.run(output, {images: orrrgb_face, dropout_rate: 1.0})

            f1 = f1+f2
            # f1 = np.reshape(f1, [1, 512])
            f1 = sklearn.preprocessing.normalize(f1)
            flag = False
            result = {}
            for i in range(faces.shape[0]):
                cnt = 0
                for j in range(faces.shape[1]):
                    dist = np.sum(np.square(np.subtract(f1, faces[i][j])), 1)
                    # 如果与库中某人距离小于阈值1
                    if dist < k:
                        cnt += 1
                # 如果与库中某个人的相似度大于阈值2，则证明是该人
                if cnt >= v:
                    # name = mylabels[i]
                    result[i] = cnt
                    # print(name)
                    flag = True
            if len(result)>0:
                listt = get_key(result, max(result.values()))
                for i in listt:
                    name = mylabels[i]
                # name = mylabels[list(result.keys())[list(result.values()).index(max(result.values()))]]
                    print(file + 'is '+name)
            if flag is False:
                print("image:" + file + 'is not in the gallery, refused!' + '\n')
            print('*' * 60)
