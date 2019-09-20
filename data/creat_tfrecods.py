import os
import tensorflow as tf
from PIL import Image
import random

cwd = os.getcwd()
file_dir = 'D:/Project/face/insightface-master/CROP_0.56/'
recordpath="D:/Project/face/insightface-master/datasets/train120.tfrecords"

result = {}

def create_record_list():

    for file in os.listdir(file_dir):
        filelist = []
        dataset_root = os.path.join(file_dir, file)
        for imgs in os.listdir(dataset_root):
            imgdir = os.path.join(dataset_root, imgs)
            filelist.append(imgdir)
        result[file] = {'label': file, 'img': filelist}
        '''
        name = file.split(sep='.')
        lable_val = 0
        if name[0] == 'cat':
            lable_val = 0
        else:
            lable_val = 1
        img_path = file_dir + file
        img = Image.open(img_path)
        img = img.resize((208, 208))
        img_raw = img.tobytes()  # 将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[lable_val])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))

        writer.write(example.SerializeToString())
        i=i+1
        print(i)
        '''
import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_annotation_dict(input_folder_path):
    label_dict = {}
    label = {}
    father_file_list = os.listdir(input_folder_path)
    for num, father_file in enumerate(father_file_list):
        full_father_file = os.path.join(input_folder_path, father_file)
        son_file_list = os.listdir(full_father_file)
        label[father_file] = num
        for image_name in son_file_list:
            label_dict[os.path.join(full_father_file, image_name)] = num
    save_obj(label,'D:/Project/face/insightface-master/datasets/label.txt')
    return label_dict


# 生成是数据文件
def create_record(filelist):
    # random.shuffle(filelist)
    i = 0
    writer = tf.python_io.TFRecordWriter(recordpath)
    for image_path, lable_val in filelist.items():

        img = Image.open(image_path)
        img = img.resize((112, 112))
        img_raw = img.tobytes()  # 将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[lable_val])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) #example对象对label和image进行封装

        writer.write(example.SerializeToString())
        print(image_path)
        print(lable_val)
    writer.close()

# 用队列形式读取文件
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [112, 112, 3])
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    # 使用shuffle_batch可以随机打乱输入
    X_train, y_train = tf.train.shuffle_batch([img, label],
                                              batch_size=8, capacity=600,
                                              min_after_dequeue=500)
    with tf.Session() as sess:
        for step in range(1000):
            tf.train.start_queue_runners()
            val, l = sess.run([X_train, y_train])
            print(val, l)
    return img, label

if __name__ == '__main__':
    list = get_annotation_dict(file_dir)
    # create_record_list()
    create_record(list)
    # read_and_decode(recordpath)