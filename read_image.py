import os
import glob
import random
import numpy as np
# 图片数据文件夹，子文件为类别

def create_image_lists(INPUT_DATA):
    result = {}
    num = 0
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir == True:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png']
        file_list = []
        dir_name = os.path.basename(sub_dir)

        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))

        num = num + len(file_list)
        label_name = dir_name.lower()
        # 初始化当前类别的训练数据集、测试数据集和验证数据集
        training_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)  # 获取当前文件名
            training_images.append(base_name)

        # 将当前类别的数据放入结果字典
        result[label_name] = {'dir': dir_name, 'training': training_images}
    return result, num



# 通过类别名称、所属数据集和图片编号获取一张图片的地址
# image_lists为所有图片信息，image_dir给出根目录，label_name为类别名称，index为图片编号，category指定图片是在哪个训练集
def get_image_path(image_lists, image_dir, label_name, index, category):
    # 获取给定类别中所有图片的信息
    label_lists = image_lists[label_name]
    # 根据所属数据集的名称获取集合中的全部图片信息
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # 获取图片的文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    # 最终的地址为数据根目录的地址加上类别的文件夹加上图片的名称
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


# 通过类别名称、所属数据集和图片编号经过inception-v3处理之后的特征向量文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    CACHE_DIR = 'D:/Project/face/INSIGHT_FACE/datasets/CROP_0.56/'
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category)


# 使用加载的训练好的网络处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    # 将当前图片作为输入，计算瓶颈张量的值
    # 这个张量的值就是这张图片的新的特征向量
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个一维数组
    bottleneck_values = np.squeeze(bottleneck_values)  # 从数组的形状中删除单维条目
    return bottleneck_values




import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_keys(d, value):
    return [k for k,v in d.items() if v == value][0]

import cv2
def cv_imread(path):
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return cv_img

import tensorflow as tf
# 随机选取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(n_classes, image_lists, how_many, category, image_size):

    ground_truths = []
    ground_labels = []
    label = load_obj('D:/Project/face/INSIGHT_FACE/datasets/label.txt')
    for _ in range(how_many):
        # 随机一个类别和图片的编号加入当前的训练数据
        label_index = random.randrange(n_classes)  # 返回指定递增基数集合中的一个随机数，基数缺省值为1，随机类别号
        # label_name = list(image_lists.keys())[label_index]
        label_name = get_keys(label, label_index)
        image_index = random.randrange(65536)
        bottleneck_path = get_bottleneck_path(image_lists, label_name, image_index, category)
        # 改变图像大小

        im1 = cv_imread(bottleneck_path)
        # im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2 = cv2.resize(im1, (image_size[0],image_size[1]))  # 为图片重新指定尺寸
        # im2 = tf.reshape(im2, [width, height, 1])
        # im2 = np.reshape(im2, [-1])
        # im2 = tf.cast(im2, tf.float32)

        # ground_truth = np.zeros(n_classes, dtype=np.float32)
        # ground_truth[label_index] = 1

        ground_truths.append(im2)
        ground_labels.append(label_index)
    ground_truths = np.reshape(ground_truths, [how_many, image_size[0],image_size[1], 3])
    # ground_labels = np.reshape(ground_labels, [how_many])
    # ground_truths = tf.cast(ground_truths, tf.float32)
    return ground_truths, ground_labels