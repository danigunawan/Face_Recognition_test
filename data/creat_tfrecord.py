import os
import tensorflow as tf
import io
from PIL import Image
import json


flags = tf.app.flags
flags.DEFINE_string('images_dir',
                    'D:/Project/face/insightface-master/CROP_0.56/',
                    'Path to image(directory)')
flags.DEFINE_string('annotation_path',
                    'D:/Project/face/insightface-master/CROP_0.56/annotations.json',
                    'Path to annotation')
flags.DEFINE_string('record_path',
                    'D:/Project/face/insightface-master/CROP_0.56/train.record',
                    'Path to TFRecord')
FLAGS = flags.FLAGS


def get_annotation_dict(input_folder_path, word2number_dict):
    label_dict = {}
    father_file_list = os.listdir(input_folder_path)
    for num, father_file in enumerate(father_file_list):
        full_father_file = os.path.join(input_folder_path, father_file)
        son_file_list = os.listdir(full_father_file)
        for image_name in son_file_list:
            label_dict[os.path.join(full_father_file, image_name)] = num
    return label_dict


def save_json(label_dict, json_path):
    with open(json_path, 'w') as json_path:
        json.dump(label_dict, json_path)
    print("label json file has been generated successfully!")



def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def process_image_channels(image):
    process_flag = False
    # process the 4 channels .png
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
        process_flag = True
    # process the channel image
    elif image.mode != 'RGB':
        image = image.convert("RGB")
        process_flag = True
    return image, process_flag

def process_image_reshape(image, resize):
    width, height = image.size
    if resize is not None:
        if width > height:
            width = int(width * resize / height)
            height = resize
        else:
            width = resize
            height = int(height * resize / width)
        image = image.resize((width, height), Image.ANTIALIAS)
    return image

def create_tf_example(image_path, label, resize=None):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encode_jpg = fid.read()
    encode_jpg_io = io.BytesIO(encode_jpg)
    image = Image.open(encode_jpg_io)
    # process png pic with four channels
    image, process_flag = process_image_channels(image)
    # reshape image
    image = process_image_reshape(image, resize)
    if process_flag == True or resize is not None:
        bytes_io = io.BytesIO()
        image.save(bytes_io, format='JPEG')
        encoded_jpg = bytes_io.getvalue()
    width, height = image.size
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': bytes_feature(encode_jpg),
                'image/format': bytes_feature(b'jpg'),
                'image/class/label': int64_feature(label),
                'image/height': int64_feature(height),
                'image/width': int64_feature(width)
            }
        ))
    return tf_example

def generate_tfrecord(annotation_dict, record_path, resize=None):
    num_tf_example = 0
    writer = tf.python_io.TFRecordWriter(record_path)
    for image_path, label in annotation_dict.items():
        if not tf.gfile.GFile(image_path):
            print("{} does not exist".format(image_path))
        tf_example = create_tf_example(image_path, label, resize)
        writer.write(tf_example.SerializeToString())
        num_tf_example += 1
        if num_tf_example % 100 == 0:
            print("Create %d TF_Example" % num_tf_example)
    writer.close()
    print("{} tf_examples has been created successfully, which are saved in {}".format(num_tf_example, record_path))

def main(_):
    word2number_dict = {
        "combinations": 0,
        "details": 1,
        "sizes": 2,
        "tags": 3,
        "models": 4,
        "tileds": 5,
        "hangs": 6
    }
    images_dir = FLAGS.images_dir
    # annotation_path = FLAGS.annotation_path
    record_path = FLAGS.record_path
    annotation_dict = get_annotation_dict(images_dir, word2number_dict)
    generate_tfrecord(annotation_dict, record_path)

if __name__ == '__main__':
    tf.app.run()