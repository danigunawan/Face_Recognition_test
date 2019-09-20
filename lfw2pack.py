# import mxnet as mx
# from mxnet import ndarray as nd
# import argparse
# import pickle
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eval'))
# import lfw
#
# parser = argparse.ArgumentParser(description='Package LFW images')
# # general
# parser.add_argument('--data-dir', default='', help='')
# parser.add_argument('--image-size', type=str, default='182,182', help='')
# parser.add_argument('--output', default='', help='path to save.')
# args = parser.parse_args()
# lfw_dir = args.data_dir
# image_size = [int(x) for x in args.image_size.split(',')]
# lfw_pairs = lfw.read_pairs(os.path.join(lfw_dir, 'D:/Project/face/insightface-master/datasets/valid.txt'))
# lfw_paths, issame_list = lfw.get_paths(lfw_dir, lfw_pairs, 'jpg')
# lfw_bins = []
# #lfw_data = nd.empty((len(lfw_paths), 3, image_size[0], image_size[1]))
# i = 0
# for path in lfw_paths:
#   with open(path, 'rb') as fin:
#     _bin = fin.read()
#     lfw_bins.append(_bin)
#     #img = mx.image.imdecode(_bin)
#     #img = nd.transpose(img, axes=(2, 0, 1))
#     #lfw_data[i][:] = img
#     i+=1
#     if i%1000==0:
#       print('loading lfw', i)
#
# with open(args.output, 'wb') as f:
#   pickle.dump((lfw_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)




#coding:utf-8

import mxnet as mx
from mxnet import ndarray as nd
import argparse
import pickle
import sys
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split(',')
            pairs.append(pair)
    return np.array(pairs)


def get_paths(pairs, same_pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    cnt = 1
    for pair in pairs:
        pair = str(pair[0]).split(' ')
        path0 = pair[0]
        path1 = pair[1]

        if cnt < same_pairs:
            issame = True
        else:
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            print('not exists', path0, path1)
            nrof_skipped_pairs += 1
        cnt += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Package  images')
    # general
    parser.add_argument('--data-dir', default='', help='')
    parser.add_argument('--image-size', type=str, default='112,112', help='')
    parser.add_argument('--output', default='./out/valid.bin', help='path to save.')
    parser.add_argument('--txtfile', default='./out/valid.txt', help='txtfile path.')
    args = parser.parse_args()
    image_size = [int(x) for x in args.image_size.split(',')]
    img_pairs = read_pairs(args.txtfile)
    img_paths, issame_list = get_paths(img_pairs, 1668/2)   # 这里的15925是相同图像对的个数，需要按照实际产生的相同图像对数量替换24702/2
    img_bins = []
    i = 0
    for path in img_paths:
        with open(path, 'rb') as fin:
            _bin = fin.read()
            img_bins.append(_bin)
            i += 1
    with open(args.output, 'wb') as f:
        pickle.dump((img_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)