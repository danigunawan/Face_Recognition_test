# face_test
根据arcface loss 写的一个face recognition程序

step1：
  运行align/face_align.py,它根据mtcnn检测的点，将数据处理成人脸正向且居中的数据。
  fg：生成../datasets/CROP_0.95_align目录下：
  CROP_0.95_align/
      张艺谋/1.png,2.png....
      赵薇/1.png,2.png....
      赵本山/1.png,2.png....
      .
      .
      .
      
      
step2：
   creat_tfrecods.py 制作tfrecoads数据代码。在对应目录下会生成CROP_0.95_align.tfrecords，label.txt.pkl两个文件，其中label.txt.pkl是label字典文     件，因为在tfrecord文件中，我们只知道label，但是并不知道label顺序，所以当最后预测的时候，比如输出一个120，如果没有字典文件，我们不知道120对应的哪个     label。

step3：
generate_image_pairs.py 制作验证集的文件对。生成valid.txt
lfw2pack.py 根据验证集文件对制作验证集bin文件。valid.bin，用来验证模型效果的，里面都是一对一对的，如果同一个人脸为0，非同一人脸为1.

step4：
train_nets.py 训练模型。

step5：
eval_ckpt_file.py 读取模型，读取制作好的验证集bin文件，找到能最好分类人脸的阈值（从0--4）之间。

step6：
create_feature.py 读取数据，根据训练好的模型，生成512维特征，此特征用来作为库，判断图片是否是库中的人。

step7：
main.py 根据模型以及阈值，测试某图像属于哪个类。
