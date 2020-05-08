"""
制作训练集的tfrecord文件
使用COVID19类，获取数据，之后按照将获取到的数据处理成tfrecord的存储格式，写入到tfrecord，保存到磁盘
"""
import os
import numpy as np
import tensorflow as tf
from Utils import COVID19


def serialize_example(x, mask, lung_mask):
    """converts x, mask, lung_mask to tf.train.Example and serialize"""
    # 1.构建用于创建Feature对象的 list
    # 注意，value只接受一维数组，所以要将二维的图片x，reshape成一维，或者将二维图片转成字符串
    # 同理，mask和lung_mask同上
    input_features = tf.train.BytesList(value=[x.tobytes()])
    mask_ = tf.train.BytesList(value=[mask.tobytes()])
    lung_mask_ = tf.train.BytesList(value=[lung_mask.tobytes()])
    shape = tf.train.Int64List(value=list(x.shape))
    # 2.构建Feature对象
    features = tf.train.Features(
        feature={
            "image": tf.train.Feature(bytes_list=input_features),
            "mask": tf.train.Feature(bytes_list=mask_),
            "lung_mask": tf.train.Feature(bytes_list=lung_mask_),
            "shape": tf.train.Feature(int64_list=shape)
        }
    )
    # 3.Features组建Example
    example = tf.train.Example(features=features)
    # 4.将Example序列化
    return example.SerializeToString()


def main(Covid19_dir, TFRecord_save_dir):
    # 存储到多个tfrecord文件中
    batch_size = 10
    covid19 = COVID19.Covid19(Covid19_dir, batch_size=batch_size)

    # 定义存储的tfrecord文件名，压缩格式
    # 定义存储路径 dir
    dir = TFRecord_save_dir
    if not os.path.exists(dir):
        os.mkdir(dir)
    # 500个样本存储成一个tfrecord文件
    num_each_file = 500
    total_file_num = int(np.ceil(covid19.length/num_each_file))
    filename_zip = "train-{:02d}-of-{:02d}.zip"

    # 定义压缩的可选项options
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    # 打开文件进行写入
    counter = 0
    for now_num in range(total_file_num):
        file_name = filename_zip.format(now_num, total_file_num)
        print(file_name)
        name = os.path.join(dir, file_name)
        with tf.io.TFRecordWriter(name, options=options) as writer:
            for i in range(int(num_each_file/batch_size)):  # 每一个tfrecord文件需要调用next_batch()的次数 500/10 = 50
                counter = counter+1
                batch_x, batch_mask, batch_lung_mask = covid19.next_batch()
                # 1.创建用于Feature对象的List
                # 将batch中的样本逐个取出
                for x, mask, lung_mask in zip(batch_x, batch_mask, batch_lung_mask):
                    # 每次写入一个样本的
                    writer.write(serialize_example(x, mask, lung_mask))
        print("counter:", counter)
    print("counter*batch_size:", counter*batch_size)
    print("covid19.length:", covid19.length)


if __name__ == "__main__":
    # 训练集的TFRecord生成
    Covid19_dir = "D:\\data\\COVID19\\"
    TFRecord_save_dir = "D:\\data\\COVID19\\TFRecord"
    # 调用main传入两个dir即可自动完成数据转换成TFRecord格式
    main(Covid19_dir, TFRecord_save_dir)

